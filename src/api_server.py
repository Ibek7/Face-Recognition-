"""
FastAPI server for face recognition system.
Provides REST API endpoints for face recognition operations.
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse, PlainTextResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import numpy as np
import cv2
import io
import base64
from PIL import Image
import tempfile
import os
import logging
from datetime import datetime
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST

# Rate limiting middleware (simple in-memory)
try:
    from rate_limiter import RateLimitMiddleware
except Exception:
    RateLimitMiddleware = None

# Prometheus metrics
REQUEST_COUNT = Counter('api_requests_total', 'Total API requests', ['method', 'endpoint', 'status'])
REQUEST_DURATION = Histogram('api_request_duration_seconds', 'API request duration', ['method', 'endpoint'])
ACTIVE_CONNECTIONS = Gauge('api_active_connections', 'Number of active connections')
RECOGNITION_COUNT = Counter('face_recognition_total', 'Total face recognition operations')
DETECTION_COUNT = Counter('face_detection_total', 'Total face detections', ['status'])

# Import face recognition components
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from database import DatabaseManager
from embeddings import FaceEmbeddingManager
from realtime import RealTimeFaceRecognizer
from monitoring import performance_monitor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pydantic models for API
class PersonCreate(BaseModel):
    name: str
    description: Optional[str] = ""

class PersonResponse(BaseModel):
    id: int
    name: str
    description: str
    created_at: str
    is_active: bool

class FaceRecognitionRequest(BaseModel):
    image_base64: str
    threshold: Optional[float] = 0.7
    top_k: Optional[int] = 5

class FaceRecognitionResult(BaseModel):
    face_id: int
    person_name: Optional[str]
    person_id: Optional[int]
    confidence: float
    quality_score: float
    bounding_box: List[int]

class FaceRecognitionResponse(BaseModel):
    success: bool
    faces: List[FaceRecognitionResult]
    processing_time: float
    message: Optional[str] = None

class SystemStats(BaseModel):
    total_persons: int
    total_embeddings: int
    total_recognitions: int
    avg_processing_time: float
    uptime: str

# FastAPI app configuration
app = FastAPI(
    title="Face Recognition API",
    description="Comprehensive face recognition system with real-time capabilities",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Attach rate limiter middleware if available
if RateLimitMiddleware is not None:
    app.add_middleware(RateLimitMiddleware)

# Global components
db_manager = None
embedding_manager = None
server_start_time = datetime.now()

def get_database():
    """Dependency to get database manager."""
    global db_manager
    if db_manager is None:
        db_manager = DatabaseManager("sqlite:///face_recognition_api.db")
    return db_manager

def get_embedding_manager():
    """Dependency to get embedding manager."""
    global embedding_manager
    if embedding_manager is None:
        embedding_manager = FaceEmbeddingManager(encoder_type="simple")
    return embedding_manager

def decode_base64_image(base64_string: str) -> np.ndarray:
    """
    Decode base64 image string to numpy array.
    
    Args:
        base64_string: Base64 encoded image
        
    Returns:
        Image as numpy array
    """
    try:
        # Remove data URL prefix if present
        if "," in base64_string:
            base64_string = base64_string.split(",")[1]
        
        # Decode base64
        image_data = base64.b64decode(base64_string)
        
        # Convert to PIL Image
        pil_image = Image.open(io.BytesIO(image_data))
        
        # Convert to numpy array (BGR format for OpenCV)
        image_array = np.array(pil_image)
        if len(image_array.shape) == 3:
            image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
        
        return image_array
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image data: {str(e)}")

def encode_image_to_base64(image: np.ndarray) -> str:
    """
    Encode numpy image array to base64 string.
    
    Args:
        image: Image as numpy array
        
    Returns:
        Base64 encoded image string
    """
    # Convert BGR to RGB for PIL
    if len(image.shape) == 3:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        image_rgb = image
    
    # Convert to PIL Image
    pil_image = Image.fromarray(image_rgb)
    
    # Encode to base64
    buffer = io.BytesIO()
    pil_image.save(buffer, format='JPEG')
    image_base64 = base64.b64encode(buffer.getvalue()).decode()
    
    return f"data:image/jpeg;base64,{image_base64}"

# API Routes

@app.on_event("startup")
async def startup_event():
    """Initialize monitoring and components on startup."""
    performance_monitor.start_system_monitoring()
    logger.info("Face Recognition API server started")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    performance_monitor.stop_system_monitoring()
    logger.info("Face Recognition API server stopped")

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Face Recognition API",
        "version": "1.0.0",
        "status": "operational",
        "endpoints": {
            "documentation": "/docs",
            "health": {
                "general": "/health",
                "liveness": "/health/live",
                "readiness": "/health/ready"
            },
            "persons": {
                "list": "/persons",
                "create": "/persons",
                "add_image": "/persons/{person_id}/images"
            },
            "recognition": {
                "recognize": "/recognize",
                "upload": "/upload"
            },
            "metrics": {
                "stats": "/stats",
                "performance": "/performance/metrics"
            }
        },
        "documentation": "Visit /docs for interactive API documentation"
    }

@app.get("/version")
async def get_version():
    """Get API version and build information."""
    return {
        "version": "1.0.0",
        "build_date": "2025-11-02",
        "python_version": sys.version.split()[0],
        "status": "stable"
    }

@app.get("/health")
async def health_check(db: DatabaseManager = Depends(get_database)):
    """
    Comprehensive health check endpoint.
    
    Returns system status, database connectivity, uptime, and resource usage.
    """
    health_status = {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "uptime": str(datetime.now() - server_start_time),
        "version": "1.0.0"
    }
    
    # Check database connectivity
    try:
        db.get_recognition_stats()
        health_status["database"] = {
            "status": "connected",
            "type": "operational"
        }
    except Exception as e:
        health_status["status"] = "degraded"
        health_status["database"] = {
            "status": "error",
            "error": str(e)
        }
        logger.error(f"Database health check failed: {str(e)}")
    
    # Check embedding manager
    try:
        embedding_mgr = get_embedding_manager()
        health_status["embedding_manager"] = {
            "status": "ready",
            "encoder_type": embedding_mgr.encoder_type if hasattr(embedding_mgr, 'encoder_type') else "unknown"
        }
    except Exception as e:
        health_status["status"] = "degraded"
        health_status["embedding_manager"] = {
            "status": "error",
            "error": str(e)
        }
        logger.error(f"Embedding manager health check failed: {str(e)}")
    
    # Get system metrics
    try:
        sys_metrics = performance_monitor.get_system_metrics_summary()
        health_status["system_metrics"] = sys_metrics
    except Exception as e:
        logger.warning(f"Could not retrieve system metrics: {str(e)}")
        health_status["system_metrics"] = {"status": "unavailable"}
    
    return health_status

@app.get("/health/live")
async def liveness_check():
    """
    Kubernetes liveness probe endpoint.
    
    Returns 200 if the application is running.
    """
    return {"status": "alive", "timestamp": datetime.now().isoformat()}

@app.get("/health/ready")
async def readiness_check(db: DatabaseManager = Depends(get_database)):
    """
    Kubernetes readiness probe endpoint.
    
    Returns 200 if the application is ready to serve requests.
    """
    try:
        # Verify critical components
        db.get_recognition_stats()
        embedding_mgr = get_embedding_manager()
        
        return {
            "status": "ready",
            "timestamp": datetime.now().isoformat(),
            "components": {
                "database": "ready",
                "embedding_manager": "ready"
            }
        }
    except Exception as e:
        logger.error(f"Readiness check failed: {str(e)}")
        raise HTTPException(
            status_code=503,
            detail=f"Service not ready: {str(e)}"
        )

@app.get("/stats", response_model=SystemStats)
async def get_system_stats(db: DatabaseManager = Depends(get_database)):
    """Get system statistics."""
    stats = db.get_recognition_stats()
    uptime = datetime.now() - server_start_time
    
    return SystemStats(
        total_persons=stats.get('total_persons', 0),
        total_embeddings=stats.get('total_embeddings', 0),
        total_recognitions=stats.get('total_recognitions', 0),
        avg_processing_time=stats.get('avg_processing_time', 0.0),
        uptime=str(uptime)
    )

@app.get("/persons", response_model=List[PersonResponse])
async def list_persons(db: DatabaseManager = Depends(get_database)):
    """List all persons in the database."""
    persons = db.list_persons()
    
    return [
        PersonResponse(
            id=person.id,
            name=person.name,
            description=person.description,
            created_at=person.created_at.isoformat(),
            is_active=person.is_active
        )
        for person in persons
    ]

@app.post("/persons", response_model=PersonResponse)
async def create_person(
    person_data: PersonCreate,
    db: DatabaseManager = Depends(get_database)
):
    """Create a new person."""
    # Check if person already exists
    existing = db.get_person_by_name(person_data.name)
    if existing:
        raise HTTPException(
            status_code=400,
            detail=f"Person '{person_data.name}' already exists"
        )
    
    # Create new person
    person = db.add_person(person_data.name, person_data.description)
    
    return PersonResponse(
        id=person.id,
        name=person.name,
        description=person.description,
        created_at=person.created_at.isoformat(),
        is_active=person.is_active
    )

@app.post("/persons/{person_id}/images")
async def add_person_image(
    person_id: int,
    file: UploadFile = File(...),
    db: DatabaseManager = Depends(get_database),
    embedding_mgr: FaceEmbeddingManager = Depends(get_embedding_manager)
):
    """Add a face image for a person."""
    # Verify person exists
    person = db.get_person(person_id)
    if not person:
        raise HTTPException(status_code=404, detail="Person not found")
    
    # Validate file type
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read and process image
        image_data = await file.read()
        
        # Save temporarily
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            tmp.write(image_data)
            tmp_path = tmp.name
        
        try:
            # Generate embeddings
            embedding_data = embedding_mgr.generate_embeddings_from_image(tmp_path)
            
            embeddings_added = 0
            for emb_info in embedding_data['embeddings']:
                db.add_face_embedding(
                    embedding=emb_info['embedding'],
                    person_id=person_id,
                    source_image_path=file.filename,
                    quality_score=emb_info['quality_score'],
                    encoder_type="simple",
                    source_type="upload"
                )
                embeddings_added += 1
            
            return {
                "success": True,
                "message": f"Added {embeddings_added} face embeddings for {person.name}",
                "embeddings_count": embeddings_added
            }
        
        finally:
            # Cleanup temporary file
            os.unlink(tmp_path)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@app.post("/recognize", response_model=FaceRecognitionResponse)
@performance_monitor.time_function("api_face_recognition")
async def recognize_faces(
    request: FaceRecognitionRequest,
    db: DatabaseManager = Depends(get_database),
    embedding_mgr: FaceEmbeddingManager = Depends(get_embedding_manager)
):
    """Recognize faces in an uploaded image."""
    import time
    start_time = time.time()
    
    try:
        # Decode image
        image = decode_base64_image(request.image_base64)
        
        # Generate embeddings
        embedding_data = embedding_mgr.pipeline.process_image_array(image)
        
        # Get face locations for bounding boxes
        face_locations = embedding_mgr.pipeline.detector.detect_faces(image)
        
        recognition_results = []
        
        for i, face_data in enumerate(embedding_data['faces']):
            # Generate embedding
            embedding = embedding_mgr.encoder.encode_face(face_data['normalized_face'])
            
            if embedding.size > 0:
                # Search for similar faces
                similar_faces = db.search_similar_faces(
                    embedding,
                    threshold=request.threshold,
                    top_k=1
                )
                
                # Get bounding box
                bbox = list(face_locations[i]) if i < len(face_locations) else [0, 0, 0, 0]
                
                if similar_faces:
                    # Found match
                    best_match, similarity = similar_faces[0]
                    person = db.get_person(best_match.person_id)
                    
                    result = FaceRecognitionResult(
                        face_id=i,
                        person_name=person.name if person else None,
                        person_id=best_match.person_id,
                        confidence=similarity,
                        quality_score=face_data['quality_score'],
                        bounding_box=bbox
                    )
                    
                    # Record recognition result
                    db.add_recognition_result(
                        person_id=best_match.person_id,
                        confidence_score=similarity,
                        source_image_path="api_upload",
                        processing_time=time.time() - start_time,
                        source_type="api"
                    )
                else:
                    # No match found
                    result = FaceRecognitionResult(
                        face_id=i,
                        person_name=None,
                        person_id=None,
                        confidence=0.0,
                        quality_score=face_data['quality_score'],
                        bounding_box=bbox
                    )
                
                recognition_results.append(result)
        
        processing_time = time.time() - start_time
        
        return FaceRecognitionResponse(
            success=True,
            faces=recognition_results,
            processing_time=processing_time
        )
    
    except Exception as e:
        logger.error(f"Error in face recognition: {str(e)}")
        return FaceRecognitionResponse(
            success=False,
            faces=[],
            processing_time=time.time() - start_time,
            message=f"Error: {str(e)}"
        )

@app.post("/upload")
async def upload_image(
    file: UploadFile = File(...),
    threshold: float = 0.7
):
    """Upload and recognize faces in an image file."""
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read image
        image_data = await file.read()
        base64_image = base64.b64encode(image_data).decode()
        
        # Create recognition request
        request = FaceRecognitionRequest(
            image_base64=base64_image,
            threshold=threshold
        )
        
        # Recognize faces
        return await recognize_faces(request)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing upload: {str(e)}")

@app.get("/performance/metrics")
async def get_performance_metrics():
    """Get performance metrics."""
    # Get recent metrics
    recent_recognition = performance_monitor.get_recent_metrics("api_face_recognition", 60)
    
    metrics = {
        "recognition_metrics": {
            "recent_count": len(recent_recognition),
            "avg_time": np.mean([m.value for m in recent_recognition]) if recent_recognition else 0,
        },
        "system_metrics": performance_monitor.get_system_metrics_summary()
    }
    
    return metrics

@app.get("/metrics")
async def metrics():
    """
    Prometheus metrics endpoint.
    
    Returns metrics in Prometheus text format for scraping.
    """
    return PlainTextResponse(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
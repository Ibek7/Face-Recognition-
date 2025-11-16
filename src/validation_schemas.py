#!/usr/bin/env python3
"""
Comprehensive Data Validation Schemas

Pydantic schemas for all API endpoints providing:
- Request/response validation
- Type safety
- Automatic documentation
- Data serialization
"""

from datetime import datetime
from typing import Optional, List, Dict, Any
from enum import Enum

from pydantic import BaseModel, Field, validator, root_validator
from pydantic import EmailStr, constr, confloat, conint


# ============================================================================
# Enums
# ============================================================================

class DetectionModel(str, Enum):
    """Face detection model types"""
    YOLOV8 = "yolov8"
    MTCNN = "mtcnn"
    RETINAFACE = "retinaface"


class RecognitionModel(str, Enum):
    """Face recognition model types"""
    FACENET = "facenet"
    ARCFACE = "arcface"
    VGGFACE = "vggface"


class ImageFormat(str, Enum):
    """Supported image formats"""
    JPEG = "jpeg"
    PNG = "png"
    WEBP = "webp"


class SortOrder(str, Enum):
    """Sort order"""
    ASC = "asc"
    DESC = "desc"


# ============================================================================
# Base Schemas
# ============================================================================

class BaseResponse(BaseModel):
    """Base response schema"""
    success: bool = Field(default=True, description="Request success status")
    message: Optional[str] = Field(None, description="Response message")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class ErrorResponse(BaseModel):
    """Error response schema"""
    success: bool = Field(default=False)
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class PaginationParams(BaseModel):
    """Pagination parameters"""
    page: conint(ge=1) = Field(1, description="Page number (1-indexed)")
    page_size: conint(ge=1, le=100) = Field(20, description="Items per page")
    sort_by: Optional[str] = Field(None, description="Field to sort by")
    sort_order: SortOrder = Field(SortOrder.ASC, description="Sort order")


class PaginatedResponse(BaseResponse):
    """Paginated response schema"""
    total: int = Field(..., description="Total number of items")
    page: int = Field(..., description="Current page")
    page_size: int = Field(..., description="Items per page")
    total_pages: int = Field(..., description="Total number of pages")


# ============================================================================
# Detection Schemas
# ============================================================================

class BoundingBox(BaseModel):
    """Bounding box coordinates"""
    x: float = Field(..., ge=0, description="X coordinate (top-left)")
    y: float = Field(..., ge=0, description="Y coordinate (top-left)")
    width: float = Field(..., gt=0, description="Box width")
    height: float = Field(..., gt=0, description="Box height")
    
    @property
    def x2(self) -> float:
        """Bottom-right X coordinate"""
        return self.x + self.width
    
    @property
    def y2(self) -> float:
        """Bottom-right Y coordinate"""
        return self.y + self.height
    
    @property
    def area(self) -> float:
        """Bounding box area"""
        return self.width * self.height


class FaceLandmarks(BaseModel):
    """Face landmark points"""
    left_eye: List[float] = Field(..., min_items=2, max_items=2)
    right_eye: List[float] = Field(..., min_items=2, max_items=2)
    nose: List[float] = Field(..., min_items=2, max_items=2)
    left_mouth: List[float] = Field(..., min_items=2, max_items=2)
    right_mouth: List[float] = Field(..., min_items=2, max_items=2)


class DetectionRequest(BaseModel):
    """Face detection request"""
    image: str = Field(
        ...,
        description="Base64 encoded image or image URL",
        max_length=10_000_000
    )
    model: DetectionModel = Field(
        DetectionModel.YOLOV8,
        description="Detection model to use"
    )
    min_confidence: confloat(ge=0.0, le=1.0) = Field(
        0.5,
        description="Minimum detection confidence"
    )
    max_faces: conint(ge=1, le=100) = Field(
        10,
        description="Maximum number of faces to detect"
    )
    return_landmarks: bool = Field(
        False,
        description="Return facial landmarks"
    )
    
    @validator('image')
    def validate_image(cls, v):
        """Validate image data"""
        if not v:
            raise ValueError("Image data cannot be empty")
        
        # Check if base64 or URL
        if v.startswith('http://') or v.startswith('https://'):
            # URL validation
            if len(v) > 2048:
                raise ValueError("Image URL too long")
        else:
            # Base64 validation
            if len(v) > 10_000_000:  # ~7.5MB
                raise ValueError("Image data too large")
        
        return v


class DetectedFace(BaseModel):
    """Detected face information"""
    face_id: str = Field(..., description="Unique face identifier")
    bbox: BoundingBox = Field(..., description="Face bounding box")
    confidence: confloat(ge=0.0, le=1.0) = Field(..., description="Detection confidence")
    landmarks: Optional[FaceLandmarks] = Field(None, description="Facial landmarks")


class DetectionResponse(BaseResponse):
    """Face detection response"""
    faces: List[DetectedFace] = Field(..., description="Detected faces")
    count: int = Field(..., description="Number of faces detected")
    model_used: DetectionModel = Field(..., description="Model used for detection")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")


# ============================================================================
# Recognition Schemas
# ============================================================================

class RecognitionRequest(BaseModel):
    """Face recognition request"""
    image: str = Field(..., description="Base64 encoded image or image URL")
    model: RecognitionModel = Field(
        RecognitionModel.FACENET,
        description="Recognition model to use"
    )
    min_confidence: confloat(ge=0.0, le=1.0) = Field(
        0.6,
        description="Minimum recognition confidence"
    )
    top_k: conint(ge=1, le=10) = Field(
        1,
        description="Return top K matches"
    )


class RecognitionMatch(BaseModel):
    """Recognition match result"""
    person_id: str = Field(..., description="Matched person ID")
    person_name: str = Field(..., description="Matched person name")
    confidence: confloat(ge=0.0, le=1.0) = Field(..., description="Match confidence")
    distance: float = Field(..., description="Embedding distance")


class RecognitionResponse(BaseResponse):
    """Face recognition response"""
    face_id: str = Field(..., description="Face identifier")
    matches: List[RecognitionMatch] = Field(..., description="Recognition matches")
    model_used: RecognitionModel = Field(..., description="Model used")
    processing_time_ms: float = Field(..., description="Processing time")


# ============================================================================
# Person Management Schemas
# ============================================================================

class PersonCreate(BaseModel):
    """Create person request"""
    name: constr(min_length=1, max_length=255) = Field(..., description="Person name")
    email: Optional[EmailStr] = Field(None, description="Person email")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")
    
    @validator('name')
    def validate_name(cls, v):
        """Validate person name"""
        v = v.strip()
        if not v:
            raise ValueError("Name cannot be empty or only whitespace")
        return v


class PersonUpdate(BaseModel):
    """Update person request"""
    name: Optional[constr(min_length=1, max_length=255)] = None
    email: Optional[EmailStr] = None
    metadata: Optional[Dict[str, Any]] = None


class PersonResponse(BaseModel):
    """Person information response"""
    id: str = Field(..., description="Person ID")
    name: str = Field(..., description="Person name")
    email: Optional[str] = Field(None, description="Person email")
    metadata: Dict[str, Any] = Field(default_factory=dict)
    embedding_count: int = Field(..., description="Number of face embeddings")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")


class PersonListResponse(PaginatedResponse):
    """List of persons response"""
    persons: List[PersonResponse] = Field(..., description="List of persons")


# ============================================================================
# Embedding Schemas
# ============================================================================

class EmbeddingCreate(BaseModel):
    """Create embedding request"""
    person_id: str = Field(..., description="Person ID")
    image: str = Field(..., description="Face image (base64 or URL)")
    model: RecognitionModel = Field(
        RecognitionModel.FACENET,
        description="Recognition model"
    )


class EmbeddingResponse(BaseModel):
    """Embedding information"""
    id: str = Field(..., description="Embedding ID")
    person_id: str = Field(..., description="Person ID")
    embedding: List[float] = Field(..., description="Face embedding vector")
    model: RecognitionModel = Field(..., description="Model used")
    created_at: datetime = Field(..., description="Creation timestamp")


class EmbeddingListResponse(PaginatedResponse):
    """List of embeddings response"""
    embeddings: List[EmbeddingResponse] = Field(..., description="Embeddings")


# ============================================================================
# Comparison Schemas
# ============================================================================

class CompareRequest(BaseModel):
    """Face comparison request"""
    image1: str = Field(..., description="First image (base64 or URL)")
    image2: str = Field(..., description="Second image (base64 or URL)")
    model: RecognitionModel = Field(
        RecognitionModel.FACENET,
        description="Recognition model"
    )


class CompareResponse(BaseResponse):
    """Face comparison response"""
    is_match: bool = Field(..., description="Whether faces match")
    similarity: confloat(ge=0.0, le=1.0) = Field(..., description="Similarity score")
    distance: float = Field(..., description="Embedding distance")
    threshold: float = Field(..., description="Matching threshold used")
    processing_time_ms: float = Field(..., description="Processing time")


# ============================================================================
# Batch Processing Schemas
# ============================================================================

class BatchDetectionRequest(BaseModel):
    """Batch face detection request"""
    images: List[str] = Field(
        ...,
        min_items=1,
        max_items=100,
        description="List of images (base64 or URLs)"
    )
    model: DetectionModel = Field(DetectionModel.YOLOV8)
    min_confidence: confloat(ge=0.0, le=1.0) = Field(0.5)


class BatchDetectionResult(BaseModel):
    """Single batch detection result"""
    image_index: int = Field(..., description="Image index in batch")
    faces: List[DetectedFace] = Field(..., description="Detected faces")
    error: Optional[str] = Field(None, description="Error message if failed")


class BatchDetectionResponse(BaseResponse):
    """Batch detection response"""
    results: List[BatchDetectionResult] = Field(..., description="Detection results")
    total_images: int = Field(..., description="Total images processed")
    successful: int = Field(..., description="Successfully processed images")
    failed: int = Field(..., description="Failed images")


# ============================================================================
# Health Check Schemas
# ============================================================================

class HealthStatus(str, Enum):
    """Health status"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


class ComponentHealth(BaseModel):
    """Individual component health"""
    name: str = Field(..., description="Component name")
    status: HealthStatus = Field(..., description="Component status")
    latency_ms: Optional[float] = Field(None, description="Component latency")
    message: Optional[str] = Field(None, description="Status message")


class HealthCheckResponse(BaseModel):
    """Health check response"""
    status: HealthStatus = Field(..., description="Overall health status")
    version: str = Field(..., description="API version")
    uptime_seconds: float = Field(..., description="Service uptime")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    components: List[ComponentHealth] = Field(..., description="Component statuses")


# ============================================================================
# Statistics Schemas
# ============================================================================

class SystemStats(BaseModel):
    """System statistics"""
    total_persons: int = Field(..., description="Total persons enrolled")
    total_embeddings: int = Field(..., description="Total face embeddings")
    total_detections: int = Field(..., description="Total detections performed")
    total_recognitions: int = Field(..., description="Total recognitions performed")
    avg_detection_time_ms: float = Field(..., description="Average detection time")
    avg_recognition_time_ms: float = Field(..., description="Average recognition time")
    uptime_seconds: float = Field(..., description="System uptime")


class StatsResponse(BaseResponse):
    """Statistics response"""
    stats: SystemStats = Field(..., description="System statistics")


# ============================================================================
# Configuration Schemas
# ============================================================================

class ModelConfig(BaseModel):
    """Model configuration"""
    detection_model: DetectionModel = Field(DetectionModel.YOLOV8)
    recognition_model: RecognitionModel = Field(RecognitionModel.FACENET)
    detection_confidence: confloat(ge=0.0, le=1.0) = Field(0.5)
    recognition_confidence: confloat(ge=0.0, le=1.0) = Field(0.6)
    max_faces_per_image: conint(ge=1, le=100) = Field(10)


class ConfigResponse(BaseResponse):
    """Configuration response"""
    config: ModelConfig = Field(..., description="Current configuration")


# Example usage and validation
if __name__ == "__main__":
    # Test schema validation
    
    # Valid detection request
    detection_req = DetectionRequest(
        image="data:image/jpeg;base64,/9j/4AAQ...",
        model=DetectionModel.YOLOV8,
        min_confidence=0.7,
        max_faces=5
    )
    print("✓ Valid detection request")
    
    # Invalid detection request (will raise validation error)
    try:
        invalid_req = DetectionRequest(
            image="",  # Empty image
            min_confidence=1.5  # Invalid confidence
        )
    except ValueError as e:
        print(f"✓ Caught validation error: {e}")
    
    # Person creation
    person = PersonCreate(
        name="John Doe",
        email="john@example.com",
        metadata={"department": "Engineering"}
    )
    print("✓ Valid person creation")
    
    print("\nAll schema validations passed!")

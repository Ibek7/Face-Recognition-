"""
Pydantic schemas and validation models for the Face Recognition API.
"""

from pydantic import BaseModel, Field, validator
from typing import List, Optional, Union, Dict, Any
from datetime import datetime
from enum import Enum

class ProcessingStatus(str, Enum):
    """Status of processing operations."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

class SourceType(str, Enum):
    """Source type for embeddings and recognitions."""
    UPLOAD = "upload"
    WEBCAM = "webcam"
    BATCH = "batch"
    API = "api"

class EncoderType(str, Enum):
    """Available encoder types."""
    SIMPLE = "simple"
    DLIB = "dlib"

# Person-related schemas
class PersonBase(BaseModel):
    """Base person schema."""
    name: str = Field(..., min_length=1, max_length=100, description="Person's name")
    description: Optional[str] = Field("", max_length=500, description="Person description")

class PersonCreate(PersonBase):
    """Schema for creating a new person."""
    pass

class PersonUpdate(BaseModel):
    """Schema for updating a person."""
    name: Optional[str] = Field(None, min_length=1, max_length=100)
    description: Optional[str] = Field(None, max_length=500)
    is_active: Optional[bool] = None

class PersonResponse(PersonBase):
    """Schema for person response."""
    id: int
    created_at: datetime
    updated_at: datetime
    is_active: bool
    embeddings_count: int = 0

    class Config:
        from_attributes = True

# Face embedding schemas
class FaceEmbeddingBase(BaseModel):
    """Base face embedding schema."""
    quality_score: float = Field(..., ge=0, le=1, description="Quality score 0-1")
    encoder_type: EncoderType = Field(default=EncoderType.SIMPLE)
    source_type: SourceType = Field(default=SourceType.UPLOAD)
    source_image_path: Optional[str] = None

class FaceEmbeddingResponse(FaceEmbeddingBase):
    """Schema for face embedding response."""
    id: int
    person_id: int
    created_at: datetime
    dimensions: int

    class Config:
        from_attributes = True

# Face recognition schemas
class BoundingBox(BaseModel):
    """Face bounding box coordinates."""
    x: int = Field(..., ge=0, description="X coordinate")
    y: int = Field(..., ge=0, description="Y coordinate")
    width: int = Field(..., gt=0, description="Width")
    height: int = Field(..., gt=0, description="Height")

class FaceRecognitionRequest(BaseModel):
    """Schema for face recognition request."""
    image_base64: str = Field(..., description="Base64 encoded image")
    threshold: float = Field(0.7, ge=0, le=1, description="Recognition threshold")
    top_k: int = Field(5, ge=1, le=20, description="Maximum number of matches")
    include_unknown: bool = Field(True, description="Include unrecognized faces")

    @validator('image_base64')
    def validate_base64_image(cls, v):
        """Validate base64 image format."""
        import base64
        import re
        
        # Remove data URL prefix if present
        if "," in v:
            v = v.split(",")[1]
        
        # Check if valid base64
        try:
            decoded = base64.b64decode(v, validate=True)
            if len(decoded) < 100:  # Minimum image size
                raise ValueError("Image too small")
        except Exception:
            raise ValueError("Invalid base64 image data")
        
        return v

class FaceResult(BaseModel):
    """Individual face recognition result."""
    face_id: int = Field(..., description="Face index in image")
    person_id: Optional[int] = Field(None, description="Matched person ID")
    person_name: Optional[str] = Field(None, description="Matched person name")
    confidence: float = Field(..., ge=0, le=1, description="Recognition confidence")
    quality_score: float = Field(..., ge=0, le=1, description="Face quality score")
    bounding_box: BoundingBox
    is_match: bool = Field(..., description="Whether face was matched")

class FaceRecognitionResponse(BaseModel):
    """Schema for face recognition response."""
    success: bool
    faces: List[FaceResult]
    processing_time: float = Field(..., ge=0, description="Processing time in seconds")
    total_faces: int = Field(..., ge=0, description="Total faces detected")
    matched_faces: int = Field(..., ge=0, description="Number of matched faces")
    message: Optional[str] = None

# Batch processing schemas
class BatchJobCreate(BaseModel):
    """Schema for creating a batch processing job."""
    name: str = Field(..., min_length=1, max_length=100)
    description: Optional[str] = Field("", max_length=500)
    image_paths: List[str] = Field(..., min_items=1, max_items=1000)
    recognition_threshold: float = Field(0.7, ge=0, le=1)

class BatchJobResponse(BaseModel):
    """Schema for batch job response."""
    id: int
    name: str
    description: str
    status: ProcessingStatus
    total_images: int
    processed_images: int
    created_at: datetime
    updated_at: datetime
    completion_percentage: float = Field(..., ge=0, le=100)

    class Config:
        from_attributes = True

# Statistics schemas
class PersonStats(BaseModel):
    """Person statistics."""
    person_id: int
    person_name: str
    embeddings_count: int
    recognitions_count: int
    avg_confidence: float
    last_recognition: Optional[datetime]

class SystemStats(BaseModel):
    """System-wide statistics."""
    total_persons: int = Field(..., ge=0)
    total_embeddings: int = Field(..., ge=0)
    total_recognitions: int = Field(..., ge=0)
    total_batch_jobs: int = Field(..., ge=0)
    avg_processing_time: float = Field(..., ge=0)
    avg_recognition_confidence: float = Field(..., ge=0, le=1)
    uptime: str
    version: str = "1.0.0"

class PerformanceMetrics(BaseModel):
    """Performance metrics."""
    recognition_metrics: Dict[str, Any]
    system_metrics: Dict[str, Any]
    recent_activity: List[Dict[str, Any]]

# Health check schemas
class HealthStatus(BaseModel):
    """Health check status."""
    status: str = Field(..., description="Service status")
    uptime: str
    database_connected: bool
    system_load: Dict[str, float]
    memory_usage: Dict[str, float]
    timestamp: datetime

# Error schemas
class ErrorResponse(BaseModel):
    """Error response schema."""
    success: bool = False
    error_code: str
    message: str
    details: Optional[Dict[str, Any]] = None
    timestamp: datetime

class ValidationErrorResponse(BaseModel):
    """Validation error response."""
    success: bool = False
    error_code: str = "VALIDATION_ERROR"
    message: str = "Validation failed"
    validation_errors: List[Dict[str, Any]]
    timestamp: datetime

# Success response schemas
class SuccessResponse(BaseModel):
    """Generic success response."""
    success: bool = True
    message: str
    data: Optional[Dict[str, Any]] = None
    timestamp: datetime = Field(default_factory=datetime.now)

class FileUploadResponse(BaseModel):
    """File upload response."""
    success: bool
    filename: str
    file_size: int
    content_type: str
    embeddings_added: int
    processing_time: float
    message: str

# Configuration schemas
class APIConfig(BaseModel):
    """API configuration schema."""
    debug: bool = False
    cors_origins: List[str] = ["*"]
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    allowed_image_types: List[str] = ["image/jpeg", "image/png", "image/bmp"]
    default_recognition_threshold: float = 0.7
    max_batch_size: int = 1000
    rate_limit_per_minute: int = 100

class DatabaseConfig(BaseModel):
    """Database configuration schema."""
    url: str = "sqlite:///face_recognition.db"
    echo: bool = False
    pool_size: int = 5
    max_overflow: int = 10

# Export all schemas
__all__ = [
    # Enums
    "ProcessingStatus", "SourceType", "EncoderType",
    # Person schemas
    "PersonBase", "PersonCreate", "PersonUpdate", "PersonResponse",
    # Face embedding schemas
    "FaceEmbeddingBase", "FaceEmbeddingResponse",
    # Recognition schemas
    "BoundingBox", "FaceRecognitionRequest", "FaceResult", "FaceRecognitionResponse",
    # Batch processing schemas
    "BatchJobCreate", "BatchJobResponse",
    # Statistics schemas
    "PersonStats", "SystemStats", "PerformanceMetrics",
    # Health and error schemas
    "HealthStatus", "ErrorResponse", "ValidationErrorResponse",
    # Success schemas
    "SuccessResponse", "FileUploadResponse",
    # Configuration schemas
    "APIConfig", "DatabaseConfig"
]
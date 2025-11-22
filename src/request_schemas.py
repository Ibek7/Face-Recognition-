"""
Input validation schemas for API requests using Pydantic.
"""

from typing import Optional, List
from pydantic import BaseModel, Field, validator
from datetime import datetime


class ImageRequest(BaseModel):
    """Image upload validation."""
    
    image_data: str = Field(..., description="Base64 encoded image")
    format: Optional[str] = Field("jpeg", description="Image format")
    
    @validator("format")
    def check_format(cls, v):
        """Validate format."""
        if v.lower() not in ["jpeg", "jpg", "png", "webp"]:
            raise ValueError("Invalid image format")
        return v.lower()


class DetectionRequest(BaseModel):
    """Face detection validation."""
    
    image_url: Optional[str] = None
    image_data: Optional[str] = None
    min_confidence: float = Field(0.5, ge=0.0, le=1.0)
    max_faces: Optional[int] = Field(None, ge=1, le=100)


class RecognitionRequest(BaseModel):
    """Face recognition validation."""
    
    image_data: str = Field(..., description="Base64 image")
    threshold: float = Field(0.6, ge=0.0, le=1.0)
    top_k: int = Field(5, ge=1, le=20)


class PersonCreate(BaseModel):
    """Person creation validation."""
    
    name: str = Field(..., min_length=1, max_length=100)
    email: Optional[str] = None
    metadata: Optional[dict] = None
    
    @validator("email")
    def check_email(cls, v):
        """Validate email."""
        if v:
            import re
            if not re.match(r'^[^@]+@[^@]+\.[^@]+$', v):
                raise ValueError("Invalid email")
        return v


class BatchRequest(BaseModel):
    """Batch processing validation."""
    
    images: List[str] = Field(..., min_items=1, max_items=50)
    min_confidence: float = Field(0.5, ge=0.0, le=1.0)
    parallel: bool = True


class ConfigUpdate(BaseModel):
    """Configuration update validation."""
    
    log_level: Optional[str] = None
    max_workers: Optional[int] = Field(None, ge=1, le=64)
    
    @validator("log_level")
    def check_log_level(cls, v):
        """Validate log level."""
        if v and v.upper() not in ["DEBUG", "INFO", "WARNING", "ERROR"]:
            raise ValueError("Invalid log level")
        return v.upper() if v else v


class HealthResponse(BaseModel):
    """Health check response."""
    
    status: str
    timestamp: datetime
    version: str
    uptime_seconds: float


class ErrorResponse(BaseModel):
    """Error response."""
    
    error: str
    detail: Optional[str] = None
    status_code: int
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# Usage:
"""
from fastapi import FastAPI
from src.request_schemas import DetectionRequest

@app.post("/detect")
async def detect(request: DetectionRequest):
    # Validated automatically
    return {"result": "ok"}
"""

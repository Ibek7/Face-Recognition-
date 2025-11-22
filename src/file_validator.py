"""
File upload validation for secure file handling.

Validates file size, type, and content for uploaded files.
"""

import os
import magic
from typing import Optional, List, Set
from fastapi import UploadFile, HTTPException, status
import hashlib
import logging

logger = logging.getLogger(__name__)


class FileValidator:
    """Validate uploaded files."""
    
    # Default allowed MIME types for images
    DEFAULT_IMAGE_TYPES = {
        'image/jpeg',
        'image/jpg',
        'image/png',
        'image/gif',
        'image/webp',
        'image/bmp'
    }
    
    # Default allowed MIME types for videos
    DEFAULT_VIDEO_TYPES = {
        'video/mp4',
        'video/mpeg',
        'video/quicktime',
        'video/x-msvideo',
        'video/webm'
    }
    
    def __init__(
        self,
        max_size_mb: int = 10,
        allowed_types: Optional[Set[str]] = None,
        check_magic_bytes: bool = True
    ):
        """
        Initialize file validator.
        
        Args:
            max_size_mb: Maximum file size in MB
            allowed_types: Set of allowed MIME types
            check_magic_bytes: Validate using magic bytes
        """
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.allowed_types = allowed_types or (
            self.DEFAULT_IMAGE_TYPES | self.DEFAULT_VIDEO_TYPES
        )
        self.check_magic_bytes = check_magic_bytes
    
    async def validate(
        self,
        file: UploadFile,
        expected_type: Optional[str] = None
    ) -> dict:
        """
        Validate uploaded file.
        
        Args:
            file: Uploaded file
            expected_type: Expected MIME type category
        
        Returns:
            File metadata
        
        Raises:
            HTTPException: If validation fails
        """
        # Read file content
        content = await file.read()
        await file.seek(0)  # Reset for later use
        
        # Check file size
        file_size = len(content)
        if file_size > self.max_size_bytes:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"File too large. Max size: {self.max_size_bytes / 1024 / 1024}MB"
            )
        
        # Detect MIME type
        if self.check_magic_bytes:
            mime_type = magic.from_buffer(content, mime=True)
        else:
            # Fallback to content type header
            mime_type = file.content_type
        
        # Validate MIME type
        if mime_type not in self.allowed_types:
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail=f"Invalid file type: {mime_type}"
            )
        
        # Check expected type category
        if expected_type:
            if expected_type == "image" and not mime_type.startswith("image/"):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Expected image file"
                )
            elif expected_type == "video" and not mime_type.startswith("video/"):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Expected video file"
                )
        
        # Calculate file hash
        file_hash = hashlib.sha256(content).hexdigest()
        
        # Validate filename
        filename = file.filename
        if not filename or ".." in filename or "/" in filename:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid filename"
            )
        
        # Get file extension
        _, ext = os.path.splitext(filename)
        
        metadata = {
            "filename": filename,
            "size": file_size,
            "mime_type": mime_type,
            "extension": ext.lower(),
            "hash": file_hash
        }
        
        logger.info(
            f"File validated: {filename} ({mime_type}, "
            f"{file_size / 1024:.1f}KB)"
        )
        
        return metadata
    
    async def validate_multiple(
        self,
        files: List[UploadFile],
        max_files: int = 10
    ) -> List[dict]:
        """
        Validate multiple uploaded files.
        
        Args:
            files: List of uploaded files
            max_files: Maximum number of files
        
        Returns:
            List of file metadata
        
        Raises:
            HTTPException: If validation fails
        """
        if len(files) > max_files:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Too many files. Max: {max_files}"
            )
        
        metadata_list = []
        
        for file in files:
            metadata = await self.validate(file)
            metadata_list.append(metadata)
        
        return metadata_list
    
    def check_duplicate(
        self,
        file_hash: str,
        existing_hashes: Set[str]
    ) -> bool:
        """
        Check if file is duplicate.
        
        Args:
            file_hash: SHA256 hash of file
            existing_hashes: Set of existing file hashes
        
        Returns:
            True if duplicate
        """
        return file_hash in existing_hashes


# Global validator instance
file_validator = FileValidator()


# Example usage in api_server.py:
"""
from fastapi import FastAPI, File, UploadFile
from src.file_validator import file_validator

app = FastAPI()

@app.post("/upload/image")
async def upload_image(file: UploadFile = File(...)):
    # Validate image
    metadata = await file_validator.validate(file, expected_type="image")
    
    # Save file
    file_path = f"data/images/{metadata['filename']}"
    
    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)
    
    return {
        "message": "File uploaded",
        "metadata": metadata
    }

@app.post("/upload/batch")
async def upload_batch(files: List[UploadFile] = File(...)):
    # Validate multiple files
    metadata_list = await file_validator.validate_multiple(files, max_files=20)
    
    return {
        "message": f"Uploaded {len(files)} files",
        "files": metadata_list
    }

# Custom validator for specific use case
custom_validator = FileValidator(
    max_size_mb=50,
    allowed_types={'image/jpeg', 'image/png'},
    check_magic_bytes=True
)
"""

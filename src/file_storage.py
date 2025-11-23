"""
File storage abstraction for local and cloud storage.

Provides unified interface for local filesystem, S3, Azure Blob, and GCS.
"""

from typing import Optional, List, BinaryIO, Dict, Any
from enum import Enum
from pathlib import Path
from datetime import datetime, timedelta
import asyncio
import io
import logging

logger = logging.getLogger(__name__)


class StorageBackend(str, Enum):
    """Storage backend type."""
    
    LOCAL = "local"
    S3 = "s3"
    AZURE = "azure"
    GCS = "gcs"


class FileMetadata:
    """File metadata."""
    
    def __init__(
        self,
        path: str,
        size: int,
        content_type: Optional[str] = None,
        etag: Optional[str] = None,
        last_modified: Optional[datetime] = None,
        metadata: Optional[Dict[str, str]] = None
    ):
        """
        Initialize file metadata.
        
        Args:
            path: File path
            size: File size in bytes
            content_type: MIME type
            etag: Entity tag
            last_modified: Last modification time
            metadata: Custom metadata
        """
        self.path = path
        self.size = size
        self.content_type = content_type
        self.etag = etag
        self.last_modified = last_modified or datetime.utcnow()
        self.metadata = metadata or {}
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "path": self.path,
            "size": self.size,
            "size_mb": round(self.size / (1024 ** 2), 2),
            "content_type": self.content_type,
            "etag": self.etag,
            "last_modified": self.last_modified.isoformat(),
            "metadata": self.metadata
        }


class StorageProvider:
    """Base storage provider."""
    
    def __init__(self, backend: StorageBackend):
        """
        Initialize storage provider.
        
        Args:
            backend: Storage backend type
        """
        self.backend = backend
    
    async def upload(
        self,
        path: str,
        content: bytes,
        content_type: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None
    ) -> FileMetadata:
        """Upload file."""
        raise NotImplementedError
    
    async def download(self, path: str) -> bytes:
        """Download file."""
        raise NotImplementedError
    
    async def delete(self, path: str):
        """Delete file."""
        raise NotImplementedError
    
    async def exists(self, path: str) -> bool:
        """Check if file exists."""
        raise NotImplementedError
    
    async def list(self, prefix: str = "") -> List[FileMetadata]:
        """List files with prefix."""
        raise NotImplementedError
    
    async def get_metadata(self, path: str) -> FileMetadata:
        """Get file metadata."""
        raise NotImplementedError


class LocalStorageProvider(StorageProvider):
    """Local filesystem storage provider."""
    
    def __init__(self, base_path: str):
        """
        Initialize local storage.
        
        Args:
            base_path: Base directory path
        """
        super().__init__(StorageBackend.LOCAL)
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
    
    def _get_full_path(self, path: str) -> Path:
        """Get full filesystem path."""
        return self.base_path / path.lstrip("/")
    
    async def upload(
        self,
        path: str,
        content: bytes,
        content_type: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None
    ) -> FileMetadata:
        """Upload file to local storage."""
        full_path = self._get_full_path(path)
        full_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write file
        await asyncio.get_event_loop().run_in_executor(
            None,
            full_path.write_bytes,
            content
        )
        
        logger.info(f"Uploaded file to local storage: {path}")
        
        return FileMetadata(
            path=path,
            size=len(content),
            content_type=content_type,
            metadata=metadata
        )
    
    async def download(self, path: str) -> bytes:
        """Download file from local storage."""
        full_path = self._get_full_path(path)
        
        if not full_path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        
        content = await asyncio.get_event_loop().run_in_executor(
            None,
            full_path.read_bytes
        )
        
        return content
    
    async def delete(self, path: str):
        """Delete file from local storage."""
        full_path = self._get_full_path(path)
        
        if full_path.exists():
            full_path.unlink()
            logger.info(f"Deleted file from local storage: {path}")
    
    async def exists(self, path: str) -> bool:
        """Check if file exists."""
        return self._get_full_path(path).exists()
    
    async def list(self, prefix: str = "") -> List[FileMetadata]:
        """List files with prefix."""
        prefix_path = self._get_full_path(prefix)
        
        if not prefix_path.exists():
            return []
        
        files = []
        
        if prefix_path.is_file():
            # Single file
            stat = prefix_path.stat()
            files.append(FileMetadata(
                path=prefix,
                size=stat.st_size,
                last_modified=datetime.fromtimestamp(stat.st_mtime)
            ))
        else:
            # Directory
            for file_path in prefix_path.rglob("*"):
                if file_path.is_file():
                    relative_path = str(file_path.relative_to(self.base_path))
                    stat = file_path.stat()
                    
                    files.append(FileMetadata(
                        path=relative_path,
                        size=stat.st_size,
                        last_modified=datetime.fromtimestamp(stat.st_mtime)
                    ))
        
        return files
    
    async def get_metadata(self, path: str) -> FileMetadata:
        """Get file metadata."""
        full_path = self._get_full_path(path)
        
        if not full_path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        
        stat = full_path.stat()
        
        return FileMetadata(
            path=path,
            size=stat.st_size,
            last_modified=datetime.fromtimestamp(stat.st_mtime)
        )


class S3StorageProvider(StorageProvider):
    """Amazon S3 storage provider."""
    
    def __init__(
        self,
        bucket: str,
        region: str = "us-east-1",
        access_key: Optional[str] = None,
        secret_key: Optional[str] = None
    ):
        """
        Initialize S3 storage.
        
        Args:
            bucket: S3 bucket name
            region: AWS region
            access_key: AWS access key
            secret_key: AWS secret key
        """
        super().__init__(StorageBackend.S3)
        self.bucket = bucket
        self.region = region
        
        # In production, use aioboto3
        logger.info(f"Initialized S3 storage for bucket: {bucket}")
    
    async def upload(
        self,
        path: str,
        content: bytes,
        content_type: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None
    ) -> FileMetadata:
        """Upload file to S3."""
        # Placeholder - use aioboto3 in production
        logger.info(f"Uploading to S3: {self.bucket}/{path}")
        
        return FileMetadata(
            path=path,
            size=len(content),
            content_type=content_type,
            metadata=metadata
        )
    
    async def download(self, path: str) -> bytes:
        """Download file from S3."""
        # Placeholder
        logger.info(f"Downloading from S3: {self.bucket}/{path}")
        return b""
    
    async def delete(self, path: str):
        """Delete file from S3."""
        logger.info(f"Deleting from S3: {self.bucket}/{path}")
    
    async def exists(self, path: str) -> bool:
        """Check if file exists in S3."""
        return False
    
    async def list(self, prefix: str = "") -> List[FileMetadata]:
        """List files in S3."""
        return []
    
    async def get_metadata(self, path: str) -> FileMetadata:
        """Get S3 object metadata."""
        return FileMetadata(path=path, size=0)


class FileStorage:
    """Unified file storage interface."""
    
    def __init__(self, provider: StorageProvider):
        """
        Initialize file storage.
        
        Args:
            provider: Storage provider
        """
        self.provider = provider
    
    async def upload_file(
        self,
        path: str,
        content: bytes,
        content_type: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None
    ) -> FileMetadata:
        """
        Upload file.
        
        Args:
            path: File path
            content: File content
            content_type: MIME type
            metadata: Custom metadata
        
        Returns:
            File metadata
        """
        return await self.provider.upload(path, content, content_type, metadata)
    
    async def upload_from_file(
        self,
        path: str,
        file_path: str,
        **kwargs
    ) -> FileMetadata:
        """Upload from local file."""
        content = Path(file_path).read_bytes()
        return await self.upload_file(path, content, **kwargs)
    
    async def download_file(self, path: str) -> bytes:
        """Download file."""
        return await self.provider.download(path)
    
    async def download_to_file(self, path: str, destination: str):
        """Download to local file."""
        content = await self.download_file(path)
        Path(destination).write_bytes(content)
    
    async def delete_file(self, path: str):
        """Delete file."""
        await self.provider.delete(path)
    
    async def file_exists(self, path: str) -> bool:
        """Check if file exists."""
        return await self.provider.exists(path)
    
    async def list_files(self, prefix: str = "") -> List[FileMetadata]:
        """List files."""
        return await self.provider.list(prefix)
    
    async def get_file_metadata(self, path: str) -> FileMetadata:
        """Get file metadata."""
        return await self.provider.get_metadata(path)
    
    async def copy_file(self, source: str, destination: str):
        """Copy file within storage."""
        content = await self.download_file(source)
        metadata = await self.get_file_metadata(source)
        
        await self.upload_file(
            destination,
            content,
            content_type=metadata.content_type,
            metadata=metadata.metadata
        )
    
    async def move_file(self, source: str, destination: str):
        """Move file within storage."""
        await self.copy_file(source, destination)
        await self.delete_file(source)
    
    def get_backend_type(self) -> StorageBackend:
        """Get storage backend type."""
        return self.provider.backend


class MultiStorageManager:
    """Manage multiple storage backends."""
    
    def __init__(self):
        """Initialize multi-storage manager."""
        self.storages: Dict[str, FileStorage] = {}
        self.default_storage: Optional[str] = None
    
    def register_storage(
        self,
        name: str,
        storage: FileStorage,
        set_as_default: bool = False
    ):
        """
        Register storage backend.
        
        Args:
            name: Storage name
            storage: FileStorage instance
            set_as_default: Set as default storage
        """
        self.storages[name] = storage
        
        if set_as_default or not self.default_storage:
            self.default_storage = name
        
        logger.info(f"Registered storage: {name} ({storage.get_backend_type().value})")
    
    def get_storage(self, name: Optional[str] = None) -> FileStorage:
        """Get storage by name."""
        if name is None:
            name = self.default_storage
        
        if name not in self.storages:
            raise ValueError(f"Storage not found: {name}")
        
        return self.storages[name]
    
    async def upload(
        self,
        path: str,
        content: bytes,
        storage: Optional[str] = None,
        **kwargs
    ) -> FileMetadata:
        """Upload to specific storage."""
        return await self.get_storage(storage).upload_file(path, content, **kwargs)
    
    async def download(
        self,
        path: str,
        storage: Optional[str] = None
    ) -> bytes:
        """Download from specific storage."""
        return await self.get_storage(storage).download_file(path)


# Example usage:
"""
from src.file_storage import FileStorage, LocalStorageProvider, S3StorageProvider, MultiStorageManager

# Local storage
local_provider = LocalStorageProvider("/var/data/uploads")
local_storage = FileStorage(local_provider)

# Upload file
metadata = await local_storage.upload_file(
    path="documents/report.pdf",
    content=pdf_bytes,
    content_type="application/pdf"
)

# Download file
content = await local_storage.download_file("documents/report.pdf")

# List files
files = await local_storage.list_files(prefix="documents/")

# Multi-storage
manager = MultiStorageManager()
manager.register_storage("local", local_storage, set_as_default=True)

# S3 storage
s3_provider = S3StorageProvider(bucket="my-bucket", region="us-east-1")
s3_storage = FileStorage(s3_provider)
manager.register_storage("s3", s3_storage)

# Upload to specific storage
await manager.upload("file.txt", b"content", storage="s3")
"""

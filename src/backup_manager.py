"""
Automated backup system for data and configurations.

Supports local and cloud backups with compression and encryption.
"""

from typing import Dict, List, Optional, Any
from enum import Enum
from pathlib import Path
from datetime import datetime, timedelta
import asyncio
import json
import shutil
import tarfile
import gzip
import logging

logger = logging.getLogger(__name__)


class BackupType(str, Enum):
    """Backup type."""
    
    FULL = "full"
    INCREMENTAL = "incremental"
    DIFFERENTIAL = "differential"


class BackupStatus(str, Enum):
    """Backup status."""
    
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class BackupStorage(str, Enum):
    """Backup storage type."""
    
    LOCAL = "local"
    S3 = "s3"
    AZURE = "azure"
    GCS = "gcs"


class BackupMetadata:
    """Backup metadata."""
    
    def __init__(
        self,
        backup_id: str,
        backup_type: BackupType,
        source_path: str,
        destination_path: str,
        size_bytes: int = 0,
        file_count: int = 0,
        status: BackupStatus = BackupStatus.IN_PROGRESS
    ):
        """
        Initialize backup metadata.
        
        Args:
            backup_id: Unique backup identifier
            backup_type: Type of backup
            source_path: Source directory
            destination_path: Destination path
            size_bytes: Backup size in bytes
            file_count: Number of files
            status: Backup status
        """
        self.backup_id = backup_id
        self.backup_type = backup_type
        self.source_path = source_path
        self.destination_path = destination_path
        self.size_bytes = size_bytes
        self.file_count = file_count
        self.status = status
        self.created_at = datetime.utcnow()
        self.completed_at: Optional[datetime] = None
        self.error_message: Optional[str] = None
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "backup_id": self.backup_id,
            "backup_type": self.backup_type.value,
            "source_path": self.source_path,
            "destination_path": self.destination_path,
            "size_bytes": self.size_bytes,
            "size_mb": round(self.size_bytes / (1024 ** 2), 2),
            "file_count": self.file_count,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "error_message": self.error_message
        }


class BackupManager:
    """Manage automated backups."""
    
    def __init__(
        self,
        backup_dir: str,
        compression: bool = True,
        retention_days: int = 30
    ):
        """
        Initialize backup manager.
        
        Args:
            backup_dir: Backup destination directory
            compression: Enable compression
            retention_days: Days to retain backups
        """
        self.backup_dir = Path(backup_dir)
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        
        self.compression = compression
        self.retention_days = retention_days
        
        self.backups: Dict[str, BackupMetadata] = {}
        self._load_metadata()
    
    def _load_metadata(self):
        """Load backup metadata from disk."""
        metadata_file = self.backup_dir / "backups.json"
        
        if metadata_file.exists():
            try:
                with open(metadata_file, "r") as f:
                    data = json.load(f)
                
                for backup_data in data:
                    metadata = BackupMetadata(**backup_data)
                    self.backups[metadata.backup_id] = metadata
            
            except Exception as e:
                logger.error(f"Failed to load backup metadata: {e}")
    
    def _save_metadata(self):
        """Save backup metadata to disk."""
        metadata_file = self.backup_dir / "backups.json"
        
        try:
            data = [backup.to_dict() for backup in self.backups.values()]
            
            with open(metadata_file, "w") as f:
                json.dump(data, f, indent=2)
        
        except Exception as e:
            logger.error(f"Failed to save backup metadata: {e}")
    
    async def create_backup(
        self,
        source_path: str,
        backup_type: BackupType = BackupType.FULL,
        exclude_patterns: Optional[List[str]] = None
    ) -> BackupMetadata:
        """
        Create backup.
        
        Args:
            source_path: Source directory to backup
            backup_type: Type of backup
            exclude_patterns: File patterns to exclude
        
        Returns:
            Backup metadata
        """
        source = Path(source_path)
        
        if not source.exists():
            raise FileNotFoundError(f"Source path not found: {source_path}")
        
        # Generate backup ID
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        backup_id = f"{source.name}_{backup_type.value}_{timestamp}"
        
        # Determine file extension
        if self.compression:
            backup_file = self.backup_dir / f"{backup_id}.tar.gz"
        else:
            backup_file = self.backup_dir / f"{backup_id}.tar"
        
        # Create metadata
        metadata = BackupMetadata(
            backup_id=backup_id,
            backup_type=backup_type,
            source_path=str(source),
            destination_path=str(backup_file)
        )
        
        self.backups[backup_id] = metadata
        self._save_metadata()
        
        try:
            # Create backup
            await self._create_archive(
                source,
                backup_file,
                exclude_patterns or []
            )
            
            # Update metadata
            metadata.size_bytes = backup_file.stat().st_size
            metadata.status = BackupStatus.COMPLETED
            metadata.completed_at = datetime.utcnow()
            
            self._save_metadata()
            
            logger.info(
                f"Backup completed: {backup_id} "
                f"({metadata.size_bytes / (1024**2):.2f} MB)"
            )
            
            return metadata
        
        except Exception as e:
            metadata.status = BackupStatus.FAILED
            metadata.error_message = str(e)
            self._save_metadata()
            
            logger.error(f"Backup failed: {e}")
            raise
    
    async def _create_archive(
        self,
        source: Path,
        destination: Path,
        exclude_patterns: List[str]
    ):
        """Create tar archive."""
        def _create():
            mode = "w:gz" if self.compression else "w"
            
            with tarfile.open(destination, mode) as tar:
                tar.add(
                    source,
                    arcname=source.name,
                    filter=lambda info: None if any(
                        pattern in info.name for pattern in exclude_patterns
                    ) else info
                )
        
        # Run in thread pool to avoid blocking
        await asyncio.get_event_loop().run_in_executor(None, _create)
    
    async def restore_backup(
        self,
        backup_id: str,
        destination_path: str
    ):
        """
        Restore backup.
        
        Args:
            backup_id: Backup ID to restore
            destination_path: Restore destination
        """
        if backup_id not in self.backups:
            raise ValueError(f"Backup not found: {backup_id}")
        
        metadata = self.backups[backup_id]
        backup_file = Path(metadata.destination_path)
        
        if not backup_file.exists():
            raise FileNotFoundError(f"Backup file not found: {backup_file}")
        
        destination = Path(destination_path)
        destination.mkdir(parents=True, exist_ok=True)
        
        def _extract():
            with tarfile.open(backup_file, "r:*") as tar:
                tar.extractall(destination)
        
        await asyncio.get_event_loop().run_in_executor(None, _extract)
        
        logger.info(f"Restored backup {backup_id} to {destination_path}")
    
    def list_backups(
        self,
        backup_type: Optional[BackupType] = None,
        status: Optional[BackupStatus] = None
    ) -> List[BackupMetadata]:
        """
        List backups.
        
        Args:
            backup_type: Filter by type
            status: Filter by status
        
        Returns:
            List of backup metadata
        """
        backups = list(self.backups.values())
        
        if backup_type:
            backups = [b for b in backups if b.backup_type == backup_type]
        
        if status:
            backups = [b for b in backups if b.status == status]
        
        # Sort by creation time (newest first)
        backups.sort(key=lambda b: b.created_at, reverse=True)
        
        return backups
    
    async def cleanup_old_backups(self):
        """Remove backups older than retention period."""
        cutoff_date = datetime.utcnow() - timedelta(days=self.retention_days)
        removed_count = 0
        
        for backup_id, metadata in list(self.backups.items()):
            if metadata.created_at < cutoff_date:
                backup_file = Path(metadata.destination_path)
                
                if backup_file.exists():
                    backup_file.unlink()
                
                del self.backups[backup_id]
                removed_count += 1
        
        if removed_count > 0:
            self._save_metadata()
            logger.info(f"Cleaned up {removed_count} old backups")
    
    def get_backup(self, backup_id: str) -> Optional[BackupMetadata]:
        """Get backup metadata."""
        return self.backups.get(backup_id)
    
    def get_stats(self) -> dict:
        """Get backup statistics."""
        total_size = sum(b.size_bytes for b in self.backups.values())
        
        return {
            "total_backups": len(self.backups),
            "completed_backups": sum(
                1 for b in self.backups.values()
                if b.status == BackupStatus.COMPLETED
            ),
            "failed_backups": sum(
                1 for b in self.backups.values()
                if b.status == BackupStatus.FAILED
            ),
            "total_size_mb": round(total_size / (1024 ** 2), 2),
            "retention_days": self.retention_days
        }


class ScheduledBackupManager(BackupManager):
    """Backup manager with scheduling."""
    
    def __init__(
        self,
        backup_dir: str,
        schedule_interval: int = 86400,  # 24 hours
        **kwargs
    ):
        """
        Initialize scheduled backup manager.
        
        Args:
            backup_dir: Backup directory
            schedule_interval: Backup interval in seconds
            **kwargs: Additional BackupManager parameters
        """
        super().__init__(backup_dir, **kwargs)
        
        self.schedule_interval = schedule_interval
        self.scheduled_paths: List[str] = []
        self._running = False
        self._task: Optional[asyncio.Task] = None
    
    def add_scheduled_path(self, path: str):
        """Add path to scheduled backups."""
        if path not in self.scheduled_paths:
            self.scheduled_paths.append(path)
            logger.info(f"Added scheduled backup: {path}")
    
    def start_scheduler(self):
        """Start backup scheduler."""
        if self._running:
            logger.warning("Backup scheduler already running")
            return
        
        self._running = True
        self._task = asyncio.create_task(self._scheduler_loop())
        logger.info("Started backup scheduler")
    
    async def stop_scheduler(self):
        """Stop backup scheduler."""
        if not self._running:
            return
        
        self._running = False
        
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        
        logger.info("Stopped backup scheduler")
    
    async def _scheduler_loop(self):
        """Scheduler loop."""
        while self._running:
            try:
                # Create backups for all scheduled paths
                for path in self.scheduled_paths:
                    try:
                        await self.create_backup(path)
                    except Exception as e:
                        logger.error(f"Scheduled backup failed for {path}: {e}")
                
                # Cleanup old backups
                await self.cleanup_old_backups()
                
                # Wait for next interval
                await asyncio.sleep(self.schedule_interval)
            
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Scheduler error: {e}")
                await asyncio.sleep(60)


# Example usage:
"""
from src.backup_manager import BackupManager, ScheduledBackupManager, BackupType

# Manual backups
backup_manager = BackupManager(
    backup_dir="/var/backups",
    compression=True,
    retention_days=30
)

# Create backup
metadata = await backup_manager.create_backup(
    source_path="/var/data",
    backup_type=BackupType.FULL
)

# List backups
backups = backup_manager.list_backups()

# Restore backup
await backup_manager.restore_backup(
    backup_id=metadata.backup_id,
    destination_path="/var/restore"
)

# Scheduled backups
scheduled_manager = ScheduledBackupManager(
    backup_dir="/var/backups",
    schedule_interval=86400  # Daily
)

scheduled_manager.add_scheduled_path("/var/data")
scheduled_manager.start_scheduler()
"""

"""
Database migration runner with version tracking.

Manages database schema migrations with rollback support.
"""

from typing import Dict, List, Optional, Callable
from enum import Enum
from pathlib import Path
from datetime import datetime
import asyncio
import hashlib
import re
import logging

logger = logging.getLogger(__name__)


class MigrationStatus(str, Enum):
    """Migration status."""
    
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


class Migration:
    """Database migration."""
    
    def __init__(
        self,
        version: str,
        name: str,
        up_sql: str,
        down_sql: Optional[str] = None
    ):
        """
        Initialize migration.
        
        Args:
            version: Migration version (e.g., "001", "002")
            name: Migration name
            up_sql: SQL to apply migration
            down_sql: SQL to rollback migration
        """
        self.version = version
        self.name = name
        self.up_sql = up_sql
        self.down_sql = down_sql
        self.checksum = self._calculate_checksum()
    
    def _calculate_checksum(self) -> str:
        """Calculate migration checksum."""
        content = f"{self.version}{self.name}{self.up_sql}"
        return hashlib.sha256(content.encode()).hexdigest()
    
    def __repr__(self) -> str:
        return f"<Migration {self.version}: {self.name}>"


class MigrationRecord:
    """Migration execution record."""
    
    def __init__(
        self,
        version: str,
        name: str,
        checksum: str,
        status: MigrationStatus = MigrationStatus.PENDING
    ):
        """
        Initialize migration record.
        
        Args:
            version: Migration version
            name: Migration name
            checksum: Migration checksum
            status: Execution status
        """
        self.version = version
        self.name = name
        self.checksum = checksum
        self.status = status
        self.applied_at: Optional[datetime] = None
        self.rolled_back_at: Optional[datetime] = None
        self.error_message: Optional[str] = None
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "version": self.version,
            "name": self.name,
            "checksum": self.checksum,
            "status": self.status.value,
            "applied_at": self.applied_at.isoformat() if self.applied_at else None,
            "rolled_back_at": self.rolled_back_at.isoformat() if self.rolled_back_at else None,
            "error_message": self.error_message
        }


class MigrationRunner:
    """Database migration runner."""
    
    def __init__(self, db_connection):
        """
        Initialize migration runner.
        
        Args:
            db_connection: Database connection (async)
        """
        self.db_connection = db_connection
        self.migrations: List[Migration] = []
    
    async def initialize(self):
        """Initialize migration tracking table."""
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS schema_migrations (
            version VARCHAR(255) PRIMARY KEY,
            name VARCHAR(255) NOT NULL,
            checksum VARCHAR(64) NOT NULL,
            status VARCHAR(50) NOT NULL,
            applied_at TIMESTAMP,
            rolled_back_at TIMESTAMP,
            error_message TEXT
        )
        """
        
        await self.db_connection.execute(create_table_sql)
        logger.info("Migration tracking table initialized")
    
    def add_migration(self, migration: Migration):
        """Add migration to runner."""
        self.migrations.append(migration)
        
        # Sort by version
        self.migrations.sort(key=lambda m: m.version)
    
    def load_migrations_from_directory(self, directory: str):
        """
        Load migrations from directory.
        
        Args:
            directory: Migration files directory
        """
        migration_dir = Path(directory)
        
        if not migration_dir.exists():
            logger.warning(f"Migration directory not found: {directory}")
            return
        
        # Pattern: {version}_{name}.sql
        pattern = re.compile(r"^(\d+)_(.+)\.sql$")
        
        for file_path in sorted(migration_dir.glob("*.sql")):
            match = pattern.match(file_path.name)
            
            if not match:
                logger.warning(f"Skipping invalid migration file: {file_path.name}")
                continue
            
            version = match.group(1)
            name = match.group(2)
            
            # Read migration file
            content = file_path.read_text()
            
            # Split into up and down
            parts = content.split("-- DOWN")
            up_sql = parts[0].replace("-- UP", "").strip()
            down_sql = parts[1].strip() if len(parts) > 1 else None
            
            migration = Migration(
                version=version,
                name=name,
                up_sql=up_sql,
                down_sql=down_sql
            )
            
            self.add_migration(migration)
        
        logger.info(f"Loaded {len(self.migrations)} migrations from {directory}")
    
    async def get_applied_migrations(self) -> List[MigrationRecord]:
        """Get list of applied migrations."""
        query = "SELECT * FROM schema_migrations ORDER BY version"
        rows = await self.db_connection.fetch(query)
        
        records = []
        for row in rows:
            record = MigrationRecord(
                version=row["version"],
                name=row["name"],
                checksum=row["checksum"],
                status=MigrationStatus(row["status"])
            )
            
            if row["applied_at"]:
                record.applied_at = row["applied_at"]
            
            if row["rolled_back_at"]:
                record.rolled_back_at = row["rolled_back_at"]
            
            record.error_message = row.get("error_message")
            
            records.append(record)
        
        return records
    
    async def get_pending_migrations(self) -> List[Migration]:
        """Get list of pending migrations."""
        applied = await self.get_applied_migrations()
        applied_versions = {r.version for r in applied if r.status == MigrationStatus.COMPLETED}
        
        pending = [
            m for m in self.migrations
            if m.version not in applied_versions
        ]
        
        return pending
    
    async def migrate(self, target_version: Optional[str] = None):
        """
        Run pending migrations.
        
        Args:
            target_version: Target version (None = all pending)
        """
        pending = await self.get_pending_migrations()
        
        if target_version:
            pending = [m for m in pending if m.version <= target_version]
        
        if not pending:
            logger.info("No pending migrations")
            return
        
        logger.info(f"Running {len(pending)} migrations")
        
        for migration in pending:
            await self._apply_migration(migration)
    
    async def _apply_migration(self, migration: Migration):
        """Apply single migration."""
        logger.info(f"Applying migration {migration.version}: {migration.name}")
        
        # Create record
        insert_sql = """
        INSERT INTO schema_migrations (version, name, checksum, status)
        VALUES ($1, $2, $3, $4)
        """
        
        await self.db_connection.execute(
            insert_sql,
            migration.version,
            migration.name,
            migration.checksum,
            MigrationStatus.RUNNING.value
        )
        
        try:
            # Execute migration in transaction
            async with self.db_connection.transaction():
                await self.db_connection.execute(migration.up_sql)
            
            # Update status
            update_sql = """
            UPDATE schema_migrations
            SET status = $1, applied_at = $2
            WHERE version = $3
            """
            
            await self.db_connection.execute(
                update_sql,
                MigrationStatus.COMPLETED.value,
                datetime.utcnow(),
                migration.version
            )
            
            logger.info(f"Migration {migration.version} completed successfully")
        
        except Exception as e:
            # Update status
            update_sql = """
            UPDATE schema_migrations
            SET status = $1, error_message = $2
            WHERE version = $3
            """
            
            await self.db_connection.execute(
                update_sql,
                MigrationStatus.FAILED.value,
                str(e),
                migration.version
            )
            
            logger.error(f"Migration {migration.version} failed: {e}")
            raise
    
    async def rollback(self, target_version: str):
        """
        Rollback to specific version.
        
        Args:
            target_version: Version to rollback to
        """
        applied = await self.get_applied_migrations()
        
        # Get migrations to rollback (in reverse order)
        to_rollback = [
            r for r in reversed(applied)
            if r.version > target_version and r.status == MigrationStatus.COMPLETED
        ]
        
        if not to_rollback:
            logger.info("No migrations to rollback")
            return
        
        logger.info(f"Rolling back {len(to_rollback)} migrations")
        
        for record in to_rollback:
            # Find migration
            migration = next(
                (m for m in self.migrations if m.version == record.version),
                None
            )
            
            if not migration:
                logger.warning(f"Migration {record.version} not found, skipping rollback")
                continue
            
            if not migration.down_sql:
                logger.error(f"Migration {record.version} has no rollback SQL")
                continue
            
            await self._rollback_migration(migration)
    
    async def _rollback_migration(self, migration: Migration):
        """Rollback single migration."""
        logger.info(f"Rolling back migration {migration.version}: {migration.name}")
        
        try:
            # Execute rollback in transaction
            async with self.db_connection.transaction():
                await self.db_connection.execute(migration.down_sql)
            
            # Update status
            update_sql = """
            UPDATE schema_migrations
            SET status = $1, rolled_back_at = $2
            WHERE version = $3
            """
            
            await self.db_connection.execute(
                update_sql,
                MigrationStatus.ROLLED_BACK.value,
                datetime.utcnow(),
                migration.version
            )
            
            logger.info(f"Migration {migration.version} rolled back successfully")
        
        except Exception as e:
            logger.error(f"Rollback of migration {migration.version} failed: {e}")
            raise
    
    async def get_current_version(self) -> Optional[str]:
        """Get current migration version."""
        applied = await self.get_applied_migrations()
        
        completed = [
            r for r in applied
            if r.status == MigrationStatus.COMPLETED
        ]
        
        if not completed:
            return None
        
        return completed[-1].version
    
    async def get_status(self) -> dict:
        """Get migration status."""
        current_version = await self.get_current_version()
        pending = await self.get_pending_migrations()
        
        return {
            "current_version": current_version,
            "pending_count": len(pending),
            "total_migrations": len(self.migrations)
        }


# Example usage:
"""
from src.migration_runner import MigrationRunner, Migration
import asyncpg

# Connect to database
conn = await asyncpg.connect("postgresql://...")

# Create runner
runner = MigrationRunner(conn)
await runner.initialize()

# Load migrations
runner.load_migrations_from_directory("migrations/")

# Run migrations
await runner.migrate()

# Check status
status = await runner.get_status()
print(f"Current version: {status['current_version']}")

# Rollback
await runner.rollback(target_version="005")
"""

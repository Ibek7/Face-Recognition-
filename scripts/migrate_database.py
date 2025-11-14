#!/usr/bin/env python3
"""
Database Migration Script

Handles database schema migrations and data transformations
for the face recognition system. Supports both SQLite and PostgreSQL.
"""

import argparse
import hashlib
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from sqlalchemy import (
    Column,
    Integer,
    String,
    Float,
    DateTime,
    JSON,
    create_engine,
    inspect,
    text,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker


Base = declarative_base()


# Migration Tracking Table
class Migration(Base):
    """Track applied migrations."""

    __tablename__ = "migrations"

    id = Column(Integer, primary_key=True)
    version = Column(String(50), unique=True, nullable=False)
    name = Column(String(255), nullable=False)
    checksum = Column(String(64), nullable=False)
    applied_at = Column(DateTime, default=datetime.utcnow)
    execution_time_ms = Column(Integer)


class MigrationManager:
    """Manages database migrations."""

    def __init__(self, database_url: str):
        """Initialize migration manager."""
        self.database_url = database_url
        self.engine = create_engine(database_url)
        self.Session = sessionmaker(bind=self.engine)
        self.migrations_dir = Path(__file__).parent.parent / "migrations"
        self.migrations_dir.mkdir(exist_ok=True)

    def _ensure_migration_table(self):
        """Create migration tracking table if it doesn't exist."""
        Base.metadata.create_all(self.engine, tables=[Migration.__table__])

    def _get_applied_migrations(self) -> List[str]:
        """Get list of applied migration versions."""
        self._ensure_migration_table()
        session = self.Session()
        try:
            migrations = session.query(Migration.version).all()
            return [m[0] for m in migrations]
        finally:
            session.close()

    def _calculate_checksum(self, content: str) -> str:
        """Calculate checksum of migration content."""
        return hashlib.sha256(content.encode()).hexdigest()

    def _get_available_migrations(self) -> List[dict]:
        """Get list of available migration files."""
        migrations = []

        # Built-in migrations
        built_in = [
            {
                "version": "001",
                "name": "initial_schema",
                "sql": self._get_initial_schema_sql(),
            },
            {
                "version": "002",
                "name": "add_person_metadata",
                "sql": self._get_add_metadata_sql(),
            },
            {
                "version": "003",
                "name": "add_embedding_indexes",
                "sql": self._get_add_indexes_sql(),
            },
            {
                "version": "004",
                "name": "add_audit_tables",
                "sql": self._get_audit_tables_sql(),
            },
        ]

        for migration in built_in:
            migrations.append(
                {
                    "version": migration["version"],
                    "name": migration["name"],
                    "sql": migration["sql"],
                    "checksum": self._calculate_checksum(migration["sql"]),
                }
            )

        # Custom migrations from files
        for file in sorted(self.migrations_dir.glob("*.sql")):
            version = file.stem.split("_")[0]
            name = "_".join(file.stem.split("_")[1:])
            with open(file) as f:
                sql = f.read()

            migrations.append(
                {
                    "version": version,
                    "name": name,
                    "sql": sql,
                    "checksum": self._calculate_checksum(sql),
                }
            )

        return sorted(migrations, key=lambda x: x["version"])

    def _get_initial_schema_sql(self) -> str:
        """Get initial schema SQL."""
        return """
        -- Persons table
        CREATE TABLE IF NOT EXISTS persons (
            id SERIAL PRIMARY KEY,
            name VARCHAR(255) NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        -- Face embeddings table
        CREATE TABLE IF NOT EXISTS embeddings (
            id SERIAL PRIMARY KEY,
            person_id INTEGER NOT NULL REFERENCES persons(id) ON DELETE CASCADE,
            embedding BYTEA NOT NULL,
            image_hash VARCHAR(64),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(person_id, image_hash)
        );

        -- Create indexes
        CREATE INDEX IF NOT EXISTS idx_persons_name ON persons(name);
        CREATE INDEX IF NOT EXISTS idx_embeddings_person_id ON embeddings(person_id);
        """

    def _get_add_metadata_sql(self) -> str:
        """Add metadata column to persons table."""
        return """
        -- Add metadata column if it doesn't exist
        DO $$ 
        BEGIN 
            IF NOT EXISTS (
                SELECT 1 FROM information_schema.columns 
                WHERE table_name='persons' AND column_name='metadata'
            ) THEN
                ALTER TABLE persons ADD COLUMN metadata JSONB;
            END IF;
        END $$;

        -- Create GIN index for JSONB metadata
        CREATE INDEX IF NOT EXISTS idx_persons_metadata ON persons USING GIN (metadata);
        """

    def _get_add_indexes_sql(self) -> str:
        """Add performance indexes."""
        return """
        -- Add composite indexes for better query performance
        CREATE INDEX IF NOT EXISTS idx_embeddings_person_created 
            ON embeddings(person_id, created_at DESC);
        
        -- Add index on created_at for persons
        CREATE INDEX IF NOT EXISTS idx_persons_created_at 
            ON persons(created_at DESC);
        """

    def _get_audit_tables_sql(self) -> str:
        """Create audit tables."""
        return """
        -- Audit log table
        CREATE TABLE IF NOT EXISTS audit_log (
            id SERIAL PRIMARY KEY,
            table_name VARCHAR(100) NOT NULL,
            record_id INTEGER NOT NULL,
            action VARCHAR(50) NOT NULL,
            old_values JSONB,
            new_values JSONB,
            user_id VARCHAR(255),
            ip_address VARCHAR(45),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        -- Create indexes
        CREATE INDEX IF NOT EXISTS idx_audit_log_table_record 
            ON audit_log(table_name, record_id);
        CREATE INDEX IF NOT EXISTS idx_audit_log_created_at 
            ON audit_log(created_at DESC);
        CREATE INDEX IF NOT EXISTS idx_audit_log_user 
            ON audit_log(user_id);
        """

    def _apply_migration(self, migration: dict) -> int:
        """Apply a single migration."""
        print(f"Applying migration {migration['version']}: {migration['name']}...")

        start_time = datetime.now()
        session = self.Session()

        try:
            # Execute migration SQL
            for statement in migration["sql"].split(";"):
                statement = statement.strip()
                if statement:
                    session.execute(text(statement))

            # Record migration
            migration_record = Migration(
                version=migration["version"],
                name=migration["name"],
                checksum=migration["checksum"],
                applied_at=datetime.utcnow(),
                execution_time_ms=int(
                    (datetime.now() - start_time).total_seconds() * 1000
                ),
            )
            session.add(migration_record)
            session.commit()

            execution_time = int(
                (datetime.now() - start_time).total_seconds() * 1000
            )
            print(f"✓ Migration {migration['version']} applied ({execution_time}ms)")
            return execution_time

        except Exception as e:
            session.rollback()
            print(f"✗ Migration {migration['version']} failed: {e}")
            raise
        finally:
            session.close()

    def migrate(self, target_version: Optional[str] = None):
        """Run pending migrations."""
        print("=" * 60)
        print("DATABASE MIGRATION")
        print("=" * 60)
        print(f"Database: {self._get_db_type()}")
        print(f"URL: {self._mask_password(self.database_url)}")
        print()

        self._ensure_migration_table()
        applied = set(self._get_applied_migrations())
        available = self._get_available_migrations()

        # Filter migrations
        pending = [m for m in available if m["version"] not in applied]

        if target_version:
            pending = [m for m in pending if m["version"] <= target_version]

        if not pending:
            print("No pending migrations.")
            return

        print(f"Found {len(pending)} pending migration(s):")
        for migration in pending:
            print(f"  - {migration['version']}: {migration['name']}")
        print()

        # Apply migrations
        total_time = 0
        for migration in pending:
            execution_time = self._apply_migration(migration)
            total_time += execution_time

        print()
        print("=" * 60)
        print(f"Migration completed successfully!")
        print(f"Applied {len(pending)} migration(s) in {total_time}ms")
        print("=" * 60)

    def rollback(self, steps: int = 1):
        """Rollback migrations."""
        print("=" * 60)
        print("MIGRATION ROLLBACK")
        print("=" * 60)

        applied = self._get_applied_migrations()
        if not applied:
            print("No migrations to rollback.")
            return

        # Get migrations to rollback
        to_rollback = sorted(applied, reverse=True)[:steps]

        print(f"Rolling back {len(to_rollback)} migration(s):")
        for version in to_rollback:
            print(f"  - {version}")
        print()

        print("⚠️  WARNING: Rollback functionality is limited.")
        print("Consider creating a database backup before proceeding.")
        print()

        # For now, just remove from migration table
        # In production, you'd have down migrations
        session = self.Session()
        try:
            for version in to_rollback:
                session.query(Migration).filter(
                    Migration.version == version
                ).delete()
            session.commit()
            print(f"✓ Rolled back {len(to_rollback)} migration(s)")
        except Exception as e:
            session.rollback()
            print(f"✗ Rollback failed: {e}")
            raise
        finally:
            session.close()

    def status(self):
        """Show migration status."""
        print("=" * 60)
        print("MIGRATION STATUS")
        print("=" * 60)
        print(f"Database: {self._get_db_type()}")
        print()

        applied = set(self._get_applied_migrations())
        available = self._get_available_migrations()

        print(f"Total migrations: {len(available)}")
        print(f"Applied: {len(applied)}")
        print(f"Pending: {len(available) - len(applied)}")
        print()

        if available:
            print("Migration History:")
            print(f"{'Version':<10} {'Name':<30} {'Status':<10}")
            print("-" * 60)
            for migration in available:
                status = "✓ Applied" if migration["version"] in applied else "Pending"
                print(
                    f"{migration['version']:<10} {migration['name']:<30} {status:<10}"
                )
        else:
            print("No migrations found.")

    def _get_db_type(self) -> str:
        """Get database type."""
        if "sqlite" in self.database_url:
            return "SQLite"
        elif "postgresql" in self.database_url:
            return "PostgreSQL"
        else:
            return "Unknown"

    def _mask_password(self, url: str) -> str:
        """Mask password in database URL."""
        if "@" in url and "://" in url:
            protocol, rest = url.split("://", 1)
            if "@" in rest:
                credentials, host = rest.split("@", 1)
                if ":" in credentials:
                    user, _ = credentials.split(":", 1)
                    return f"{protocol}://{user}:****@{host}"
        return url


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Database migration tool for Face Recognition"
    )
    parser.add_argument(
        "--database-url",
        default=os.getenv(
            "DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/face_recognition"
        ),
        help="Database URL",
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Migrate command
    migrate_parser = subparsers.add_parser("migrate", help="Run pending migrations")
    migrate_parser.add_argument(
        "--target", help="Target migration version", default=None
    )

    # Rollback command
    rollback_parser = subparsers.add_parser("rollback", help="Rollback migrations")
    rollback_parser.add_argument(
        "--steps", type=int, default=1, help="Number of migrations to rollback"
    )

    # Status command
    subparsers.add_parser("status", help="Show migration status")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    manager = MigrationManager(args.database_url)

    try:
        if args.command == "migrate":
            manager.migrate(args.target)
        elif args.command == "rollback":
            manager.rollback(args.steps)
        elif args.command == "status":
            manager.status()
    except Exception as e:
        print(f"\n✗ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

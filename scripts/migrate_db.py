#!/usr/bin/env python3
"""
Database migration script template.

Usage:
    python scripts/migrate_db.py --action create --name "add_user_table"
    python scripts/migrate_db.py --action upgrade
    python scripts/migrate_db.py --action downgrade
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

try:
    import psycopg2
    from psycopg2 import sql
except ImportError:
    print("Warning: psycopg2 not installed. Install with: pip install psycopg2-binary")
    psycopg2 = None


MIGRATIONS_DIR = Path(__file__).parent.parent / "migrations"
MIGRATIONS_DIR.mkdir(exist_ok=True)


class MigrationManager:
    """Handle database migrations."""
    
    def __init__(self, db_url: str):
        """
        Initialize migration manager.
        
        Args:
            db_url: Database connection URL
        """
        self.db_url = db_url
        self._ensure_migrations_table()
    
    def _get_connection(self):
        """Get database connection."""
        if not psycopg2:
            raise ImportError("psycopg2 required for migrations")
        return psycopg2.connect(self.db_url)
    
    def _ensure_migrations_table(self):
        """Create migrations tracking table if not exists."""
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS schema_migrations (
                        id SERIAL PRIMARY KEY,
                        version VARCHAR(255) UNIQUE NOT NULL,
                        applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                conn.commit()
    
    def create_migration(self, name: str) -> Path:
        """
        Create a new migration file.
        
        Args:
            name: Migration name
            
        Returns:
            Path to created migration file
        """
        timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        version = f"{timestamp}_{name}"
        filepath = MIGRATIONS_DIR / f"{version}.sql"
        
        template = f"""-- Migration: {name}
-- Created: {datetime.utcnow().isoformat()}

-- Upgrade
-- ========================================

-- Add your upgrade SQL here


-- Downgrade
-- ========================================
-- Add your downgrade SQL here (optional)

"""
        filepath.write_text(template)
        print(f"Created migration: {filepath}")
        return filepath
    
    def get_pending_migrations(self) -> list:
        """Get list of pending migrations."""
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT version FROM schema_migrations")
                applied = {row[0] for row in cur.fetchall()}
        
        all_migrations = sorted([
            f.stem for f in MIGRATIONS_DIR.glob("*.sql")
        ])
        
        return [m for m in all_migrations if m not in applied]
    
    def apply_migration(self, version: str):
        """
        Apply a migration.
        
        Args:
            version: Migration version to apply
        """
        filepath = MIGRATIONS_DIR / f"{version}.sql"
        if not filepath.exists():
            raise FileNotFoundError(f"Migration not found: {filepath}")
        
        sql_content = filepath.read_text()
        
        # Extract upgrade section
        if "-- Upgrade" in sql_content:
            parts = sql_content.split("-- Downgrade")
            upgrade_sql = parts[0].split("-- Upgrade")[1].strip()
        else:
            upgrade_sql = sql_content.strip()
        
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                # Apply migration
                cur.execute(upgrade_sql)
                
                # Record migration
                cur.execute(
                    "INSERT INTO schema_migrations (version) VALUES (%s)",
                    (version,)
                )
                
                conn.commit()
        
        print(f"Applied migration: {version}")
    
    def upgrade(self):
        """Apply all pending migrations."""
        pending = self.get_pending_migrations()
        
        if not pending:
            print("No pending migrations")
            return
        
        print(f"Applying {len(pending)} migrations...")
        for version in pending:
            self.apply_migration(version)
        
        print("All migrations applied successfully")
    
    def status(self):
        """Show migration status."""
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT version, applied_at 
                    FROM schema_migrations 
                    ORDER BY applied_at DESC
                """)
                applied = cur.fetchall()
        
        print(f"\nApplied migrations: {len(applied)}")
        for version, applied_at in applied[:10]:  # Show last 10
            print(f"  {version} - {applied_at}")
        
        pending = self.get_pending_migrations()
        print(f"\nPending migrations: {len(pending)}")
        for version in pending:
            print(f"  {version}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Database migration tool")
    
    parser.add_argument(
        "--action",
        choices=["create", "upgrade", "status"],
        required=True,
        help="Migration action"
    )
    
    parser.add_argument(
        "--name",
        help="Migration name (for create action)"
    )
    
    parser.add_argument(
        "--db-url",
        default="postgresql://localhost/face_recognition",
        help="Database URL"
    )
    
    args = parser.parse_args()
    
    if args.action == "create":
        if not args.name:
            print("Error: --name required for create action")
            sys.exit(1)
        
        manager = MigrationManager(args.db_url)
        manager.create_migration(args.name)
    
    elif args.action == "upgrade":
        manager = MigrationManager(args.db_url)
        manager.upgrade()
    
    elif args.action == "status":
        manager = MigrationManager(args.db_url)
        manager.status()


if __name__ == "__main__":
    main()

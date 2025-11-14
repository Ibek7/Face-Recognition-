# Database Migration Guide

## Overview

This guide explains how to manage database schema changes and migrations.

## Migration Tools

We use Alembic for database migrations (to be integrated).

### Installation

```bash
pip install alembic
```

### Initialize Alembic

```bash
alembic init alembic
```

## Manual Migration Steps

### SQLite to PostgreSQL

#### 1. Export Data from SQLite

```python
from src.database import DatabaseManager
import json

# Connect to SQLite
db = DatabaseManager("sqlite:///face_recognition.db")

# Export persons
persons = db.list_persons(active_only=False)
persons_data = [
    {
        "id": p.id,
        "name": p.name,
        "description": p.description,
        "created_at": str(p.created_at),
        "updated_at": str(p.updated_at),
        "is_active": p.is_active
    }
    for p in persons
]

with open("export_persons.json", "w") as f:
    json.dump(persons_data, f, indent=2)
```

#### 2. Import Data to PostgreSQL

```python
from src.database import DatabaseManager
import json

# Connect to PostgreSQL
db = DatabaseManager("postgresql://user:password@localhost/face_recognition")

# Import persons
with open("export_persons.json", "r") as f:
    persons_data = json.load(f)

for person_data in persons_data:
    db.add_person(
        name=person_data["name"],
        description=person_data["description"]
    )
```

## Schema Versions

### v1.0 (Current)

**Tables:**
- `persons` - Person records
- `face_embeddings` - Face embedding vectors
- `recognition_results` - Recognition history
- `datasets` - Training datasets
- `processing_jobs` - Batch processing jobs

**Indexes:**
- `idx_persons_name` - Fast person lookup by name
- `idx_face_embeddings_person_id` - Fast embedding lookup
- `idx_recognition_results_timestamp` - Time-based queries

### Future Migrations

#### v1.1 - Add User Authentication

```sql
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

ALTER TABLE persons ADD COLUMN user_id INTEGER REFERENCES users(id);
```

#### v1.2 - Add Face Landmarks

```sql
ALTER TABLE face_embeddings ADD COLUMN landmarks JSONB;
CREATE INDEX idx_face_embeddings_landmarks ON face_embeddings USING gin(landmarks);
```

## Backup and Restore

### Backup SQLite

```bash
# Create backup
cp face_recognition.db face_recognition_backup_$(date +%Y%m%d).db

# Or use SQLite backup command
sqlite3 face_recognition.db ".backup face_recognition_backup.db"
```

### Backup PostgreSQL

```bash
# Full backup
pg_dump -U user -d face_recognition > backup_$(date +%Y%m%d).sql

# Compressed backup
pg_dump -U user -d face_recognition | gzip > backup_$(date +%Y%m%d).sql.gz
```

### Restore PostgreSQL

```bash
# From uncompressed backup
psql -U user -d face_recognition < backup_20251113.sql

# From compressed backup
gunzip -c backup_20251113.sql.gz | psql -U user -d face_recognition
```

## Database Maintenance

### Vacuum (PostgreSQL)

```sql
-- Reclaim storage and update statistics
VACUUM ANALYZE;

-- Full vacuum (locks table)
VACUUM FULL;
```

### Reindex

```sql
-- Rebuild specific index
REINDEX INDEX idx_persons_name;

-- Rebuild all indexes on table
REINDEX TABLE persons;
```

### Cleanup Old Data

```python
from src.database import DatabaseManager

db = DatabaseManager()

# Remove recognition results older than 30 days
deleted_count = db.cleanup_old_results(days=30)
print(f"Deleted {deleted_count} old records")
```

## Performance Optimization

### Add Indexes

```sql
-- Speed up person lookups
CREATE INDEX idx_persons_name ON persons(name);

-- Speed up embedding queries
CREATE INDEX idx_face_embeddings_person_id ON face_embeddings(person_id);

-- Speed up recognition history queries
CREATE INDEX idx_recognition_results_timestamp ON recognition_results(recognized_at);
```

### Analyze Query Performance

```sql
-- PostgreSQL
EXPLAIN ANALYZE SELECT * FROM persons WHERE name = 'John Doe';

-- SQLite
EXPLAIN QUERY PLAN SELECT * FROM persons WHERE name = 'John Doe';
```

## Connection Pooling

For production, use connection pooling:

```python
from sqlalchemy import create_engine, pool

engine = create_engine(
    "postgresql://user:password@localhost/face_recognition",
    poolclass=pool.QueuePool,
    pool_size=10,
    max_overflow=20
)
```

## Monitoring

### Check Database Size

```sql
-- PostgreSQL
SELECT pg_size_pretty(pg_database_size('face_recognition'));

-- SQLite
SELECT page_count * page_size as size FROM pragma_page_count(), pragma_page_size();
```

### Active Connections

```sql
-- PostgreSQL
SELECT count(*) FROM pg_stat_activity WHERE datname = 'face_recognition';
```

### Slow Queries

```sql
-- Enable logging
ALTER DATABASE face_recognition SET log_min_duration_statement = 1000;  -- Log queries > 1s
```

## Troubleshooting

### Connection Issues

```bash
# Test PostgreSQL connection
psql -h localhost -U user -d face_recognition -c "SELECT 1;"

# Check SQLite file permissions
ls -la face_recognition.db
```

### Lock Issues

```sql
-- Find blocking queries (PostgreSQL)
SELECT pid, usename, pg_blocking_pids(pid) as blocked_by, query 
FROM pg_stat_activity 
WHERE cardinality(pg_blocking_pids(pid)) > 0;
```

### Corruption Recovery

```bash
# SQLite integrity check
sqlite3 face_recognition.db "PRAGMA integrity_check;"

# Restore from backup if corrupted
cp face_recognition_backup.db face_recognition.db
```

## Best Practices

1. **Always backup before migrations**
2. **Test migrations on a copy first**
3. **Use transactions for data modifications**
4. **Monitor database performance regularly**
5. **Keep connection pools sized appropriately**
6. **Clean up old data periodically**
7. **Document all schema changes**

## Resources

- [SQLAlchemy Documentation](https://docs.sqlalchemy.org/)
- [Alembic Tutorial](https://alembic.sqlalchemy.org/en/latest/tutorial.html)
- [PostgreSQL Performance Tips](https://wiki.postgresql.org/wiki/Performance_Optimization)

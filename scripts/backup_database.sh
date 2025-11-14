#!/bin/bash

###############################################################################
# Database Backup Script
# 
# Automated backup script for the face recognition database.
# Supports both SQLite and PostgreSQL databases with compression and rotation.
###############################################################################

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
BACKUP_DIR="${PROJECT_ROOT}/backups/db"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RETENTION_DAYS=30

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Create backup directory if it doesn't exist
mkdir -p "$BACKUP_DIR"

# Load environment variables
if [ -f "${PROJECT_ROOT}/.env" ]; then
    set -a
    source "${PROJECT_ROOT}/.env"
    set +a
    log_info "Loaded environment variables from .env"
fi

# Determine database type
DB_TYPE="${DATABASE_TYPE:-sqlite}"

###############################################################################
# SQLite Backup
###############################################################################
backup_sqlite() {
    local db_file="${1:-${PROJECT_ROOT}/data/face_recognition.db}"
    
    if [ ! -f "$db_file" ]; then
        log_error "SQLite database file not found: $db_file"
        return 1
    fi
    
    local backup_file="${BACKUP_DIR}/sqlite_backup_${TIMESTAMP}.db"
    
    log_info "Backing up SQLite database..."
    log_info "Source: $db_file"
    log_info "Destination: $backup_file"
    
    # Create backup using SQLite's backup command
    sqlite3 "$db_file" ".backup '${backup_file}'"
    
    if [ $? -eq 0 ]; then
        log_info "SQLite backup created successfully"
        
        # Compress the backup
        log_info "Compressing backup..."
        gzip "$backup_file"
        backup_file="${backup_file}.gz"
        
        local size=$(du -h "$backup_file" | cut -f1)
        log_info "Compressed backup size: $size"
        log_info "Backup location: $backup_file"
        
        return 0
    else
        log_error "SQLite backup failed"
        return 1
    fi
}

###############################################################################
# PostgreSQL Backup
###############################################################################
backup_postgresql() {
    local db_host="${POSTGRES_HOST:-localhost}"
    local db_port="${POSTGRES_PORT:-5432}"
    local db_name="${POSTGRES_DB:-face_recognition}"
    local db_user="${POSTGRES_USER:-postgres}"
    
    log_info "Backing up PostgreSQL database..."
    log_info "Host: $db_host:$db_port"
    log_info "Database: $db_name"
    log_info "User: $db_user"
    
    local backup_file="${BACKUP_DIR}/postgres_backup_${TIMESTAMP}.sql"
    
    # Check if pg_dump is available
    if ! command -v pg_dump &> /dev/null; then
        log_error "pg_dump command not found. Please install PostgreSQL client tools."
        return 1
    fi
    
    # Create backup using pg_dump
    PGPASSWORD="${POSTGRES_PASSWORD}" pg_dump \
        -h "$db_host" \
        -p "$db_port" \
        -U "$db_user" \
        -d "$db_name" \
        --verbose \
        --format=plain \
        --no-owner \
        --no-acl \
        -f "$backup_file"
    
    if [ $? -eq 0 ]; then
        log_info "PostgreSQL backup created successfully"
        
        # Compress the backup
        log_info "Compressing backup..."
        gzip "$backup_file"
        backup_file="${backup_file}.gz"
        
        local size=$(du -h "$backup_file" | cut -f1)
        log_info "Compressed backup size: $size"
        log_info "Backup location: $backup_file"
        
        return 0
    else
        log_error "PostgreSQL backup failed"
        return 1
    fi
}

###############################################################################
# Custom Format PostgreSQL Backup (for large databases)
###############################################################################
backup_postgresql_custom() {
    local db_host="${POSTGRES_HOST:-localhost}"
    local db_port="${POSTGRES_PORT:-5432}"
    local db_name="${POSTGRES_DB:-face_recognition}"
    local db_user="${POSTGRES_USER:-postgres}"
    
    log_info "Creating custom format PostgreSQL backup..."
    
    local backup_file="${BACKUP_DIR}/postgres_backup_${TIMESTAMP}.dump"
    
    # Create backup using pg_dump with custom format
    PGPASSWORD="${POSTGRES_PASSWORD}" pg_dump \
        -h "$db_host" \
        -p "$db_port" \
        -U "$db_user" \
        -d "$db_name" \
        --verbose \
        --format=custom \
        --compress=9 \
        -f "$backup_file"
    
    if [ $? -eq 0 ]; then
        log_info "Custom format backup created successfully"
        
        local size=$(du -h "$backup_file" | cut -f1)
        log_info "Backup size: $size"
        log_info "Backup location: $backup_file"
        
        return 0
    else
        log_error "Custom format backup failed"
        return 1
    fi
}

###############################################################################
# Cleanup old backups
###############################################################################
cleanup_old_backups() {
    log_info "Cleaning up backups older than ${RETENTION_DAYS} days..."
    
    local deleted_count=0
    
    # Find and delete old backup files
    while IFS= read -r -d '' file; do
        rm -f "$file"
        deleted_count=$((deleted_count + 1))
        log_info "Deleted old backup: $(basename "$file")"
    done < <(find "$BACKUP_DIR" -type f -name "*backup_*.{db.gz,sql.gz,dump}" -mtime +${RETENTION_DAYS} -print0)
    
    if [ $deleted_count -gt 0 ]; then
        log_info "Deleted $deleted_count old backup(s)"
    else
        log_info "No old backups to delete"
    fi
}

###############################################################################
# Verify backup integrity
###############################################################################
verify_backup() {
    local backup_file="$1"
    
    log_info "Verifying backup integrity..."
    
    if [[ "$backup_file" == *.gz ]]; then
        if gzip -t "$backup_file" 2>/dev/null; then
            log_info "Backup integrity verified successfully"
            return 0
        else
            log_error "Backup integrity check failed"
            return 1
        fi
    else
        # For custom format dumps, use pg_restore --list
        if [[ "$backup_file" == *.dump ]]; then
            if pg_restore --list "$backup_file" > /dev/null 2>&1; then
                log_info "Backup integrity verified successfully"
                return 0
            else
                log_error "Backup integrity check failed"
                return 1
            fi
        fi
    fi
    
    return 0
}

###############################################################################
# Main execution
###############################################################################
main() {
    log_info "Starting database backup process..."
    log_info "Database type: $DB_TYPE"
    
    local backup_success=false
    
    case "$DB_TYPE" in
        sqlite)
            if backup_sqlite; then
                backup_success=true
            fi
            ;;
        postgresql|postgres)
            # Check if we should use custom format for large databases
            if [ "${USE_CUSTOM_FORMAT:-false}" = "true" ]; then
                if backup_postgresql_custom; then
                    backup_success=true
                fi
            else
                if backup_postgresql; then
                    backup_success=true
                fi
            fi
            ;;
        *)
            log_error "Unknown database type: $DB_TYPE"
            log_error "Supported types: sqlite, postgresql"
            exit 1
            ;;
    esac
    
    if [ "$backup_success" = true ]; then
        # Find the most recent backup file
        local latest_backup=$(find "$BACKUP_DIR" -type f -name "*backup_${TIMESTAMP}.*" | head -n 1)
        
        if [ -n "$latest_backup" ]; then
            verify_backup "$latest_backup"
        fi
        
        # Cleanup old backups
        cleanup_old_backups
        
        log_info "Database backup completed successfully"
        
        # Show backup summary
        echo ""
        log_info "=== Backup Summary ==="
        log_info "Total backups: $(find "$BACKUP_DIR" -type f | wc -l)"
        log_info "Backup directory size: $(du -sh "$BACKUP_DIR" | cut -f1)"
        
        exit 0
    else
        log_error "Database backup failed"
        exit 1
    fi
}

# Run main function
main "$@"

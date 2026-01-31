#!/bin/bash
# Database backup script for Tuner-UI
# Backs up PostgreSQL database and artifacts directory

set -e

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="${BACKUP_DIR:-/backups}"
DB_NAME="${POSTGRES_DB:-tunerui}"
DB_USER="${POSTGRES_USER:-tuner_user}"

echo "Starting backup at $TIMESTAMP"

# Create backup directory
mkdir -p "$BACKUP_DIR"

# PostgreSQL backup
echo "Backing up PostgreSQL database..."
docker exec tuner-ui-postgres pg_dump -U "$DB_USER" "$DB_NAME" | gzip > "$BACKUP_DIR/db_$TIMESTAMP.sql.gz"

# Artifacts backup
if [ -d "/app/artifacts" ]; then
    echo "Backing up artifacts..."
    tar -czf "$BACKUP_DIR/artifacts_$TIMESTAMP.tar.gz" /app/artifacts
fi

# Cleanup old backups (keep last 30 days)
echo "Cleaning up old backups..."
find "$BACKUP_DIR" -name "db_*.sql.gz" -mtime +30 -delete
find "$BACKUP_DIR" -name "artifacts_*.tar.gz" -mtime +30 -delete

echo "Backup completed successfully: $TIMESTAMP"
echo "Database: $BACKUP_DIR/db_$TIMESTAMP.sql.gz"
echo "Artifacts: $BACKUP_DIR/artifacts_$TIMESTAMP.tar.gz"

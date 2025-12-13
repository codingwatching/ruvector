#!/bin/bash
# Start PostgreSQL with ruvector extension, then run the Node.js actor

set -e

export POSTGRES_PASSWORD="${POSTGRES_PASSWORD:-secret}"
export POSTGRES_USER="${POSTGRES_USER:-postgres}"
export POSTGRES_DB="${POSTGRES_DB:-ruvector}"
export PGPASSWORD="$POSTGRES_PASSWORD"

echo "=== RuVector PostgreSQL Actor ==="
echo "Environment: USER=$POSTGRES_USER DB=$POSTGRES_DB"

# Start PostgreSQL (from base image entrypoint)
echo "Starting PostgreSQL with ruvector extension..."
docker-entrypoint.sh postgres &
PG_PID=$!

# Wait for PostgreSQL to be ready with better error handling
echo "Waiting for PostgreSQL to be ready..."
READY=false
for i in $(seq 1 60); do
    if pg_isready -h localhost -U "$POSTGRES_USER" -d "$POSTGRES_DB" -q 2>/dev/null; then
        echo "PostgreSQL is ready after ${i}s!"
        READY=true
        break
    fi
    echo "Waiting... ($i/60)"
    sleep 1
done

if [ "$READY" != "true" ]; then
    echo "ERROR: PostgreSQL failed to start within 60 seconds"
    exit 1
fi

# Enable ruvector extension (use -X to skip psqlrc, < /dev/null to avoid blocking)
echo "Enabling ruvector extension..."
psql -X -h localhost -U "$POSTGRES_USER" -d "$POSTGRES_DB" -c "CREATE EXTENSION IF NOT EXISTS ruvector;" < /dev/null 2>&1 || {
    echo "Warning: Could not create ruvector extension, trying pgvector..."
    psql -X -h localhost -U "$POSTGRES_USER" -d "$POSTGRES_DB" -c "CREATE EXTENSION IF NOT EXISTS vector;" < /dev/null 2>&1 || true
}
echo "Extension enabled."

# Show version
echo "Checking extension version..."
psql -X -h localhost -U "$POSTGRES_USER" -d "$POSTGRES_DB" -t -c "SELECT ruvector_version();" < /dev/null 2>&1 || echo "Extension ready (pgvector mode)"
echo "Version check complete."

# Export connection string for Node.js
export DATABASE_URL="postgresql://${POSTGRES_USER}:${POSTGRES_PASSWORD}@localhost:5432/${POSTGRES_DB}"
echo "DATABASE_URL set (local)"

echo "=== Starting Node.js Actor ==="
echo "Working directory: $(pwd)"
cd /app

# Run Node.js actor
echo "Launching Node.js..."
node src/main.js
EXIT_CODE=$?

echo "=== Actor completed with exit code: $EXIT_CODE ==="
exit $EXIT_CODE

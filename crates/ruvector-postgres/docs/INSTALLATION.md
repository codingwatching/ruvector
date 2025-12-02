# RuVector-Postgres Installation Guide

## Overview

This guide covers installation of RuVector-Postgres on various platforms including standard PostgreSQL, Neon, Supabase, and containerized environments.

## Prerequisites

### System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| PostgreSQL | 14+ | 16+ |
| RAM | 4 GB | 16+ GB |
| CPU | x86_64 or ARM64 | x86_64 with AVX2+ |
| Disk | 10 GB | SSD recommended |

### Build Requirements

| Tool | Version |
|------|---------|
| Rust | 1.75+ |
| Cargo | 1.75+ |
| pgrx | 0.12+ |
| PostgreSQL Dev | 14-18 |
| clang | 14+ |
| pkg-config | any |

## Installation Methods

### Method 1: Build from Source (Recommended)

#### Step 1: Install Rust

```bash
# Install Rust via rustup
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env

# Verify installation
rustc --version
cargo --version
```

#### Step 2: Install pgrx

```bash
# Install pgrx CLI
cargo install --locked cargo-pgrx@0.12.9

# Initialize pgrx for your PostgreSQL version
cargo pgrx init --pg16 $(which pg_config)

# Or for multiple versions:
cargo pgrx init \
    --pg14 /usr/lib/postgresql/14/bin/pg_config \
    --pg15 /usr/lib/postgresql/15/bin/pg_config \
    --pg16 /usr/lib/postgresql/16/bin/pg_config
```

#### Step 3: Build the Extension

```bash
# Clone the repository
git clone https://github.com/ruvnet/ruvector.git
cd ruvector/crates/ruvector-postgres

# Build for your PostgreSQL version
cargo pgrx package --pg-config $(which pg_config)

# The built extension will be in:
# target/release/ruvector-pg16/
```

#### Step 4: Install the Extension

```bash
# Copy files to PostgreSQL directories
sudo cp target/release/ruvector-pg16/usr/share/postgresql/16/extension/* \
    /usr/share/postgresql/16/extension/

sudo cp target/release/ruvector-pg16/usr/lib/postgresql/16/lib/* \
    /usr/lib/postgresql/16/lib/

# Restart PostgreSQL
sudo systemctl restart postgresql
```

#### Step 5: Enable in Database

```sql
-- Connect to your database
psql -U postgres -d your_database

-- Create the extension
CREATE EXTENSION ruvector;

-- Verify installation
SELECT ruvector_version();
```

### Method 2: Pre-built Packages

#### Debian/Ubuntu

```bash
# Add repository (when available)
curl -fsSL https://packages.ruvector.io/gpg | sudo gpg --dearmor -o /usr/share/keyrings/ruvector.gpg
echo "deb [signed-by=/usr/share/keyrings/ruvector.gpg] https://packages.ruvector.io/apt stable main" | \
    sudo tee /etc/apt/sources.list.d/ruvector.list

# Install
sudo apt update
sudo apt install postgresql-16-ruvector

# Enable
sudo -u postgres psql -c "CREATE EXTENSION ruvector;"
```

#### RHEL/CentOS/Fedora

```bash
# Add repository (when available)
sudo dnf config-manager --add-repo https://packages.ruvector.io/rpm/ruvector.repo

# Install
sudo dnf install postgresql16-ruvector

# Enable
sudo -u postgres psql -c "CREATE EXTENSION ruvector;"
```

#### macOS (Homebrew)

```bash
# Install PostgreSQL if needed
brew install postgresql@16

# Install ruvector (when available)
brew install ruvector-postgres

# Enable
psql -c "CREATE EXTENSION ruvector;"
```

### Method 3: Docker

#### Using Pre-built Image

```bash
# Pull the image
docker pull ruvector/postgres:16

# Run container
docker run -d \
    --name ruvector-postgres \
    -e POSTGRES_PASSWORD=mysecretpassword \
    -p 5432:5432 \
    ruvector/postgres:16

# Connect and enable
docker exec -it ruvector-postgres psql -U postgres -c "CREATE EXTENSION ruvector;"
```

#### Building Custom Image

```dockerfile
# Dockerfile
FROM postgres:16

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    pkg-config \
    libssl-dev \
    libclang-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Rust
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# Install pgrx
RUN cargo install --locked cargo-pgrx@0.12.9
RUN cargo pgrx init --pg16 /usr/lib/postgresql/16/bin/pg_config

# Copy and build extension
COPY crates/ruvector-postgres /app/ruvector-postgres
WORKDIR /app/ruvector-postgres
RUN cargo pgrx install --release --pg-config /usr/lib/postgresql/16/bin/pg_config

# Clean up build dependencies
RUN apt-get remove -y build-essential && apt-get autoremove -y
```

```bash
# Build and run
docker build -t ruvector-postgres:custom .
docker run -d -e POSTGRES_PASSWORD=secret -p 5432:5432 ruvector-postgres:custom
```

### Method 4: Cloud Platforms

#### Neon

See [NEON_COMPATIBILITY.md](./NEON_COMPATIBILITY.md) for detailed instructions.

1. Contact Neon support (Scale plan required)
2. Request custom extension upload
3. Provide pre-built artifacts
4. Enable via `CREATE EXTENSION ruvector;`

#### Supabase

```sql
-- Supabase supports custom extensions via SQL
-- Contact Supabase support for custom extension installation

-- Once installed:
CREATE EXTENSION ruvector;
```

#### AWS RDS

```bash
# RDS doesn't support custom extensions directly
# Use RDS for PostgreSQL with pgvector, or
# Deploy on EC2 with self-managed PostgreSQL

# For EC2 deployment:
# Follow Method 1 (Build from Source)
```

#### Google Cloud SQL

```bash
# Cloud SQL supports limited extensions
# For custom extensions, use Compute Engine with self-managed PostgreSQL

# Deploy on GCE:
# Follow Method 1 (Build from Source)
```

## Configuration

### PostgreSQL Configuration

Add to `postgresql.conf`:

```ini
# RuVector settings
shared_preload_libraries = 'ruvector'  # Optional, for background features

# Memory settings for vector operations
maintenance_work_mem = '2GB'           # For index builds
work_mem = '256MB'                     # For queries

# Parallel query settings
max_parallel_workers_per_gather = 4
max_parallel_maintenance_workers = 8
```

### Extension Settings (GUCs)

```sql
-- Search quality (higher = better recall, slower)
SET ruvector.ef_search = 100;          -- Default: 40

-- IVFFlat probes (higher = better recall, slower)
SET ruvector.probes = 10;              -- Default: 1

-- Maximum index memory
SET ruvector.max_index_memory = '1GB'; -- Default: unlimited

-- Enable/disable quantization
SET ruvector.enable_quantization = on; -- Default: on

-- Query plan caching
SET ruvector.plan_cache = on;          -- Default: on
```

### Per-Session Settings

```sql
-- For high-recall queries
SET ruvector.ef_search = 200;
SET ruvector.probes = 20;

-- For low-latency queries
SET ruvector.ef_search = 20;
SET ruvector.probes = 1;
```

## Verification

### Check Installation

```sql
-- Verify extension is installed
SELECT * FROM pg_extension WHERE extname = 'ruvector';

-- Check version
SELECT ruvector_version();
-- Expected: 0.1.0

-- Check SIMD capabilities
SELECT ruvector_simd_info();
-- Expected: avx2 or avx512 on modern x86_64
```

### Basic Functionality Test

```sql
-- Create test table
CREATE TABLE test_vectors (
    id SERIAL PRIMARY KEY,
    embedding ruvector(3)
);

-- Insert vectors
INSERT INTO test_vectors (embedding) VALUES
    ('[1, 2, 3]'),
    ('[4, 5, 6]'),
    ('[7, 8, 9]');

-- Test distance calculation
SELECT id, embedding <-> '[1, 1, 1]'::ruvector AS distance
FROM test_vectors
ORDER BY distance
LIMIT 3;

-- Clean up
DROP TABLE test_vectors;
```

### Index Creation Test

```sql
-- Create table with embeddings
CREATE TABLE items (
    id SERIAL PRIMARY KEY,
    embedding ruvector(1536)
);

-- Insert sample data
INSERT INTO items (embedding)
SELECT ('[' || array_to_string(array_agg(random()), ',') || ']')::ruvector
FROM generate_series(1, 1536) d
CROSS JOIN generate_series(1, 10000) i
GROUP BY i;

-- Create HNSW index
CREATE INDEX items_embedding_idx ON items
USING ruhnsw (embedding ruvector_l2_ops)
WITH (m = 16, ef_construction = 100);

-- Test search
EXPLAIN ANALYZE
SELECT * FROM items
ORDER BY embedding <-> (SELECT embedding FROM items LIMIT 1)
LIMIT 10;

-- Verify index usage
-- Should show "Index Scan using items_embedding_idx"
```

## Upgrading

### Minor Version Upgrade (0.1.0 → 0.1.1)

```sql
-- Check current version
SELECT ruvector_version();

-- Upgrade extension
ALTER EXTENSION ruvector UPDATE TO '0.1.1';

-- Verify
SELECT ruvector_version();
```

### Major Version Upgrade (0.1.x → 0.2.x)

```bash
# Stop PostgreSQL
sudo systemctl stop postgresql

# Install new version
cd ruvector/crates/ruvector-postgres
git pull
cargo pgrx package --pg-config $(which pg_config)
sudo cp target/release/ruvector-pg16/usr/lib/postgresql/16/lib/* \
    /usr/lib/postgresql/16/lib/

# Start PostgreSQL
sudo systemctl start postgresql

# Upgrade in database
psql -U postgres -d your_database -c "ALTER EXTENSION ruvector UPDATE;"
```

### Migration from pgvector

```sql
-- Keep existing pgvector data
-- CREATE EXTENSION IF NOT EXISTS vector;

-- Install ruvector
CREATE EXTENSION ruvector;

-- Migrate data (type cast works automatically)
CREATE TABLE items_new AS
SELECT id, embedding::ruvector AS embedding, metadata
FROM items;

-- Create new index
CREATE INDEX ON items_new USING ruhnsw (embedding ruvector_l2_ops);

-- Swap tables
BEGIN;
ALTER TABLE items RENAME TO items_old;
ALTER TABLE items_new RENAME TO items;
COMMIT;

-- Validate
SELECT COUNT(*) FROM items;
SELECT COUNT(*) FROM items_old;

-- Drop old table when ready
-- DROP TABLE items_old;
```

## Uninstallation

```sql
-- Drop all dependent objects first
DROP INDEX IF EXISTS items_embedding_idx;

-- Drop extension
DROP EXTENSION ruvector CASCADE;
```

```bash
# Remove library files
sudo rm /usr/lib/postgresql/16/lib/ruvector.so
sudo rm /usr/share/postgresql/16/extension/ruvector*
```

## Troubleshooting

### Extension Won't Load

```bash
# Check library path
pg_config --pkglibdir
ls -la /usr/lib/postgresql/16/lib/ruvector*

# Check shared_preload_libraries
psql -c "SHOW shared_preload_libraries;"

# Check PostgreSQL logs
sudo tail -100 /var/log/postgresql/postgresql-16-main.log
```

### SIMD Not Detected

```sql
-- Check detected SIMD
SELECT ruvector_simd_info();

-- If showing 'scalar', check CPU capabilities:
-- Linux:
-- cat /proc/cpuinfo | grep -E 'avx2|avx512'

-- macOS:
-- sysctl -a | grep machdep.cpu.features
```

### Index Build Slow

```sql
-- Increase maintenance memory
SET maintenance_work_mem = '8GB';

-- Increase parallelism
SET max_parallel_maintenance_workers = 16;

-- Use CONCURRENTLY for non-blocking builds
CREATE INDEX CONCURRENTLY ON items
USING ruhnsw (embedding ruvector_l2_ops);
```

### Out of Memory

```sql
-- Reduce index memory limit
SET ruvector.max_index_memory = '512MB';

-- Use more aggressive quantization
CREATE INDEX ON items USING ruhnsw (embedding ruvector_l2_ops)
WITH (quantization = 'pq32');  -- 32 subspaces, maximum compression
```

### Connection Issues

```bash
# Check PostgreSQL is running
sudo systemctl status postgresql

# Check listen addresses
grep listen_addresses /etc/postgresql/16/main/postgresql.conf

# Check pg_hba.conf for authentication
sudo cat /etc/postgresql/16/main/pg_hba.conf
```

## Support

- **Documentation**: https://github.com/ruvnet/ruvector/tree/main/crates/ruvector-postgres/docs
- **Issues**: https://github.com/ruvnet/ruvector/issues
- **Discussions**: https://github.com/ruvnet/ruvector/discussions

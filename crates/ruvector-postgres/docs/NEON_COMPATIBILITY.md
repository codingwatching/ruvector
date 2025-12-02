# Neon Postgres Compatibility Guide

## Overview

RuVector-Postgres is designed with first-class support for Neon's serverless PostgreSQL platform. This guide covers deployment, configuration, and optimization for Neon environments.

## Neon Platform Overview

Neon is a serverless PostgreSQL platform with:
- **Separation of storage and compute**: Compute nodes are stateless
- **Scale to zero**: Instances suspend when idle
- **Instant branching**: Copy-on-write database branches
- **Dynamic Extension Loading**: Custom extensions loaded on demand

## Compatibility Matrix

| Neon Feature | RuVector Support | Notes |
|--------------|------------------|-------|
| PostgreSQL 14 | ✓ | Full support |
| PostgreSQL 15 | ✓ | Full support |
| PostgreSQL 16 | ✓ | Full support |
| PostgreSQL 17 | ✓ | Full support |
| PostgreSQL 18 | ✓ | Full support |
| Scale to Zero | ✓ | Fast cold start |
| Branching | ✓ | Index state preserved |
| Connection Pooling | ✓ | Thread-safe |
| Read Replicas | ✓ | Consistent reads |

## Design Considerations for Neon

### 1. Stateless Compute

Neon compute nodes are ephemeral. RuVector-Postgres handles this by:

```rust
// No global mutable state that requires persistence
// All state lives in PostgreSQL's shared memory or storage

#[pg_guard]
pub fn _PG_init() {
    // Lightweight initialization - no disk I/O
    // SIMD feature detection cached in thread-local
    init_simd_dispatch();

    // Register GUCs
    register_gucs();

    // No background workers that assume persistence
}
```

### 2. Fast Cold Start

Critical for scale-to-zero. RuVector-Postgres achieves sub-100ms initialization:

```
┌─────────────────────────────────────────────────────────────────┐
│                    Cold Start Timeline                           │
├─────────────────────────────────────────────────────────────────┤
│  0ms  │ Extension loaded via CREATE EXTENSION                   │
│  5ms  │ SIMD feature detection complete                         │
│ 10ms  │ GUC registration complete                               │
│ 15ms  │ Operator/function registration complete                 │
│ 20ms  │ Index access method registration complete               │
│ 50ms  │ First query ready                                       │
└─────────────────────────────────────────────────────────────────┘
```

**Optimization techniques:**
- Lazy index loading (mmap on demand)
- No precomputed tables at startup
- Minimal heap allocations in init path

### 3. Memory Efficiency

Neon compute has memory limits. RuVector-Postgres is memory-conscious:

```sql
-- Check memory usage
SELECT * FROM ruvector_memory_stats();

┌──────────────────────────────────────────────────────────┐
│                  Memory Statistics                        │
├──────────────────────────────────────────────────────────┤
│ index_memory_mb        │ 256                             │
│ vector_cache_mb        │ 64                              │
│ quantization_tables_mb │ 8                               │
│ total_extension_mb     │ 328                             │
└──────────────────────────────────────────────────────────┘
```

**Memory optimization options:**
```sql
-- Limit index memory (useful for smaller Neon instances)
SET ruvector.max_index_memory = '256MB';

-- Use quantization to reduce memory footprint
CREATE INDEX ON items USING ruhnsw (embedding ruvector_l2_ops)
WITH (quantization = 'sq8');  -- 4x memory reduction
```

### 4. No Background Workers

Neon restricts background workers. RuVector-Postgres avoids them:

```rust
// ❌ NOT used: Background workers
// pg_background_worker_register(...)

// ✓ Used: On-demand operations
// All maintenance happens during queries or explicit commands
```

**Alternative patterns:**
```sql
-- Explicit index maintenance (instead of background vacuum)
SELECT ruvector_index_maintenance('items_embedding_idx');

-- Manual statistics update
ANALYZE items;
```

## Requesting RuVector on Neon

### For Scale Plan Customers

1. **Open Support Ticket**
   - Navigate to Neon Console → Support
   - Request: "Custom extension: ruvector-postgres"
   - Provide: PostgreSQL version requirements

2. **Provide Extension Artifacts**
   - Pre-built `.so` files for each PG version
   - Control file and SQL scripts
   - Security documentation

3. **Security Review**
   - Neon engineers review code safety
   - SIMD usage validated
   - Memory management audited

4. **Deployment**
   - Extension uploaded to Neon's extension storage
   - Available via Dynamic Extension Loading

### For Free Plan Users

1. **Submit Discord Request**
   - Join Neon Discord: discord.gg/92vNTzKDGp
   - Post in #feedback channel
   - Include use case description

2. **Alternative: Use pgvector**
   - pgvector is pre-installed on all Neon instances
   - RuVector provides migration path

## Installation on Neon

Once approved by Neon:

```sql
-- Enable the extension
CREATE EXTENSION ruvector;

-- Verify installation
SELECT ruvector_version();
-- Returns: 0.1.0

-- Check SIMD capabilities
SELECT ruvector_simd_info();
-- Returns: avx2 (or avx512, neon, scalar)
```

## Migration from pgvector

RuVector-Postgres is API-compatible with pgvector:

### Step 1: Create Parallel Tables

```sql
-- Keep existing pgvector table
-- CREATE TABLE items_pgvector AS SELECT * FROM items;

-- Create new table with ruvector
CREATE TABLE items_ruvector (
    id SERIAL PRIMARY KEY,
    embedding ruvector(1536)
);

-- Copy data (automatic type conversion)
INSERT INTO items_ruvector (id, embedding)
SELECT id, embedding::ruvector FROM items;
```

### Step 2: Rebuild Indexes

```sql
-- Create optimized HNSW index
CREATE INDEX items_embedding_ruhnsw_idx ON items_ruvector
USING ruhnsw (embedding ruvector_l2_ops)
WITH (m = 32, ef_construction = 200);
```

### Step 3: Validate Compatibility

```sql
-- Compare search results
WITH pgvector_results AS (
    SELECT id, embedding <-> '[...]'::vector AS dist
    FROM items ORDER BY dist LIMIT 10
),
ruvector_results AS (
    SELECT id, embedding <-> '[...]'::ruvector AS dist
    FROM items_ruvector ORDER BY dist LIMIT 10
)
SELECT
    p.id = r.id AS id_match,
    abs(p.dist - r.dist) < 0.0001 AS dist_match
FROM pgvector_results p
JOIN ruvector_results r ON p.id = r.id;
```

### Step 4: Switch Over

```sql
-- Rename tables
ALTER TABLE items RENAME TO items_old;
ALTER TABLE items_ruvector RENAME TO items;

-- Drop old table after validation
DROP TABLE items_old;
```

## Performance Tuning for Neon

### Instance Size Recommendations

| Neon Compute | Max Vectors | Recommended Settings |
|--------------|-------------|---------------------|
| 0.25 CU | 100K | m=8, ef=64, sq8 quant |
| 0.5 CU | 500K | m=16, ef=100, sq8 quant |
| 1 CU | 2M | m=24, ef=150, optional quant |
| 2 CU | 5M | m=32, ef=200 |
| 4 CU | 10M+ | m=48, ef=300 |

### Query Optimization

```sql
-- Adjust search parameters for recall vs latency
SET ruvector.ef_search = 100;  -- Default: 40

-- Enable query plan caching
SET ruvector.plan_cache = on;

-- Use LIMIT for early termination
SELECT * FROM items
ORDER BY embedding <-> query_vector
LIMIT 10;  -- Essential for performance
```

### Index Build on Neon

```sql
-- For large datasets, use non-blocking build
CREATE INDEX CONCURRENTLY items_embedding_idx ON items
USING ruhnsw (embedding ruvector_l2_ops)
WITH (m = 32, ef_construction = 200);

-- Monitor progress
SELECT * FROM pg_stat_progress_create_index;
```

## Neon Branching with RuVector

### Branch Creation

```sql
-- Indexes are preserved in branches
-- No special handling required

-- In parent branch
CREATE INDEX ... ON items USING ruhnsw (...);

-- Create child branch via Neon Console or API
-- Index is instantly available in child (copy-on-write)
```

### Branch-Specific Tuning

```sql
-- Development branch: Faster builds, lower recall
SET ruvector.ef_search = 20;

-- Production branch: Higher recall
SET ruvector.ef_search = 200;
```

## Monitoring on Neon

### Extension Metrics

```sql
-- Index statistics
SELECT * FROM ruvector_index_stats();

┌────────────────────────────────────────────────────────────────┐
│                    Index Statistics                             │
├────────────────────────────────────────────────────────────────┤
│ index_name              │ items_embedding_idx                  │
│ index_size_mb           │ 512                                  │
│ vector_count            │ 1000000                              │
│ dimensions              │ 1536                                 │
│ build_time_seconds      │ 45.2                                 │
│ last_vacuum             │ 2024-01-15 10:30:00                  │
│ fragmentation_pct       │ 2.3                                  │
└────────────────────────────────────────────────────────────────┘
```

### Query Performance

```sql
-- Explain analyze for vector queries
EXPLAIN (ANALYZE, BUFFERS)
SELECT * FROM items
ORDER BY embedding <-> '[0.1, 0.2, ...]'::ruvector
LIMIT 10;

-- Output includes:
-- - Distance calculations performed
-- - Nodes visited in HNSW graph
-- - Buffer hits/misses
```

## Troubleshooting

### Cold Start Issues

```sql
-- Check extension load time
SELECT * FROM pg_extension WHERE extname = 'ruvector';

-- Verify SIMD detection
SELECT ruvector_simd_info();
-- Expected: 'avx2' on most Neon instances
```

### Memory Pressure

```sql
-- Reduce index memory
SET ruvector.max_index_memory = '128MB';

-- Use aggressive quantization
CREATE INDEX ... WITH (quantization = 'pq16');

-- Check current usage
SELECT * FROM ruvector_memory_stats();
```

### Connection Pool Compatibility

```sql
-- RuVector is connection-pool safe
-- No connection-specific state

-- Works with PgBouncer in transaction mode
-- Works with Neon's built-in connection pooling
```

## Limitations on Neon

1. **No Custom Background Workers**
   - Index maintenance is synchronous
   - Use `CONCURRENTLY` for non-blocking builds

2. **Memory Constraints**
   - Use quantization for large indexes
   - Monitor with `ruvector_memory_stats()`

3. **Compute Suspension**
   - First query after suspension may be slower
   - Indexes are memory-mapped from storage

4. **Extension Updates**
   - Requires Neon support ticket
   - Plan for migration during updates

## Support Resources

- **Neon Documentation**: https://neon.tech/docs
- **RuVector Issues**: https://github.com/ruvnet/ruvector/issues
- **Neon Discord**: discord.gg/92vNTzKDGp
- **Neon Support**: console.neon.tech → Support

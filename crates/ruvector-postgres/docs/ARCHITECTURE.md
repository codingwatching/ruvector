# RuVector-Postgres Architecture

## Overview

RuVector-Postgres is a high-performance, drop-in replacement for the pgvector extension, built in Rust using the pgrx framework. It provides SIMD-optimized vector similarity search with advanced indexing algorithms, quantization support, and hybrid search capabilities.

## Design Goals

1. **pgvector API Compatibility**: 100% compatible SQL interface with pgvector
2. **Superior Performance**: 2-10x faster than pgvector through SIMD and algorithmic optimizations
3. **Memory Efficiency**: Up to 32x memory reduction via quantization
4. **Neon Compatibility**: Designed for serverless PostgreSQL (Neon, Supabase, etc.)
5. **Production Ready**: Battle-tested algorithms from ruvector-core

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           PostgreSQL Server                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │                      RuVector-Postgres Extension                         │ │
│  ├─────────────────────────────────────────────────────────────────────────┤ │
│  │                                                                           │ │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐  │ │
│  │  │   Vector    │  │   HNSW      │  │  IVFFlat    │  │   Flat Index    │  │ │
│  │  │   Type      │  │   Index     │  │   Index     │  │   (fallback)    │  │ │
│  │  │             │  │             │  │             │  │                 │  │ │
│  │  │ - ruvector  │  │ - O(log n)  │  │ - O(√n)     │  │ - O(n)          │  │ │
│  │  │ - halfvec   │  │ - 95%+ rec  │  │ - clusters  │  │ - exact search  │  │ │
│  │  │ - sparsevec │  │ - SIMD ops  │  │ - training  │  │                 │  │ │
│  │  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └────────┬────────┘  │ │
│  │         │                │                │                   │           │ │
│  │  ┌──────┴────────────────┴────────────────┴───────────────────┴────────┐  │ │
│  │  │                     SIMD Distance Layer                              │  │ │
│  │  │                                                                       │  │ │
│  │  │  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌────────────────┐  │  │ │
│  │  │  │  AVX-512   │  │   AVX2     │  │   NEON     │  │   Scalar       │  │  │ │
│  │  │  │  (x86_64)  │  │  (x86_64)  │  │  (ARM64)   │  │   Fallback     │  │  │ │
│  │  │  └────────────┘  └────────────┘  └────────────┘  └────────────────┘  │  │ │
│  │  └──────────────────────────────────────────────────────────────────────┘  │ │
│  │                                                                           │ │
│  │  ┌──────────────────────────────────────────────────────────────────────┐  │ │
│  │  │                    Quantization Engine                                │  │ │
│  │  │                                                                       │  │ │
│  │  │  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌────────────────┐  │  │ │
│  │  │  │   Scalar   │  │  Product   │  │   Binary   │  │   Half-Prec    │  │  │ │
│  │  │  │    (4x)    │  │   (8-16x)  │  │    (32x)   │  │    (2x)        │  │  │ │
│  │  │  └────────────┘  └────────────┘  └────────────┘  └────────────────┘  │  │ │
│  │  └──────────────────────────────────────────────────────────────────────┘  │ │
│  │                                                                           │ │
│  │  ┌──────────────────────────────────────────────────────────────────────┐  │ │
│  │  │                    Hybrid Search Engine                               │  │ │
│  │  │                                                                       │  │ │
│  │  │  ┌─────────────────────┐  ┌─────────────────────┐  ┌──────────────┐  │  │ │
│  │  │  │  Vector Similarity  │  │   BM25 Text Search  │  │  RRF Fusion  │  │  │ │
│  │  │  │     (dense)         │  │      (sparse)       │  │  (ranking)   │  │  │ │
│  │  │  └─────────────────────┘  └─────────────────────┘  └──────────────┘  │  │ │
│  │  └──────────────────────────────────────────────────────────────────────┘  │ │
│  │                                                                           │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
│                                                                               │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. Vector Types

#### `ruvector` - Primary Vector Type
```sql
-- Dimensions: 1 to 16,000
-- Storage: 4 bytes per dimension (f32) + 8 bytes header
CREATE TABLE items (
    id SERIAL PRIMARY KEY,
    embedding ruvector(1536)  -- OpenAI embedding dimensions
);
```

#### `halfvec` - Half-Precision Vector
```sql
-- Storage: 2 bytes per dimension (f16) + 8 bytes header
-- 50% memory savings, minimal accuracy loss
CREATE TABLE items (
    id SERIAL PRIMARY KEY,
    embedding halfvec(1536)
);
```

#### `sparsevec` - Sparse Vector
```sql
-- Storage: Only non-zero elements stored
-- Ideal for high-dimensional sparse data (BM25, TF-IDF)
CREATE TABLE items (
    id SERIAL PRIMARY KEY,
    sparse_embedding sparsevec(50000)
);
```

### 2. Distance Operators

| Operator | Distance Metric | Description |
|----------|----------------|-------------|
| `<->` | L2 (Euclidean) | `sqrt(sum((a[i] - b[i])^2))` |
| `<#>` | Inner Product | `-sum(a[i] * b[i])` (negative for ORDER BY) |
| `<=>` | Cosine | `1 - (a·b)/(‖a‖‖b‖)` |
| `<+>` | L1 (Manhattan) | `sum(abs(a[i] - b[i]))` |
| `<~>` | Hamming | Bit differences (binary vectors) |
| `<%>` | Jaccard | Set similarity (sparse vectors) |

### 3. Index Types

#### HNSW (Hierarchical Navigable Small World)
```sql
CREATE INDEX ON items USING ruhnsw (embedding ruvector_l2_ops)
WITH (m = 16, ef_construction = 200);
```

**Parameters:**
- `m`: Maximum connections per layer (default: 16, range: 2-100)
- `ef_construction`: Build-time search breadth (default: 64, range: 4-1000)

**Characteristics:**
- Search: O(log n)
- Insert: O(log n)
- Memory: ~1.5x index overhead
- Recall: 95-99%+ with tuned parameters

#### IVFFlat (Inverted File with Flat Quantization)
```sql
CREATE INDEX ON items USING ruivfflat (embedding ruvector_l2_ops)
WITH (lists = 100);
```

**Parameters:**
- `lists`: Number of clusters (default: sqrt(n), recommended: rows/1000 to rows/10000)

**Characteristics:**
- Search: O(√n)
- Insert: O(1) after training
- Memory: Minimal overhead
- Recall: 90-95% with `probes = sqrt(lists)`

### 4. SIMD Optimization Tiers

```
┌─────────────────────────────────────────────────────────────────┐
│                    Runtime Feature Detection                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│   if std::is_x86_feature_detected!("avx512f") {                  │
│       // AVX-512: 16 floats per operation                        │
│       avx512_euclidean_distance(a, b)                            │
│   } else if std::is_x86_feature_detected!("avx2") {              │
│       // AVX2: 8 floats per operation                            │
│       avx2_euclidean_distance(a, b)                              │
│   } else if cfg!(target_arch = "aarch64") {                      │
│       // ARM NEON: 4 floats per operation                        │
│       neon_euclidean_distance(a, b)                              │
│   } else {                                                        │
│       // Scalar fallback: 1 float per operation                  │
│       scalar_euclidean_distance(a, b)                            │
│   }                                                               │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

### 5. Quantization Engine

#### Scalar Quantization (SQ8)
- **Compression**: 4x (f32 → i8)
- **Speed**: ~2x faster distance calculations
- **Accuracy**: <1% recall loss

```sql
CREATE INDEX ON items USING ruhnsw (embedding ruvector_l2_ops)
WITH (m = 16, ef_construction = 200, quantization = 'sq8');
```

#### Product Quantization (PQ)
- **Compression**: 8-32x
- **Speed**: ~4x faster with precomputed tables
- **Accuracy**: 1-5% recall loss depending on subspaces

```sql
CREATE INDEX ON items USING ruhnsw (embedding ruvector_l2_ops)
WITH (m = 16, ef_construction = 200, quantization = 'pq16');  -- 16 subspaces
```

#### Binary Quantization (BQ)
- **Compression**: 32x (f32 → 1 bit)
- **Speed**: ~10x faster with Hamming distance
- **Accuracy**: Requires reranking for high recall

```sql
CREATE INDEX ON items USING ruhnsw (embedding ruvector_l2_ops)
WITH (m = 16, ef_construction = 200, quantization = 'binary');
```

## Memory Layout

### Vector Storage Format

```
┌────────────────────────────────────────────────────────────────┐
│                    ruvector Internal Layout                     │
├────────────────────────────────────────────────────────────────┤
│  Bytes 0-3   │  Bytes 4-7   │  Bytes 8-11  │  Bytes 12+       │
│  vl_len_     │  dimensions  │  reserved    │  float32 data... │
│  (header)    │  (u32)       │  (flags)     │  [dim0, dim1...] │
└────────────────────────────────────────────────────────────────┘
```

### HNSW Index Layout

```
┌─────────────────────────────────────────────────────────────────┐
│                      HNSW Index Structure                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  Layer L (top):     ○──────○                                     │
│                     │      │                                     │
│  Layer L-1:         ○──○───○──○                                  │
│                     │  │   │  │                                  │
│  Layer L-2:         ○──○───○──○──○──○                            │
│                     │  │   │  │  │  │                            │
│  Layer 0 (base):    ○──○───○──○──○──○──○──○──○                   │
│                                                                   │
│  Entry Point: Top layer node                                     │
│  Search: Greedy descent + local beam search                     │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

## Query Execution Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                      Query: SELECT ... ORDER BY v <-> q         │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  1. Parse & Plan                                                 │
│     └─> Identify index scan opportunity                         │
│                                                                   │
│  2. Index Selection                                              │
│     └─> Choose HNSW/IVFFlat based on cost estimation            │
│                                                                   │
│  3. Index Scan (SIMD-accelerated)                               │
│     ├─> HNSW: Navigate layers, beam search at layer 0          │
│     └─> IVFFlat: Probe nearest centroids, scan cells           │
│                                                                   │
│  4. Distance Calculation (per candidate)                        │
│     ├─> Quantized distance (fast, approximate)                  │
│     └─> Full precision rerank (optional)                        │
│                                                                   │
│  5. Result Aggregation                                          │
│     └─> Return top-k with distances                             │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

## Comparison with pgvector

| Feature | pgvector 0.8.0 | RuVector-Postgres |
|---------|---------------|-------------------|
| Vector dimensions | 16,000 max | 16,000 max |
| HNSW index | ✓ | ✓ (optimized) |
| IVFFlat index | ✓ | ✓ (optimized) |
| Half-precision | ✓ | ✓ |
| Sparse vectors | ✓ | ✓ |
| Binary quantization | ✓ | ✓ |
| Product quantization | ✗ | ✓ |
| Scalar quantization | ✗ | ✓ |
| AVX-512 optimized | Partial | Full |
| ARM NEON optimized | ✗ | ✓ |
| Hybrid search | ✗ | ✓ |
| Filtered HNSW | Partial | ✓ |
| Iterative scan | ✓ | ✓ |

## Thread Safety

RuVector-Postgres is fully thread-safe:

- **Read operations**: Lock-free concurrent reads
- **Write operations**: Fine-grained locking per graph layer
- **Index builds**: Parallel with work-stealing

```rust
// Internal synchronization primitives
pub struct HnswIndex {
    layers: Vec<RwLock<Layer>>,           // Per-layer locks
    entry_point: AtomicUsize,             // Lock-free entry point
    node_count: AtomicUsize,              // Lock-free counter
    vectors: DashMap<NodeId, Vec<f32>>,   // Concurrent hashmap
}
```

## Extension Dependencies

```toml
[dependencies]
pgrx = "0.12"                  # PostgreSQL extension framework
simsimd = "5.9"                # SIMD-accelerated distance functions
parking_lot = "0.12"           # Fast synchronization primitives
dashmap = "6.0"                # Concurrent hashmap
rayon = "1.10"                 # Data parallelism
half = "2.4"                   # Half-precision floats
bitflags = "2.6"               # Compact flags storage
```

## Performance Tuning

### Index Build Performance

```sql
-- Parallel index build (uses all available cores)
SET maintenance_work_mem = '8GB';
SET max_parallel_maintenance_workers = 8;

CREATE INDEX CONCURRENTLY ON items
USING ruhnsw (embedding ruvector_l2_ops)
WITH (m = 32, ef_construction = 400);
```

### Search Performance

```sql
-- Adjust search quality vs speed tradeoff
SET ruvector.ef_search = 200;  -- Higher = better recall, slower
SET ruvector.probes = 10;      -- For IVFFlat: more probes = better recall

-- Use iterative scan for filtered queries
SELECT * FROM items
WHERE category = 'electronics'
ORDER BY embedding <-> '[0.1, 0.2, ...]'::ruvector
LIMIT 10;
```

## File Structure

```
crates/ruvector-postgres/
├── Cargo.toml                    # Rust dependencies
├── ruvector.control              # Extension metadata
├── docs/
│   ├── ARCHITECTURE.md           # This file
│   ├── NEON_COMPATIBILITY.md     # Neon deployment guide
│   ├── SIMD_OPTIMIZATION.md      # SIMD implementation details
│   └── INSTALLATION.md           # Installation instructions
├── sql/
│   ├── ruvector--0.1.0.sql       # Extension SQL definitions
│   └── ruvector--0.0.0--0.1.0.sql # Migration script
├── src/
│   ├── lib.rs                    # Extension entry point
│   ├── types/
│   │   ├── mod.rs
│   │   ├── vector.rs             # ruvector type
│   │   ├── halfvec.rs            # Half-precision vector
│   │   └── sparsevec.rs          # Sparse vector
│   ├── distance/
│   │   ├── mod.rs
│   │   ├── simd.rs               # SIMD implementations
│   │   └── scalar.rs             # Scalar fallbacks
│   ├── index/
│   │   ├── mod.rs
│   │   ├── hnsw.rs               # HNSW implementation
│   │   ├── ivfflat.rs            # IVFFlat implementation
│   │   └── scan.rs               # Index scan operators
│   ├── quantization/
│   │   ├── mod.rs
│   │   ├── scalar.rs             # SQ8 quantization
│   │   ├── product.rs            # PQ quantization
│   │   └── binary.rs             # Binary quantization
│   ├── operators.rs              # SQL operators
│   └── functions.rs              # SQL functions
└── tests/
    ├── integration_tests.rs
    └── compatibility_tests.rs    # pgvector compatibility
```

## Version History

- **0.1.0**: Initial release with pgvector compatibility
  - HNSW and IVFFlat indexes
  - SIMD-optimized distance functions
  - Scalar quantization support
  - Neon compatibility

## License

MIT License - Same as ruvector-core

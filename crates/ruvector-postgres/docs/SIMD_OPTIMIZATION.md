# SIMD Optimization in RuVector-Postgres

## Overview

RuVector-Postgres leverages Single Instruction Multiple Data (SIMD) instructions for vectorized distance calculations, achieving 4-16x speedup over scalar implementations. This document details the SIMD architecture, implementations, and performance characteristics.

## SIMD Architecture

### Instruction Set Hierarchy

```
┌─────────────────────────────────────────────────────────────────┐
│                    SIMD Instruction Sets                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │ AVX-512 (x86_64)                                             │ │
│  │ • 512-bit registers (ZMM0-ZMM31)                            │ │
│  │ • 16 floats per operation                                   │ │
│  │ • Available: Intel Skylake-X+, AMD Zen4+                    │ │
│  │ • Speedup: ~8-16x over scalar                               │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                           │                                       │
│                           ▼                                       │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │ AVX2 (x86_64)                                                │ │
│  │ • 256-bit registers (YMM0-YMM15)                            │ │
│  │ • 8 floats per operation                                    │ │
│  │ • Available: Intel Haswell+, AMD Excavator+                 │ │
│  │ • Speedup: ~4-8x over scalar                                │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                           │                                       │
│                           ▼                                       │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │ ARM NEON (aarch64)                                           │ │
│  │ • 128-bit registers (V0-V31)                                │ │
│  │ • 4 floats per operation                                    │ │
│  │ • Available: All ARM64 (Apple M1+, AWS Graviton)            │ │
│  │ • Speedup: ~2-4x over scalar                                │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                           │                                       │
│                           ▼                                       │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │ Scalar Fallback                                              │ │
│  │ • Standard floating-point operations                        │ │
│  │ • Available: All platforms                                  │ │
│  │ • Baseline performance                                      │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

### Runtime Feature Detection

```rust
use std::arch::x86_64::*;

/// Cached SIMD capability detection
#[derive(Clone, Copy)]
pub enum SimdCapability {
    Avx512,
    Avx2,
    Neon,
    Scalar,
}

/// Thread-local cached capability (checked once per thread)
thread_local! {
    static SIMD_CAP: SimdCapability = detect_simd_capability();
}

fn detect_simd_capability() -> SimdCapability {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") &&
           is_x86_feature_detected!("avx512vl") {
            return SimdCapability::Avx512;
        }
        if is_x86_feature_detected!("avx2") &&
           is_x86_feature_detected!("fma") {
            return SimdCapability::Avx2;
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        // NEON is mandatory on aarch64
        return SimdCapability::Neon;
    }

    SimdCapability::Scalar
}
```

## Distance Function Implementations

### Euclidean Distance (L2)

#### AVX-512 Implementation

```rust
#[target_feature(enable = "avx512f")]
#[inline]
pub unsafe fn euclidean_distance_avx512(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());

    let n = a.len();
    let mut sum = _mm512_setzero_ps();

    // Process 16 floats at a time
    let chunks = n / 16;
    for i in 0..chunks {
        let offset = i * 16;
        let va = _mm512_loadu_ps(a.as_ptr().add(offset));
        let vb = _mm512_loadu_ps(b.as_ptr().add(offset));
        let diff = _mm512_sub_ps(va, vb);
        sum = _mm512_fmadd_ps(diff, diff, sum);  // FMA: sum += diff * diff
    }

    // Horizontal sum
    let mut result = _mm512_reduce_add_ps(sum);

    // Handle remainder
    for i in (chunks * 16)..n {
        let diff = a[i] - b[i];
        result += diff * diff;
    }

    result.sqrt()
}
```

#### AVX2 Implementation

```rust
#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
pub unsafe fn euclidean_distance_avx2(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());

    let n = a.len();
    let mut sum = _mm256_setzero_ps();

    // Process 8 floats at a time
    let chunks = n / 8;
    for i in 0..chunks {
        let offset = i * 8;
        let va = _mm256_loadu_ps(a.as_ptr().add(offset));
        let vb = _mm256_loadu_ps(b.as_ptr().add(offset));
        let diff = _mm256_sub_ps(va, vb);
        sum = _mm256_fmadd_ps(diff, diff, sum);  // FMA: sum += diff * diff
    }

    // Horizontal sum (AVX2 requires more steps)
    let sum_high = _mm256_extractf128_ps(sum, 1);
    let sum_low = _mm256_castps256_ps128(sum);
    let sum128 = _mm_add_ps(sum_high, sum_low);
    let sum64 = _mm_add_ps(sum128, _mm_movehl_ps(sum128, sum128));
    let sum32 = _mm_add_ss(sum64, _mm_shuffle_ps(sum64, sum64, 1));

    let mut result = _mm_cvtss_f32(sum32);

    // Handle remainder
    for i in (chunks * 8)..n {
        let diff = a[i] - b[i];
        result += diff * diff;
    }

    result.sqrt()
}
```

#### ARM NEON Implementation

```rust
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
pub unsafe fn euclidean_distance_neon(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::aarch64::*;

    debug_assert_eq!(a.len(), b.len());

    let n = a.len();
    let mut sum = vdupq_n_f32(0.0);

    // Process 4 floats at a time
    let chunks = n / 4;
    for i in 0..chunks {
        let offset = i * 4;
        let va = vld1q_f32(a.as_ptr().add(offset));
        let vb = vld1q_f32(b.as_ptr().add(offset));
        let diff = vsubq_f32(va, vb);
        sum = vfmaq_f32(sum, diff, diff);  // FMA: sum += diff * diff
    }

    // Horizontal sum
    let mut result = vaddvq_f32(sum);

    // Handle remainder
    for i in (chunks * 4)..n {
        let diff = a[i] - b[i];
        result += diff * diff;
    }

    result.sqrt()
}
```

### Cosine Distance

```rust
/// Cosine distance = 1 - (a·b) / (||a|| * ||b||)
#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
pub unsafe fn cosine_distance_avx2(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());

    let n = a.len();
    let mut dot = _mm256_setzero_ps();
    let mut norm_a = _mm256_setzero_ps();
    let mut norm_b = _mm256_setzero_ps();

    let chunks = n / 8;
    for i in 0..chunks {
        let offset = i * 8;
        let va = _mm256_loadu_ps(a.as_ptr().add(offset));
        let vb = _mm256_loadu_ps(b.as_ptr().add(offset));

        // Compute all three sums simultaneously
        dot = _mm256_fmadd_ps(va, vb, dot);       // dot += a * b
        norm_a = _mm256_fmadd_ps(va, va, norm_a); // norm_a += a * a
        norm_b = _mm256_fmadd_ps(vb, vb, norm_b); // norm_b += b * b
    }

    // Horizontal sums
    let dot_sum = horizontal_sum_avx2(dot);
    let norm_a_sum = horizontal_sum_avx2(norm_a);
    let norm_b_sum = horizontal_sum_avx2(norm_b);

    // Handle remainder
    let (mut dot_r, mut norm_a_r, mut norm_b_r) = (0.0f32, 0.0f32, 0.0f32);
    for i in (chunks * 8)..n {
        dot_r += a[i] * b[i];
        norm_a_r += a[i] * a[i];
        norm_b_r += b[i] * b[i];
    }

    let dot_total = dot_sum + dot_r;
    let norm_a_total = (norm_a_sum + norm_a_r).sqrt();
    let norm_b_total = (norm_b_sum + norm_b_r).sqrt();

    1.0 - (dot_total / (norm_a_total * norm_b_total))
}
```

### Inner Product (Dot Product)

```rust
/// Negative dot product for ORDER BY compatibility
#[target_feature(enable = "avx512f")]
#[inline]
pub unsafe fn inner_product_distance_avx512(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len();
    let mut sum = _mm512_setzero_ps();

    let chunks = n / 16;
    for i in 0..chunks {
        let offset = i * 16;
        let va = _mm512_loadu_ps(a.as_ptr().add(offset));
        let vb = _mm512_loadu_ps(b.as_ptr().add(offset));
        sum = _mm512_fmadd_ps(va, vb, sum);  // sum += a * b
    }

    let mut result = _mm512_reduce_add_ps(sum);

    for i in (chunks * 16)..n {
        result += a[i] * b[i];
    }

    -result  // Negative for ORDER BY ASC
}
```

## Dispatch Mechanism

### Static Dispatch (Compile-Time)

```rust
/// Compile-time dispatch using cfg
#[inline]
pub fn euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
    {
        unsafe { euclidean_distance_avx512(a, b) }
    }

    #[cfg(all(target_arch = "x86_64", target_feature = "avx2",
              not(target_feature = "avx512f")))]
    {
        unsafe { euclidean_distance_avx2(a, b) }
    }

    #[cfg(target_arch = "aarch64")]
    {
        unsafe { euclidean_distance_neon(a, b) }
    }

    #[cfg(not(any(
        all(target_arch = "x86_64", target_feature = "avx2"),
        target_arch = "aarch64"
    )))]
    {
        euclidean_distance_scalar(a, b)
    }
}
```

### Dynamic Dispatch (Runtime)

```rust
/// Runtime dispatch with cached capability
#[inline]
pub fn euclidean_distance_dynamic(a: &[f32], b: &[f32]) -> f32 {
    SIMD_CAP.with(|cap| {
        match *cap {
            SimdCapability::Avx512 => unsafe { euclidean_distance_avx512(a, b) },
            SimdCapability::Avx2 => unsafe { euclidean_distance_avx2(a, b) },
            SimdCapability::Neon => unsafe { euclidean_distance_neon(a, b) },
            SimdCapability::Scalar => euclidean_distance_scalar(a, b),
        }
    })
}
```

### Function Pointer Table (For Hot Paths)

```rust
/// Static function pointers initialized at extension load
pub struct DistanceFunctions {
    pub euclidean: fn(&[f32], &[f32]) -> f32,
    pub cosine: fn(&[f32], &[f32]) -> f32,
    pub inner_product: fn(&[f32], &[f32]) -> f32,
    pub manhattan: fn(&[f32], &[f32]) -> f32,
}

impl DistanceFunctions {
    pub fn detect() -> Self {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx512f") {
                return Self::avx512();
            }
            if is_x86_feature_detected!("avx2") {
                return Self::avx2();
            }
        }

        #[cfg(target_arch = "aarch64")]
        {
            return Self::neon();
        }

        Self::scalar()
    }

    fn avx512() -> Self {
        Self {
            euclidean: |a, b| unsafe { euclidean_distance_avx512(a, b) },
            cosine: |a, b| unsafe { cosine_distance_avx512(a, b) },
            inner_product: |a, b| unsafe { inner_product_distance_avx512(a, b) },
            manhattan: |a, b| unsafe { manhattan_distance_avx512(a, b) },
        }
    }

    // ... similar for avx2(), neon(), scalar()
}

/// Global distance functions (initialized once at load)
static DISTANCE_FNS: OnceLock<DistanceFunctions> = OnceLock::new();
```

## Batch Distance Calculations

### Parallel Batch with Rayon

```rust
use rayon::prelude::*;

/// Calculate distances from query to multiple vectors
pub fn batch_distances(
    query: &[f32],
    vectors: &[Vec<f32>],
    metric: DistanceMetric,
) -> Vec<f32> {
    let distance_fn = DISTANCE_FNS.get().unwrap();

    vectors
        .par_iter()
        .map(|v| match metric {
            DistanceMetric::Euclidean => (distance_fn.euclidean)(query, v),
            DistanceMetric::Cosine => (distance_fn.cosine)(query, v),
            DistanceMetric::InnerProduct => (distance_fn.inner_product)(query, v),
            DistanceMetric::Manhattan => (distance_fn.manhattan)(query, v),
        })
        .collect()
}
```

### Prefetching for Cache Efficiency

```rust
#[target_feature(enable = "avx2")]
pub unsafe fn batch_euclidean_prefetch(
    query: &[f32],
    vectors: &[&[f32]],
    results: &mut [f32],
) {
    use std::arch::x86_64::_mm_prefetch;

    const PREFETCH_DISTANCE: usize = 4;

    for (i, vector) in vectors.iter().enumerate() {
        // Prefetch upcoming vectors
        if i + PREFETCH_DISTANCE < vectors.len() {
            let future = vectors[i + PREFETCH_DISTANCE].as_ptr();
            _mm_prefetch(future as *const i8, _MM_HINT_T0);
        }

        results[i] = euclidean_distance_avx2(query, vector);
    }
}
```

## Performance Benchmarks

### Micro-benchmarks by Dimension

| Dimensions | Scalar | AVX2 | AVX-512 | Speedup |
|------------|--------|------|---------|---------|
| 128 | 180 ns | 28 ns | 18 ns | 10x |
| 384 | 520 ns | 72 ns | 42 ns | 12x |
| 768 | 1050 ns | 135 ns | 78 ns | 13x |
| 1536 | 2100 ns | 260 ns | 145 ns | 14x |
| 3072 | 4200 ns | 510 ns | 285 ns | 15x |

### Throughput (queries per second)

```
┌─────────────────────────────────────────────────────────────────┐
│          Query Throughput (1M vectors, 1536 dims)               │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  Scalar:    ████                                 ~2,500 q/s      │
│                                                                   │
│  AVX2:      ██████████████████████              ~18,000 q/s      │
│                                                                   │
│  AVX-512:   ████████████████████████████████    ~32,000 q/s      │
│                                                                   │
│  AVX-512    ████████████████████████████████████████████████     │
│  + Quant:                                       ~95,000 q/s      │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

## Memory Alignment

### Aligned Allocation

```rust
/// 64-byte aligned vector storage for optimal SIMD performance
#[repr(C, align(64))]
pub struct AlignedVector {
    data: Box<[f32]>,
}

impl AlignedVector {
    pub fn new(dimensions: usize) -> Self {
        use std::alloc::{alloc_zeroed, Layout};

        let layout = Layout::from_size_align(
            dimensions * std::mem::size_of::<f32>(),
            64,  // AVX-512 cache line alignment
        ).unwrap();

        let ptr = unsafe { alloc_zeroed(layout) as *mut f32 };
        let slice = unsafe {
            std::slice::from_raw_parts_mut(ptr, dimensions)
        };

        Self { data: unsafe { Box::from_raw(slice) } }
    }
}
```

### Unaligned Load Handling

```rust
/// Safe unaligned load with fallback
#[inline]
pub unsafe fn load_unaligned_256(ptr: *const f32) -> __m256 {
    if ptr as usize % 32 == 0 {
        _mm256_load_ps(ptr)  // Aligned load (faster)
    } else {
        _mm256_loadu_ps(ptr)  // Unaligned load
    }
}
```

## Half-Precision (FP16) Support

### AVX-512 FP16 Conversion

```rust
#[target_feature(enable = "avx512f", enable = "avx512bw")]
pub unsafe fn euclidean_distance_fp16(a: &[f16], b: &[f16]) -> f32 {
    let n = a.len();
    let mut sum = _mm512_setzero_ps();

    let chunks = n / 16;
    for i in 0..chunks {
        let offset = i * 16;

        // Load 16 half-precision values and convert to float
        let va_half = _mm256_loadu_si256(a.as_ptr().add(offset) as *const __m256i);
        let vb_half = _mm256_loadu_si256(b.as_ptr().add(offset) as *const __m256i);

        let va = _mm512_cvtph_ps(va_half);
        let vb = _mm512_cvtph_ps(vb_half);

        let diff = _mm512_sub_ps(va, vb);
        sum = _mm512_fmadd_ps(diff, diff, sum);
    }

    _mm512_reduce_add_ps(sum).sqrt()
}
```

## Quantized Distance Calculations

### Binary Quantization with POPCNT

```rust
/// Hamming distance for binary quantized vectors
#[target_feature(enable = "popcnt")]
pub unsafe fn hamming_distance(a: &[u64], b: &[u64]) -> u32 {
    let mut count = 0u32;

    for i in 0..a.len() {
        let xor = a[i] ^ b[i];
        count += _popcnt64(xor as i64) as u32;
    }

    count
}
```

### Scalar Quantization with SIMD

```rust
/// Distance on SQ8 quantized vectors
#[target_feature(enable = "avx2")]
pub unsafe fn sq8_distance_avx2(
    a: &[i8],
    b: &[i8],
    scale_a: f32,
    scale_b: f32,
) -> f32 {
    let n = a.len();
    let mut sum = _mm256_setzero_si256();

    let chunks = n / 32;
    for i in 0..chunks {
        let offset = i * 32;

        let va = _mm256_loadu_si256(a.as_ptr().add(offset) as *const __m256i);
        let vb = _mm256_loadu_si256(b.as_ptr().add(offset) as *const __m256i);

        // Subtract with saturation
        let diff = _mm256_subs_epi8(va, vb);

        // Multiply and accumulate (using maddubs trick)
        let abs_diff = _mm256_abs_epi8(diff);
        sum = _mm256_add_epi32(sum, _mm256_sad_epu8(abs_diff, _mm256_setzero_si256()));
    }

    // Extract sum and apply scaling
    let arr: [i32; 8] = std::mem::transmute(sum);
    let total: i32 = arr.iter().sum();

    (total as f32) * scale_a.max(scale_b)
}
```

## Compiler Optimizations

### Build Configuration

```toml
# Cargo.toml
[profile.release]
opt-level = 3
lto = "fat"
codegen-units = 1
target-cpu = "native"  # Use best CPU features available

[target.x86_64-unknown-linux-gnu]
rustflags = ["-C", "target-feature=+avx2,+fma"]

[target.aarch64-unknown-linux-gnu]
rustflags = ["-C", "target-feature=+neon"]
```

### Multi-Version Compilation

```rust
// Compile multiple versions for different CPUs
#[multiversion::multiversion(
    targets("x86_64+avx512f+avx512vl",
            "x86_64+avx2+fma",
            "aarch64+neon",
            "x86_64")
)]
pub fn euclidean_distance_multiversion(a: &[f32], b: &[f32]) -> f32 {
    // Implementation automatically selected at runtime
    // based on CPU capabilities
    euclidean_distance_impl(a, b)
}
```

## Verification and Testing

### SIMD Correctness Tests

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_simd_matches_scalar() {
        let a: Vec<f32> = (0..1536).map(|i| i as f32 * 0.001).collect();
        let b: Vec<f32> = (0..1536).map(|i| (1536 - i) as f32 * 0.001).collect();

        let scalar = euclidean_distance_scalar(&a, &b);

        #[cfg(target_arch = "x86_64")]
        if is_x86_feature_detected!("avx2") {
            let simd = unsafe { euclidean_distance_avx2(&a, &b) };
            assert_relative_eq!(scalar, simd, epsilon = 1e-5);
        }

        #[cfg(target_arch = "x86_64")]
        if is_x86_feature_detected!("avx512f") {
            let simd = unsafe { euclidean_distance_avx512(&a, &b) };
            assert_relative_eq!(scalar, simd, epsilon = 1e-5);
        }
    }

    #[test]
    fn test_edge_cases() {
        // Empty vectors
        assert_eq!(euclidean_distance(&[], &[]), 0.0);

        // Single element
        assert_relative_eq!(euclidean_distance(&[1.0], &[2.0]), 1.0);

        // Non-aligned length
        let a: Vec<f32> = vec![1.0; 137];  // Not divisible by 8 or 16
        let b: Vec<f32> = vec![2.0; 137];
        let _ = euclidean_distance(&a, &b);  // Should not panic
    }
}
```

## Debugging SIMD Issues

### Compile-Time Assertions

```rust
// Ensure alignment at compile time
const_assert!(std::mem::align_of::<AlignedVector>() >= 64);

// Ensure correct size
const_assert!(std::mem::size_of::<__m256>() == 32);
const_assert!(std::mem::size_of::<__m512>() == 64);
```

### Runtime Diagnostics

```sql
-- Check which SIMD path is active
SELECT ruvector_simd_info();

-- Returns detailed info:
┌─────────────────────────────────────────────────────────────────┐
│                    SIMD Configuration                            │
├─────────────────────────────────────────────────────────────────┤
│ architecture          │ x86_64                                  │
│ active_simd           │ avx512f                                 │
│ available_features    │ avx512f, avx512vl, avx2, fma, sse4.2   │
│ vector_width_bits     │ 512                                     │
│ floats_per_op         │ 16                                      │
│ estimated_speedup     │ 14x                                     │
└─────────────────────────────────────────────────────────────────┘
```

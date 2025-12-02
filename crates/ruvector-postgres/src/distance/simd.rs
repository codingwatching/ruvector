//! SIMD-optimized distance implementations
//!
//! Provides AVX-512, AVX2, and ARM NEON implementations of distance functions.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use super::scalar;

// ============================================================================
// AVX-512 Implementations
// ============================================================================

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
#[inline]
unsafe fn euclidean_distance_avx512(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len();
    let mut sum = _mm512_setzero_ps();

    let chunks = n / 16;
    for i in 0..chunks {
        let offset = i * 16;
        let va = _mm512_loadu_ps(a.as_ptr().add(offset));
        let vb = _mm512_loadu_ps(b.as_ptr().add(offset));
        let diff = _mm512_sub_ps(va, vb);
        sum = _mm512_fmadd_ps(diff, diff, sum);
    }

    let mut result = _mm512_reduce_add_ps(sum);

    // Handle remainder
    for i in (chunks * 16)..n {
        let diff = a[i] - b[i];
        result += diff * diff;
    }

    result.sqrt()
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
#[inline]
unsafe fn cosine_distance_avx512(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len();
    let mut dot = _mm512_setzero_ps();
    let mut norm_a = _mm512_setzero_ps();
    let mut norm_b = _mm512_setzero_ps();

    let chunks = n / 16;
    for i in 0..chunks {
        let offset = i * 16;
        let va = _mm512_loadu_ps(a.as_ptr().add(offset));
        let vb = _mm512_loadu_ps(b.as_ptr().add(offset));

        dot = _mm512_fmadd_ps(va, vb, dot);
        norm_a = _mm512_fmadd_ps(va, va, norm_a);
        norm_b = _mm512_fmadd_ps(vb, vb, norm_b);
    }

    let mut dot_sum = _mm512_reduce_add_ps(dot);
    let mut norm_a_sum = _mm512_reduce_add_ps(norm_a);
    let mut norm_b_sum = _mm512_reduce_add_ps(norm_b);

    for i in (chunks * 16)..n {
        dot_sum += a[i] * b[i];
        norm_a_sum += a[i] * a[i];
        norm_b_sum += b[i] * b[i];
    }

    let denominator = (norm_a_sum * norm_b_sum).sqrt();
    if denominator == 0.0 {
        return 1.0;
    }

    1.0 - (dot_sum / denominator)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
#[inline]
unsafe fn inner_product_avx512(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len();
    let mut sum = _mm512_setzero_ps();

    let chunks = n / 16;
    for i in 0..chunks {
        let offset = i * 16;
        let va = _mm512_loadu_ps(a.as_ptr().add(offset));
        let vb = _mm512_loadu_ps(b.as_ptr().add(offset));
        sum = _mm512_fmadd_ps(va, vb, sum);
    }

    let mut result = _mm512_reduce_add_ps(sum);

    for i in (chunks * 16)..n {
        result += a[i] * b[i];
    }

    -result
}

// ============================================================================
// AVX2 Implementations
// ============================================================================

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
unsafe fn euclidean_distance_avx2(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len();
    let mut sum = _mm256_setzero_ps();

    let chunks = n / 8;
    for i in 0..chunks {
        let offset = i * 8;
        let va = _mm256_loadu_ps(a.as_ptr().add(offset));
        let vb = _mm256_loadu_ps(b.as_ptr().add(offset));
        let diff = _mm256_sub_ps(va, vb);
        sum = _mm256_fmadd_ps(diff, diff, sum);
    }

    // Horizontal sum
    let sum_high = _mm256_extractf128_ps(sum, 1);
    let sum_low = _mm256_castps256_ps128(sum);
    let sum128 = _mm_add_ps(sum_high, sum_low);
    let sum64 = _mm_add_ps(sum128, _mm_movehl_ps(sum128, sum128));
    let sum32 = _mm_add_ss(sum64, _mm_shuffle_ps(sum64, sum64, 1));

    let mut result = _mm_cvtss_f32(sum32);

    for i in (chunks * 8)..n {
        let diff = a[i] - b[i];
        result += diff * diff;
    }

    result.sqrt()
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
unsafe fn cosine_distance_avx2(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len();
    let mut dot = _mm256_setzero_ps();
    let mut norm_a = _mm256_setzero_ps();
    let mut norm_b = _mm256_setzero_ps();

    let chunks = n / 8;
    for i in 0..chunks {
        let offset = i * 8;
        let va = _mm256_loadu_ps(a.as_ptr().add(offset));
        let vb = _mm256_loadu_ps(b.as_ptr().add(offset));

        dot = _mm256_fmadd_ps(va, vb, dot);
        norm_a = _mm256_fmadd_ps(va, va, norm_a);
        norm_b = _mm256_fmadd_ps(vb, vb, norm_b);
    }

    // Horizontal sums
    let dot_sum = horizontal_sum_256(dot);
    let norm_a_sum = horizontal_sum_256(norm_a);
    let norm_b_sum = horizontal_sum_256(norm_b);

    let mut dot_total = dot_sum;
    let mut norm_a_total = norm_a_sum;
    let mut norm_b_total = norm_b_sum;

    for i in (chunks * 8)..n {
        dot_total += a[i] * b[i];
        norm_a_total += a[i] * a[i];
        norm_b_total += b[i] * b[i];
    }

    let denominator = (norm_a_total * norm_b_total).sqrt();
    if denominator == 0.0 {
        return 1.0;
    }

    1.0 - (dot_total / denominator)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
unsafe fn inner_product_avx2(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len();
    let mut sum = _mm256_setzero_ps();

    let chunks = n / 8;
    for i in 0..chunks {
        let offset = i * 8;
        let va = _mm256_loadu_ps(a.as_ptr().add(offset));
        let vb = _mm256_loadu_ps(b.as_ptr().add(offset));
        sum = _mm256_fmadd_ps(va, vb, sum);
    }

    let mut result = horizontal_sum_256(sum);

    for i in (chunks * 8)..n {
        result += a[i] * b[i];
    }

    -result
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn manhattan_distance_avx2(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len();
    let sign_mask = _mm256_set1_ps(-0.0); // Sign bit mask
    let mut sum = _mm256_setzero_ps();

    let chunks = n / 8;
    for i in 0..chunks {
        let offset = i * 8;
        let va = _mm256_loadu_ps(a.as_ptr().add(offset));
        let vb = _mm256_loadu_ps(b.as_ptr().add(offset));
        let diff = _mm256_sub_ps(va, vb);
        let abs_diff = _mm256_andnot_ps(sign_mask, diff); // Clear sign bit
        sum = _mm256_add_ps(sum, abs_diff);
    }

    let mut result = horizontal_sum_256(sum);

    for i in (chunks * 8)..n {
        result += (a[i] - b[i]).abs();
    }

    result
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn horizontal_sum_256(v: __m256) -> f32 {
    let sum_high = _mm256_extractf128_ps(v, 1);
    let sum_low = _mm256_castps256_ps128(v);
    let sum128 = _mm_add_ps(sum_high, sum_low);
    let sum64 = _mm_add_ps(sum128, _mm_movehl_ps(sum128, sum128));
    let sum32 = _mm_add_ss(sum64, _mm_shuffle_ps(sum64, sum64, 1));
    _mm_cvtss_f32(sum32)
}

// ============================================================================
// ARM NEON Implementations
// ============================================================================

#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn euclidean_distance_neon(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::aarch64::*;

    let n = a.len();
    let mut sum = vdupq_n_f32(0.0);

    let chunks = n / 4;
    for i in 0..chunks {
        let offset = i * 4;
        let va = vld1q_f32(a.as_ptr().add(offset));
        let vb = vld1q_f32(b.as_ptr().add(offset));
        let diff = vsubq_f32(va, vb);
        sum = vfmaq_f32(sum, diff, diff);
    }

    let mut result = vaddvq_f32(sum);

    for i in (chunks * 4)..n {
        let diff = a[i] - b[i];
        result += diff * diff;
    }

    result.sqrt()
}

#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn cosine_distance_neon(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::aarch64::*;

    let n = a.len();
    let mut dot = vdupq_n_f32(0.0);
    let mut norm_a = vdupq_n_f32(0.0);
    let mut norm_b = vdupq_n_f32(0.0);

    let chunks = n / 4;
    for i in 0..chunks {
        let offset = i * 4;
        let va = vld1q_f32(a.as_ptr().add(offset));
        let vb = vld1q_f32(b.as_ptr().add(offset));

        dot = vfmaq_f32(dot, va, vb);
        norm_a = vfmaq_f32(norm_a, va, va);
        norm_b = vfmaq_f32(norm_b, vb, vb);
    }

    let mut dot_sum = vaddvq_f32(dot);
    let mut norm_a_sum = vaddvq_f32(norm_a);
    let mut norm_b_sum = vaddvq_f32(norm_b);

    for i in (chunks * 4)..n {
        dot_sum += a[i] * b[i];
        norm_a_sum += a[i] * a[i];
        norm_b_sum += b[i] * b[i];
    }

    let denominator = (norm_a_sum * norm_b_sum).sqrt();
    if denominator == 0.0 {
        return 1.0;
    }

    1.0 - (dot_sum / denominator)
}

#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn inner_product_neon(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::aarch64::*;

    let n = a.len();
    let mut sum = vdupq_n_f32(0.0);

    let chunks = n / 4;
    for i in 0..chunks {
        let offset = i * 4;
        let va = vld1q_f32(a.as_ptr().add(offset));
        let vb = vld1q_f32(b.as_ptr().add(offset));
        sum = vfmaq_f32(sum, va, vb);
    }

    let mut result = vaddvq_f32(sum);

    for i in (chunks * 4)..n {
        result += a[i] * b[i];
    }

    -result
}

// ============================================================================
// Public Wrapper Functions
// ============================================================================

// AVX-512 wrappers
#[cfg(target_arch = "x86_64")]
pub fn euclidean_distance_avx512_wrapper(a: &[f32], b: &[f32]) -> f32 {
    if is_x86_feature_detected!("avx512f") {
        unsafe { euclidean_distance_avx512(a, b) }
    } else {
        scalar::euclidean_distance(a, b)
    }
}

#[cfg(not(target_arch = "x86_64"))]
pub fn euclidean_distance_avx512_wrapper(a: &[f32], b: &[f32]) -> f32 {
    scalar::euclidean_distance(a, b)
}

#[cfg(target_arch = "x86_64")]
pub fn cosine_distance_avx512_wrapper(a: &[f32], b: &[f32]) -> f32 {
    if is_x86_feature_detected!("avx512f") {
        unsafe { cosine_distance_avx512(a, b) }
    } else {
        scalar::cosine_distance(a, b)
    }
}

#[cfg(not(target_arch = "x86_64"))]
pub fn cosine_distance_avx512_wrapper(a: &[f32], b: &[f32]) -> f32 {
    scalar::cosine_distance(a, b)
}

#[cfg(target_arch = "x86_64")]
pub fn inner_product_avx512_wrapper(a: &[f32], b: &[f32]) -> f32 {
    if is_x86_feature_detected!("avx512f") {
        unsafe { inner_product_avx512(a, b) }
    } else {
        scalar::inner_product_distance(a, b)
    }
}

#[cfg(not(target_arch = "x86_64"))]
pub fn inner_product_avx512_wrapper(a: &[f32], b: &[f32]) -> f32 {
    scalar::inner_product_distance(a, b)
}

// AVX2 wrappers
#[cfg(target_arch = "x86_64")]
pub fn euclidean_distance_avx2_wrapper(a: &[f32], b: &[f32]) -> f32 {
    if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
        unsafe { euclidean_distance_avx2(a, b) }
    } else {
        scalar::euclidean_distance(a, b)
    }
}

#[cfg(not(target_arch = "x86_64"))]
pub fn euclidean_distance_avx2_wrapper(a: &[f32], b: &[f32]) -> f32 {
    scalar::euclidean_distance(a, b)
}

#[cfg(target_arch = "x86_64")]
pub fn cosine_distance_avx2_wrapper(a: &[f32], b: &[f32]) -> f32 {
    if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
        unsafe { cosine_distance_avx2(a, b) }
    } else {
        scalar::cosine_distance(a, b)
    }
}

#[cfg(not(target_arch = "x86_64"))]
pub fn cosine_distance_avx2_wrapper(a: &[f32], b: &[f32]) -> f32 {
    scalar::cosine_distance(a, b)
}

#[cfg(target_arch = "x86_64")]
pub fn inner_product_avx2_wrapper(a: &[f32], b: &[f32]) -> f32 {
    if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
        unsafe { inner_product_avx2(a, b) }
    } else {
        scalar::inner_product_distance(a, b)
    }
}

#[cfg(not(target_arch = "x86_64"))]
pub fn inner_product_avx2_wrapper(a: &[f32], b: &[f32]) -> f32 {
    scalar::inner_product_distance(a, b)
}

#[cfg(target_arch = "x86_64")]
pub fn manhattan_distance_avx2_wrapper(a: &[f32], b: &[f32]) -> f32 {
    if is_x86_feature_detected!("avx2") {
        unsafe { manhattan_distance_avx2(a, b) }
    } else {
        scalar::manhattan_distance(a, b)
    }
}

#[cfg(not(target_arch = "x86_64"))]
pub fn manhattan_distance_avx2_wrapper(a: &[f32], b: &[f32]) -> f32 {
    scalar::manhattan_distance(a, b)
}

// NEON wrappers
#[cfg(target_arch = "aarch64")]
pub fn euclidean_distance_neon_wrapper(a: &[f32], b: &[f32]) -> f32 {
    unsafe { euclidean_distance_neon(a, b) }
}

#[cfg(not(target_arch = "aarch64"))]
pub fn euclidean_distance_neon_wrapper(a: &[f32], b: &[f32]) -> f32 {
    scalar::euclidean_distance(a, b)
}

#[cfg(target_arch = "aarch64")]
pub fn cosine_distance_neon_wrapper(a: &[f32], b: &[f32]) -> f32 {
    unsafe { cosine_distance_neon(a, b) }
}

#[cfg(not(target_arch = "aarch64"))]
pub fn cosine_distance_neon_wrapper(a: &[f32], b: &[f32]) -> f32 {
    scalar::cosine_distance(a, b)
}

#[cfg(target_arch = "aarch64")]
pub fn inner_product_neon_wrapper(a: &[f32], b: &[f32]) -> f32 {
    unsafe { inner_product_neon(a, b) }
}

#[cfg(not(target_arch = "aarch64"))]
pub fn inner_product_neon_wrapper(a: &[f32], b: &[f32]) -> f32 {
    scalar::inner_product_distance(a, b)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_avx2_euclidean() {
        let a: Vec<f32> = (0..128).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..128).map(|i| (i + 1) as f32).collect();

        let scalar = scalar::euclidean_distance(&a, &b);
        let simd = euclidean_distance_avx2_wrapper(&a, &b);

        assert!((scalar - simd).abs() < 1e-4, "scalar={}, simd={}", scalar, simd);
    }

    #[test]
    fn test_avx2_cosine() {
        let a: Vec<f32> = (0..128).map(|i| i as f32 * 0.01).collect();
        let b: Vec<f32> = (0..128).map(|i| (128 - i) as f32 * 0.01).collect();

        let scalar = scalar::cosine_distance(&a, &b);
        let simd = cosine_distance_avx2_wrapper(&a, &b);

        assert!((scalar - simd).abs() < 1e-4, "scalar={}, simd={}", scalar, simd);
    }

    #[test]
    fn test_avx2_inner_product() {
        let a: Vec<f32> = (0..128).map(|i| i as f32 * 0.01).collect();
        let b: Vec<f32> = (0..128).map(|i| (128 - i) as f32 * 0.01).collect();

        let scalar = scalar::inner_product_distance(&a, &b);
        let simd = inner_product_avx2_wrapper(&a, &b);

        assert!((scalar - simd).abs() < 1e-3, "scalar={}, simd={}", scalar, simd);
    }

    #[test]
    fn test_avx2_manhattan() {
        let a: Vec<f32> = (0..128).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..128).map(|i| (i + 1) as f32).collect();

        let scalar = scalar::manhattan_distance(&a, &b);
        let simd = manhattan_distance_avx2_wrapper(&a, &b);

        assert!((scalar - simd).abs() < 1e-4, "scalar={}, simd={}", scalar, simd);
    }

    #[test]
    fn test_remainder_handling() {
        // Test with non-aligned sizes
        for size in [1, 3, 5, 7, 9, 15, 17, 31, 33, 63, 65, 127, 129] {
            let a: Vec<f32> = (0..size).map(|i| i as f32).collect();
            let b: Vec<f32> = (0..size).map(|i| (size - i) as f32).collect();

            let scalar = scalar::euclidean_distance(&a, &b);
            let simd = euclidean_distance_avx2_wrapper(&a, &b);

            assert!(
                (scalar - simd).abs() < 1e-3,
                "size={}, scalar={}, simd={}",
                size,
                scalar,
                simd
            );
        }
    }
}

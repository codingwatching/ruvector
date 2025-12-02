//! SQL operators and distance functions for vector similarity search
//!
//! Provides both array-based and zero-copy RuVector distance functions for SQL queries.
//! The RuVector functions are preferred for better performance (zero-copy access).

use pgrx::prelude::*;

use crate::distance::{
    cosine_distance, euclidean_distance, inner_product_distance, manhattan_distance,
};
use crate::types::RuVector;

// ============================================================================
// Zero-Copy Distance Functions (RuVector-based) - PREFERRED
// ============================================================================

/// Compute L2 (Euclidean) distance between two RuVector instances
///
/// This is the zero-copy version that operates directly on RuVector types.
/// Uses SIMD optimizations (AVX-512, AVX2, or NEON) automatically.
///
/// PARALLEL SAFE: This function can be executed by parallel workers
#[pg_extern(immutable, strict, parallel_safe, name = "ruvector_l2_distance")]
pub fn ruvector_l2_distance(a: RuVector, b: RuVector) -> f32 {
    if a.dimensions() != b.dimensions() {
        pgrx::error!(
            "Cannot compute distance between vectors of different dimensions ({} vs {})",
            a.dimensions(),
            b.dimensions()
        );
    }
    // Zero-copy: as_slice() returns &[f32] without allocation
    euclidean_distance(a.as_slice(), b.as_slice())
}

/// Compute inner product distance (negative dot product) between two RuVector instances
///
/// Returns -(a·b) for use with ORDER BY ASC in nearest neighbor queries.
/// Uses SIMD optimizations automatically.
///
/// PARALLEL SAFE: This function can be executed by parallel workers
#[pg_extern(immutable, strict, parallel_safe, name = "ruvector_ip_distance")]
pub fn ruvector_ip_distance(a: RuVector, b: RuVector) -> f32 {
    if a.dimensions() != b.dimensions() {
        pgrx::error!(
            "Cannot compute distance between vectors of different dimensions ({} vs {})",
            a.dimensions(),
            b.dimensions()
        );
    }
    inner_product_distance(a.as_slice(), b.as_slice())
}

/// Compute cosine distance between two RuVector instances
///
/// Returns 1 - (a·b)/(‖a‖‖b‖). Zero distance means vectors point in same direction.
/// Uses SIMD optimizations automatically.
///
/// PARALLEL SAFE: This function can be executed by parallel workers
#[pg_extern(immutable, strict, parallel_safe, name = "ruvector_cosine_distance")]
pub fn ruvector_cosine_distance(a: RuVector, b: RuVector) -> f32 {
    if a.dimensions() != b.dimensions() {
        pgrx::error!(
            "Cannot compute distance between vectors of different dimensions ({} vs {})",
            a.dimensions(),
            b.dimensions()
        );
    }
    cosine_distance(a.as_slice(), b.as_slice())
}

/// Compute L1 (Manhattan) distance between two RuVector instances
///
/// Returns sum of absolute differences. Uses SIMD optimizations automatically.
///
/// PARALLEL SAFE: This function can be executed by parallel workers
#[pg_extern(immutable, strict, parallel_safe, name = "ruvector_l1_distance")]
pub fn ruvector_l1_distance(a: RuVector, b: RuVector) -> f32 {
    if a.dimensions() != b.dimensions() {
        pgrx::error!(
            "Cannot compute distance between vectors of different dimensions ({} vs {})",
            a.dimensions(),
            b.dimensions()
        );
    }
    manhattan_distance(a.as_slice(), b.as_slice())
}

// ============================================================================
// SQL Operators for RuVector Distance Functions
// ============================================================================

/// SQL operator <-> for L2 (Euclidean) distance
///
/// Example: SELECT * FROM items ORDER BY embedding <-> '[1,2,3]' LIMIT 10;
#[pg_operator(immutable, parallel_safe)]
#[opname(<->)]
pub fn ruvector_l2_dist_op(a: RuVector, b: RuVector) -> f32 {
    ruvector_l2_distance(a, b)
}

/// SQL operator <#> for negative inner product distance
///
/// Example: SELECT * FROM items ORDER BY embedding <#> '[1,2,3]' LIMIT 10;
#[pg_operator(immutable, parallel_safe)]
#[opname(<#>)]
pub fn ruvector_neg_ip_op(a: RuVector, b: RuVector) -> f32 {
    ruvector_ip_distance(a, b)
}

/// SQL operator <=> for cosine distance
///
/// Example: SELECT * FROM items ORDER BY embedding <=> '[1,2,3]' LIMIT 10;
#[pg_operator(immutable, parallel_safe)]
#[opname(<=>)]
pub fn ruvector_cosine_dist_op(a: RuVector, b: RuVector) -> f32 {
    ruvector_cosine_distance(a, b)
}

/// SQL operator <+> for L1 (Manhattan) distance
///
/// Example: SELECT * FROM items ORDER BY embedding <+> '[1,2,3]' LIMIT 10;
#[pg_operator(immutable, parallel_safe)]
#[opname(<+>)]
pub fn ruvector_l1_dist_op(a: RuVector, b: RuVector) -> f32 {
    ruvector_l1_distance(a, b)
}

// ============================================================================
// Distance Functions (Array-based) - LEGACY
// ============================================================================

/// Compute L2 (Euclidean) distance between two float arrays
#[pg_extern(immutable, parallel_safe)]
pub fn l2_distance_arr(a: Vec<f32>, b: Vec<f32>) -> f32 {
    if a.len() != b.len() {
        pgrx::error!(
            "Cannot compute distance between vectors of different dimensions ({} vs {})",
            a.len(),
            b.len()
        );
    }
    euclidean_distance(&a, &b)
}

/// Compute inner product between two float arrays
#[pg_extern(immutable, parallel_safe)]
pub fn inner_product_arr(a: Vec<f32>, b: Vec<f32>) -> f32 {
    if a.len() != b.len() {
        pgrx::error!(
            "Cannot compute distance between vectors of different dimensions ({} vs {})",
            a.len(),
            b.len()
        );
    }
    -inner_product_distance(&a, &b)
}

/// Compute negative inner product (for ORDER BY ASC nearest neighbor)
#[pg_extern(immutable, parallel_safe)]
pub fn neg_inner_product_arr(a: Vec<f32>, b: Vec<f32>) -> f32 {
    if a.len() != b.len() {
        pgrx::error!(
            "Cannot compute distance between vectors of different dimensions ({} vs {})",
            a.len(),
            b.len()
        );
    }
    inner_product_distance(&a, &b)
}

/// Compute cosine distance between two float arrays
#[pg_extern(immutable, parallel_safe)]
pub fn cosine_distance_arr(a: Vec<f32>, b: Vec<f32>) -> f32 {
    if a.len() != b.len() {
        pgrx::error!(
            "Cannot compute distance between vectors of different dimensions ({} vs {})",
            a.len(),
            b.len()
        );
    }
    cosine_distance(&a, &b)
}

/// Compute cosine similarity between two float arrays
#[pg_extern(immutable, parallel_safe)]
pub fn cosine_similarity_arr(a: Vec<f32>, b: Vec<f32>) -> f32 {
    1.0 - cosine_distance_arr(a, b)
}

/// Compute L1 (Manhattan) distance between two float arrays
#[pg_extern(immutable, parallel_safe)]
pub fn l1_distance_arr(a: Vec<f32>, b: Vec<f32>) -> f32 {
    if a.len() != b.len() {
        pgrx::error!(
            "Cannot compute distance between vectors of different dimensions ({} vs {})",
            a.len(),
            b.len()
        );
    }
    manhattan_distance(&a, &b)
}

// ============================================================================
// Vector Utility Functions
// ============================================================================

/// Normalize a vector to unit length
#[pg_extern(immutable, parallel_safe)]
pub fn vector_normalize(v: Vec<f32>) -> Vec<f32> {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm == 0.0 {
        return v;
    }
    v.iter().map(|x| x / norm).collect()
}

/// Add two vectors element-wise
#[pg_extern(immutable, parallel_safe)]
pub fn vector_add(a: Vec<f32>, b: Vec<f32>) -> Vec<f32> {
    if a.len() != b.len() {
        pgrx::error!("Vectors must have the same dimensions");
    }
    a.iter().zip(b.iter()).map(|(x, y)| x + y).collect()
}

/// Subtract two vectors element-wise
#[pg_extern(immutable, parallel_safe)]
pub fn vector_sub(a: Vec<f32>, b: Vec<f32>) -> Vec<f32> {
    if a.len() != b.len() {
        pgrx::error!("Vectors must have the same dimensions");
    }
    a.iter().zip(b.iter()).map(|(x, y)| x - y).collect()
}

/// Multiply vector by scalar
#[pg_extern(immutable, parallel_safe)]
pub fn vector_mul_scalar(v: Vec<f32>, scalar: f32) -> Vec<f32> {
    v.iter().map(|x| x * scalar).collect()
}

/// Get vector dimensions
#[pg_extern(immutable, parallel_safe)]
pub fn vector_dims(v: Vec<f32>) -> i32 {
    v.len() as i32
}

/// Get vector L2 norm
#[pg_extern(immutable, parallel_safe)]
pub fn vector_norm(v: Vec<f32>) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

/// Average two vectors
#[pg_extern(immutable, parallel_safe)]
pub fn vector_avg2(a: Vec<f32>, b: Vec<f32>) -> Vec<f32> {
    if a.len() != b.len() {
        pgrx::error!("Vectors must have the same dimensions");
    }
    a.iter().zip(b.iter()).map(|(x, y)| (x + y) / 2.0).collect()
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(any(test, feature = "pg_test"))]
#[pg_schema]
mod tests {
    use super::*;

    // ========================================================================
    // Zero-Copy RuVector Function Tests
    // ========================================================================

    #[pg_test]
    fn test_ruvector_l2_distance() {
        let a = RuVector::from_slice(&[0.0, 0.0, 0.0]);
        let b = RuVector::from_slice(&[3.0, 4.0, 0.0]);
        let dist = ruvector_l2_distance(a, b);
        assert!((dist - 5.0).abs() < 1e-5, "Expected 5.0, got {}", dist);
    }

    #[pg_test]
    fn test_ruvector_cosine_distance() {
        let a = RuVector::from_slice(&[1.0, 0.0, 0.0]);
        let b = RuVector::from_slice(&[1.0, 0.0, 0.0]);
        let dist = ruvector_cosine_distance(a, b);
        assert!(dist.abs() < 1e-5, "Same vectors should have 0 distance, got {}", dist);
    }

    #[pg_test]
    fn test_ruvector_cosine_orthogonal() {
        let a = RuVector::from_slice(&[1.0, 0.0]);
        let b = RuVector::from_slice(&[0.0, 1.0]);
        let dist = ruvector_cosine_distance(a, b);
        assert!((dist - 1.0).abs() < 1e-5, "Orthogonal vectors should have distance 1.0, got {}", dist);
    }

    #[pg_test]
    fn test_ruvector_ip_distance() {
        let a = RuVector::from_slice(&[1.0, 2.0, 3.0]);
        let b = RuVector::from_slice(&[4.0, 5.0, 6.0]);
        let dist = ruvector_ip_distance(a, b);
        // -(1*4 + 2*5 + 3*6) = -32
        assert!((dist - (-32.0)).abs() < 1e-5, "Expected -32.0, got {}", dist);
    }

    #[pg_test]
    fn test_ruvector_l1_distance() {
        let a = RuVector::from_slice(&[1.0, 2.0, 3.0]);
        let b = RuVector::from_slice(&[4.0, 6.0, 8.0]);
        let dist = ruvector_l1_distance(a, b);
        // |4-1| + |6-2| + |8-3| = 3 + 4 + 5 = 12
        assert!((dist - 12.0).abs() < 1e-5, "Expected 12.0, got {}", dist);
    }

    #[pg_test]
    fn test_ruvector_operators() {
        let a = RuVector::from_slice(&[0.0, 0.0]);
        let b = RuVector::from_slice(&[3.0, 4.0]);

        // Test L2 operator <->
        let l2 = ruvector_l2_dist_op(a.clone(), b.clone());
        assert!((l2 - 5.0).abs() < 1e-5);

        // Test that operators give same results as functions
        let l2_fn = ruvector_l2_distance(a.clone(), b.clone());
        assert!((l2 - l2_fn).abs() < 1e-10);
    }

    #[pg_test]
    fn test_ruvector_large_vectors() {
        // Test with larger vectors to exercise SIMD paths
        let size = 1024;
        let a_data: Vec<f32> = (0..size).map(|i| i as f32 * 0.01).collect();
        let b_data: Vec<f32> = (0..size).map(|i| (size - i) as f32 * 0.01).collect();

        let a = RuVector::from_slice(&a_data);
        let b = RuVector::from_slice(&b_data);

        // Just verify it doesn't panic and returns reasonable values
        let l2 = ruvector_l2_distance(a.clone(), b.clone());
        assert!(l2 > 0.0 && l2.is_finite(), "L2 distance should be positive and finite");

        let cosine = ruvector_cosine_distance(a.clone(), b.clone());
        assert!(cosine >= 0.0 && cosine <= 2.0, "Cosine distance should be in [0,2]");

        let ip = ruvector_ip_distance(a.clone(), b.clone());
        assert!(ip.is_finite(), "Inner product should be finite");

        let l1 = ruvector_l1_distance(a.clone(), b.clone());
        assert!(l1 > 0.0 && l1.is_finite(), "L1 distance should be positive and finite");
    }

    #[pg_test]
    #[should_panic(expected = "Cannot compute distance between vectors of different dimensions")]
    fn test_ruvector_dimension_mismatch() {
        let a = RuVector::from_slice(&[1.0, 2.0, 3.0]);
        let b = RuVector::from_slice(&[1.0, 2.0]);
        let _ = ruvector_l2_distance(a, b);
    }

    #[pg_test]
    fn test_ruvector_zero_vectors() {
        let a = RuVector::from_slice(&[0.0, 0.0, 0.0]);
        let b = RuVector::from_slice(&[0.0, 0.0, 0.0]);

        let l2 = ruvector_l2_distance(a.clone(), b.clone());
        assert!(l2.abs() < 1e-5, "Distance between zero vectors should be 0");

        let cosine = ruvector_cosine_distance(a.clone(), b.clone());
        assert!(cosine.abs() <= 1.0, "Cosine distance should handle zero vectors");
    }

    #[pg_test]
    fn test_ruvector_simd_alignment() {
        // Test various sizes to ensure SIMD remainder handling works
        for size in [1, 3, 7, 8, 15, 16, 31, 32, 63, 64, 127, 128, 256] {
            let a_data: Vec<f32> = (0..size).map(|i| i as f32).collect();
            let b_data: Vec<f32> = (0..size).map(|i| (i + 1) as f32).collect();

            let a = RuVector::from_slice(&a_data);
            let b = RuVector::from_slice(&b_data);

            let dist = ruvector_l2_distance(a, b);
            assert!(dist.is_finite() && dist > 0.0,
                "L2 distance failed for size {}", size);
        }
    }

    // ========================================================================
    // Legacy Array-Based Function Tests
    // ========================================================================

    #[pg_test]
    fn test_l2_distance() {
        let a = vec![0.0, 0.0, 0.0];
        let b = vec![3.0, 4.0, 0.0];
        let dist = l2_distance_arr(a, b);
        assert!((dist - 5.0).abs() < 1e-5);
    }

    #[pg_test]
    fn test_cosine_distance() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        let dist = cosine_distance_arr(a, b);
        assert!(dist.abs() < 1e-5);
    }

    #[pg_test]
    fn test_inner_product() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        let ip = inner_product_arr(a, b);
        assert!((ip - 32.0).abs() < 1e-5);
    }

    #[pg_test]
    fn test_vector_normalize() {
        let v = vec![3.0, 4.0];
        let n = vector_normalize(v);
        let norm: f32 = n.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5);
    }
}

//! SQL operators and distance functions for vector similarity search
//!
//! Provides array-based distance functions for use in SQL queries.

use pgrx::prelude::*;

use crate::distance::{
    cosine_distance, euclidean_distance, inner_product_distance, manhattan_distance,
};

// ============================================================================
// Distance Functions (Array-based)
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

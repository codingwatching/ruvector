//! Vector type implementations for PostgreSQL
//!
//! This module provides the core vector types:
//! - `RuVector`: Primary f32 vector type (pgvector compatible)
//! - `HalfVec`: Half-precision (f16) vector for memory savings
//! - `SparseVec`: Sparse vector for high-dimensional data

mod vector;
mod halfvec;
mod sparsevec;

pub use vector::RuVector;
pub use halfvec::HalfVec;
pub use sparsevec::SparseVec;

use std::sync::atomic::{AtomicUsize, Ordering};

/// Global vector cache memory tracking
static VECTOR_CACHE_BYTES: AtomicUsize = AtomicUsize::new(0);

/// Get current vector cache memory usage in MB
pub fn get_vector_cache_memory_mb() -> f64 {
    VECTOR_CACHE_BYTES.load(Ordering::Relaxed) as f64 / (1024.0 * 1024.0)
}

/// Track memory allocation
pub(crate) fn track_allocation(bytes: usize) {
    VECTOR_CACHE_BYTES.fetch_add(bytes, Ordering::Relaxed);
}

/// Track memory deallocation
pub(crate) fn track_deallocation(bytes: usize) {
    VECTOR_CACHE_BYTES.fetch_sub(bytes, Ordering::Relaxed);
}

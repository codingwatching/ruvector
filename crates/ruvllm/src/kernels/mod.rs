//! NEON-Optimized LLM Kernels for Mac M4 Pro
//!
//! This module provides highly optimized SIMD kernels for LLM operations,
//! specifically tuned for Apple Silicon (M1/M2/M3/M4) using ARM NEON intrinsics.
//!
//! ## Kernel Categories
//!
//! - [`attention`]: Flash Attention 2, Paged Attention, MQA/GQA
//! - [`rope`]: Rotary Position Embeddings (RoPE)
//! - [`norm`]: RMSNorm, LayerNorm
//! - [`matmul`]: Batched GEMM operations
//!
//! ## Performance Optimizations
//!
//! All kernels implement:
//! - 4x loop unrolling for instruction-level parallelism
//! - FMA instructions for improved throughput
//! - Pointer caching to reduce address calculations
//! - Efficient horizontal reductions via `vaddvq_f32`
//! - Software prefetching for large tensors
//!
//! ## Usage
//!
//! ```rust,ignore
//! use ruvllm::kernels::{flash_attention_neon, apply_rope_neon, rms_norm_neon};
//!
//! // Flash attention with NEON SIMD
//! let output = flash_attention_neon(&query, &key, &value, scale, true);
//!
//! // Apply RoPE to query/key tensors
//! apply_rope_neon(&mut qk, &positions, head_dim, 10000.0);
//!
//! // RMSNorm normalization
//! rms_norm_neon(&mut hidden, &weight, 1e-6);
//! ```

pub mod attention;
pub mod matmul;
pub mod norm;
pub mod rope;

// Re-exports for convenience
pub use attention::{
    flash_attention_neon, grouped_query_attention_neon, multi_query_attention_neon,
    paged_attention_neon, PagedKvCache,
};
pub use matmul::{batched_gemm_neon, gemm_neon, gemv_neon};
pub use norm::{layer_norm_neon, rms_norm_neon};
pub use rope::{apply_rope_neon, precompute_rope_tables, RopeConfig};

/// SIMD lane width for NEON (128-bit = 4 floats)
pub const NEON_LANE_WIDTH: usize = 4;

/// Optimal unroll factor for M4 Pro's 6-wide superscalar core
pub const UNROLL_FACTOR: usize = 4;

/// Prefetch distance in cache lines (64 bytes = 16 floats)
pub const PREFETCH_DISTANCE: usize = 64;

/// Check if NEON is available at runtime
#[inline(always)]
pub fn is_neon_available() -> bool {
    #[cfg(target_arch = "aarch64")]
    {
        true // NEON is always available on aarch64
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        false
    }
}

/// Kernel configuration for attention operations
#[derive(Debug, Clone, Copy)]
pub struct AttentionConfig {
    /// Number of query heads
    pub num_heads: usize,
    /// Number of key-value heads (for GQA)
    pub num_kv_heads: usize,
    /// Dimension per head
    pub head_dim: usize,
    /// Maximum sequence length
    pub max_seq_len: usize,
    /// Whether to use causal masking
    pub causal: bool,
    /// Softmax scale factor (typically 1/sqrt(head_dim))
    pub scale: f32,
}

impl Default for AttentionConfig {
    fn default() -> Self {
        Self {
            num_heads: 32,
            num_kv_heads: 8,
            head_dim: 128,
            max_seq_len: 4096,
            causal: true,
            scale: 0.0, // Will be computed from head_dim if 0
        }
    }
}

impl AttentionConfig {
    /// Get the effective scale (computes from head_dim if not set)
    #[inline(always)]
    pub fn effective_scale(&self) -> f32 {
        if self.scale == 0.0 {
            1.0 / (self.head_dim as f32).sqrt()
        } else {
            self.scale
        }
    }

    /// Get the GQA ratio (num_heads / num_kv_heads)
    #[inline(always)]
    pub fn gqa_ratio(&self) -> usize {
        self.num_heads / self.num_kv_heads
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_attention_config() {
        let config = AttentionConfig::default();
        assert_eq!(config.gqa_ratio(), 4);
        assert!((config.effective_scale() - 0.088388).abs() < 0.001);
    }

    #[test]
    fn test_neon_available() {
        #[cfg(target_arch = "aarch64")]
        assert!(is_neon_available());
    }
}

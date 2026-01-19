//! Metal GPU Acceleration for Apple Silicon M4 Pro
//!
//! This module provides GPU-accelerated compute shaders for LLM operations,
//! specifically optimized for Apple Silicon's Metal Performance Shaders and
//! the M4 Pro's matrix coprocessor (AMX/SME).
//!
//! ## Features
//!
//! - **Flash Attention**: Tiled attention with O(N) memory complexity
//! - **GEMM**: Optimized matrix multiplication using simdgroup_matrix
//! - **RMSNorm/LayerNorm**: Parallel normalization with warp-level reductions
//! - **RoPE**: Rotary position embedding application
//!
//! ## M4 Pro Optimizations
//!
//! - Uses `simdgroup_half8x8` for tensor core acceleration
//! - Optimized for 16KB threadgroup memory
//! - FP16 operations for 2x throughput
//! - Coalesced memory access patterns
//!
//! ## Usage
//!
//! ```rust,ignore
//! use ruvllm::metal::{MetalContext, MetalConfig};
//!
//! let ctx = MetalContext::new(MetalConfig::default())?;
//!
//! // Flash attention
//! let output = ctx.flash_attention(&q, &k, &v, &config)?;
//!
//! // Matrix multiplication
//! let c = ctx.gemm_f16(&a, &b, m, n, k)?;
//! ```

#[cfg(target_os = "macos")]
mod context;
#[cfg(target_os = "macos")]
mod pipelines;
#[cfg(target_os = "macos")]
mod buffers;
#[cfg(target_os = "macos")]
mod operations;

#[cfg(target_os = "macos")]
pub use context::{MetalContext, MetalConfig};
#[cfg(target_os = "macos")]
pub use pipelines::{MetalPipelines, PipelineCache};
#[cfg(target_os = "macos")]
pub use buffers::{MetalBuffer, MetalBufferPool};
#[cfg(target_os = "macos")]
pub use operations::*;

use crate::error::{Result, RuvLLMError};
use crate::kernels::AttentionConfig;

/// Attention parameters for Metal shaders
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct AttentionParams {
    /// Number of query heads
    pub num_heads: u32,
    /// Number of key-value heads
    pub num_kv_heads: u32,
    /// Dimension per head
    pub head_dim: u32,
    /// Sequence length (query)
    pub seq_len: u32,
    /// KV sequence length
    pub kv_len: u32,
    /// Softmax scale factor
    pub scale: f32,
    /// Whether to apply causal mask
    pub causal: u32,
    /// Padding for alignment
    pub _padding: u32,
}

impl AttentionParams {
    /// Create attention params from config
    pub fn from_config(config: &AttentionConfig, seq_len: usize, kv_len: usize) -> Self {
        Self {
            num_heads: config.num_heads as u32,
            num_kv_heads: config.num_kv_heads as u32,
            head_dim: config.head_dim as u32,
            seq_len: seq_len as u32,
            kv_len: kv_len as u32,
            scale: config.effective_scale(),
            causal: config.causal as u32,
            _padding: 0,
        }
    }
}

/// GEMM parameters for Metal shaders
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct GemmParams {
    /// M dimension (rows of A/C)
    pub m: u32,
    /// N dimension (cols of B/C)
    pub n: u32,
    /// K dimension (cols of A, rows of B)
    pub k: u32,
    /// Leading dimension of A
    pub lda: u32,
    /// Leading dimension of B
    pub ldb: u32,
    /// Leading dimension of C
    pub ldc: u32,
    /// Alpha scalar
    pub alpha: f32,
    /// Beta scalar
    pub beta: f32,
}

impl GemmParams {
    /// Create GEMM params for C = alpha * A @ B + beta * C
    pub fn new(m: usize, n: usize, k: usize) -> Self {
        Self {
            m: m as u32,
            n: n as u32,
            k: k as u32,
            lda: k as u32,  // Row-major
            ldb: n as u32,
            ldc: n as u32,
            alpha: 1.0,
            beta: 0.0,
        }
    }
}

/// Normalization parameters for Metal shaders
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct NormParams {
    /// Hidden dimension
    pub hidden_size: u32,
    /// Epsilon for numerical stability
    pub eps: f32,
    /// Number of elements per thread
    pub elements_per_thread: u32,
    /// Padding for alignment
    pub _padding: u32,
}

impl NormParams {
    /// Create norm params
    pub fn new(hidden_size: usize, eps: f32) -> Self {
        let elements_per_thread = (hidden_size + 255) / 256; // Distribute across 256 threads
        Self {
            hidden_size: hidden_size as u32,
            eps,
            elements_per_thread: elements_per_thread as u32,
            _padding: 0,
        }
    }
}

/// RoPE parameters for Metal shaders
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct RopeParams {
    /// Head dimension (must be even)
    pub head_dim: u32,
    /// Number of heads
    pub num_heads: u32,
    /// Position offset
    pub position: u32,
    /// Base for frequency calculation (default 10000)
    pub theta_base: f32,
}

impl RopeParams {
    /// Create RoPE params
    pub fn new(head_dim: usize, num_heads: usize, position: usize, theta_base: f32) -> Self {
        Self {
            head_dim: head_dim as u32,
            num_heads: num_heads as u32,
            position: position as u32,
            theta_base,
        }
    }
}

/// Tile sizes optimized for M4 Pro
pub mod tile_sizes {
    /// Attention tile size (fits in 16KB threadgroup memory)
    pub const ATTENTION_TILE: usize = 64;
    /// GEMM tile M dimension
    pub const GEMM_TILE_M: usize = 64;
    /// GEMM tile N dimension
    pub const GEMM_TILE_N: usize = 64;
    /// GEMM tile K dimension
    pub const GEMM_TILE_K: usize = 32;
    /// Number of threads per SIMD group
    pub const SIMD_SIZE: usize = 32;
    /// Maximum threads per threadgroup
    pub const MAX_THREADS_PER_THREADGROUP: usize = 1024;
}

/// Check if Metal is available on this system
#[cfg(target_os = "macos")]
pub fn is_metal_available() -> bool {
    metal::Device::system_default().is_some()
}

#[cfg(not(target_os = "macos"))]
pub fn is_metal_available() -> bool {
    false
}

/// Get Metal device information
#[cfg(target_os = "macos")]
pub fn get_device_info() -> Option<MetalDeviceInfo> {
    metal::Device::system_default().map(|device| MetalDeviceInfo {
        name: device.name().to_string(),
        registry_id: device.registry_id(),
        max_threads_per_threadgroup: device.max_threads_per_threadgroup().width as usize,
        max_buffer_length: device.max_buffer_length() as usize,
        has_unified_memory: device.has_unified_memory(),
        recommended_max_working_set_size: device.recommended_max_working_set_size() as usize,
    })
}

#[cfg(not(target_os = "macos"))]
pub fn get_device_info() -> Option<MetalDeviceInfo> {
    None
}

/// Metal device information
#[derive(Debug, Clone)]
pub struct MetalDeviceInfo {
    /// Device name (e.g., "Apple M4 Pro")
    pub name: String,
    /// Registry ID
    pub registry_id: u64,
    /// Maximum threads per threadgroup
    pub max_threads_per_threadgroup: usize,
    /// Maximum buffer length
    pub max_buffer_length: usize,
    /// Whether device has unified memory
    pub has_unified_memory: bool,
    /// Recommended working set size
    pub recommended_max_working_set_size: usize,
}

/// Embedded shader source code
pub mod shader_source {
    /// Flash Attention shader source
    pub const ATTENTION: &str = include_str!("shaders/attention.metal");
    /// GEMM shader source
    pub const GEMM: &str = include_str!("shaders/gemm.metal");
    /// Normalization shader source
    pub const NORM: &str = include_str!("shaders/norm.metal");
    /// RoPE shader source
    pub const ROPE: &str = include_str!("shaders/rope.metal");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_attention_params() {
        let config = AttentionConfig {
            num_heads: 32,
            num_kv_heads: 8,
            head_dim: 128,
            max_seq_len: 4096,
            causal: true,
            scale: 0.0,
        };

        let params = AttentionParams::from_config(&config, 1, 100);
        assert_eq!(params.num_heads, 32);
        assert_eq!(params.num_kv_heads, 8);
        assert_eq!(params.head_dim, 128);
        assert!(params.scale > 0.0);
    }

    #[test]
    fn test_gemm_params() {
        let params = GemmParams::new(64, 128, 256);
        assert_eq!(params.m, 64);
        assert_eq!(params.n, 128);
        assert_eq!(params.k, 256);
        assert_eq!(params.alpha, 1.0);
        assert_eq!(params.beta, 0.0);
    }

    #[test]
    fn test_norm_params() {
        let params = NormParams::new(4096, 1e-6);
        assert_eq!(params.hidden_size, 4096);
        assert!((params.eps - 1e-6).abs() < 1e-10);
    }

    #[test]
    fn test_rope_params() {
        let params = RopeParams::new(128, 32, 0, 10000.0);
        assert_eq!(params.head_dim, 128);
        assert_eq!(params.num_heads, 32);
        assert_eq!(params.theta_base, 10000.0);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn test_metal_available() {
        // Metal should be available on macOS
        let available = is_metal_available();
        println!("Metal available: {}", available);

        if available {
            let info = get_device_info().unwrap();
            println!("Device: {}", info.name);
            println!("Unified memory: {}", info.has_unified_memory);
        }
    }
}

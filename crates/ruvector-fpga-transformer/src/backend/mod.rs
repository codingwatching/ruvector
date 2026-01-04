//! Backend implementations for FPGA Transformer
//!
//! All backends implement the `TransformerBackend` trait for uniform API.

use crate::artifact::ModelArtifact;
use crate::error::Result;
use crate::types::{InferenceRequest, InferenceResult, ModelId};

#[cfg(feature = "native_sim")]
pub mod native_sim;

#[cfg(feature = "daemon")]
pub mod fpga_daemon;

#[cfg(feature = "pcie")]
pub mod fpga_pcie;

#[cfg(feature = "wasm")]
pub mod wasm_sim;

/// Trait for transformer inference backends
///
/// All backends must be thread-safe and implement the same API.
pub trait TransformerBackend: Send + Sync {
    /// Load a model artifact and return its ID
    ///
    /// The artifact is validated, test vectors are run, and
    /// the model is prepared for inference.
    fn load(&self, artifact: &ModelArtifact) -> Result<ModelId>;

    /// Run inference on the given request
    ///
    /// The request must specify a model that has been loaded.
    /// Returns the inference result with witness log.
    fn infer(&self, req: InferenceRequest) -> Result<InferenceResult>;

    /// Unload a model to free resources
    fn unload(&self, model: ModelId) -> Result<()>;

    /// Check if a model is loaded
    fn is_loaded(&self, model: ModelId) -> bool;

    /// Get the backend kind
    fn kind(&self) -> crate::types::BackendKind;

    /// Get backend-specific statistics
    fn stats(&self) -> BackendStats {
        BackendStats::default()
    }
}

/// Backend statistics
#[derive(Debug, Clone, Default)]
pub struct BackendStats {
    /// Number of models currently loaded
    pub models_loaded: usize,
    /// Total inferences performed
    pub total_inferences: u64,
    /// Total cycles consumed (FPGA only)
    pub total_cycles: u64,
    /// Average latency in nanoseconds
    pub avg_latency_ns: u64,
    /// P99 latency in nanoseconds
    pub p99_latency_ns: u64,
    /// Number of early exits
    pub early_exits: u64,
    /// Number of skipped inferences
    pub skipped: u64,
}

/// Protocol constants for daemon/PCIe communication
pub mod protocol {
    /// Magic number for frame validation
    pub const MAGIC: u32 = 0x5256_5846; // "RVXF" - RuVector FPGA

    /// Current protocol version
    pub const VERSION: u16 = 1;

    /// Frame header size in bytes
    pub const HEADER_SIZE: usize = 24;

    /// Maximum payload size
    pub const MAX_PAYLOAD: usize = 1024 * 1024; // 1MB

    /// Request flags
    pub mod flags {
        /// Return only top-K predictions
        pub const TOPK_ONLY: u16 = 0x0001;
        /// Use LUT-based softmax
        pub const LUT_SOFTMAX: u16 = 0x0002;
        /// Enable early exit
        pub const EARLY_EXIT: u16 = 0x0004;
        /// Return detailed witness
        pub const WITNESS_DETAIL: u16 = 0x0008;
    }

    /// Response status codes
    pub mod status {
        /// Success
        pub const OK: u16 = 0;
        /// Model not found
        pub const MODEL_NOT_FOUND: u16 = 1;
        /// Shape mismatch
        pub const SHAPE_MISMATCH: u16 = 2;
        /// Gate blocked
        pub const GATE_BLOCKED: u16 = 3;
        /// Internal error
        pub const INTERNAL_ERROR: u16 = 0xFFFF;
    }
}

/// Request frame for wire protocol
#[repr(C, packed)]
#[derive(Debug, Clone, Copy)]
pub struct RequestFrame {
    /// Magic number (MAGIC)
    pub magic: u32,
    /// Protocol version
    pub protocol: u16,
    /// Sequence length
    pub seq_len: u16,
    /// Model dimension
    pub d_model: u16,
    /// Vocabulary size
    pub vocab: u16,
    /// Model ID (lower 32 bits)
    pub model_id_low: u32,
    /// Model ID (upper 32 bits)
    pub model_id_high: u32,
    /// Request flags
    pub flags: u16,
    /// Top-K count (if TOPK_ONLY flag set)
    pub topk: u16,
}

impl RequestFrame {
    /// Create a new request frame
    pub fn new(
        seq_len: u16,
        d_model: u16,
        vocab: u32,
        model_id: &ModelId,
        flags: u16,
        topk: u16,
    ) -> Self {
        let id_bytes = model_id.as_bytes();
        let model_id_low = u32::from_le_bytes([id_bytes[0], id_bytes[1], id_bytes[2], id_bytes[3]]);
        let model_id_high =
            u32::from_le_bytes([id_bytes[4], id_bytes[5], id_bytes[6], id_bytes[7]]);

        Self {
            magic: protocol::MAGIC,
            protocol: protocol::VERSION,
            seq_len,
            d_model,
            vocab: (vocab & 0xFFFF) as u16,
            model_id_low,
            model_id_high,
            flags,
            topk,
        }
    }

    /// Serialize to bytes
    pub fn to_bytes(&self) -> [u8; protocol::HEADER_SIZE] {
        let mut bytes = [0u8; protocol::HEADER_SIZE];
        bytes[0..4].copy_from_slice(&self.magic.to_le_bytes());
        bytes[4..6].copy_from_slice(&self.protocol.to_le_bytes());
        bytes[6..8].copy_from_slice(&self.seq_len.to_le_bytes());
        bytes[8..10].copy_from_slice(&self.d_model.to_le_bytes());
        bytes[10..12].copy_from_slice(&self.vocab.to_le_bytes());
        bytes[12..16].copy_from_slice(&self.model_id_low.to_le_bytes());
        bytes[16..20].copy_from_slice(&self.model_id_high.to_le_bytes());
        bytes[20..22].copy_from_slice(&self.flags.to_le_bytes());
        bytes[22..24].copy_from_slice(&self.topk.to_le_bytes());
        bytes
    }
}

/// Response frame from wire protocol
#[repr(C, packed)]
#[derive(Debug, Clone, Copy)]
pub struct ResponseFrame {
    /// Status code
    pub status: u16,
    /// Latency in nanoseconds
    pub latency_ns: u32,
    /// Compute cycles
    pub cycles: u32,
    /// Gate decision (packed)
    pub gate_decision: u8,
    /// Exit layer (if early exit)
    pub exit_layer: u8,
    /// Skip reason (if skipped)
    pub skip_reason: u8,
    /// Reserved
    pub reserved: u8,
}

impl ResponseFrame {
    /// Parse from bytes
    pub fn from_bytes(bytes: &[u8; 14]) -> Self {
        Self {
            status: u16::from_le_bytes([bytes[0], bytes[1]]),
            latency_ns: u32::from_le_bytes([bytes[2], bytes[3], bytes[4], bytes[5]]),
            cycles: u32::from_le_bytes([bytes[6], bytes[7], bytes[8], bytes[9]]),
            gate_decision: bytes[10],
            exit_layer: bytes[11],
            skip_reason: bytes[12],
            reserved: bytes[13],
        }
    }

    /// Convert gate decision to enum
    pub fn to_gate_decision(&self) -> crate::types::GateDecision {
        match self.gate_decision {
            0 => crate::types::GateDecision::RanFull,
            1 => crate::types::GateDecision::EarlyExit {
                layer: self.exit_layer,
            },
            2 => crate::types::GateDecision::Skipped {
                reason: match self.skip_reason {
                    0 => crate::types::SkipReason::LowCoherence,
                    1 => crate::types::SkipReason::PolicyDenied,
                    _ => crate::types::SkipReason::BudgetExceeded,
                },
            },
            _ => crate::types::GateDecision::RanFull,
        }
    }
}

/// Calculate CRC32 checksum for frame validation
pub fn crc32(data: &[u8]) -> u32 {
    // Simple CRC32 implementation (could use crc32fast crate in production)
    let mut crc: u32 = 0xFFFFFFFF;
    for &byte in data {
        crc ^= byte as u32;
        for _ in 0..8 {
            crc = if crc & 1 != 0 {
                (crc >> 1) ^ 0xEDB88320
            } else {
                crc >> 1
            };
        }
    }
    !crc
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_request_frame_roundtrip() {
        let model_id = ModelId::new([0x42u8; 32]);
        let frame = RequestFrame::new(64, 256, 32000, &model_id, 0, 16);
        let bytes = frame.to_bytes();

        assert_eq!(bytes.len(), protocol::HEADER_SIZE);
        assert_eq!(&bytes[0..4], &protocol::MAGIC.to_le_bytes());
    }

    #[test]
    fn test_crc32() {
        let data = b"test data";
        let crc = crc32(data);
        // CRC should be consistent
        assert_eq!(crc, crc32(data));
    }
}

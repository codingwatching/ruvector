//! Witness frame generation for audit trails.
//!
//! Each `tick` produces a witness frame containing:
//! - Hash of inputs
//! - Hash of state delta
//! - Hash of outputs
//!
//! Frames chain via prev_hash to form a tamper-evident log.
//! Replaying the same input sequence on any host must produce
//! an identical witness chain root.

use alloc::vec::Vec;
use sha2::{Sha256, Digest};

/// A single witness frame in the audit chain.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WitnessFrame {
    /// Hash of the previous frame (all zeros for genesis).
    pub prev_hash: [u8; 32],
    /// SHA-256 hash of tick inputs.
    pub input_hash: [u8; 32],
    /// SHA-256 hash of state delta (put operations in this tick).
    pub state_delta_hash: [u8; 32],
    /// SHA-256 hash of tick outputs.
    pub output_hash: [u8; 32],
    /// Epoch number for this frame.
    pub epoch: u32,
    /// Sequence number within the epoch.
    pub sequence: u32,
}

impl WitnessFrame {
    /// Compute the hash of this frame for chaining.
    pub fn hash(&self) -> [u8; 32] {
        let mut hasher = Sha256::new();
        hasher.update(&self.prev_hash);
        hasher.update(&self.input_hash);
        hasher.update(&self.state_delta_hash);
        hasher.update(&self.output_hash);
        hasher.update(&self.epoch.to_le_bytes());
        hasher.update(&self.sequence.to_le_bytes());
        let result = hasher.finalize();
        let mut out = [0u8; 32];
        out.copy_from_slice(&result);
        out
    }

    /// Serialized size of a witness frame.
    pub const SERIALIZED_SIZE: usize = 32 + 32 + 32 + 32 + 4 + 4; // 136 bytes

    /// Serialize the frame to bytes (little-endian).
    pub fn to_bytes(&self) -> [u8; Self::SERIALIZED_SIZE] {
        let mut buf = [0u8; Self::SERIALIZED_SIZE];
        buf[0..32].copy_from_slice(&self.prev_hash);
        buf[32..64].copy_from_slice(&self.input_hash);
        buf[64..96].copy_from_slice(&self.state_delta_hash);
        buf[96..128].copy_from_slice(&self.output_hash);
        buf[128..132].copy_from_slice(&self.epoch.to_le_bytes());
        buf[132..136].copy_from_slice(&self.sequence.to_le_bytes());
        buf
    }

    /// Deserialize a frame from bytes.
    pub fn from_bytes(data: &[u8]) -> Option<Self> {
        if data.len() < Self::SERIALIZED_SIZE {
            return None;
        }
        let mut prev_hash = [0u8; 32];
        prev_hash.copy_from_slice(&data[0..32]);
        let mut input_hash = [0u8; 32];
        input_hash.copy_from_slice(&data[32..64]);
        let mut state_delta_hash = [0u8; 32];
        state_delta_hash.copy_from_slice(&data[64..96]);
        let mut output_hash = [0u8; 32];
        output_hash.copy_from_slice(&data[96..128]);
        let epoch = u32::from_le_bytes(data[128..132].try_into().ok()?);
        let sequence = u32::from_le_bytes(data[132..136].try_into().ok()?);
        Some(Self {
            prev_hash,
            input_hash,
            state_delta_hash,
            output_hash,
            epoch,
            sequence,
        })
    }
}

/// Witness chain accumulator.
pub struct WitnessChain {
    frames: Vec<WitnessFrame>,
    last_hash: [u8; 32],
    current_epoch: u32,
    current_sequence: u32,
}

impl WitnessChain {
    /// Create a new empty witness chain.
    pub fn new() -> Self {
        Self {
            frames: Vec::new(),
            last_hash: [0u8; 32],
            current_epoch: 0,
            current_sequence: 0,
        }
    }

    /// Append a frame from tick data.
    pub fn append_tick(
        &mut self,
        input: &[u8],
        state_delta: &[u8],
        output: &[u8],
    ) -> WitnessFrame {
        let frame = WitnessFrame {
            prev_hash: self.last_hash,
            input_hash: sha256(input),
            state_delta_hash: sha256(state_delta),
            output_hash: sha256(output),
            epoch: self.current_epoch,
            sequence: self.current_sequence,
        };
        self.last_hash = frame.hash();
        self.current_sequence += 1;
        self.frames.push(frame.clone());
        frame
    }

    /// Seal the current epoch, advancing to the next.
    pub fn seal_epoch(&mut self) {
        self.current_epoch += 1;
        self.current_sequence = 0;
    }

    /// Get the current witness chain root hash.
    pub fn root(&self) -> [u8; 32] {
        self.last_hash
    }

    /// Get the current epoch number.
    pub fn epoch(&self) -> u32 {
        self.current_epoch
    }

    /// Get all frames in the chain.
    pub fn frames(&self) -> &[WitnessFrame] {
        &self.frames
    }

    /// Verify the chain integrity.
    pub fn verify(&self) -> bool {
        let mut expected_prev = [0u8; 32];
        for frame in &self.frames {
            if frame.prev_hash != expected_prev {
                return false;
            }
            expected_prev = frame.hash();
        }
        true
    }

    /// Serialize the entire chain to bytes.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(self.frames.len() * WitnessFrame::SERIALIZED_SIZE);
        for frame in &self.frames {
            out.extend_from_slice(&frame.to_bytes());
        }
        out
    }
}

impl Default for WitnessChain {
    fn default() -> Self {
        Self::new()
    }
}

/// Compute SHA-256 hash of data.
pub(crate) fn sha256(data: &[u8]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(data);
    let result = hasher.finalize();
    let mut out = [0u8; 32];
    out.copy_from_slice(&result);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn witness_chain_deterministic() {
        let mut c1 = WitnessChain::new();
        let mut c2 = WitnessChain::new();

        c1.append_tick(b"input1", b"delta1", b"output1");
        c2.append_tick(b"input1", b"delta1", b"output1");
        assert_eq!(c1.root(), c2.root());

        c1.append_tick(b"input2", b"delta2", b"output2");
        c2.append_tick(b"input2", b"delta2", b"output2");
        assert_eq!(c1.root(), c2.root());
    }

    #[test]
    fn witness_chain_verification() {
        let mut chain = WitnessChain::new();
        chain.append_tick(b"a", b"b", b"c");
        chain.append_tick(b"d", b"e", b"f");
        chain.seal_epoch();
        chain.append_tick(b"g", b"h", b"i");
        assert!(chain.verify());
    }

    #[test]
    fn frame_serialization_roundtrip() {
        let frame = WitnessFrame {
            prev_hash: [1u8; 32],
            input_hash: [2u8; 32],
            state_delta_hash: [3u8; 32],
            output_hash: [4u8; 32],
            epoch: 42,
            sequence: 7,
        };
        let bytes = frame.to_bytes();
        let decoded = WitnessFrame::from_bytes(&bytes).unwrap();
        assert_eq!(frame, decoded);
    }
}

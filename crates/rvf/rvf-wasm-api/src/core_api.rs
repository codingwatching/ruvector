//! RVF v1 Core API: init, tick, seal, version.
//!
//! `rvf.core.init(manifest) -> Result<()>`
//! `rvf.core.tick(input, out) -> Result<u32>`
//! `rvf.core.seal() -> Result<()>`
//! `rvf.core.version() -> u32`
//!
//! The container is the central runtime. It owns state, witness chain,
//! DRBG, and crypto keys. It delegates I/O and telemetry to host-provided
//! backends via capability injection.

use alloc::vec::Vec;

use crate::drbg::Drbg;
use crate::error::{ApiError, ApiResult};
use crate::host::HostCapabilities;
use crate::state::StateStore;
use crate::witness::WitnessChain;
use crate::ABI_VERSION;

/// Manifest header parsed from init data.
#[derive(Debug, Clone)]
pub struct Manifest {
    /// ABI version expected by the manifest.
    pub abi_version: u32,
    /// Number of RVF segments in the manifest.
    pub segment_count: u32,
    /// Raw segment data.
    pub segments: Vec<u8>,
}

impl Manifest {
    /// Parse manifest from raw bytes.
    ///
    /// Format: [abi_version: u32 LE][segment_count: u32 LE][segments...]
    pub fn parse(data: &[u8]) -> ApiResult<Self> {
        if data.len() < 8 {
            return Err(ApiError::InvalidManifest);
        }
        let abi_version = u32::from_le_bytes(data[0..4].try_into().unwrap());
        let segment_count = u32::from_le_bytes(data[4..8].try_into().unwrap());
        let segments = data[8..].to_vec();
        Ok(Self {
            abi_version,
            segment_count,
            segments,
        })
    }
}

/// The main cognitive container runtime.
///
/// Owns state, witness chain, DRBG, and processes ticks.
pub struct RvfContainer {
    /// Parsed manifest.
    manifest: Option<Manifest>,
    /// Key-value state store with Merkle commitment.
    state: StateStore,
    /// Witness chain for audit trail.
    witness: WitnessChain,
    /// Deterministic RNG.
    drbg: Option<Drbg>,
    /// Ed25519 signing key bytes (if provided by host).
    #[cfg(feature = "ed25519")]
    crypto_keys: Option<crate::crypto::ed25519::Ed25519Keys>,
    /// Whether the current epoch is sealed.
    sealed: bool,
    /// Total ticks processed.
    tick_count: u64,
}

impl RvfContainer {
    /// Create a new uninitialized container.
    pub fn new() -> Self {
        Self {
            manifest: None,
            state: StateStore::new(),
            witness: WitnessChain::new(),
            drbg: None,
            #[cfg(feature = "ed25519")]
            crypto_keys: None,
            sealed: false,
            tick_count: 0,
        }
    }

    /// Initialize the container with manifest data and host capabilities.
    ///
    /// This is `rvf.core.init`. Must be called before any `tick`.
    pub fn init(&mut self, manifest_data: &[u8], caps: &HostCapabilities) -> ApiResult<()> {
        let manifest = Manifest::parse(manifest_data)?;

        // Validate ABI compatibility
        if manifest.abi_version > ABI_VERSION {
            return Err(ApiError::InvalidManifest);
        }

        self.manifest = Some(manifest);
        self.drbg = Some(Drbg::new(&caps.drbg_seed));

        #[cfg(feature = "ed25519")]
        if let Some(ref sk) = caps.signing_key {
            self.crypto_keys = Some(crate::crypto::ed25519::Ed25519Keys::from_bytes(sk));
        }

        self.sealed = false;
        self.tick_count = 0;

        Ok(())
    }

    /// Process a single tick.
    ///
    /// This is `rvf.core.tick`. Takes input bytes, produces output bytes.
    /// Automatically appends a witness frame.
    /// Returns the number of output bytes written.
    pub fn tick(&mut self, input: &[u8], output: &mut [u8]) -> ApiResult<u32> {
        if self.manifest.is_none() {
            return Err(ApiError::NotInitialized);
        }
        if self.sealed {
            return Err(ApiError::AlreadySealed);
        }

        // Process input: for now, echo-hash as output
        // Real implementations override this with segment-specific logic
        let hash = crate::crypto::sha256(input);
        let write_len = core::cmp::min(hash.len(), output.len());
        output[..write_len].copy_from_slice(&hash[..write_len]);

        // Record witness frame
        let state_delta = self.state.delta_bytes();
        self.witness
            .append_tick(input, &state_delta, &output[..write_len]);

        self.tick_count += 1;

        Ok(write_len as u32)
    }

    /// Seal the current epoch.
    ///
    /// This is `rvf.core.seal`. Commits state and closes the witness frame.
    pub fn seal(&mut self) -> ApiResult<()> {
        if self.manifest.is_none() {
            return Err(ApiError::NotInitialized);
        }
        if self.sealed {
            return Err(ApiError::AlreadySealed);
        }

        self.state.commit();
        self.witness.seal_epoch();
        self.sealed = true;

        Ok(())
    }

    /// Unseal to begin a new epoch (internal, called after seal).
    pub fn unseal(&mut self) {
        self.sealed = false;
    }

    /// Get the ABI version.
    ///
    /// This is `rvf.core.version`.
    pub fn version(&self) -> u32 {
        ABI_VERSION
    }

    /// Access the state store.
    pub fn state(&self) -> &StateStore {
        &self.state
    }

    /// Access the state store mutably.
    pub fn state_mut(&mut self) -> &mut StateStore {
        &mut self.state
    }

    /// Access the witness chain.
    pub fn witness(&self) -> &WitnessChain {
        &self.witness
    }

    /// Access the DRBG (for deterministic randomness).
    pub fn drbg(&mut self) -> ApiResult<&mut Drbg> {
        self.drbg.as_mut().ok_or(ApiError::NoSeed)
    }

    /// Get the Ed25519 public key bytes.
    #[cfg(feature = "ed25519")]
    pub fn ed25519_pubkey(&self) -> ApiResult<[u8; 32]> {
        self.crypto_keys
            .as_ref()
            .map(|k| k.pubkey())
            .ok_or(ApiError::CryptoError)
    }

    /// Sign a message with Ed25519.
    #[cfg(feature = "ed25519")]
    pub fn ed25519_sign(&self, message: &[u8]) -> ApiResult<[u8; 64]> {
        self.crypto_keys
            .as_ref()
            .ok_or(ApiError::CryptoError)?
            .sign(message)
    }

    /// Get the total number of ticks processed.
    pub fn tick_count(&self) -> u64 {
        self.tick_count
    }

    /// Check if the container is initialized.
    pub fn is_initialized(&self) -> bool {
        self.manifest.is_some()
    }

    /// Check if the current epoch is sealed.
    pub fn is_sealed(&self) -> bool {
        self.sealed
    }
}

impl Default for RvfContainer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::host::HostCapabilities;

    fn make_manifest() -> Vec<u8> {
        let mut m = Vec::new();
        m.extend_from_slice(&ABI_VERSION.to_le_bytes());
        m.extend_from_slice(&0u32.to_le_bytes()); // 0 segments
        m
    }

    #[test]
    fn init_and_version() {
        let mut c = RvfContainer::new();
        let caps = HostCapabilities::minimal(b"seed");
        c.init(&make_manifest(), &caps).unwrap();
        assert_eq!(c.version(), ABI_VERSION);
        assert!(c.is_initialized());
    }

    #[test]
    fn tick_requires_init() {
        let mut c = RvfContainer::new();
        let mut out = [0u8; 64];
        assert!(c.tick(b"hello", &mut out).is_err());
    }

    #[test]
    fn tick_produces_output() {
        let mut c = RvfContainer::new();
        let caps = HostCapabilities::minimal(b"seed");
        c.init(&make_manifest(), &caps).unwrap();
        let mut out = [0u8; 64];
        let n = c.tick(b"hello", &mut out).unwrap();
        assert_eq!(n, 32); // SHA-256 output
    }

    #[test]
    fn seal_prevents_tick() {
        let mut c = RvfContainer::new();
        let caps = HostCapabilities::minimal(b"seed");
        c.init(&make_manifest(), &caps).unwrap();
        c.seal().unwrap();
        let mut out = [0u8; 64];
        assert!(c.tick(b"hello", &mut out).is_err());
    }

    #[test]
    fn double_seal_fails() {
        let mut c = RvfContainer::new();
        let caps = HostCapabilities::minimal(b"seed");
        c.init(&make_manifest(), &caps).unwrap();
        c.seal().unwrap();
        assert!(c.seal().is_err());
    }

    #[test]
    fn tick_witness_chain_grows() {
        let mut c = RvfContainer::new();
        let caps = HostCapabilities::minimal(b"seed");
        c.init(&make_manifest(), &caps).unwrap();
        let mut out = [0u8; 64];
        c.tick(b"a", &mut out).unwrap();
        c.tick(b"b", &mut out).unwrap();
        assert_eq!(c.witness().frames().len(), 2);
        assert!(c.witness().verify());
    }

    #[test]
    fn manifest_parse_too_short() {
        assert!(Manifest::parse(b"abc").is_err());
    }
}

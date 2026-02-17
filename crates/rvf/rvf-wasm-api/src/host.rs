//! Host capability traits for the RVF v1 API.
//!
//! The host (browser, WASI runtime, appliance) implements these traits
//! and passes them to the container at `init` time. No global mutable
//! singletons — everything is capability-passed.

use crate::error::ApiResult;

/// I/O capability descriptor provided by the host.
#[derive(Debug, Clone)]
pub struct IoCapability {
    /// Unique capability identifier.
    pub cap_id: u32,
    /// Human-readable name for logging.
    pub name: alloc::string::String,
    /// Whether reads are allowed.
    pub can_read: bool,
    /// Whether writes are allowed.
    pub can_write: bool,
    /// Maximum offset accessible (0 = unlimited).
    pub max_offset: u64,
}

/// Host-provided capabilities injected at container init.
///
/// This is the dependency-injection boundary. The container never
/// reaches outside these capabilities.
pub struct HostCapabilities {
    /// DRBG seed for deterministic randomness.
    pub drbg_seed: alloc::vec::Vec<u8>,
    /// Ed25519 signing key bytes (32 bytes), if available.
    pub signing_key: Option<[u8; 32]>,
    /// I/O capabilities granted to this container.
    pub io_caps: alloc::vec::Vec<IoCapability>,
    /// Whether telemetry emission is enabled.
    pub telemetry_enabled: bool,
}

impl HostCapabilities {
    /// Create minimal capabilities with just a DRBG seed.
    pub fn minimal(seed: &[u8]) -> Self {
        Self {
            drbg_seed: alloc::vec::Vec::from(seed),
            signing_key: None,
            io_caps: alloc::vec::Vec::new(),
            telemetry_enabled: false,
        }
    }

    /// Look up an I/O capability by ID.
    pub fn find_io_cap(&self, cap_id: u32) -> Option<&IoCapability> {
        self.io_caps.iter().find(|c| c.cap_id == cap_id)
    }
}

/// Trait for host-provided I/O backend.
///
/// Hosts implement this to provide file/network/storage access
/// behind capability gates.
pub trait IoBackend {
    /// Read bytes from the given capability.
    fn read(&self, cap_id: u32, offset: u64, buf: &mut [u8]) -> ApiResult<usize>;
    /// Write bytes to the given capability.
    fn write(&self, cap_id: u32, offset: u64, data: &[u8]) -> ApiResult<()>;
}

/// Trait for host-provided telemetry sink.
pub trait TelemetrySink {
    /// Emit a named metric with an f64 value.
    fn emit(&self, metric_name: &str, value: f64) -> ApiResult<()>;
}

/// No-op I/O backend for containers without I/O capabilities.
pub struct NoIo;

impl IoBackend for NoIo {
    fn read(&self, _cap_id: u32, _offset: u64, _buf: &mut [u8]) -> ApiResult<usize> {
        Err(crate::error::ApiError::CapabilityDenied)
    }

    fn write(&self, _cap_id: u32, _offset: u64, _data: &[u8]) -> ApiResult<()> {
        Err(crate::error::ApiError::CapabilityDenied)
    }
}

/// No-op telemetry sink that silently drops metrics.
pub struct NoTelemetry;

impl TelemetrySink for NoTelemetry {
    fn emit(&self, _metric_name: &str, _value: f64) -> ApiResult<()> {
        Ok(())
    }
}

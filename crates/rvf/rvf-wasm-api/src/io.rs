//! RVF v1 I/O API: capability-gated read/write.
//!
//! `rvf.io.read_cap(cap_id, offset, len, out) -> u32`
//! `rvf.io.write_cap(cap_id, offset, data) -> Result<()>`
//!
//! All I/O is mediated through host-provided capabilities.
//! Without the `io` feature or without granted capabilities,
//! all I/O calls fail deterministically with CapabilityDenied.

use crate::error::{ApiError, ApiResult};
use crate::host::{HostCapabilities, IoBackend};

/// Capability-gated I/O operations.
pub struct IoApi<'a, B: IoBackend> {
    caps: &'a HostCapabilities,
    backend: &'a B,
}

impl<'a, B: IoBackend> IoApi<'a, B> {
    /// Create a new I/O API with the given capabilities and backend.
    pub fn new(caps: &'a HostCapabilities, backend: &'a B) -> Self {
        Self { caps, backend }
    }

    /// Read from a capability.
    pub fn read_cap(&self, cap_id: u32, offset: u64, buf: &mut [u8]) -> ApiResult<usize> {
        let cap = self
            .caps
            .find_io_cap(cap_id)
            .ok_or(ApiError::InvalidCapability)?;
        if !cap.can_read {
            return Err(ApiError::CapabilityDenied);
        }
        if cap.max_offset > 0 && offset + buf.len() as u64 > cap.max_offset {
            return Err(ApiError::BufferTooSmall);
        }
        self.backend.read(cap_id, offset, buf)
    }

    /// Write to a capability.
    pub fn write_cap(&self, cap_id: u32, offset: u64, data: &[u8]) -> ApiResult<()> {
        let cap = self
            .caps
            .find_io_cap(cap_id)
            .ok_or(ApiError::InvalidCapability)?;
        if !cap.can_write {
            return Err(ApiError::CapabilityDenied);
        }
        if cap.max_offset > 0 && offset + data.len() as u64 > cap.max_offset {
            return Err(ApiError::BufferTooSmall);
        }
        self.backend.write(cap_id, offset, data)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::host::{IoCapability, NoIo};
    use alloc::string::String;

    #[test]
    fn no_io_denies_all() {
        let caps = HostCapabilities::minimal(b"seed");
        let backend = NoIo;
        let api = IoApi::new(&caps, &backend);
        let mut buf = [0u8; 32];
        assert!(matches!(
            api.read_cap(0, 0, &mut buf),
            Err(ApiError::InvalidCapability)
        ));
        assert!(matches!(
            api.write_cap(0, 0, &buf),
            Err(ApiError::InvalidCapability)
        ));
    }

    #[test]
    fn read_only_cap_denies_write() {
        let mut caps = HostCapabilities::minimal(b"seed");
        caps.io_caps.push(IoCapability {
            cap_id: 1,
            name: String::from("test"),
            can_read: true,
            can_write: false,
            max_offset: 0,
        });
        let backend = NoIo;
        let api = IoApi::new(&caps, &backend);
        assert!(matches!(
            api.write_cap(1, 0, b"data"),
            Err(ApiError::CapabilityDenied)
        ));
    }
}

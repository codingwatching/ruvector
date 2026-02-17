//! Migration shim: forwards legacy RVF API calls to the v1 API.
//!
//! This module provides backward-compatible wrappers that map
//! the old rvf-wasm tile API (rvf_init, rvf_load_query, etc.)
//! to the new v1 cognitive container API.
//!
//! Usage: Enable the `migration` module and call the shim functions
//! instead of the legacy exports. Mark old APIs as deprecated.

use alloc::vec::Vec;
use crate::core_api::RvfContainer;
use crate::error::{ApiError, ApiResult};
use crate::host::HostCapabilities;

/// Legacy API adapter that wraps an RvfContainer.
///
/// Maps old-style pointer-based tile operations to the
/// v1 API's init/tick/seal model.
pub struct LegacyAdapter {
    container: RvfContainer,
    initialized: bool,
}

impl LegacyAdapter {
    /// Create a new legacy adapter.
    pub fn new() -> Self {
        Self {
            container: RvfContainer::new(),
            initialized: false,
        }
    }

    /// Legacy init: wraps to v1 init with a default manifest.
    pub fn legacy_init(&mut self, config: &[u8], seed: &[u8]) -> ApiResult<()> {
        // Build a v1-compatible manifest from legacy config
        let mut manifest = Vec::with_capacity(8 + config.len());
        manifest.extend_from_slice(&crate::ABI_VERSION.to_le_bytes());
        manifest.extend_from_slice(&0u32.to_le_bytes()); // 0 segments for legacy
        manifest.extend_from_slice(config);

        let caps = HostCapabilities::minimal(seed);
        self.container.init(&manifest, &caps)?;
        self.initialized = true;
        Ok(())
    }

    /// Legacy query: wraps to v1 tick with query data as input.
    pub fn legacy_query(&mut self, query_data: &[u8], output: &mut [u8]) -> ApiResult<u32> {
        if !self.initialized {
            return Err(ApiError::NotInitialized);
        }
        self.container.tick(query_data, output)
    }

    /// Legacy seal: wraps to v1 seal.
    pub fn legacy_seal(&mut self) -> ApiResult<()> {
        self.container.seal()
    }

    /// Access the underlying v1 container.
    pub fn container(&self) -> &RvfContainer {
        &self.container
    }

    /// Access the underlying v1 container mutably.
    pub fn container_mut(&mut self) -> &mut RvfContainer {
        &mut self.container
    }
}

impl Default for LegacyAdapter {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn legacy_init_query_seal() {
        let mut adapter = LegacyAdapter::new();
        adapter.legacy_init(b"config-data", b"seed").unwrap();
        let mut out = [0u8; 64];
        let n = adapter.legacy_query(b"query", &mut out).unwrap();
        assert!(n > 0);
        adapter.legacy_seal().unwrap();
    }

    #[test]
    fn legacy_query_before_init_fails() {
        let mut adapter = LegacyAdapter::new();
        let mut out = [0u8; 64];
        assert!(adapter.legacy_query(b"query", &mut out).is_err());
    }
}

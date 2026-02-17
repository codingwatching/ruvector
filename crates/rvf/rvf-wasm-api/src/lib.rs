//! RVF v1 Minimal WASM API for Cognitive Containers.
//!
//! Implements the stable v1 API surface defined in ADR-040:
//! - `rvf.core`: init, tick, seal, version
//! - `rvf.state`: get, put, commit (Merkle-based)
//! - `rvf.crypto`: ed25519_pubkey, ed25519_sign, sha256
//! - `rvf.io`: capability-gated read/write (optional)
//! - `rvf.telemetry`: metric emission (optional)
//!
//! Constraints:
//! - No global mutable singletons
//! - Deterministic: no wall-clock, no RNG without host-provided seed
//! - Fixed-width types; little-endian
//! - Streaming via `tick` as only progress primitive

#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

pub mod core_api;
pub mod state;
pub mod crypto;
pub mod error;
pub mod host;
pub mod witness;
pub mod drbg;

#[cfg(feature = "io")]
pub mod io;

#[cfg(feature = "telemetry")]
pub mod telemetry;

pub mod migration;

/// ABI version constant: v1.0.0 encoded as 0x00010000
pub const ABI_VERSION: u32 = 0x0001_0000;

pub use core_api::RvfContainer;
pub use error::{ApiError, ApiResult};
pub use host::{HostCapabilities, IoCapability};
pub use state::StateStore;
pub use witness::WitnessFrame;

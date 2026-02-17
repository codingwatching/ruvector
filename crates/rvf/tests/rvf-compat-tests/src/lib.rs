//! RVF v1 Compatibility Test Suite (ADR-040).
//!
//! These tests guarantee identity, auditability, and reproducibility
//! across all hosts (browser, WASI, appliance). Every test must pass
//! on all target triples before release.
//!
//! Test categories:
//! 1. Ed25519 identity stability
//! 2. Witness log determinism
//! 3. Deterministic boot
//! 4. State model round-trips
//! 5. Stress & streaming
//! 6. Capability fencing
//! 7. Cross-target matrix (same binary, same roots)

pub mod report;

#[cfg(test)]
mod tests;

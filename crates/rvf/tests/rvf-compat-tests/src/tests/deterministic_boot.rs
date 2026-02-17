//! Test 3: Deterministic boot.
//!
//! Cold boot init with fixed manifest & DRBG seed -> identical
//! commit() root prior to any tick.

use rvf_wasm_api::core_api::RvfContainer;
use rvf_wasm_api::host::HostCapabilities;
use rvf_wasm_api::ABI_VERSION;

fn make_manifest() -> Vec<u8> {
    let mut m = Vec::new();
    m.extend_from_slice(&ABI_VERSION.to_le_bytes());
    m.extend_from_slice(&0u32.to_le_bytes());
    m
}

#[test]
fn cold_boot_identical_state_root() {
    let manifest = make_manifest();
    let caps = HostCapabilities::minimal(b"deterministic-boot-seed");

    let mut c1 = RvfContainer::new();
    c1.init(&manifest, &caps).unwrap();
    let root1 = c1.state_mut().commit();

    let mut c2 = RvfContainer::new();
    c2.init(&manifest, &caps).unwrap();
    let root2 = c2.state_mut().commit();

    assert_eq!(
        root1, root2,
        "cold boot with same manifest + seed must yield identical state root"
    );
}

#[test]
fn cold_boot_state_root_is_zero_when_empty() {
    let manifest = make_manifest();
    let caps = HostCapabilities::minimal(b"empty-boot");

    let mut container = RvfContainer::new();
    container.init(&manifest, &caps).unwrap();
    let root = container.state_mut().commit();

    assert_eq!(root, [0u8; 32], "empty state commit must be all zeros");
}

#[test]
fn drbg_deterministic_across_boots() {
    let manifest = make_manifest();
    let caps = HostCapabilities::minimal(b"drbg-test-seed");

    let mut c1 = RvfContainer::new();
    c1.init(&manifest, &caps).unwrap();
    let r1 = c1.drbg().unwrap().next_32();

    let mut c2 = RvfContainer::new();
    c2.init(&manifest, &caps).unwrap();
    let r2 = c2.drbg().unwrap().next_32();

    assert_eq!(r1, r2, "DRBG output must be identical for same seed");
}

#[test]
fn different_seeds_different_drbg() {
    let manifest = make_manifest();

    let mut c1 = RvfContainer::new();
    c1.init(&manifest, &HostCapabilities::minimal(b"seed-a")).unwrap();
    let r1 = c1.drbg().unwrap().next_32();

    let mut c2 = RvfContainer::new();
    c2.init(&manifest, &HostCapabilities::minimal(b"seed-b")).unwrap();
    let r2 = c2.drbg().unwrap().next_32();

    assert_ne!(r1, r2, "different seeds must produce different DRBG output");
}

#[test]
fn state_after_puts_is_deterministic() {
    let manifest = make_manifest();
    let caps = HostCapabilities::minimal(b"put-seed");

    let mut c1 = RvfContainer::new();
    c1.init(&manifest, &caps).unwrap();
    c1.state_mut().put(b"key-a", b"val-1");
    c1.state_mut().put(b"key-b", b"val-2");
    let root1 = c1.state_mut().commit();

    let mut c2 = RvfContainer::new();
    c2.init(&manifest, &caps).unwrap();
    c2.state_mut().put(b"key-a", b"val-1");
    c2.state_mut().put(b"key-b", b"val-2");
    let root2 = c2.state_mut().commit();

    assert_eq!(root1, root2, "identical puts must produce identical Merkle root");
}

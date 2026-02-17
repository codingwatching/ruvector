//! Test 2: Witness log determinism.
//!
//! Each tick appends a frame: inputs hash, state delta hash, output hash.
//! Replay the same input sequence across hosts => identical witness chain root.

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
fn identical_inputs_identical_witness_root() {
    let caps = HostCapabilities::minimal(b"witness-seed");
    let manifest = make_manifest();

    let mut c1 = RvfContainer::new();
    c1.init(&manifest, &caps).unwrap();

    let mut c2 = RvfContainer::new();
    c2.init(&manifest, &caps).unwrap();

    let inputs = [b"input-a".as_slice(), b"input-b", b"input-c"];
    let mut out1 = [0u8; 64];
    let mut out2 = [0u8; 64];

    for input in &inputs {
        c1.tick(input, &mut out1).unwrap();
        c2.tick(input, &mut out2).unwrap();
    }

    assert_eq!(
        c1.witness().root(),
        c2.witness().root(),
        "same input sequence must produce identical witness chain root"
    );
}

#[test]
fn different_inputs_different_witness_root() {
    let caps = HostCapabilities::minimal(b"seed");
    let manifest = make_manifest();

    let mut c1 = RvfContainer::new();
    c1.init(&manifest, &caps).unwrap();
    let mut c2 = RvfContainer::new();
    c2.init(&manifest, &caps).unwrap();

    let mut out = [0u8; 64];
    c1.tick(b"alpha", &mut out).unwrap();
    c2.tick(b"beta", &mut out).unwrap();

    assert_ne!(c1.witness().root(), c2.witness().root());
}

#[test]
fn witness_chain_is_verifiable() {
    let caps = HostCapabilities::minimal(b"verify-seed");
    let manifest = make_manifest();

    let mut container = RvfContainer::new();
    container.init(&manifest, &caps).unwrap();

    let mut out = [0u8; 64];
    for i in 0..10 {
        let input = format!("tick-{}", i);
        container.tick(input.as_bytes(), &mut out).unwrap();
    }

    assert!(
        container.witness().verify(),
        "witness chain must pass integrity verification"
    );
}

#[test]
fn seal_advances_epoch_in_witness() {
    let caps = HostCapabilities::minimal(b"epoch-seed");
    let manifest = make_manifest();

    let mut container = RvfContainer::new();
    container.init(&manifest, &caps).unwrap();

    let mut out = [0u8; 64];
    container.tick(b"e0-tick", &mut out).unwrap();
    assert_eq!(container.witness().epoch(), 0);

    container.seal().unwrap();
    assert_eq!(container.witness().epoch(), 1);
}

#[test]
fn witness_frame_serialization_roundtrip() {
    let caps = HostCapabilities::minimal(b"serial-seed");
    let manifest = make_manifest();

    let mut container = RvfContainer::new();
    container.init(&manifest, &caps).unwrap();

    let mut out = [0u8; 64];
    container.tick(b"serialize-test", &mut out).unwrap();

    let chain_bytes = container.witness().to_bytes();
    assert!(!chain_bytes.is_empty());

    // Verify frame can be deserialized
    let frame = rvf_wasm_api::witness::WitnessFrame::from_bytes(&chain_bytes).unwrap();
    assert_eq!(frame.epoch, 0);
    assert_eq!(frame.sequence, 0);
}

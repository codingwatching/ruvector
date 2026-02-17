//! Test 5: Stress & streaming.
//!
//! Variable-size tick inputs, back-pressure on out buffer.
//! Must not allocate unbounded memory or diverge.

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
fn small_input_tick() {
    let caps = HostCapabilities::minimal(b"stress-small");
    let manifest = make_manifest();

    let mut container = RvfContainer::new();
    container.init(&manifest, &caps).unwrap();

    let input = vec![0xFFu8; 1]; // 1 byte
    let mut out = [0u8; 64];
    let n = container.tick(&input, &mut out).unwrap();
    assert!(n > 0);
}

#[test]
fn medium_input_tick() {
    let caps = HostCapabilities::minimal(b"stress-medium");
    let manifest = make_manifest();

    let mut container = RvfContainer::new();
    container.init(&manifest, &caps).unwrap();

    let input = vec![0xABu8; 1024]; // 1 KB
    let mut out = [0u8; 64];
    let n = container.tick(&input, &mut out).unwrap();
    assert_eq!(n, 32); // SHA-256 output
}

#[test]
fn large_input_tick() {
    let caps = HostCapabilities::minimal(b"stress-large");
    let manifest = make_manifest();

    let mut container = RvfContainer::new();
    container.init(&manifest, &caps).unwrap();

    let input = vec![0xCDu8; 8 * 1024 * 1024]; // 8 MB
    let mut out = [0u8; 64];
    let n = container.tick(&input, &mut out).unwrap();
    assert_eq!(n, 32);
}

#[test]
fn many_ticks_no_divergence() {
    let caps = HostCapabilities::minimal(b"stress-many");
    let manifest = make_manifest();

    let mut container = RvfContainer::new();
    container.init(&manifest, &caps).unwrap();

    let mut out = [0u8; 64];
    for i in 0..1000 {
        let input = format!("tick-{}", i);
        container.tick(input.as_bytes(), &mut out).unwrap();
    }

    assert_eq!(container.tick_count(), 1000);
    assert!(container.witness().verify());
    assert_eq!(container.witness().frames().len(), 1000);
}

#[test]
fn small_output_buffer_caps_at_buffer_size() {
    let caps = HostCapabilities::minimal(b"stress-cap");
    let manifest = make_manifest();

    let mut container = RvfContainer::new();
    container.init(&manifest, &caps).unwrap();

    let mut out = [0u8; 16]; // Smaller than SHA-256 output
    let n = container.tick(b"test", &mut out).unwrap();
    assert_eq!(n, 16, "output should be capped at buffer size");
}

#[test]
fn variable_size_inputs_deterministic() {
    let caps = HostCapabilities::minimal(b"var-size");
    let manifest = make_manifest();
    let sizes = [1, 64, 256, 1024, 4096, 16384, 65536];

    let mut c1 = RvfContainer::new();
    c1.init(&manifest, &caps).unwrap();
    let mut c2 = RvfContainer::new();
    c2.init(&manifest, &caps).unwrap();

    let mut out1 = [0u8; 64];
    let mut out2 = [0u8; 64];

    for &size in &sizes {
        let input = vec![0x42u8; size];
        c1.tick(&input, &mut out1).unwrap();
        c2.tick(&input, &mut out2).unwrap();
    }

    assert_eq!(c1.witness().root(), c2.witness().root());
}

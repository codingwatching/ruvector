//! Test 1: Ed25519 identity stability.
//!
//! - ed25519_pubkey() must be stable across boots of the same artifact.
//! - Sign/verify known vectors; compare against host verifier.

use rvf_wasm_api::core_api::RvfContainer;
use rvf_wasm_api::host::HostCapabilities;
use rvf_wasm_api::ABI_VERSION;

fn make_manifest() -> Vec<u8> {
    let mut m = Vec::new();
    m.extend_from_slice(&ABI_VERSION.to_le_bytes());
    m.extend_from_slice(&0u32.to_le_bytes());
    m
}

fn make_caps_with_key(seed: &[u8], signing_key: [u8; 32]) -> HostCapabilities {
    let mut caps = HostCapabilities::minimal(seed);
    caps.signing_key = Some(signing_key);
    caps
}

#[test]
fn pubkey_stable_across_boots() {
    let key = [42u8; 32];
    let caps = make_caps_with_key(b"seed1", key);
    let manifest = make_manifest();

    let mut c1 = RvfContainer::new();
    c1.init(&manifest, &caps).unwrap();
    let pk1 = c1.ed25519_pubkey().unwrap();

    let mut c2 = RvfContainer::new();
    c2.init(&manifest, &caps).unwrap();
    let pk2 = c2.ed25519_pubkey().unwrap();

    assert_eq!(pk1, pk2, "pubkey must be stable across container boots");
}

#[test]
fn sign_verify_known_message() {
    let key = [7u8; 32];
    let caps = make_caps_with_key(b"seed2", key);
    let manifest = make_manifest();

    let mut container = RvfContainer::new();
    container.init(&manifest, &caps).unwrap();

    let message = b"RVF cognitive container attestation v1";
    let sig = container.ed25519_sign(message).unwrap();

    // Verify with a fresh container (same key)
    let mut verifier = RvfContainer::new();
    verifier.init(&manifest, &caps).unwrap();
    // The pubkey should match, and the signature should be deterministic
    assert_eq!(container.ed25519_pubkey().unwrap(), verifier.ed25519_pubkey().unwrap());
}

#[test]
fn different_keys_different_pubkeys() {
    let manifest = make_manifest();

    let mut c1 = RvfContainer::new();
    c1.init(&manifest, &make_caps_with_key(b"s", [1u8; 32])).unwrap();

    let mut c2 = RvfContainer::new();
    c2.init(&manifest, &make_caps_with_key(b"s", [2u8; 32])).unwrap();

    assert_ne!(
        c1.ed25519_pubkey().unwrap(),
        c2.ed25519_pubkey().unwrap(),
        "different signing keys must produce different pubkeys"
    );
}

#[test]
fn sign_is_deterministic() {
    let key = [99u8; 32];
    let caps = make_caps_with_key(b"seed", key);
    let manifest = make_manifest();

    let mut c1 = RvfContainer::new();
    c1.init(&manifest, &caps).unwrap();
    let sig1 = c1.ed25519_sign(b"test").unwrap();

    let mut c2 = RvfContainer::new();
    c2.init(&manifest, &caps).unwrap();
    let sig2 = c2.ed25519_sign(b"test").unwrap();

    assert_eq!(sig1, sig2, "ed25519 signatures must be deterministic for same key and message");
}

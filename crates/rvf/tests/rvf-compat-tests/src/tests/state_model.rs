//! Test 4: State model round-trips.
//!
//! Put/Get/Commit round-trips with seeded keys/values.
//! Snapshot/Restore yields the same Merkle root.

use rvf_wasm_api::state::StateStore;

#[test]
fn put_get_commit_roundtrip() {
    let mut store = StateStore::new();

    for i in 0..100u32 {
        let key = format!("key-{:04}", i);
        let val = format!("value-{:08}", i * 7);
        store.put(key.as_bytes(), val.as_bytes());
    }

    let root = store.commit();
    assert_ne!(root, [0u8; 32], "non-empty state must have non-zero root");

    // Verify all values readable
    for i in 0..100u32 {
        let key = format!("key-{:04}", i);
        let expected = format!("value-{:08}", i * 7);
        assert_eq!(store.get(key.as_bytes()).unwrap(), expected.as_bytes());
    }
}

#[test]
fn snapshot_restore_yields_same_root() {
    let mut store = StateStore::new();
    store.put(b"alpha", b"111");
    store.put(b"beta", b"222");
    store.put(b"gamma", b"333");
    let root = store.commit();
    let snapshot = store.snapshot();

    let mut restored = StateStore::new();
    restored.restore(&snapshot).unwrap();
    let restored_root = restored.commit();

    assert_eq!(root, restored_root, "snapshot/restore must preserve Merkle root");
}

#[test]
fn overwrite_changes_root() {
    let mut store = StateStore::new();
    store.put(b"k", b"v1");
    let root1 = store.commit();

    store.put(b"k", b"v2");
    let root2 = store.commit();

    assert_ne!(root1, root2, "overwriting a value must change the Merkle root");
}

#[test]
fn insertion_order_independent() {
    let mut s1 = StateStore::new();
    s1.put(b"z", b"26");
    s1.put(b"a", b"1");
    s1.put(b"m", b"13");
    let r1 = s1.commit();

    let mut s2 = StateStore::new();
    s2.put(b"a", b"1");
    s2.put(b"m", b"13");
    s2.put(b"z", b"26");
    let r2 = s2.commit();

    assert_eq!(r1, r2, "insertion order must not affect Merkle root");
}

#[test]
fn large_state_commit() {
    let mut store = StateStore::new();
    for i in 0..1000u32 {
        let key = i.to_le_bytes();
        let val = (i * 31337).to_le_bytes();
        store.put(&key, &val);
    }
    let root = store.commit();
    assert_ne!(root, [0u8; 32]);

    // Re-commit without changes should yield same root
    let root2 = store.commit();
    assert_eq!(root, root2, "commit without changes must be idempotent");
}

#[test]
fn snapshot_restore_large_state() {
    let mut store = StateStore::new();
    for i in 0..500u32 {
        store.put(&i.to_le_bytes(), &(i * 42).to_le_bytes());
    }
    let root = store.commit();
    let snap = store.snapshot();

    let mut restored = StateStore::new();
    restored.restore(&snap).unwrap();
    let restored_root = restored.commit();

    assert_eq!(root, restored_root);
    assert_eq!(restored.len(), 500);
}

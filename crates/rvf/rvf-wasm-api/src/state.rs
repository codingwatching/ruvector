//! RVF v1 State API: key-scoped state with Merkle commit.
//!
//! `rvf.state.get(key) -> value`
//! `rvf.state.put(key, value)`
//! `rvf.state.commit() -> [u8; 32]` (Merkle root)
//!
//! State is maintained as a sorted key-value map. On `commit()`,
//! a Merkle tree is computed over all entries to produce a
//! deterministic root hash.

use alloc::collections::BTreeMap;
use alloc::vec::Vec;
use sha2::{Sha256, Digest};

use crate::error::{ApiError, ApiResult};

/// Key-value state store with Merkle commitment.
pub struct StateStore {
    /// Current state entries.
    entries: BTreeMap<Vec<u8>, Vec<u8>>,
    /// Delta entries accumulated since last commit (for witness).
    delta: Vec<(Vec<u8>, Vec<u8>)>,
    /// Last committed Merkle root.
    last_root: [u8; 32],
    /// Number of commits performed.
    commit_count: u64,
}

impl StateStore {
    /// Create a new empty state store.
    pub fn new() -> Self {
        Self {
            entries: BTreeMap::new(),
            delta: Vec::new(),
            last_root: [0u8; 32],
            commit_count: 0,
        }
    }

    /// Get a value by key. Returns the number of bytes, or KeyNotFound.
    pub fn get(&self, key: &[u8]) -> ApiResult<&[u8]> {
        self.entries
            .get(key)
            .map(|v| v.as_slice())
            .ok_or(ApiError::KeyNotFound)
    }

    /// Put a key-value pair. Overwrites existing values.
    pub fn put(&mut self, key: &[u8], value: &[u8]) {
        self.delta.push((key.to_vec(), value.to_vec()));
        self.entries.insert(key.to_vec(), value.to_vec());
    }

    /// Commit state and return the Merkle root.
    ///
    /// Computes a binary Merkle tree over all sorted key-value pairs.
    /// The delta is cleared after commit.
    pub fn commit(&mut self) -> [u8; 32] {
        let root = self.compute_merkle_root();
        self.last_root = root;
        self.delta.clear();
        self.commit_count += 1;
        root
    }

    /// Get the delta since last commit (for witness frame).
    pub fn delta_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        for (k, v) in &self.delta {
            out.extend_from_slice(&(k.len() as u32).to_le_bytes());
            out.extend_from_slice(k);
            out.extend_from_slice(&(v.len() as u32).to_le_bytes());
            out.extend_from_slice(v);
        }
        out
    }

    /// Get the last committed Merkle root.
    pub fn last_root(&self) -> [u8; 32] {
        self.last_root
    }

    /// Get the number of commits performed.
    pub fn commit_count(&self) -> u64 {
        self.commit_count
    }

    /// Get the number of entries.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Check if the store is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Compute Merkle root over all sorted entries.
    fn compute_merkle_root(&self) -> [u8; 32] {
        if self.entries.is_empty() {
            return [0u8; 32];
        }

        // Hash each leaf: H(key || value)
        let mut leaves: Vec<[u8; 32]> = self
            .entries
            .iter()
            .map(|(k, v)| {
                let mut hasher = Sha256::new();
                hasher.update(&(k.len() as u32).to_le_bytes());
                hasher.update(k);
                hasher.update(&(v.len() as u32).to_le_bytes());
                hasher.update(v);
                let result = hasher.finalize();
                let mut out = [0u8; 32];
                out.copy_from_slice(&result);
                out
            })
            .collect();

        // Build Merkle tree bottom-up
        while leaves.len() > 1 {
            let mut next_level = Vec::with_capacity((leaves.len() + 1) / 2);
            let mut i = 0;
            while i < leaves.len() {
                if i + 1 < leaves.len() {
                    let mut hasher = Sha256::new();
                    hasher.update(&leaves[i]);
                    hasher.update(&leaves[i + 1]);
                    let result = hasher.finalize();
                    let mut out = [0u8; 32];
                    out.copy_from_slice(&result);
                    next_level.push(out);
                } else {
                    // Odd leaf gets promoted
                    next_level.push(leaves[i]);
                }
                i += 2;
            }
            leaves = next_level;
        }

        leaves[0]
    }

    /// Snapshot the current state into serialized bytes.
    pub fn snapshot(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(&(self.entries.len() as u32).to_le_bytes());
        for (k, v) in &self.entries {
            out.extend_from_slice(&(k.len() as u32).to_le_bytes());
            out.extend_from_slice(k);
            out.extend_from_slice(&(v.len() as u32).to_le_bytes());
            out.extend_from_slice(v);
        }
        out
    }

    /// Restore state from a snapshot.
    pub fn restore(&mut self, data: &[u8]) -> ApiResult<()> {
        if data.len() < 4 {
            return Err(ApiError::InvalidManifest);
        }
        let count = u32::from_le_bytes(data[0..4].try_into().unwrap()) as usize;
        let mut offset = 4;
        self.entries.clear();
        self.delta.clear();

        for _ in 0..count {
            if offset + 4 > data.len() {
                return Err(ApiError::InvalidManifest);
            }
            let klen = u32::from_le_bytes(data[offset..offset + 4].try_into().unwrap()) as usize;
            offset += 4;
            if offset + klen > data.len() {
                return Err(ApiError::InvalidManifest);
            }
            let key = data[offset..offset + klen].to_vec();
            offset += klen;

            if offset + 4 > data.len() {
                return Err(ApiError::InvalidManifest);
            }
            let vlen = u32::from_le_bytes(data[offset..offset + 4].try_into().unwrap()) as usize;
            offset += 4;
            if offset + vlen > data.len() {
                return Err(ApiError::InvalidManifest);
            }
            let value = data[offset..offset + vlen].to_vec();
            offset += vlen;

            self.entries.insert(key, value);
        }

        self.last_root = self.compute_merkle_root();
        Ok(())
    }
}

impl Default for StateStore {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_commit_returns_zeros() {
        let mut store = StateStore::new();
        assert_eq!(store.commit(), [0u8; 32]);
    }

    #[test]
    fn put_get_roundtrip() {
        let mut store = StateStore::new();
        store.put(b"key1", b"value1");
        assert_eq!(store.get(b"key1").unwrap(), b"value1");
    }

    #[test]
    fn missing_key_returns_error() {
        let store = StateStore::new();
        assert!(store.get(b"nonexistent").is_err());
    }

    #[test]
    fn commit_is_deterministic() {
        let mut s1 = StateStore::new();
        let mut s2 = StateStore::new();
        s1.put(b"a", b"1");
        s1.put(b"b", b"2");
        s2.put(b"a", b"1");
        s2.put(b"b", b"2");
        assert_eq!(s1.commit(), s2.commit());
    }

    #[test]
    fn order_independent_commit() {
        let mut s1 = StateStore::new();
        let mut s2 = StateStore::new();
        s1.put(b"b", b"2");
        s1.put(b"a", b"1");
        s2.put(b"a", b"1");
        s2.put(b"b", b"2");
        assert_eq!(s1.commit(), s2.commit());
    }

    #[test]
    fn snapshot_restore_preserves_root() {
        let mut store = StateStore::new();
        store.put(b"x", b"10");
        store.put(b"y", b"20");
        let root = store.commit();
        let snap = store.snapshot();

        let mut restored = StateStore::new();
        restored.restore(&snap).unwrap();
        let restored_root = restored.commit();
        assert_eq!(root, restored_root);
    }

    #[test]
    fn delta_cleared_after_commit() {
        let mut store = StateStore::new();
        store.put(b"k", b"v");
        assert!(!store.delta_bytes().is_empty());
        store.commit();
        assert!(store.delta_bytes().is_empty());
    }
}

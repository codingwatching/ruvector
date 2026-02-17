//! Deterministic Random Bit Generator (HMAC-DRBG style).
//!
//! All randomness in a cognitive container MUST come from this DRBG,
//! seeded by the host at init time via the witness seed.
//! This ensures deterministic, reproducible execution.

use sha2::{Sha256, Digest};

/// HMAC-DRBG-SHA256 for deterministic randomness.
///
/// Follows NIST SP 800-90A simplified for our use case.
/// The container never accesses system RNG directly.
pub struct Drbg {
    key: [u8; 32],
    value: [u8; 32],
    reseed_counter: u64,
}

impl Drbg {
    /// Create a new DRBG from a seed.
    pub fn new(seed: &[u8]) -> Self {
        let mut key = [0u8; 32];
        let mut value = [0x01u8; 32];

        // Initial seeding: K = HMAC(0x00..., V || 0x00 || seed)
        let k_input = Self::build_input(&value, 0x00, seed);
        key = Self::hmac_sha256(&[0u8; 32], &k_input);
        // V = HMAC(K, V)
        value = Self::hmac_sha256(&key, &value);
        // K = HMAC(K, V || 0x01 || seed)
        let k_input = Self::build_input(&value, 0x01, seed);
        key = Self::hmac_sha256(&key, &k_input);
        // V = HMAC(K, V)
        value = Self::hmac_sha256(&key, &value);

        Self {
            key,
            value,
            reseed_counter: 1,
        }
    }

    /// Generate `len` deterministic random bytes.
    pub fn generate(&mut self, output: &mut [u8]) {
        let mut offset = 0;
        while offset < output.len() {
            self.value = Self::hmac_sha256(&self.key, &self.value);
            let copy_len = core::cmp::min(32, output.len() - offset);
            output[offset..offset + copy_len].copy_from_slice(&self.value[..copy_len]);
            offset += copy_len;
        }
        // Update K and V
        let k_input = Self::build_input(&self.value, 0x00, &[]);
        self.key = Self::hmac_sha256(&self.key, &k_input);
        self.value = Self::hmac_sha256(&self.key, &self.value);
        self.reseed_counter += 1;
    }

    /// Generate a fixed 32-byte block.
    pub fn next_32(&mut self) -> [u8; 32] {
        let mut out = [0u8; 32];
        self.generate(&mut out);
        out
    }

    fn build_input(v: &[u8; 32], separator: u8, additional: &[u8]) -> alloc::vec::Vec<u8> {
        let mut buf = alloc::vec::Vec::with_capacity(32 + 1 + additional.len());
        buf.extend_from_slice(v);
        buf.push(separator);
        buf.extend_from_slice(additional);
        buf
    }

    fn hmac_sha256(key: &[u8; 32], data: &[u8]) -> [u8; 32] {
        // HMAC-SHA256: H((K ^ opad) || H((K ^ ipad) || message))
        let mut ipad = [0x36u8; 64];
        let mut opad = [0x5cu8; 64];
        for i in 0..32 {
            ipad[i] ^= key[i];
            opad[i] ^= key[i];
        }

        let mut inner = Sha256::new();
        inner.update(&ipad);
        inner.update(data);
        let inner_hash = inner.finalize();

        let mut outer = Sha256::new();
        outer.update(&opad);
        outer.update(&inner_hash);
        let result = outer.finalize();

        let mut out = [0u8; 32];
        out.copy_from_slice(&result);
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic_output() {
        let mut d1 = Drbg::new(b"test-seed-42");
        let mut d2 = Drbg::new(b"test-seed-42");
        assert_eq!(d1.next_32(), d2.next_32());
        assert_eq!(d1.next_32(), d2.next_32());
    }

    #[test]
    fn different_seeds_different_output() {
        let mut d1 = Drbg::new(b"seed-a");
        let mut d2 = Drbg::new(b"seed-b");
        assert_ne!(d1.next_32(), d2.next_32());
    }

    #[test]
    fn generates_nonzero() {
        let mut d = Drbg::new(b"nonzero-test");
        let out = d.next_32();
        assert_ne!(out, [0u8; 32]);
    }
}

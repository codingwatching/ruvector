//! RVF v1 Crypto API: ed25519 and SHA-256.
//!
//! `rvf.crypto.ed25519_pubkey(out_ptr) -> u32`
//! `rvf.crypto.ed25519_sign(msg, sig_out) -> Result<()>`
//! `rvf.crypto.sha256(data, out) -> Result<()>`
//!
//! All crypto operations are deterministic and use no system RNG.

use sha2::{Sha256, Digest};
use crate::error::{ApiError, ApiResult};

/// Compute SHA-256 hash of data.
pub fn sha256(data: &[u8]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(data);
    let result = hasher.finalize();
    let mut out = [0u8; 32];
    out.copy_from_slice(&result);
    out
}

/// Compute SHA-256 hash, writing result to output buffer.
pub fn sha256_to_buf(data: &[u8], out: &mut [u8]) -> ApiResult<()> {
    if out.len() < 32 {
        return Err(ApiError::BufferTooSmall);
    }
    let hash = sha256(data);
    out[..32].copy_from_slice(&hash);
    Ok(())
}

/// Ed25519 cryptographic operations.
///
/// Only available with the `ed25519` feature.
#[cfg(feature = "ed25519")]
pub mod ed25519 {
    use super::*;
    use ed25519_dalek::{SigningKey, Signer, VerifyingKey, Verifier, Signature};

    /// Ed25519 key pair holder.
    pub struct Ed25519Keys {
        signing_key: SigningKey,
    }

    impl Ed25519Keys {
        /// Create from raw 32-byte secret key.
        pub fn from_bytes(secret: &[u8; 32]) -> Self {
            let signing_key = SigningKey::from_bytes(secret);
            Self { signing_key }
        }

        /// Get the public key bytes (32 bytes).
        pub fn pubkey(&self) -> [u8; 32] {
            let vk: VerifyingKey = (&self.signing_key).into();
            vk.to_bytes()
        }

        /// Sign a message, writing the 64-byte signature to output.
        pub fn sign(&self, message: &[u8]) -> ApiResult<[u8; 64]> {
            let sig = self.signing_key.sign(message);
            Ok(sig.to_bytes())
        }

        /// Verify a signature against a message.
        pub fn verify(&self, message: &[u8], signature: &[u8; 64]) -> ApiResult<bool> {
            let vk: VerifyingKey = (&self.signing_key).into();
            let sig = Signature::from_bytes(signature);
            Ok(vk.verify(message, &sig).is_ok())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sha256_known_vector() {
        // SHA-256("") = e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
        let hash = sha256(b"");
        assert_eq!(
            hash,
            [
                0xe3, 0xb0, 0xc4, 0x42, 0x98, 0xfc, 0x1c, 0x14,
                0x9a, 0xfb, 0xf4, 0xc8, 0x99, 0x6f, 0xb9, 0x24,
                0x27, 0xae, 0x41, 0xe4, 0x64, 0x9b, 0x93, 0x4c,
                0xa4, 0x95, 0x99, 0x1b, 0x78, 0x52, 0xb8, 0x55,
            ]
        );
    }

    #[test]
    fn sha256_deterministic() {
        let h1 = sha256(b"test data");
        let h2 = sha256(b"test data");
        assert_eq!(h1, h2);
    }

    #[test]
    fn sha256_to_buf_works() {
        let mut buf = [0u8; 32];
        sha256_to_buf(b"hello", &mut buf).unwrap();
        assert_eq!(buf, sha256(b"hello"));
    }

    #[test]
    fn sha256_to_buf_short_buffer() {
        let mut buf = [0u8; 16];
        assert!(sha256_to_buf(b"hello", &mut buf).is_err());
    }

    #[cfg(feature = "ed25519")]
    mod ed25519_tests {
        use super::super::ed25519::*;

        #[test]
        fn pubkey_is_stable() {
            let secret = [42u8; 32];
            let k1 = Ed25519Keys::from_bytes(&secret);
            let k2 = Ed25519Keys::from_bytes(&secret);
            assert_eq!(k1.pubkey(), k2.pubkey());
        }

        #[test]
        fn sign_verify_roundtrip() {
            let secret = [7u8; 32];
            let keys = Ed25519Keys::from_bytes(&secret);
            let msg = b"RVF cognitive container attestation";
            let sig = keys.sign(msg).unwrap();
            assert!(keys.verify(msg, &sig).unwrap());
        }

        #[test]
        fn wrong_message_fails_verify() {
            let secret = [7u8; 32];
            let keys = Ed25519Keys::from_bytes(&secret);
            let sig = keys.sign(b"original").unwrap();
            assert!(!keys.verify(b"tampered", &sig).unwrap());
        }
    }
}

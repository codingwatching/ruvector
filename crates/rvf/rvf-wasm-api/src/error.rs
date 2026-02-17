//! Error types for the RVF v1 WASM API.

use alloc::string::String;
use core::fmt;

/// API error codes following the v1 specification.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ApiError {
    /// Invalid manifest data provided to init.
    InvalidManifest,
    /// Buffer too small for the requested operation.
    BufferTooSmall,
    /// State key not found.
    KeyNotFound,
    /// Cryptographic operation failed.
    CryptoError,
    /// I/O capability not available.
    CapabilityDenied,
    /// I/O capability ID not recognized.
    InvalidCapability,
    /// Container not initialized.
    NotInitialized,
    /// Container already sealed for this epoch.
    AlreadySealed,
    /// DRBG seed not provided.
    NoSeed,
    /// Internal error with description.
    Internal(String),
}

/// Convenience result type.
pub type ApiResult<T> = Result<T, ApiError>;

impl fmt::Display for ApiError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ApiError::InvalidManifest => write!(f, "invalid manifest"),
            ApiError::BufferTooSmall => write!(f, "buffer too small"),
            ApiError::KeyNotFound => write!(f, "key not found"),
            ApiError::CryptoError => write!(f, "crypto error"),
            ApiError::CapabilityDenied => write!(f, "capability denied"),
            ApiError::InvalidCapability => write!(f, "invalid capability"),
            ApiError::NotInitialized => write!(f, "container not initialized"),
            ApiError::AlreadySealed => write!(f, "epoch already sealed"),
            ApiError::NoSeed => write!(f, "DRBG seed not provided"),
            ApiError::Internal(msg) => write!(f, "internal: {}", msg),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for ApiError {}

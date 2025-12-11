//! TRM Configuration
//!
//! Configuration structures for the TRM recursive reasoning engine.

use serde::{Deserialize, Serialize};
use super::error::TrmError;

/// Configuration for TRM recursive engine
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TrmConfig {
    /// Hidden dimension for latent state
    pub hidden_dim: usize,

    /// Embedding dimension (input/output)
    pub embedding_dim: usize,

    /// Maximum answer sequence length
    pub max_answer_len: usize,

    /// Maximum recursion depth (K)
    pub max_k: usize,

    /// Default recursion depth
    pub default_k: usize,

    /// Latent updates per K iteration (n)
    pub latent_iterations: usize,

    /// Use attention variant (true) or MLP variant (false)
    pub use_attention: bool,

    /// Number of attention heads (if using attention)
    pub num_heads: usize,

    /// Dropout probability (for training)
    pub dropout: f32,

    /// Confidence threshold for early stopping
    pub confidence_threshold: f32,

    /// Enable SIMD optimizations (reserved for future use)
    #[serde(default)]
    pub use_simd: bool,

    /// Residual scale for answer refinement
    pub residual_scale: f32,

    /// Enable early stopping
    pub early_stopping: bool,

    /// Minimum iterations before early stopping
    pub min_iterations: usize,

    /// Use entropy-based confidence scoring
    pub use_entropy_confidence: bool,

    /// Convergence threshold for plateauing detection
    pub convergence_threshold: f32,
}

impl Default for TrmConfig {
    fn default() -> Self {
        Self {
            hidden_dim: 256,
            embedding_dim: 256,
            max_answer_len: 64,
            max_k: 20,
            default_k: 5,
            latent_iterations: 3,
            use_attention: false,
            num_heads: 8,
            dropout: 0.1,
            confidence_threshold: 0.95,
            use_simd: true,
            residual_scale: 0.1,
            early_stopping: true,
            min_iterations: 1,
            use_entropy_confidence: false,
            convergence_threshold: 0.001,
        }
    }
}

impl TrmConfig {
    /// Create a new builder
    pub fn builder() -> TrmConfigBuilder {
        TrmConfigBuilder::new()
    }

    /// Validate the configuration
    pub fn validate(&self) -> Result<(), TrmError> {
        if self.hidden_dim == 0 {
            return Err(TrmError::InvalidConfig("hidden_dim must be > 0".to_string()));
        }

        if self.embedding_dim == 0 {
            return Err(TrmError::InvalidConfig("embedding_dim must be > 0".to_string()));
        }

        if self.max_k == 0 {
            return Err(TrmError::InvalidConfig("max_k must be > 0".to_string()));
        }

        if self.default_k > self.max_k {
            return Err(TrmError::InvalidConfig(format!(
                "default_k ({}) cannot exceed max_k ({})",
                self.default_k, self.max_k
            )));
        }

        if self.use_attention && self.hidden_dim % self.num_heads != 0 {
            return Err(TrmError::InvalidConfig(format!(
                "hidden_dim ({}) must be divisible by num_heads ({}) for attention",
                self.hidden_dim, self.num_heads
            )));
        }

        if self.confidence_threshold < 0.0 || self.confidence_threshold > 1.0 {
            return Err(TrmError::InvalidConfig(format!(
                "confidence_threshold ({}) must be in [0, 1]",
                self.confidence_threshold
            )));
        }

        if self.latent_iterations == 0 {
            return Err(TrmError::InvalidConfig("latent_iterations must be > 0".to_string()));
        }

        Ok(())
    }
}

/// Builder for TrmConfig
#[derive(Clone, Debug)]
pub struct TrmConfigBuilder {
    config: TrmConfig,
}

impl TrmConfigBuilder {
    /// Create a new builder with default values
    pub fn new() -> Self {
        Self {
            config: TrmConfig::default(),
        }
    }

    /// Set hidden dimension
    pub fn hidden_dim(mut self, dim: usize) -> Self {
        self.config.hidden_dim = dim;
        self
    }

    /// Set embedding dimension
    pub fn embedding_dim(mut self, dim: usize) -> Self {
        self.config.embedding_dim = dim;
        self
    }

    /// Set maximum answer length
    pub fn max_answer_len(mut self, len: usize) -> Self {
        self.config.max_answer_len = len;
        self
    }

    /// Set maximum K
    pub fn max_k(mut self, k: usize) -> Self {
        self.config.max_k = k;
        self
    }

    /// Set default K
    pub fn default_k(mut self, k: usize) -> Self {
        self.config.default_k = k;
        self
    }

    /// Set latent iterations per K step
    pub fn latent_iterations(mut self, n: usize) -> Self {
        self.config.latent_iterations = n;
        self
    }

    /// Use attention variant
    pub fn use_attention(mut self, use_attn: bool) -> Self {
        self.config.use_attention = use_attn;
        self
    }

    /// Set number of attention heads
    pub fn num_heads(mut self, heads: usize) -> Self {
        self.config.num_heads = heads;
        self
    }

    /// Set dropout probability
    pub fn dropout(mut self, p: f32) -> Self {
        self.config.dropout = p;
        self
    }

    /// Set confidence threshold for early stopping
    pub fn confidence_threshold(mut self, threshold: f32) -> Self {
        self.config.confidence_threshold = threshold;
        self
    }

    /// Enable/disable SIMD optimizations
    pub fn use_simd(mut self, use_simd: bool) -> Self {
        self.config.use_simd = use_simd;
        self
    }

    /// Set residual scale for answer refinement
    pub fn residual_scale(mut self, scale: f32) -> Self {
        self.config.residual_scale = scale;
        self
    }

    /// Enable/disable early stopping
    pub fn early_stopping(mut self, enabled: bool) -> Self {
        self.config.early_stopping = enabled;
        self
    }

    /// Set minimum iterations before early stopping
    pub fn min_iterations(mut self, min: usize) -> Self {
        self.config.min_iterations = min;
        self
    }

    /// Use entropy-based confidence scoring
    pub fn use_entropy_confidence(mut self, use_entropy: bool) -> Self {
        self.config.use_entropy_confidence = use_entropy;
        self
    }

    /// Set convergence threshold for plateau detection
    pub fn convergence_threshold(mut self, threshold: f32) -> Self {
        self.config.convergence_threshold = threshold;
        self
    }

    /// Build the configuration (validates before returning)
    pub fn build(self) -> Result<TrmConfig, TrmError> {
        self.config.validate()?;
        Ok(self.config)
    }

    /// Build without validation (for testing)
    pub fn build_unchecked(self) -> TrmConfig {
        self.config
    }
}

impl Default for TrmConfigBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = TrmConfig::default();
        assert_eq!(config.hidden_dim, 256);
        assert_eq!(config.max_k, 20);
        assert_eq!(config.default_k, 5);
        assert!(!config.use_attention);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_builder() {
        let config = TrmConfig::builder()
            .hidden_dim(512)
            .max_k(30)
            .use_attention(true)
            .num_heads(8)
            .build()
            .unwrap();

        assert_eq!(config.hidden_dim, 512);
        assert_eq!(config.max_k, 30);
        assert!(config.use_attention);
    }

    #[test]
    fn test_validation_hidden_dim() {
        let result = TrmConfig::builder()
            .hidden_dim(0)
            .build();
        assert!(result.is_err());
    }

    #[test]
    fn test_validation_default_k_exceeds_max() {
        let result = TrmConfig::builder()
            .max_k(10)
            .default_k(20)
            .build();
        assert!(result.is_err());
    }

    #[test]
    fn test_validation_attention_heads() {
        // 256 is not divisible by 7
        let result = TrmConfig::builder()
            .hidden_dim(256)
            .use_attention(true)
            .num_heads(7)
            .build();
        assert!(result.is_err());
    }

    #[test]
    fn test_validation_confidence_threshold() {
        let result = TrmConfig::builder()
            .confidence_threshold(1.5)
            .build();
        assert!(result.is_err());
    }

    #[test]
    fn test_serialization() {
        let config = TrmConfig::default();
        let json = serde_json::to_string(&config).unwrap();
        let deserialized: TrmConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(config.hidden_dim, deserialized.hidden_dim);
        assert_eq!(config.max_k, deserialized.max_k);
    }
}

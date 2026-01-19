//! LLM inference backends for RuvLLM
//!
//! This module provides pluggable backend implementations for LLM inference.
//! Currently supported backends:
//!
//! - **Candle** (Rust-native HuggingFace): Full Rust implementation with Metal acceleration
//!
//! ## Architecture Support
//!
//! The Candle backend supports the following model architectures:
//! - Mistral (7B, Codestral)
//! - Llama (1B-70B, Llama 2, Llama 3)
//! - Phi (1.5, 2, 3)
//!
//! ## Quantization
//!
//! Supports GGUF quantization formats:
//! - Q4_0, Q4_1, Q4_K (4-bit quantization)
//! - Q8_0, Q8_1 (8-bit quantization)
//! - F16, F32 (full precision)
//!
//! ## Example
//!
//! ```rust,ignore
//! use ruvllm::backends::{CandleBackend, ModelConfig, GenerateParams};
//!
//! let mut backend = CandleBackend::new()?;
//!
//! let config = ModelConfig {
//!     architecture: ModelArchitecture::Mistral,
//!     quantization: Some(Quantization::Q4K),
//!     use_flash_attention: true,
//!     ..Default::default()
//! };
//!
//! backend.load_model("mistralai/Mistral-7B-v0.1", config)?;
//!
//! let params = GenerateParams::default()
//!     .with_max_tokens(256)
//!     .with_temperature(0.7);
//!
//! let response = backend.generate("Hello, world!", params)?;
//! ```

#[cfg(feature = "candle")]
mod candle_backend;

#[cfg(feature = "candle")]
pub use candle_backend::*;

use crate::error::{Result, RuvLLMError};
use std::sync::Arc;

/// Model architecture types supported by RuvLLM
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ModelArchitecture {
    /// Mistral architecture (7B, Codestral)
    Mistral,
    /// Llama architecture (1B-70B)
    Llama,
    /// Phi architecture (1.5, 2, 3)
    Phi,
    /// Qwen architecture
    Qwen,
    /// Gemma architecture
    Gemma,
}

impl Default for ModelArchitecture {
    fn default() -> Self {
        Self::Llama
    }
}

impl ModelArchitecture {
    /// Get architecture name for HuggingFace model config
    pub fn config_name(&self) -> &'static str {
        match self {
            Self::Mistral => "mistral",
            Self::Llama => "llama",
            Self::Phi => "phi",
            Self::Qwen => "qwen2",
            Self::Gemma => "gemma",
        }
    }
}

/// Quantization formats for model weights
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Quantization {
    /// No quantization (FP32)
    None,
    /// Half precision (FP16)
    F16,
    /// Brain float (BF16)
    Bf16,
    /// 8-bit quantization
    Q8,
    /// 4-bit K-quants (higher quality)
    Q4K,
    /// 4-bit quantization (standard)
    Q4,
    /// 2-bit quantization (experimental)
    Q2K,
}

impl Default for Quantization {
    fn default() -> Self {
        Self::Q4K
    }
}

impl Quantization {
    /// Get bytes per weight element
    pub fn bytes_per_weight(&self) -> f32 {
        match self {
            Self::None => 4.0,
            Self::F16 | Self::Bf16 => 2.0,
            Self::Q8 => 1.0,
            Self::Q4K | Self::Q4 => 0.5,
            Self::Q2K => 0.25,
        }
    }

    /// Check if this is a GGUF quantization format
    pub fn is_gguf(&self) -> bool {
        matches!(self, Self::Q8 | Self::Q4K | Self::Q4 | Self::Q2K)
    }
}

/// Configuration for loading and running a model
#[derive(Debug, Clone)]
pub struct ModelConfig {
    /// Model architecture
    pub architecture: ModelArchitecture,
    /// Quantization format
    pub quantization: Option<Quantization>,
    /// Use Flash Attention for memory efficiency
    pub use_flash_attention: bool,
    /// Maximum sequence length
    pub max_sequence_length: usize,
    /// Number of key-value heads (for GQA)
    pub num_kv_heads: Option<usize>,
    /// Hidden dimension size
    pub hidden_size: Option<usize>,
    /// Number of layers
    pub num_layers: Option<usize>,
    /// Vocabulary size
    pub vocab_size: Option<usize>,
    /// Rope theta (for rotary embeddings)
    pub rope_theta: Option<f64>,
    /// Use sliding window attention
    pub sliding_window: Option<usize>,
    /// Device to load model on (metal, cpu)
    pub device: DeviceType,
    /// Data type for inference
    pub dtype: DType,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            architecture: ModelArchitecture::default(),
            quantization: Some(Quantization::Q4K),
            use_flash_attention: true,
            max_sequence_length: 4096,
            num_kv_heads: None,
            hidden_size: None,
            num_layers: None,
            vocab_size: None,
            rope_theta: None,
            sliding_window: None,
            device: DeviceType::default(),
            dtype: DType::default(),
        }
    }
}

/// Device type for inference
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum DeviceType {
    /// CPU inference
    Cpu,
    /// Metal (Apple Silicon) - default on macOS
    #[default]
    Metal,
    /// CUDA (NVIDIA GPUs)
    Cuda(usize),
}

/// Data type for tensor operations
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum DType {
    /// 32-bit floating point
    F32,
    /// 16-bit floating point (default)
    #[default]
    F16,
    /// Brain float 16
    Bf16,
}

/// Parameters for text generation
#[derive(Debug, Clone)]
pub struct GenerateParams {
    /// Maximum number of tokens to generate
    pub max_tokens: usize,
    /// Temperature for sampling (0.0 = deterministic)
    pub temperature: f32,
    /// Top-p (nucleus) sampling
    pub top_p: f32,
    /// Top-k sampling (0 = disabled)
    pub top_k: usize,
    /// Repetition penalty
    pub repetition_penalty: f32,
    /// Frequency penalty
    pub frequency_penalty: f32,
    /// Presence penalty
    pub presence_penalty: f32,
    /// Stop sequences
    pub stop_sequences: Vec<String>,
    /// Seed for reproducibility
    pub seed: Option<u64>,
}

impl Default for GenerateParams {
    fn default() -> Self {
        Self {
            max_tokens: 256,
            temperature: 0.7,
            top_p: 0.9,
            top_k: 40,
            repetition_penalty: 1.1,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
            stop_sequences: Vec::new(),
            seed: None,
        }
    }
}

impl GenerateParams {
    /// Set maximum tokens
    pub fn with_max_tokens(mut self, max_tokens: usize) -> Self {
        self.max_tokens = max_tokens;
        self
    }

    /// Set temperature
    pub fn with_temperature(mut self, temperature: f32) -> Self {
        self.temperature = temperature;
        self
    }

    /// Set top-p sampling
    pub fn with_top_p(mut self, top_p: f32) -> Self {
        self.top_p = top_p;
        self
    }

    /// Set top-k sampling
    pub fn with_top_k(mut self, top_k: usize) -> Self {
        self.top_k = top_k;
        self
    }

    /// Set repetition penalty
    pub fn with_repetition_penalty(mut self, penalty: f32) -> Self {
        self.repetition_penalty = penalty;
        self
    }

    /// Add stop sequence
    pub fn with_stop_sequence(mut self, stop: impl Into<String>) -> Self {
        self.stop_sequences.push(stop.into());
        self
    }

    /// Set seed for reproducibility
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }
}

/// Token generated during streaming
#[derive(Debug, Clone)]
pub struct GeneratedToken {
    /// Token ID
    pub id: u32,
    /// Token text
    pub text: String,
    /// Log probability
    pub logprob: Option<f32>,
    /// Is this a special token
    pub is_special: bool,
}

/// Backend trait for LLM inference
///
/// This trait defines the interface that all inference backends must implement.
/// It provides methods for model loading, text generation, and embedding extraction.
pub trait LlmBackend: Send + Sync {
    /// Load a model from path or HuggingFace Hub
    ///
    /// # Arguments
    ///
    /// * `model_id` - Path to local model or HuggingFace model ID
    /// * `config` - Model configuration
    ///
    /// # Errors
    ///
    /// Returns an error if the model cannot be loaded (not found, invalid format, etc.)
    fn load_model(&mut self, model_id: &str, config: ModelConfig) -> Result<()>;

    /// Generate text from a prompt
    ///
    /// # Arguments
    ///
    /// * `prompt` - Input text prompt
    /// * `params` - Generation parameters
    ///
    /// # Returns
    ///
    /// Generated text (excluding the input prompt)
    fn generate(&self, prompt: &str, params: GenerateParams) -> Result<String>;

    /// Generate text with streaming output
    ///
    /// # Arguments
    ///
    /// * `prompt` - Input text prompt
    /// * `params` - Generation parameters
    ///
    /// # Returns
    ///
    /// Iterator over generated tokens
    fn generate_stream(
        &self,
        prompt: &str,
        params: GenerateParams,
    ) -> Result<Box<dyn Iterator<Item = Result<GeneratedToken>> + Send + '_>>;

    /// Extract embeddings from text
    ///
    /// Uses the model's embedding layer to generate dense vector representations.
    ///
    /// # Arguments
    ///
    /// * `text` - Input text
    ///
    /// # Returns
    ///
    /// Vector of embeddings (hidden_size dimension)
    fn get_embeddings(&self, text: &str) -> Result<Vec<f32>>;

    /// Get the tokenizer for this backend
    fn tokenizer(&self) -> Option<&dyn Tokenizer>;

    /// Check if a model is loaded
    fn is_model_loaded(&self) -> bool;

    /// Get model information
    fn model_info(&self) -> Option<ModelInfo>;

    /// Unload the current model and free memory
    fn unload_model(&mut self);
}

/// Tokenizer trait for text encoding/decoding
pub trait Tokenizer: Send + Sync {
    /// Encode text to token IDs
    fn encode(&self, text: &str) -> Result<Vec<u32>>;

    /// Decode token IDs to text
    fn decode(&self, tokens: &[u32]) -> Result<String>;

    /// Get vocabulary size
    fn vocab_size(&self) -> usize;

    /// Get special tokens
    fn special_tokens(&self) -> SpecialTokens;
}

/// Special token IDs
#[derive(Debug, Clone, Default)]
pub struct SpecialTokens {
    /// Beginning of sequence token
    pub bos_token_id: Option<u32>,
    /// End of sequence token
    pub eos_token_id: Option<u32>,
    /// Padding token
    pub pad_token_id: Option<u32>,
    /// Unknown token
    pub unk_token_id: Option<u32>,
}

/// Information about a loaded model
#[derive(Debug, Clone)]
pub struct ModelInfo {
    /// Model name/ID
    pub name: String,
    /// Model architecture
    pub architecture: ModelArchitecture,
    /// Number of parameters (approximate)
    pub num_parameters: usize,
    /// Vocabulary size
    pub vocab_size: usize,
    /// Hidden dimension
    pub hidden_size: usize,
    /// Number of layers
    pub num_layers: usize,
    /// Maximum context length
    pub max_context_length: usize,
    /// Quantization applied
    pub quantization: Option<Quantization>,
    /// Memory usage in bytes
    pub memory_usage: usize,
}

/// A placeholder backend for when no real backend is available
pub struct NoopBackend;

impl LlmBackend for NoopBackend {
    fn load_model(&mut self, _model_id: &str, _config: ModelConfig) -> Result<()> {
        Err(RuvLLMError::Config(
            "No inference backend enabled. Enable 'candle' feature.".to_string(),
        ))
    }

    fn generate(&self, _prompt: &str, _params: GenerateParams) -> Result<String> {
        Err(RuvLLMError::Config(
            "No inference backend enabled.".to_string(),
        ))
    }

    fn generate_stream(
        &self,
        _prompt: &str,
        _params: GenerateParams,
    ) -> Result<Box<dyn Iterator<Item = Result<GeneratedToken>> + Send + '_>> {
        Err(RuvLLMError::Config(
            "No inference backend enabled.".to_string(),
        ))
    }

    fn get_embeddings(&self, _text: &str) -> Result<Vec<f32>> {
        Err(RuvLLMError::Config(
            "No inference backend enabled.".to_string(),
        ))
    }

    fn tokenizer(&self) -> Option<&dyn Tokenizer> {
        None
    }

    fn is_model_loaded(&self) -> bool {
        false
    }

    fn model_info(&self) -> Option<ModelInfo> {
        None
    }

    fn unload_model(&mut self) {}
}

/// Create a backend instance based on available features
pub fn create_backend() -> Box<dyn LlmBackend> {
    #[cfg(feature = "candle")]
    {
        Box::new(CandleBackend::new().unwrap_or_else(|_| CandleBackend::default()))
    }

    #[cfg(not(feature = "candle"))]
    {
        Box::new(NoopBackend)
    }
}

/// Thread-safe backend wrapper
pub type SharedBackend = Arc<dyn LlmBackend>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantization_bytes() {
        assert_eq!(Quantization::None.bytes_per_weight(), 4.0);
        assert_eq!(Quantization::F16.bytes_per_weight(), 2.0);
        assert_eq!(Quantization::Q4K.bytes_per_weight(), 0.5);
    }

    #[test]
    fn test_generate_params_builder() {
        let params = GenerateParams::default()
            .with_max_tokens(512)
            .with_temperature(0.5)
            .with_top_p(0.95)
            .with_seed(42);

        assert_eq!(params.max_tokens, 512);
        assert_eq!(params.temperature, 0.5);
        assert_eq!(params.top_p, 0.95);
        assert_eq!(params.seed, Some(42));
    }

    #[test]
    fn test_model_architecture() {
        assert_eq!(ModelArchitecture::Mistral.config_name(), "mistral");
        assert_eq!(ModelArchitecture::Llama.config_name(), "llama");
    }
}

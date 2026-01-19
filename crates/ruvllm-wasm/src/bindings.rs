//! JavaScript/WASM Bindings for RuvLLM
//!
//! This module provides JavaScript-friendly wrappers around the RuvLLM
//! inference runtime. All types are designed to work seamlessly with
//! JavaScript through wasm-bindgen.
//!
//! # Example (JavaScript)
//!
//! ```javascript
//! import init, { RuvLLMWasm, GenerateConfig, KvCacheWasm } from 'ruvllm-wasm';
//!
//! await init();
//!
//! // Create inference engine
//! const llm = new RuvLLMWasm();
//!
//! // Configure generation
//! const config = new GenerateConfig();
//! config.maxTokens = 256;
//! config.temperature = 0.7;
//!
//! // Generate text
//! const result = await llm.generate("Hello, world!", config);
//! console.log(result);
//! ```

use crate::utils::{log, result_to_js};
use ruvllm_integration::{
    kv_cache::{KvCacheConfig, KvCacheStats, TwoTierKvCache},
    memory_pool::{ArenaStats, BufferPool, BufferPoolStats, BufferSize, InferenceArena},
    tokenizer::{ChatMessage, ChatTemplate, Role},
    types::{ModelSize, Precision},
};
use serde::{Deserialize, Serialize};
use wasm_bindgen::prelude::*;

// ============================================================================
// Configuration Types
// ============================================================================

/// Generation configuration for text generation.
///
/// Controls sampling parameters and output constraints.
/// TypeScript-friendly with getter/setter methods.
#[wasm_bindgen]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerateConfig {
    /// Maximum tokens to generate
    #[wasm_bindgen(skip)]
    pub max_tokens: usize,
    /// Temperature for sampling (0.0 = deterministic)
    #[wasm_bindgen(skip)]
    pub temperature: f32,
    /// Top-p (nucleus) sampling threshold
    #[wasm_bindgen(skip)]
    pub top_p: f32,
    /// Top-k sampling (0 = disabled)
    #[wasm_bindgen(skip)]
    pub top_k: usize,
    /// Repetition penalty (1.0 = no penalty)
    #[wasm_bindgen(skip)]
    pub repetition_penalty: f32,
    /// Stop sequences (JSON array of strings)
    #[wasm_bindgen(skip)]
    pub stop_sequences: Vec<String>,
}

#[wasm_bindgen]
impl GenerateConfig {
    /// Create a new GenerateConfig with default values.
    #[wasm_bindgen(constructor)]
    pub fn new() -> GenerateConfig {
        GenerateConfig {
            max_tokens: 256,
            temperature: 0.7,
            top_p: 0.9,
            top_k: 40,
            repetition_penalty: 1.1,
            stop_sequences: Vec::new(),
        }
    }

    /// Get maximum tokens.
    #[wasm_bindgen(getter, js_name = maxTokens)]
    pub fn max_tokens(&self) -> usize {
        self.max_tokens
    }

    /// Set maximum tokens.
    #[wasm_bindgen(setter, js_name = maxTokens)]
    pub fn set_max_tokens(&mut self, value: usize) {
        self.max_tokens = value;
    }

    /// Get temperature.
    #[wasm_bindgen(getter)]
    pub fn temperature(&self) -> f32 {
        self.temperature
    }

    /// Set temperature.
    #[wasm_bindgen(setter)]
    pub fn set_temperature(&mut self, value: f32) {
        self.temperature = value;
    }

    /// Get top-p value.
    #[wasm_bindgen(getter, js_name = topP)]
    pub fn top_p(&self) -> f32 {
        self.top_p
    }

    /// Set top-p value.
    #[wasm_bindgen(setter, js_name = topP)]
    pub fn set_top_p(&mut self, value: f32) {
        self.top_p = value;
    }

    /// Get top-k value.
    #[wasm_bindgen(getter, js_name = topK)]
    pub fn top_k(&self) -> usize {
        self.top_k
    }

    /// Set top-k value.
    #[wasm_bindgen(setter, js_name = topK)]
    pub fn set_top_k(&mut self, value: usize) {
        self.top_k = value;
    }

    /// Get repetition penalty.
    #[wasm_bindgen(getter, js_name = repetitionPenalty)]
    pub fn repetition_penalty(&self) -> f32 {
        self.repetition_penalty
    }

    /// Set repetition penalty.
    #[wasm_bindgen(setter, js_name = repetitionPenalty)]
    pub fn set_repetition_penalty(&mut self, value: f32) {
        self.repetition_penalty = value;
    }

    /// Add a stop sequence.
    #[wasm_bindgen(js_name = addStopSequence)]
    pub fn add_stop_sequence(&mut self, sequence: &str) {
        self.stop_sequences.push(sequence.to_string());
    }

    /// Clear all stop sequences.
    #[wasm_bindgen(js_name = clearStopSequences)]
    pub fn clear_stop_sequences(&mut self) {
        self.stop_sequences.clear();
    }

    /// Convert to JSON string.
    #[wasm_bindgen(js_name = toJson)]
    pub fn to_json(&self) -> Result<String, JsValue> {
        serde_json::to_string(self).map_err(|e| JsValue::from_str(&e.to_string()))
    }

    /// Create from JSON string.
    #[wasm_bindgen(js_name = fromJson)]
    pub fn from_json(json: &str) -> Result<GenerateConfig, JsValue> {
        serde_json::from_str(json).map_err(|e| JsValue::from_str(&e.to_string()))
    }
}

impl Default for GenerateConfig {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Chat Message Types
// ============================================================================

/// Chat message for instruction-tuned models.
///
/// Used to construct conversations for chat-based inference.
#[wasm_bindgen]
#[derive(Debug, Clone)]
pub struct ChatMessageWasm {
    inner: ChatMessage,
}

#[wasm_bindgen]
impl ChatMessageWasm {
    /// Create a system message.
    #[wasm_bindgen(js_name = system)]
    pub fn system(content: &str) -> ChatMessageWasm {
        ChatMessageWasm {
            inner: ChatMessage::system(content),
        }
    }

    /// Create a user message.
    #[wasm_bindgen(js_name = user)]
    pub fn user(content: &str) -> ChatMessageWasm {
        ChatMessageWasm {
            inner: ChatMessage::user(content),
        }
    }

    /// Create an assistant message.
    #[wasm_bindgen(js_name = assistant)]
    pub fn assistant(content: &str) -> ChatMessageWasm {
        ChatMessageWasm {
            inner: ChatMessage::assistant(content),
        }
    }

    /// Get the role as a string.
    #[wasm_bindgen(getter)]
    pub fn role(&self) -> String {
        self.inner.role.as_str().to_string()
    }

    /// Get the message content.
    #[wasm_bindgen(getter)]
    pub fn content(&self) -> String {
        self.inner.content.clone()
    }
}

/// Chat template for formatting conversations.
#[wasm_bindgen]
#[derive(Debug, Clone)]
pub struct ChatTemplateWasm {
    inner: ChatTemplate,
}

#[wasm_bindgen]
impl ChatTemplateWasm {
    /// Create a Llama 3 chat template.
    #[wasm_bindgen(js_name = llama3)]
    pub fn llama3() -> ChatTemplateWasm {
        ChatTemplateWasm {
            inner: ChatTemplate::Llama3,
        }
    }

    /// Create a Mistral chat template.
    #[wasm_bindgen(js_name = mistral)]
    pub fn mistral() -> ChatTemplateWasm {
        ChatTemplateWasm {
            inner: ChatTemplate::Mistral,
        }
    }

    /// Create a Qwen/ChatML chat template.
    #[wasm_bindgen(js_name = chatml)]
    pub fn chatml() -> ChatTemplateWasm {
        ChatTemplateWasm {
            inner: ChatTemplate::ChatML,
        }
    }

    /// Create a Phi chat template.
    #[wasm_bindgen(js_name = phi)]
    pub fn phi() -> ChatTemplateWasm {
        ChatTemplateWasm {
            inner: ChatTemplate::Phi,
        }
    }

    /// Create a Gemma chat template.
    #[wasm_bindgen(js_name = gemma)]
    pub fn gemma() -> ChatTemplateWasm {
        ChatTemplateWasm {
            inner: ChatTemplate::Gemma,
        }
    }

    /// Create a custom chat template.
    #[wasm_bindgen(js_name = custom)]
    pub fn custom(template: &str) -> ChatTemplateWasm {
        ChatTemplateWasm {
            inner: ChatTemplate::Custom(template.to_string()),
        }
    }

    /// Detect template from model ID.
    #[wasm_bindgen(js_name = detectFromModelId)]
    pub fn detect_from_model_id(model_id: &str) -> ChatTemplateWasm {
        ChatTemplateWasm {
            inner: ChatTemplate::detect_from_model_id(model_id),
        }
    }

    /// Format messages using this template.
    #[wasm_bindgen(js_name = format)]
    pub fn format(&self, messages: Vec<ChatMessageWasm>) -> String {
        let inner_messages: Vec<ChatMessage> = messages.into_iter().map(|m| m.inner).collect();
        self.inner.format(&inner_messages)
    }

    /// Get the template name.
    #[wasm_bindgen(getter)]
    pub fn name(&self) -> String {
        match &self.inner {
            ChatTemplate::Llama3 => "llama3".to_string(),
            ChatTemplate::Llama2 => "llama2".to_string(),
            ChatTemplate::Mistral => "mistral".to_string(),
            ChatTemplate::Qwen => "qwen".to_string(),
            ChatTemplate::ChatML => "chatml".to_string(),
            ChatTemplate::Phi => "phi".to_string(),
            ChatTemplate::Gemma => "gemma".to_string(),
            ChatTemplate::Custom(_) => "custom".to_string(),
        }
    }
}

// ============================================================================
// KV Cache
// ============================================================================

/// KV cache configuration for WASM.
#[wasm_bindgen]
#[derive(Debug, Clone)]
pub struct KvCacheConfigWasm {
    /// Number of tokens in high-precision tail
    tail_length: usize,
    /// Maximum tokens to cache
    max_tokens: usize,
    /// Number of KV heads
    num_kv_heads: usize,
    /// Head dimension
    head_dim: usize,
}

#[wasm_bindgen]
impl KvCacheConfigWasm {
    /// Create a new KV cache configuration.
    #[wasm_bindgen(constructor)]
    pub fn new() -> KvCacheConfigWasm {
        KvCacheConfigWasm {
            tail_length: 256,
            max_tokens: 4096,
            num_kv_heads: 8,
            head_dim: 128,
        }
    }

    /// Get tail length.
    #[wasm_bindgen(getter, js_name = tailLength)]
    pub fn tail_length(&self) -> usize {
        self.tail_length
    }

    /// Set tail length.
    #[wasm_bindgen(setter, js_name = tailLength)]
    pub fn set_tail_length(&mut self, value: usize) {
        self.tail_length = value;
    }

    /// Get max tokens.
    #[wasm_bindgen(getter, js_name = maxTokens)]
    pub fn max_tokens(&self) -> usize {
        self.max_tokens
    }

    /// Set max tokens.
    #[wasm_bindgen(setter, js_name = maxTokens)]
    pub fn set_max_tokens(&mut self, value: usize) {
        self.max_tokens = value;
    }

    /// Get number of KV heads.
    #[wasm_bindgen(getter, js_name = numKvHeads)]
    pub fn num_kv_heads(&self) -> usize {
        self.num_kv_heads
    }

    /// Set number of KV heads.
    #[wasm_bindgen(setter, js_name = numKvHeads)]
    pub fn set_num_kv_heads(&mut self, value: usize) {
        self.num_kv_heads = value;
    }

    /// Get head dimension.
    #[wasm_bindgen(getter, js_name = headDim)]
    pub fn head_dim(&self) -> usize {
        self.head_dim
    }

    /// Set head dimension.
    #[wasm_bindgen(setter, js_name = headDim)]
    pub fn set_head_dim(&mut self, value: usize) {
        self.head_dim = value;
    }

    /// Convert to internal config.
    pub(crate) fn to_internal(&self) -> KvCacheConfig {
        KvCacheConfig {
            tail_length: self.tail_length,
            tail_precision: Precision::FP16,
            store_precision: Precision::Q4,
            max_tokens: self.max_tokens,
            num_kv_heads: self.num_kv_heads,
            head_dim: self.head_dim,
            migration_batch: 64,
        }
    }
}

impl Default for KvCacheConfigWasm {
    fn default() -> Self {
        Self::new()
    }
}

/// KV cache statistics.
#[wasm_bindgen]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KvCacheStatsWasm {
    /// Total tokens cached
    pub total_tokens: usize,
    /// Tokens in high-precision tail
    pub tail_tokens: usize,
    /// Tokens in quantized store
    pub store_tokens: usize,
    /// Bytes used by tail
    pub tail_bytes: usize,
    /// Bytes used by store
    pub store_bytes: usize,
    /// Compression ratio
    pub compression_ratio: f32,
}

#[wasm_bindgen]
impl KvCacheStatsWasm {
    /// Get total tokens.
    #[wasm_bindgen(getter, js_name = totalTokens)]
    pub fn total_tokens(&self) -> usize {
        self.total_tokens
    }

    /// Get tail tokens.
    #[wasm_bindgen(getter, js_name = tailTokens)]
    pub fn tail_tokens(&self) -> usize {
        self.tail_tokens
    }

    /// Get store tokens.
    #[wasm_bindgen(getter, js_name = storeTokens)]
    pub fn store_tokens(&self) -> usize {
        self.store_tokens
    }

    /// Get compression ratio.
    #[wasm_bindgen(getter, js_name = compressionRatio)]
    pub fn compression_ratio(&self) -> f32 {
        self.compression_ratio
    }

    /// Convert to JSON.
    #[wasm_bindgen(js_name = toJson)]
    pub fn to_json(&self) -> Result<String, JsValue> {
        serde_json::to_string(self).map_err(|e| JsValue::from_str(&e.to_string()))
    }
}

/// Two-tier KV cache for WASM.
///
/// Provides memory-efficient caching with a high-precision tail
/// and quantized store for older tokens.
#[wasm_bindgen]
pub struct KvCacheWasm {
    inner: TwoTierKvCache,
}

#[wasm_bindgen]
impl KvCacheWasm {
    /// Create a new KV cache with the given configuration.
    #[wasm_bindgen(constructor)]
    pub fn new(config: &KvCacheConfigWasm) -> KvCacheWasm {
        KvCacheWasm {
            inner: TwoTierKvCache::new(config.to_internal()),
        }
    }

    /// Create with default configuration.
    #[wasm_bindgen(js_name = withDefaults)]
    pub fn with_defaults() -> KvCacheWasm {
        KvCacheWasm {
            inner: TwoTierKvCache::new(KvCacheConfig::default()),
        }
    }

    /// Append KV pairs to the cache.
    ///
    /// # Arguments
    ///
    /// * `keys` - Key tensor as Float32Array
    /// * `values` - Value tensor as Float32Array
    #[wasm_bindgen]
    pub fn append(&self, keys: &[f32], values: &[f32]) -> Result<(), JsValue> {
        self.inner.append(keys, values).map_err(|e| JsValue::from_str(&e.to_string()))
    }

    /// Get all cached KV pairs.
    ///
    /// Returns an object with `keys` and `values` Float32Arrays.
    #[wasm_bindgen(js_name = getAllKv)]
    pub fn get_all_kv(&self) -> Result<JsValue, JsValue> {
        let (keys, values) = self.inner.get_all_kv();

        let obj = js_sys::Object::new();
        let keys_array = js_sys::Float32Array::from(keys.as_slice());
        let values_array = js_sys::Float32Array::from(values.as_slice());

        js_sys::Reflect::set(&obj, &"keys".into(), &keys_array)?;
        js_sys::Reflect::set(&obj, &"values".into(), &values_array)?;

        Ok(obj.into())
    }

    /// Get cache statistics.
    #[wasm_bindgen]
    pub fn stats(&self) -> KvCacheStatsWasm {
        let stats = self.inner.stats();
        KvCacheStatsWasm {
            total_tokens: stats.total_tokens,
            tail_tokens: stats.tail_tokens,
            store_tokens: stats.store_tokens,
            tail_bytes: stats.tail_bytes,
            store_bytes: stats.store_bytes,
            compression_ratio: stats.compression_ratio,
        }
    }

    /// Clear the cache.
    #[wasm_bindgen]
    pub fn clear(&self) {
        self.inner.clear();
    }

    /// Get the total number of cached tokens.
    #[wasm_bindgen(getter, js_name = tokenCount)]
    pub fn token_count(&self) -> usize {
        self.inner.stats().total_tokens
    }
}

// ============================================================================
// Memory Arena
// ============================================================================

/// Arena allocator for inference buffers.
///
/// Provides fast bump allocation with O(1) reset for
/// generation-step temporaries.
#[wasm_bindgen]
pub struct InferenceArenaWasm {
    inner: InferenceArena,
}

#[wasm_bindgen]
impl InferenceArenaWasm {
    /// Create a new arena with the specified capacity in bytes.
    #[wasm_bindgen(constructor)]
    pub fn new(capacity: usize) -> InferenceArenaWasm {
        InferenceArenaWasm {
            inner: InferenceArena::new(capacity),
        }
    }

    /// Create an arena sized for model dimensions.
    #[wasm_bindgen(js_name = forModel)]
    pub fn for_model(hidden_dim: usize, vocab_size: usize, batch_size: usize) -> InferenceArenaWasm {
        InferenceArenaWasm {
            inner: InferenceArena::for_model(hidden_dim, vocab_size, batch_size),
        }
    }

    /// Reset the arena, making all memory available for reuse.
    #[wasm_bindgen]
    pub fn reset(&self) {
        self.inner.reset();
    }

    /// Get current bytes used.
    #[wasm_bindgen(getter)]
    pub fn used(&self) -> usize {
        self.inner.used()
    }

    /// Get total capacity.
    #[wasm_bindgen(getter)]
    pub fn capacity(&self) -> usize {
        self.inner.capacity()
    }

    /// Get remaining available bytes.
    #[wasm_bindgen(getter)]
    pub fn remaining(&self) -> usize {
        self.inner.remaining()
    }

    /// Get high water mark (maximum bytes ever used).
    #[wasm_bindgen(getter, js_name = highWaterMark)]
    pub fn high_water_mark(&self) -> usize {
        self.inner.high_water_mark()
    }

    /// Get statistics as JSON.
    #[wasm_bindgen(js_name = statsJson)]
    pub fn stats_json(&self) -> Result<String, JsValue> {
        let stats = self.inner.stats();
        serde_json::to_string(&ArenaStatsJson {
            capacity: stats.capacity,
            used: stats.used,
            remaining: stats.remaining,
            high_water_mark: stats.high_water_mark,
            allocation_count: stats.allocation_count,
            utilization: stats.utilization,
        })
        .map_err(|e| JsValue::from_str(&e.to_string()))
    }
}

#[derive(Serialize)]
struct ArenaStatsJson {
    capacity: usize,
    used: usize,
    remaining: usize,
    high_water_mark: usize,
    allocation_count: usize,
    utilization: f64,
}

// ============================================================================
// Buffer Pool
// ============================================================================

/// Buffer pool for efficient memory reuse.
///
/// Maintains free lists for multiple size classes to
/// minimize allocation overhead during inference.
#[wasm_bindgen]
pub struct BufferPoolWasm {
    inner: BufferPool,
}

#[wasm_bindgen]
impl BufferPoolWasm {
    /// Create a new buffer pool with default settings.
    #[wasm_bindgen(constructor)]
    pub fn new() -> BufferPoolWasm {
        BufferPoolWasm {
            inner: BufferPool::new(),
        }
    }

    /// Create with specified max buffers per size class.
    #[wasm_bindgen(js_name = withCapacity)]
    pub fn with_capacity(max_buffers_per_class: usize) -> BufferPoolWasm {
        BufferPoolWasm {
            inner: BufferPool::with_capacity(max_buffers_per_class),
        }
    }

    /// Pre-warm the pool by allocating buffers.
    #[wasm_bindgen(js_name = prewarmAll)]
    pub fn prewarm_all(&self, count_per_class: usize) {
        self.inner.prewarm_all(count_per_class);
    }

    /// Get pool statistics as JSON.
    #[wasm_bindgen(js_name = statsJson)]
    pub fn stats_json(&self) -> Result<String, JsValue> {
        let stats = self.inner.stats();
        serde_json::to_string(&PoolStatsJson {
            hits: stats.hits,
            misses: stats.misses,
            allocations: stats.allocations,
            returns: stats.returns,
            drops: stats.drops,
            free_buffers: stats.free_buffers.to_vec(),
            hit_rate: stats.hit_rate,
        })
        .map_err(|e| JsValue::from_str(&e.to_string()))
    }

    /// Get the hit rate (0.0 - 1.0).
    #[wasm_bindgen(getter, js_name = hitRate)]
    pub fn hit_rate(&self) -> f64 {
        self.inner.stats().hit_rate
    }

    /// Clear all pooled buffers.
    #[wasm_bindgen]
    pub fn clear(&self) {
        self.inner.clear();
    }
}

impl Default for BufferPoolWasm {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Serialize)]
struct PoolStatsJson {
    hits: u64,
    misses: u64,
    allocations: u64,
    returns: u64,
    drops: u64,
    free_buffers: Vec<usize>,
    hit_rate: f64,
}

// ============================================================================
// Main RuvLLM WASM Interface
// ============================================================================

/// Main RuvLLM WASM interface.
///
/// Provides the primary entry point for LLM inference in the browser.
/// Manages KV cache, memory pools, and inference state.
///
/// # Example (JavaScript)
///
/// ```javascript
/// const llm = new RuvLLMWasm();
/// await llm.initialize();
///
/// const result = await llm.generate("Hello, ", config);
/// console.log(result);
/// ```
#[wasm_bindgen]
pub struct RuvLLMWasm {
    /// KV cache for attention
    kv_cache: Option<TwoTierKvCache>,
    /// Buffer pool for memory management
    buffer_pool: BufferPool,
    /// Whether the engine is initialized
    initialized: bool,
}

#[wasm_bindgen]
impl RuvLLMWasm {
    /// Create a new RuvLLM WASM instance.
    #[wasm_bindgen(constructor)]
    pub fn new() -> RuvLLMWasm {
        crate::utils::set_panic_hook();

        RuvLLMWasm {
            kv_cache: None,
            buffer_pool: BufferPool::new(),
            initialized: false,
        }
    }

    /// Initialize the engine with default configuration.
    #[wasm_bindgen]
    pub fn initialize(&mut self) -> Result<(), JsValue> {
        self.initialize_with_config(&KvCacheConfigWasm::default())
    }

    /// Initialize with custom KV cache configuration.
    #[wasm_bindgen(js_name = initializeWithConfig)]
    pub fn initialize_with_config(&mut self, config: &KvCacheConfigWasm) -> Result<(), JsValue> {
        log("Initializing RuvLLM WASM...");

        // Create KV cache
        self.kv_cache = Some(TwoTierKvCache::new(config.to_internal()));

        // Pre-warm buffer pool
        self.buffer_pool.prewarm_all(4);

        self.initialized = true;
        log("RuvLLM WASM initialized successfully");

        Ok(())
    }

    /// Check if the engine is initialized.
    #[wasm_bindgen(getter, js_name = isInitialized)]
    pub fn is_initialized(&self) -> bool {
        self.initialized
    }

    /// Get the KV cache (if initialized).
    #[wasm_bindgen(js_name = getKvCache)]
    pub fn get_kv_cache(&self) -> Option<KvCacheWasm> {
        self.kv_cache.as_ref().map(|cache| KvCacheWasm {
            inner: TwoTierKvCache::new(KvCacheConfig::default()),
        })
    }

    /// Get buffer pool statistics.
    #[wasm_bindgen(js_name = getPoolStats)]
    pub fn get_pool_stats(&self) -> Result<String, JsValue> {
        let stats = self.buffer_pool.stats();
        serde_json::to_string(&PoolStatsJson {
            hits: stats.hits,
            misses: stats.misses,
            allocations: stats.allocations,
            returns: stats.returns,
            drops: stats.drops,
            free_buffers: stats.free_buffers.to_vec(),
            hit_rate: stats.hit_rate,
        })
        .map_err(|e| JsValue::from_str(&e.to_string()))
    }

    /// Clear all caches and reset state.
    #[wasm_bindgen]
    pub fn reset(&mut self) {
        if let Some(cache) = &self.kv_cache {
            cache.clear();
        }
        self.buffer_pool.clear();
        log("RuvLLM WASM state reset");
    }

    /// Get version information.
    #[wasm_bindgen(js_name = version)]
    pub fn version() -> String {
        "2.0.0".to_string()
    }

    /// Format a chat conversation using a template.
    #[wasm_bindgen(js_name = formatChat)]
    pub fn format_chat(
        template: &ChatTemplateWasm,
        messages: Vec<ChatMessageWasm>,
    ) -> String {
        let inner_messages: Vec<ChatMessage> = messages.into_iter().map(|m| m.inner).collect();
        template.inner.format(&inner_messages)
    }
}

impl Default for RuvLLMWasm {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Utility Exports
// ============================================================================

/// Get the WASM module version.
#[wasm_bindgen(js_name = getVersion)]
pub fn get_version() -> String {
    "2.0.0".to_string()
}

/// Check if the WASM module is ready.
#[wasm_bindgen(js_name = isReady)]
pub fn is_ready() -> bool {
    true
}

/// Detect chat template from model ID.
#[wasm_bindgen(js_name = detectChatTemplate)]
pub fn detect_chat_template(model_id: &str) -> ChatTemplateWasm {
    ChatTemplateWasm::detect_from_model_id(model_id)
}

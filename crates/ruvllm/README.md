# RuvLLM - High-Performance LLM Inference for Rust

RuvLLM is a Rust-native LLM inference engine optimized for Apple Silicon (M4 Pro), featuring real-time fine-tuning, NEON SIMD acceleration, and integration with the SONA self-optimizing neural architecture.

## Features

### Multiple Backends
- **Candle Backend**: HuggingFace's Candle framework with Metal GPU acceleration
- **mistral-rs**: Alternative backend for Mistral model family

### Optimized Kernels
- **NEON SIMD**: ARM64-optimized kernels with 4x loop unrolling and FMA instructions
- **Flash Attention 2**: Memory-efficient attention with O(N) complexity
- **Paged Attention**: Efficient KV cache management for inference

### Real-Time Learning
- **MicroLoRA**: Per-request fine-tuning with rank 1-2 adapters (<1ms latency)
- **EWC++**: Elastic Weight Consolidation to prevent catastrophic forgetting
- **SONA Integration**: Self-optimizing neural architecture with 3-tier learning loops

### Memory Efficiency
- **Two-Tier KV Cache**: FP16 tail + Q4/Q8 quantized store
- **Grouped-Query Attention (GQA)**: 4-8x KV memory reduction
- **Speculative Decoding**: 2-3x faster inference with draft models

## Quick Start

```rust
use ruvllm::prelude::*;

// Initialize backend with Metal GPU
let mut backend = CandleBackend::with_device(DeviceType::Metal)?;

// Load a model
backend.load_model("Qwen/Qwen2.5-7B-Instruct", ModelConfig::default())?;

// Generate text
let response = backend.generate("Explain quantum computing in simple terms.",
    GenerateParams {
        max_tokens: 256,
        temperature: 0.7,
        top_p: 0.9,
        ..Default::default()
    }
)?;

println!("{}", response);
```

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
ruvllm = { version = "0.1", features = ["candle", "metal"] }
```

### Feature Flags

| Feature | Description |
|---------|-------------|
| `candle` | Enable Candle backend (HuggingFace) |
| `metal` | Apple Silicon GPU acceleration |
| `cuda` | NVIDIA GPU acceleration |
| `inference-metal` | Full Metal inference stack (recommended for Mac) |
| `inference-cuda` | Full CUDA inference stack (recommended for NVIDIA) |
| `async-runtime` | Tokio async support |
| `wasm` | WebAssembly support |

## Architecture

```
+------------------------+
|     Application        |
+------------------------+
           |
+------------------------+
|    RuvLLM Backend      |
|  +------------------+  |
|  | Candle / mistral |  |
|  +------------------+  |
|           |            |
|  +------------------+  |
|  | SONA Learning    |  |
|  | - Instant (<1ms) |  |
|  | - Background     |  |
|  | - Deep           |  |
|  +------------------+  |
|           |            |
|  +------------------+  |
|  | NEON Kernels     |  |
|  | - Flash Attn     |  |
|  | - Paged Attn     |  |
|  | - RMSNorm/RoPE   |  |
|  +------------------+  |
+------------------------+
           |
+------------------------+
|  Metal GPU / CUDA      |
+------------------------+
```

## Supported Models

| Model Family | Sizes | Backend |
|--------------|-------|---------|
| Qwen 2.5 | 0.5B-72B | Candle |
| Mistral | 7B | Candle |
| Phi-3 | 3.8B | Candle |
| Llama 3.x | 8B-70B | Candle |

## Performance

Benchmarks on Apple M4 Pro (14-core):

| Model | Quantization | Prefill (tok/s) | Decode (tok/s) | Memory |
|-------|--------------|-----------------|----------------|--------|
| Qwen2.5-7B | Q4K | 2,400 | 85 | 4.2 GB |
| Qwen2.5-7B | Q8 | 1,800 | 62 | 7.8 GB |
| Mistral-7B | Q4K | 2,200 | 78 | 4.1 GB |
| Phi-3.8B | Q4K | 3,100 | 120 | 2.3 GB |

## MicroLoRA Real-Time Adaptation

RuvLLM supports per-request fine-tuning using MicroLoRA:

```rust
use ruvllm::lora::{MicroLoRA, MicroLoraConfig, AdaptFeedback};

// Create MicroLoRA adapter
let config = MicroLoraConfig::for_hidden_dim(4096);
let lora = MicroLoRA::new(config);

// Adapt on user feedback
let feedback = AdaptFeedback::from_quality(0.9);
lora.adapt(&input_embedding, feedback)?;

// Apply learned updates
lora.apply_updates(0.01); // learning rate
```

## SONA Learning Loops

Three-tier learning for continuous improvement:

1. **Instant Loop** (<1ms): MicroLoRA per-request adaptation
2. **Background Loop** (~100ms): Pattern consolidation, adapter merging
3. **Deep Loop** (minutes): Full fine-tuning, knowledge distillation

```rust
use ruvllm::optimization::SonaLlm;

let sona = SonaLlm::new(SonaLlmConfig::default());

// Record feedback for instant learning
let result = sona.instant_adapt("user query", "model response", 0.85);

// Periodically consolidate in background
if let Some(bg_result) = sona.maybe_background() {
    println!("Background consolidated {} samples", bg_result.samples_used);
}
```

## Two-Tier KV Cache

Memory-efficient caching with automatic tiering:

```rust
use ruvllm::kv_cache::{TwoTierKvCache, KvCacheConfig};

let config = KvCacheConfig {
    tail_length: 256,         // Recent tokens in FP16
    tail_precision: Precision::FP16,
    store_precision: Precision::Q4,  // Older tokens in Q4
    max_tokens: 4096,
    ..Default::default()
};

let cache = TwoTierKvCache::new(config);
cache.append(&keys, &values)?;

// Automatic migration from tail to quantized store
let stats = cache.stats();
println!("Tail: {} tokens, Store: {} tokens, Ratio: {:.2}x",
    stats.tail_tokens, stats.store_tokens, stats.compression_ratio);
```

## NEON-Optimized Attention

High-performance attention implementations:

```rust
use ruvllm::kernels::attention::{flash_attention_neon, AttentionConfig};

let config = AttentionConfig {
    num_heads: 32,
    num_kv_heads: 8,  // GQA: 4:1 ratio
    head_dim: 128,
    causal: true,
    ..Default::default()
};

// Flash Attention with online softmax
let output = flash_attention_neon(&query, &key, &value, scale, true);

// Grouped-Query Attention
let output = grouped_query_attention_neon(&queries, &keys, &values, &config);
```

## Error Handling

RuvLLM uses a comprehensive error hierarchy:

```rust
use ruvllm::error::{Result, RuvLLMError};

match backend.generate(prompt, params) {
    Ok(response) => println!("{}", response),
    Err(RuvLLMError::Model(e)) => eprintln!("Model error: {}", e),
    Err(RuvLLMError::OutOfMemory(e)) => eprintln!("OOM: {}", e),
    Err(RuvLLMError::Generation(e)) => eprintln!("Generation failed: {}", e),
    Err(e) => eprintln!("Error: {}", e),
}
```

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `RUVLLM_CACHE_DIR` | Model cache directory | `~/.cache/ruvllm` |
| `RUVLLM_LOG_LEVEL` | Logging level | `info` |
| `RUVLLM_METAL_DEVICE` | Metal device index | `0` |

### Model Configuration

```rust
let config = ModelConfig {
    max_context: 4096,
    use_flash_attention: true,
    quantization: Quantization::Q4K,
    kv_cache_config: KvCacheConfig::default(),
    ..Default::default()
};
```

## Benchmarks

Run benchmarks with:

```bash
# Attention benchmarks
cargo bench --bench attention_bench

# LoRA benchmarks
cargo bench --bench lora_bench

# End-to-end inference
cargo bench --bench e2e_bench
```

## Examples

See the `/examples` directory for:

- Basic inference
- Streaming generation
- MicroLoRA adaptation
- Multi-turn chat
- Speculative decoding

## Documentation

- [Architecture Guide](../../docs/ruvllm/ARCHITECTURE.md)
- [API Reference](../../docs/ruvllm/API_REFERENCE.md)
- [Fine-Tuning Guide](../../docs/ruvllm/FINE_TUNING.md)
- [Optimization Guide](../../docs/ruvllm/OPTIMIZATION.md)

## License

Apache-2.0 / MIT dual license.

## Contributing

Contributions welcome! Please see [CONTRIBUTING.md](../../CONTRIBUTING.md) for guidelines.

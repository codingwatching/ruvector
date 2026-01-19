//! # RuvLLM WASM - Browser-Compatible LLM Inference Runtime
//!
//! This crate provides WebAssembly bindings for the RuvLLM inference runtime,
//! enabling LLM inference directly in web browsers.
//!
//! ## Features
//!
//! - **KV Cache Management**: Two-tier KV cache with FP16 tail and quantized store
//! - **Memory Pooling**: Efficient buffer reuse for minimal allocation overhead
//! - **Chat Templates**: Support for Llama3, Mistral, Qwen, Phi, Gemma formats
//! - **TypeScript-Friendly**: All types have getter/setter methods for easy JS interop
//!
//! ## Quick Start (JavaScript)
//!
//! ```javascript
//! import init, { RuvLLMWasm, GenerateConfig, ChatMessageWasm, ChatTemplateWasm } from 'ruvllm-wasm';
//!
//! async function main() {
//!     // Initialize WASM module
//!     await init();
//!
//!     // Create inference engine
//!     const llm = new RuvLLMWasm();
//!     llm.initialize();
//!
//!     // Format a chat conversation
//!     const template = ChatTemplateWasm.llama3();
//!     const messages = [
//!         ChatMessageWasm.system("You are a helpful assistant."),
//!         ChatMessageWasm.user("What is WebAssembly?"),
//!     ];
//!     const prompt = template.format(messages);
//!
//!     console.log("Formatted prompt:", prompt);
//!
//!     // KV Cache management
//!     const kvCache = llm.getKvCache();
//!     if (kvCache) {
//!         const stats = kvCache.stats();
//!         console.log("Cache stats:", stats.toJson());
//!     }
//! }
//!
//! main();
//! ```
//!
//! ## Building
//!
//! ```bash
//! # Build for browser (bundler target)
//! wasm-pack build --target bundler
//!
//! # Build for Node.js
//! wasm-pack build --target nodejs
//!
//! # Build for web (no bundler)
//! wasm-pack build --target web
//! ```
//!
//! ## Architecture
//!
//! ```text
//! +-------------------+     +-------------------+
//! | JavaScript/TS     |---->| wasm-bindgen      |
//! | Application       |     | Bindings          |
//! +-------------------+     +-------------------+
//!                                   |
//!                                   v
//!                           +-------------------+
//!                           | RuvLLM Core       |
//!                           | (Rust WASM)       |
//!                           +-------------------+
//!                                   |
//!                                   v
//!                           +-------------------+
//!                           | Memory Pool       |
//!                           | KV Cache          |
//!                           | Tokenizer         |
//!                           +-------------------+
//! ```
//!
//! ## Memory Management
//!
//! The WASM module uses efficient memory management strategies:
//!
//! - **Arena Allocator**: O(1) bump allocation for inference temporaries
//! - **Buffer Pool**: Pre-allocated buffers in size classes (1KB-256KB)
//! - **Two-Tier KV Cache**: FP16 tail + Q4 quantized store
//!
//! ## Browser Compatibility
//!
//! Requires browsers with WebAssembly support:
//! - Chrome 57+
//! - Firefox 52+
//! - Safari 11+
//! - Edge 16+

#![warn(missing_docs)]
#![warn(clippy::all)]

use wasm_bindgen::prelude::*;

pub mod bindings;
pub mod utils;

// Re-export all bindings
pub use bindings::*;
pub use utils::{log, warn, error, now_ms, Timer, set_panic_hook};

/// Initialize the WASM module.
///
/// This should be called once at application startup to set up
/// panic hooks and any other initialization.
#[wasm_bindgen(start)]
pub fn init() {
    utils::set_panic_hook();
}

/// Perform a simple health check.
///
/// Returns true if the WASM module is functioning correctly.
#[wasm_bindgen(js_name = healthCheck)]
pub fn health_check() -> bool {
    // Try to create a small arena to verify memory allocation works
    let arena = ruvllm_integration::memory_pool::InferenceArena::new(1024);
    arena.capacity() == 1024
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_health_check() {
        // In non-WASM tests, this verifies the logic works
        let arena = ruvllm_integration::memory_pool::InferenceArena::new(1024);
        assert!(arena.capacity() >= 1024);
    }
}

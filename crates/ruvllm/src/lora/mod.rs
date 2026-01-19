//! MicroLoRA Fine-tuning Pipeline for Real-time Per-request Adaptation
//!
//! This module provides an ultra-lightweight LoRA implementation optimized for
//! real-time adaptation with minimal overhead (<1MB per adapter).
//!
//! ## Architecture
//!
//! ```text
//! +-------------------+     +-------------------+
//! | Request           |---->| MicroLoRA         |
//! | (input tensor)    |     | - Rank 1-2        |
//! +-------------------+     | - <1ms forward    |
//!                           | - Per-request     |
//!                           +--------+----------+
//!                                    |
//!                                    v (async feedback)
//!                           +--------+----------+
//!                           | Training Pipeline |
//!                           | - EWC++ regul.    |
//!                           | - Single-example  |
//!                           | - LR scheduling   |
//!                           +--------+----------+
//!                                    |
//!                                    v
//!                           +--------+----------+
//!                           | Adapter Manager   |
//!                           | - Hot-swapping    |
//!                           | - Composition     |
//!                           | - Persistence     |
//!                           +-------------------+
//! ```
//!
//! ## Features
//!
//! - **Ultra-lightweight**: Rank 1-2 adapters with <1MB memory footprint
//! - **Real-time**: Per-request adaptation with <1ms forward pass
//! - **EWC++ Integration**: Prevents catastrophic forgetting during adaptation
//! - **NEON/SIMD Optimized**: Hardware-accelerated forward and backward passes
//! - **Async Adaptation**: Non-blocking training with feedback loops
//! - **Hot-swapping**: Seamlessly switch adapters without model reload

pub mod adapter;
pub mod micro_lora;
pub mod training;

// Re-exports
pub use adapter::{
    AdapterComposer, AdapterHandle, AdapterPool, AdapterRegistry, CompositionStrategy,
};
pub use micro_lora::{
    AdaptFeedback, LoraAdapter, MicroLoRA, MicroLoraConfig, TargetModule,
};
pub use training::{
    EwcRegularizer, GradientAccumulator, LearningRateSchedule, TrainingConfig, TrainingPipeline,
};

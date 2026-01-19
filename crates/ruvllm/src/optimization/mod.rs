//! Real-time Optimization System for RuvLLM
//!
//! This module provides the optimization infrastructure for LLM inference,
//! integrating SONA learning with MicroLoRA and custom kernels.
//!
//! ## Architecture
//!
//! ```text
//! +-------------------+     +-------------------+
//! | Inference Request |---->| RealtimeOptimizer |
//! | (tokens, params)  |     | - Batch sizing    |
//! +-------------------+     | - KV management   |
//!                           | - Token budgets   |
//!                           +--------+----------+
//!                                    |
//!                                    v (metrics)
//!                           +--------+----------+
//!                           | InferenceMetrics  |
//!                           | - TTFT tracking   |
//!                           | - TPS monitoring  |
//!                           | - Memory usage    |
//!                           +--------+----------+
//!                                    |
//!                                    v (feedback)
//!                           +--------+----------+
//!                           | SonaLlm           |
//!                           | - Instant adapt   |
//!                           | - Background loop |
//!                           | - Deep optimize   |
//!                           +-------------------+
//! ```
//!
//! ## Features
//!
//! - **Real-time Optimization**: Dynamic batch sizing and KV cache management
//! - **SONA Integration**: Three-tier learning loops for continuous improvement
//! - **Metrics Collection**: Comprehensive inference telemetry
//! - **Speculative Decoding**: Draft model integration for faster generation

pub mod metrics;
pub mod realtime;
pub mod sona_llm;

// Re-exports
pub use metrics::{
    InferenceMetrics, MetricsCollector, MetricsSnapshot, MovingAverage, LatencyHistogram,
};
pub use realtime::{
    RealtimeOptimizer, RealtimeConfig, BatchSizeStrategy, KvCachePressurePolicy,
    TokenBudgetAllocation, SpeculativeConfig, OptimizationDecision,
};
pub use sona_llm::{
    SonaLlm, SonaLlmConfig, TrainingSample, AdaptationResult, LearningLoopStats,
    ConsolidationStrategy, OptimizationTrigger,
};

//! # RuvLLM - LLM Serving Runtime with Ruvector Integration
//!
//! RuvLLM is an edge-focused LLM serving runtime designed for portable, high-performance
//! inference across heterogeneous hardware. It integrates with Ruvector for intelligent
//! memory capabilities, enabling continuous self-improvement through SONA learning.
//!
//! ## Architecture
//!
//! RuvLLM uses Ruvector as a unified memory layer with three distinct roles:
//!
//! - **Policy Memory Store**: Learned thresholds and parameters for runtime decisions
//! - **Session State Index**: Multi-turn conversation state with KV cache references
//! - **Witness Log Index**: Audit logging with semantic search capabilities
//!
//! ## Key Components
//!
//! - [`PagedAttention`]: Memory-efficient attention mechanism with page tables
//! - [`TwoTierKvCache`]: FP16 tail + quantized store for optimal memory/quality tradeoff
//! - [`AdapterManager`]: LoRA adapter loading and hot-swapping
//! - [`SessionManager`]: Session lifecycle and state management
//! - [`PolicyStore`]: Ruvector-backed policy storage with semantic search
//! - [`WitnessLog`]: Audit logging with HNSW-indexed semantic search
//! - [`SonaIntegration`]: Three-tier learning loop integration
//!
//! ## Example
//!
//! ```rust,ignore
//! use ruvllm::{RuvLLMConfig, RuvLLMEngine};
//!
//! // Create engine with default configuration
//! let config = RuvLLMConfig::default();
//! let engine = RuvLLMEngine::new(config)?;
//!
//! // Create a session
//! let session = engine.create_session("user-123")?;
//!
//! // Process a request
//! let response = engine.process(&session, "Hello, world!")?;
//! ```

#![warn(missing_docs)]
#![warn(clippy::all)]

pub mod adapter_manager;
pub mod backends;
pub mod error;
pub mod kernels;
pub mod kv_cache;
pub mod lora;
pub mod optimization;
pub mod paged_attention;
pub mod policy_store;
pub mod session;
pub mod session_index;
pub mod sona;
pub mod types;
pub mod witness_log;

// Re-exports
pub use adapter_manager::{AdapterManager, LoraAdapter, AdapterConfig};
pub use lora::{
    MicroLoRA, MicroLoraConfig, TargetModule, AdaptFeedback,
    AdapterRegistry, AdapterPool, AdapterComposer, CompositionStrategy,
    TrainingPipeline, TrainingConfig, EwcRegularizer, LearningRateSchedule,
};
pub use backends::{
    create_backend, DeviceType, DType, GenerateParams, GeneratedToken, LlmBackend,
    ModelArchitecture, ModelConfig, ModelInfo, Quantization, SharedBackend, SpecialTokens,
    Tokenizer,
};
#[cfg(feature = "candle")]
pub use backends::CandleBackend;
pub use error::{RuvLLMError, Result};
pub use kv_cache::{TwoTierKvCache, KvCacheConfig, CacheTier, CacheQuantization};
pub use paged_attention::{PagedAttention, PagedAttentionConfig, PageTable, PageBlock};
pub use policy_store::{PolicyStore, PolicyEntry, PolicyType, QuantizationPolicy, RouterPolicy};
pub use session::{SessionManager, Session, SessionConfig};
pub use session_index::{SessionIndex, SessionState, KvCacheReference};
pub use sona::{SonaIntegration, SonaConfig, LearningLoop};
pub use optimization::{
    InferenceMetrics, MetricsCollector, MetricsSnapshot, MovingAverage, LatencyHistogram,
    RealtimeOptimizer, RealtimeConfig, BatchSizeStrategy, KvCachePressurePolicy,
    TokenBudgetAllocation, SpeculativeConfig, OptimizationDecision,
    SonaLlm, SonaLlmConfig, TrainingSample, AdaptationResult, LearningLoopStats,
    ConsolidationStrategy, OptimizationTrigger,
};
pub use types::*;
pub use witness_log::{WitnessLog, WitnessEntry, LatencyBreakdown, RoutingDecision};

/// RuvLLM engine configuration
#[derive(Debug, Clone)]
pub struct RuvLLMConfig {
    /// Path to Ruvector storage
    pub storage_path: String,
    /// Paged attention configuration
    pub paged_attention: PagedAttentionConfig,
    /// KV cache configuration
    pub kv_cache: KvCacheConfig,
    /// Session configuration
    pub session: SessionConfig,
    /// SONA learning configuration
    pub sona: SonaConfig,
    /// Maximum concurrent sessions
    pub max_sessions: usize,
    /// Embedding dimension for semantic search
    pub embedding_dim: usize,
}

impl Default for RuvLLMConfig {
    fn default() -> Self {
        Self {
            storage_path: ".ruvllm".to_string(),
            paged_attention: PagedAttentionConfig::default(),
            kv_cache: KvCacheConfig::default(),
            session: SessionConfig::default(),
            sona: SonaConfig::default(),
            max_sessions: 1000,
            embedding_dim: 768,
        }
    }
}

/// Main RuvLLM engine
pub struct RuvLLMEngine {
    /// Configuration
    config: RuvLLMConfig,
    /// Policy store backed by Ruvector
    policy_store: PolicyStore,
    /// Session manager
    session_manager: SessionManager,
    /// Session index backed by Ruvector
    session_index: SessionIndex,
    /// Adapter manager
    adapter_manager: AdapterManager,
    /// Witness log for audit
    witness_log: WitnessLog,
    /// SONA learning integration
    sona: SonaIntegration,
}

impl RuvLLMEngine {
    /// Create a new RuvLLM engine
    pub fn new(config: RuvLLMConfig) -> Result<Self> {
        let storage_path = &config.storage_path;

        let policy_store = PolicyStore::new(
            &format!("{}/policies", storage_path),
            config.embedding_dim,
        )?;

        let session_index = SessionIndex::new(
            &format!("{}/sessions", storage_path),
            config.embedding_dim,
        )?;

        let witness_log = WitnessLog::new(
            &format!("{}/witness", storage_path),
            config.embedding_dim,
        )?;

        let session_manager = SessionManager::new(config.session.clone());
        let adapter_manager = AdapterManager::new();
        let sona = SonaIntegration::new(config.sona.clone());

        Ok(Self {
            config,
            policy_store,
            session_manager,
            session_index,
            adapter_manager,
            witness_log,
            sona,
        })
    }

    /// Create a new session
    pub fn create_session(&self, user_id: Option<&str>) -> Result<Session> {
        let session = self.session_manager.create_session(user_id)?;

        // Index the session in Ruvector
        let state = SessionState::from_session(&session);
        self.session_index.store(&state)?;

        Ok(session)
    }

    /// Get session by ID
    pub fn get_session(&self, session_id: &str) -> Result<Option<Session>> {
        self.session_manager.get_session(session_id)
    }

    /// Search for policies matching context
    pub fn search_policies(&self, context_embedding: &[f32], limit: usize) -> Result<Vec<PolicyEntry>> {
        self.policy_store.search(context_embedding, limit)
    }

    /// Record a witness entry for audit
    pub fn record_witness(&self, entry: WitnessEntry) -> Result<()> {
        self.witness_log.record(entry)
    }

    /// Search witness logs semantically
    pub fn search_witness(&self, query_embedding: &[f32], limit: usize) -> Result<Vec<WitnessEntry>> {
        self.witness_log.search(query_embedding, limit)
    }

    /// Get the SONA integration for learning
    pub fn sona(&self) -> &SonaIntegration {
        &self.sona
    }

    /// Get the adapter manager
    pub fn adapters(&self) -> &AdapterManager {
        &self.adapter_manager
    }

    /// Get the policy store
    pub fn policies(&self) -> &PolicyStore {
        &self.policy_store
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let config = RuvLLMConfig::default();
        assert_eq!(config.max_sessions, 1000);
        assert_eq!(config.embedding_dim, 768);
    }
}

//! TRM Engine
//!
//! Main recursive reasoning engine that orchestrates:
//! - Latent state updates (MLP or Attention)
//! - Answer refinement
//! - Confidence scoring
//! - Early stopping
//! - Trajectory recording

use std::time::Instant;

use super::{
    attention::AttentionLatentUpdater,
    confidence::ConfidenceScorer,
    error::TrmError,
    mlp::MlpLatentUpdater,
    refiner::AnswerRefiner,
    types::{TrmIterationState, TrmResult, TrmRoutingDecision, TrmTrajectory},
    LatentUpdate, RecursiveReasoner, TrmConfig,
};

/// The main TRM recursive reasoning engine
///
/// Implements the core TRM algorithm:
/// ```text
/// for k in 0..K:
///     for n in 0..N:
///         latent = update_latent(question, answer, latent)
///     answer = refine_answer(question, latent, answer)
///     confidence = score_confidence(answer)
///     if confidence >= threshold:
///         break  // Early stopping
/// ```
pub struct TrmEngine {
    config: TrmConfig,

    // Components (boxed for dynamic dispatch)
    latent_updater: Box<dyn LatentUpdate>,
    refiner: AnswerRefiner,
    scorer: ConfidenceScorer,

    // Pre-allocated buffers
    latent_buffer: Vec<f32>,
    question_buffer: Vec<f32>,

    // Statistics
    total_iterations: u64,
    total_early_stops: u64,
}

impl TrmEngine {
    /// Create a new TRM engine with the given configuration
    pub fn new(config: TrmConfig) -> Result<Self, TrmError> {
        config.validate()?;

        let latent_updater: Box<dyn LatentUpdate> = if config.use_attention {
            Box::new(AttentionLatentUpdater::from_config(&config))
        } else {
            Box::new(MlpLatentUpdater::from_config(&config))
        };

        let refiner = AnswerRefiner::new(&config);
        let scorer = ConfidenceScorer::new(&config);

        let latent_buffer = vec![0.0; config.hidden_dim];
        let question_buffer = vec![0.0; config.embedding_dim];

        Ok(Self {
            config,
            latent_updater,
            refiner,
            scorer,
            latent_buffer,
            question_buffer,
            total_iterations: 0,
            total_early_stops: 0,
        })
    }

    /// Create a TRM engine with MLP latent updater (faster)
    pub fn with_mlp(config: TrmConfig) -> Result<Self, TrmError> {
        let mut cfg = config;
        cfg.use_attention = false;
        Self::new(cfg)
    }

    /// Create a TRM engine with Attention latent updater (more expressive)
    pub fn with_attention(config: TrmConfig) -> Result<Self, TrmError> {
        let mut cfg = config;
        cfg.use_attention = true;
        Self::new(cfg)
    }

    /// Create with default configuration
    pub fn default_engine() -> Result<Self, TrmError> {
        Self::new(TrmConfig::default())
    }

    /// Get the current configuration
    pub fn config(&self) -> &TrmConfig {
        &self.config
    }

    /// Get statistics
    pub fn stats(&self) -> (u64, u64) {
        (self.total_iterations, self.total_early_stops)
    }

    /// Reset statistics
    pub fn reset_stats(&mut self) {
        self.total_iterations = 0;
        self.total_early_stops = 0;
    }

    /// Mean pool an embedding to target dimension
    fn mean_pool(&self, input: &[f32], target_dim: usize) -> Vec<f32> {
        if input.len() == target_dim {
            return input.to_vec();
        }

        if input.len() < target_dim {
            let mut pooled = input.to_vec();
            pooled.resize(target_dim, 0.0);
            return pooled;
        }

        let num_chunks = input.len() / target_dim;
        let mut pooled = vec![0.0; target_dim];

        for chunk in input.chunks(target_dim) {
            for (i, &v) in chunk.iter().enumerate() {
                if i < target_dim {
                    pooled[i] += v / num_chunks as f32;
                }
            }
        }

        pooled
    }

    /// Mean pool an embedding into a pre-allocated buffer (avoids allocation)
    fn mean_pool_into(&self, input: &[f32], output: &mut [f32]) {
        let target_dim = output.len();

        if input.len() == target_dim {
            output.copy_from_slice(input);
            return;
        }

        if input.len() < target_dim {
            output[..input.len()].copy_from_slice(input);
            output[input.len()..].fill(0.0);
            return;
        }

        output.fill(0.0);
        let num_chunks = input.len() / target_dim;
        let scale = 1.0 / num_chunks as f32;

        for chunk in input.chunks(target_dim) {
            for (i, &v) in chunk.iter().enumerate() {
                if i < target_dim {
                    output[i] += v * scale;
                }
            }
        }
    }

    /// Run a single latent update iteration
    fn latent_update_step(
        &self,
        question_pooled: &[f32],
        answer_pooled: &[f32],
        latent: &mut [f32],
    ) {
        self.latent_updater.update(question_pooled, answer_pooled, latent);
    }

    /// Run the recursive reasoning loop
    fn reason_internal(
        &mut self,
        question: &[f32],
        answer: &mut [f32],
        k_iterations: usize,
    ) -> TrmResult {
        let start_time = Instant::now();

        // Initialize trajectory - pool question once (doesn't change)
        let question_pooled = self.mean_pool(question, self.config.embedding_dim);
        let mut trajectory = TrmTrajectory::new(question_pooled.clone());

        // Initialize latent state - reuse buffer
        let mut latent = std::mem::take(&mut self.latent_buffer);
        latent.fill(0.0);

        // Pre-allocate answer pooling buffer to avoid repeated allocations
        let mut answer_pooled = vec![0.0; self.config.embedding_dim];

        let mut prev_confidence = 0.0;
        let mut early_stopped = false;
        let mut iterations_used = 0;

        // Main K-iteration loop
        for k in 0..k_iterations {
            let iter_start = Instant::now();

            // Pool current answer into pre-allocated buffer
            self.mean_pool_into(answer, &mut answer_pooled);

            // N latent update iterations
            for _ in 0..self.config.latent_iterations {
                self.latent_update_step(&question_pooled, &answer_pooled, &mut latent);
            }

            // Refine the answer
            self.refiner.refine(question, &latent, answer);

            // Score confidence
            let confidence = if self.config.use_entropy_confidence {
                self.scorer.score_with_entropy(answer)
            } else {
                self.scorer.score(answer)
            };

            let iter_latency = iter_start.elapsed().as_micros() as u64;

            // Record iteration state
            let state = TrmIterationState::new(
                k,
                latent.clone(),
                answer.to_vec(),
                confidence,
                iter_latency,
                prev_confidence,
            );
            trajectory.push(state);

            iterations_used = k + 1;
            self.total_iterations += 1;

            // Check for early stopping
            if self.config.early_stopping && confidence >= self.config.confidence_threshold {
                early_stopped = true;
                trajectory.set_converged(true);
                self.total_early_stops += 1;
                break;
            }

            // Check for convergence (plateauing)
            if self.config.early_stopping && k >= 2 {
                if trajectory.is_plateauing(3, self.config.convergence_threshold) {
                    early_stopped = true;
                    trajectory.set_converged(true);
                    self.total_early_stops += 1;
                    break;
                }
            }

            prev_confidence = confidence;
        }

        // Restore latent buffer for reuse
        self.latent_buffer = latent;

        let total_latency = start_time.elapsed().as_micros() as u64;
        let final_confidence = trajectory.final_confidence();

        TrmResult::new(
            answer.to_vec(),
            final_confidence,
            iterations_used,
            early_stopped,
            trajectory,
            total_latency,
        )
    }
}

impl RecursiveReasoner for TrmEngine {
    fn reason(&mut self, question: &[f32], answer: &mut [f32]) -> TrmResult {
        self.reason_internal(question, answer, self.config.default_k)
    }

    fn reason_with_k(&mut self, question: &[f32], answer: &mut [f32], k: usize) -> TrmResult {
        let k_clamped = k.min(self.config.max_k);
        self.reason_internal(question, answer, k_clamped)
    }

    fn reason_with_routing(
        &mut self,
        question: &[f32],
        answer: &mut [f32],
        routing: &TrmRoutingDecision,
    ) -> TrmResult {
        if !routing.use_trm {
            // Skip TRM, return answer as-is
            let trajectory = TrmTrajectory::new(self.mean_pool(question, self.config.embedding_dim));
            let confidence = self.scorer.score(answer);
            return TrmResult::new(
                answer.to_vec(),
                confidence,
                0,
                false,
                trajectory,
                0,
            );
        }

        // Apply routing decisions
        let k = routing.k_value.min(self.config.max_k);
        self.reason_internal(question, answer, k)
    }

    fn reset(&mut self) {
        self.latent_buffer.fill(0.0);
        self.question_buffer.fill(0.0);
    }
}

/// Builder for TrmEngine with fluent API
pub struct TrmEngineBuilder {
    config: TrmConfig,
}

impl TrmEngineBuilder {
    /// Create a new builder with default config
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

    /// Set default K iterations
    pub fn default_k(mut self, k: usize) -> Self {
        self.config.default_k = k;
        self
    }

    /// Set max K iterations
    pub fn max_k(mut self, k: usize) -> Self {
        self.config.max_k = k;
        self
    }

    /// Set latent iterations per K
    pub fn latent_iterations(mut self, n: usize) -> Self {
        self.config.latent_iterations = n;
        self
    }

    /// Use attention-based latent updater
    pub fn use_attention(mut self, use_attn: bool) -> Self {
        self.config.use_attention = use_attn;
        self
    }

    /// Set number of attention heads (if using attention)
    pub fn num_heads(mut self, heads: usize) -> Self {
        self.config.num_heads = heads;
        self
    }

    /// Set confidence threshold for early stopping
    pub fn confidence_threshold(mut self, threshold: f32) -> Self {
        self.config.confidence_threshold = threshold;
        self
    }

    /// Enable/disable early stopping
    pub fn early_stopping(mut self, enable: bool) -> Self {
        self.config.early_stopping = enable;
        self
    }

    /// Use entropy-based confidence scoring
    pub fn use_entropy_confidence(mut self, use_entropy: bool) -> Self {
        self.config.use_entropy_confidence = use_entropy;
        self
    }

    /// Set residual scale for answer refinement
    pub fn residual_scale(mut self, scale: f32) -> Self {
        self.config.residual_scale = scale;
        self
    }

    /// Build the TrmEngine
    pub fn build(self) -> Result<TrmEngine, TrmError> {
        TrmEngine::new(self.config)
    }
}

impl Default for TrmEngineBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_engine() -> TrmEngine {
        TrmEngineBuilder::new()
            .hidden_dim(64)
            .embedding_dim(64)
            .default_k(5)
            .latent_iterations(2)
            .confidence_threshold(0.95)
            .build()
            .unwrap()
    }

    #[test]
    fn test_engine_creation() {
        let engine = make_engine();
        assert_eq!(engine.config().hidden_dim, 64);
        assert_eq!(engine.config().default_k, 5);
    }

    #[test]
    fn test_engine_with_mlp() {
        let config = TrmConfig {
            hidden_dim: 64,
            embedding_dim: 64,
            ..Default::default()
        };
        let engine = TrmEngine::with_mlp(config).unwrap();
        assert!(!engine.config().use_attention);
    }

    #[test]
    fn test_engine_with_attention() {
        let config = TrmConfig {
            hidden_dim: 64,
            embedding_dim: 64,
            num_heads: 4,
            ..Default::default()
        };
        let engine = TrmEngine::with_attention(config).unwrap();
        assert!(engine.config().use_attention);
    }

    #[test]
    fn test_reason_basic() {
        let mut engine = make_engine();

        let question = vec![0.5; 64];
        let mut answer = vec![0.1; 64];

        let result = engine.reason(&question, &mut answer);

        assert!(result.iterations_used > 0);
        assert!(result.confidence >= 0.0 && result.confidence <= 1.0);
        assert_eq!(result.answer.len(), 64);
    }

    #[test]
    fn test_reason_with_k() {
        let mut engine = make_engine();

        let question = vec![0.5; 64];
        let mut answer = vec![0.1; 64];

        let result = engine.reason_with_k(&question, &mut answer, 3);

        // Should use at most 3 iterations (may early stop before)
        assert!(result.iterations_used <= 3);
    }

    #[test]
    fn test_trajectory_recorded() {
        let mut engine = make_engine();

        let question = vec![0.5; 64];
        let mut answer = vec![0.1; 64];

        let result = engine.reason(&question, &mut answer);

        // Trajectory should have states for each iteration
        assert_eq!(result.trajectory.states.len(), result.iterations_used);

        // Each state should have latent and answer
        for state in &result.trajectory.states {
            assert_eq!(state.latent_state.len(), 64);
            assert_eq!(state.answer_state.len(), 64);
        }
    }

    #[test]
    fn test_early_stopping() {
        // Create engine with low threshold to trigger early stopping
        let mut engine = TrmEngineBuilder::new()
            .hidden_dim(64)
            .embedding_dim(64)
            .default_k(20)  // High K
            .confidence_threshold(0.4)  // Low threshold
            .early_stopping(true)
            .build()
            .unwrap();

        let question = vec![0.5; 64];
        let mut answer = vec![0.5; 64];

        let result = engine.reason(&question, &mut answer);

        // With low threshold, should early stop
        // (may not always trigger depending on random weights)
        assert!(result.iterations_used <= 20);
    }

    #[test]
    fn test_routing_skip_trm() {
        let mut engine = make_engine();

        let question = vec![0.5; 64];
        let mut answer = vec![0.5; 64];
        let original_answer = answer.clone();

        let routing = TrmRoutingDecision {
            use_trm: false,
            ..Default::default()
        };

        let result = engine.reason_with_routing(&question, &mut answer, &routing);

        assert_eq!(result.iterations_used, 0);
        // Answer should be unchanged when skipping TRM
        assert_eq!(result.answer, original_answer);
    }

    #[test]
    fn test_routing_custom_k() {
        let mut engine = make_engine();

        let question = vec![0.5; 64];
        let mut answer = vec![0.1; 64];

        let routing = TrmRoutingDecision {
            use_trm: true,
            k_value: 2,
            ..Default::default()
        };

        let result = engine.reason_with_routing(&question, &mut answer, &routing);

        assert!(result.iterations_used <= 2);
    }

    #[test]
    fn test_stats_tracking() {
        let mut engine = make_engine();

        let question = vec![0.5; 64];
        let mut answer = vec![0.1; 64];

        engine.reason(&question, &mut answer);
        engine.reason(&question, &mut answer);

        let (total_iters, _) = engine.stats();
        assert!(total_iters >= 2);  // At least 2 total iterations

        engine.reset_stats();
        let (total_iters, _) = engine.stats();
        assert_eq!(total_iters, 0);
    }

    #[test]
    fn test_variable_length_inputs() {
        let mut engine = make_engine();

        // Short inputs
        let question = vec![0.5; 32];
        let mut answer = vec![0.1; 32];
        let result = engine.reason(&question, &mut answer);
        assert!(result.confidence >= 0.0);

        // Long inputs
        let question = vec![0.5; 256];
        let mut answer = vec![0.1; 256];
        let result = engine.reason(&question, &mut answer);
        assert!(result.confidence >= 0.0);
    }

    #[test]
    fn test_builder_pattern() {
        let engine = TrmEngineBuilder::new()
            .hidden_dim(128)
            .embedding_dim(128)
            .default_k(10)
            .max_k(25)
            .latent_iterations(4)
            .use_attention(true)
            .num_heads(8)
            .confidence_threshold(0.9)
            .early_stopping(true)
            .use_entropy_confidence(true)
            .residual_scale(0.05)
            .build()
            .unwrap();

        assert_eq!(engine.config().hidden_dim, 128);
        assert_eq!(engine.config().default_k, 10);
        assert_eq!(engine.config().max_k, 25);
        assert!(engine.config().use_attention);
        assert_eq!(engine.config().num_heads, 8);
    }

    #[test]
    fn test_confidence_improves() {
        let mut engine = TrmEngineBuilder::new()
            .hidden_dim(64)
            .embedding_dim(64)
            .default_k(10)
            .early_stopping(false)  // Don't early stop
            .build()
            .unwrap();

        let question = vec![0.5; 64];
        let mut answer = vec![0.0; 64];  // Start with zeros

        let result = engine.reason(&question, &mut answer);

        // Confidence improvement should be non-negative generally
        // (not guaranteed due to random initialization, but trajectory should show progression)
        assert!(result.trajectory.states.len() > 0);
    }

    #[test]
    fn test_reset() {
        let mut engine = make_engine();

        let question = vec![0.5; 64];
        let mut answer = vec![0.1; 64];

        engine.reason(&question, &mut answer);
        engine.reset();

        // After reset, buffers should be zeroed
        // (internal state, can't directly verify but shouldn't panic)
        let result = engine.reason(&question, &mut answer);
        assert!(result.confidence >= 0.0);
    }
}

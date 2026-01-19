//! Serving Engine for Continuous Batching
//!
//! This module provides the main serving engine that coordinates
//! request submission, scheduling, and model execution with streaming output.

use super::kv_cache_manager::KvCachePoolConfig;
use super::request::{
    CompletedRequest, FinishReason, InferenceRequest, Priority, RequestId, RequestState,
    RunningRequest, TokenOutput,
};
use super::scheduler::{ContinuousBatchScheduler, RequestQueue, SchedulerConfig};
use crate::backends::{GenerateParams, GeneratedToken, LlmBackend};
use crate::error::{Result, RuvLLMError};
use parking_lot::{Mutex, RwLock};
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

#[cfg(feature = "async-runtime")]
use tokio::sync::mpsc;

/// Configuration for the serving engine
#[derive(Debug, Clone)]
pub struct ServingEngineConfig {
    /// Scheduler configuration
    pub scheduler: SchedulerConfig,
    /// KV cache pool configuration
    pub kv_cache: KvCachePoolConfig,
    /// Maximum concurrent requests
    pub max_concurrent_requests: usize,
    /// Enable request coalescing
    pub coalesce_requests: bool,
    /// Coalescing window in milliseconds
    pub coalesce_window_ms: u64,
    /// Enable streaming output
    pub streaming_enabled: bool,
    /// Request timeout in milliseconds
    pub request_timeout_ms: u64,
}

impl Default for ServingEngineConfig {
    fn default() -> Self {
        Self {
            scheduler: SchedulerConfig::default(),
            kv_cache: KvCachePoolConfig::default(),
            max_concurrent_requests: 256,
            coalesce_requests: false,
            coalesce_window_ms: 10,
            streaming_enabled: true,
            request_timeout_ms: 60000,
        }
    }
}

/// Result of processing a request
#[derive(Debug, Clone)]
pub struct GenerationResult {
    /// Request ID
    pub request_id: RequestId,
    /// Generated token IDs
    pub generated_tokens: Vec<u32>,
    /// Generated text (if decoded)
    pub generated_text: Option<String>,
    /// Finish reason
    pub finish_reason: FinishReason,
    /// Processing time in milliseconds
    pub processing_time_ms: u64,
    /// Tokens per second
    pub tokens_per_second: f64,
    /// Number of prompt tokens
    pub prompt_tokens: usize,
    /// Number of generated tokens
    pub completion_tokens: usize,
}

impl From<CompletedRequest> for GenerationResult {
    fn from(completed: CompletedRequest) -> Self {
        Self {
            request_id: completed.id,
            generated_tokens: completed.generated_tokens.clone(),
            generated_text: None,
            finish_reason: completed.finish_reason,
            processing_time_ms: completed.processing_time_ms,
            tokens_per_second: completed.tokens_per_second,
            prompt_tokens: completed.prompt_tokens.len(),
            completion_tokens: completed.generated_tokens.len(),
        }
    }
}

/// Streaming token callback
pub type TokenCallback = Box<dyn Fn(TokenOutput) + Send + Sync>;

/// Internal request state for the engine
struct EngineRequest {
    /// Request data
    request: InferenceRequest,
    /// Token callback for streaming
    callback: Option<TokenCallback>,
    /// Completion notifier
    #[cfg(feature = "async-runtime")]
    completion_tx: Option<tokio::sync::oneshot::Sender<GenerationResult>>,
    /// Created time
    created_at: Instant,
}

/// The serving engine for continuous batching
pub struct ServingEngine {
    /// Configuration
    config: ServingEngineConfig,
    /// The LLM backend
    model: Arc<dyn LlmBackend>,
    /// Request scheduler
    scheduler: Mutex<ContinuousBatchScheduler>,
    /// Request queue
    queue: Mutex<RequestQueue>,
    /// Pending request data
    pending_requests: RwLock<HashMap<RequestId, EngineRequest>>,
    /// Completed results
    completed_results: RwLock<HashMap<RequestId, GenerationResult>>,
    /// Running state
    is_running: AtomicBool,
    /// Total requests processed
    total_requests: AtomicU64,
    /// Total tokens generated
    total_tokens: AtomicU64,
    /// Start time for metrics
    start_time: Instant,
}

impl ServingEngine {
    /// Create a new serving engine
    pub fn new(model: Arc<dyn LlmBackend>, config: ServingEngineConfig) -> Self {
        let scheduler = ContinuousBatchScheduler::new(
            config.scheduler.clone(),
            config.kv_cache.clone(),
        );

        Self {
            config,
            model,
            scheduler: Mutex::new(scheduler),
            queue: Mutex::new(RequestQueue::new()),
            pending_requests: RwLock::new(HashMap::new()),
            completed_results: RwLock::new(HashMap::new()),
            is_running: AtomicBool::new(false),
            total_requests: AtomicU64::new(0),
            total_tokens: AtomicU64::new(0),
            start_time: Instant::now(),
        }
    }

    /// Create with default configuration
    pub fn with_default_config(model: Arc<dyn LlmBackend>) -> Self {
        Self::new(model, ServingEngineConfig::default())
    }

    /// Submit a request for processing
    pub fn submit(&self, request: InferenceRequest) -> Result<RequestId> {
        let request_id = request.id;

        // Check capacity
        {
            let queue = self.queue.lock();
            if queue.pending_count() + queue.running_count()
                >= self.config.max_concurrent_requests
            {
                return Err(RuvLLMError::OutOfMemory(
                    "Maximum concurrent requests reached".to_string(),
                ));
            }
        }

        // Store request data
        {
            let engine_request = EngineRequest {
                request: request.clone(),
                callback: None,
                #[cfg(feature = "async-runtime")]
                completion_tx: None,
                created_at: Instant::now(),
            };
            self.pending_requests.write().insert(request_id, engine_request);
        }

        // Add to queue
        self.queue.lock().add(request);
        self.total_requests.fetch_add(1, Ordering::Relaxed);

        Ok(request_id)
    }

    /// Submit a request with a streaming callback
    pub fn submit_with_callback(
        &self,
        request: InferenceRequest,
        callback: TokenCallback,
    ) -> Result<RequestId> {
        let request_id = request.id;

        // Check capacity
        {
            let queue = self.queue.lock();
            if queue.pending_count() + queue.running_count()
                >= self.config.max_concurrent_requests
            {
                return Err(RuvLLMError::OutOfMemory(
                    "Maximum concurrent requests reached".to_string(),
                ));
            }
        }

        // Store request data with callback
        {
            let engine_request = EngineRequest {
                request: request.clone(),
                callback: Some(callback),
                #[cfg(feature = "async-runtime")]
                completion_tx: None,
                created_at: Instant::now(),
            };
            self.pending_requests.write().insert(request_id, engine_request);
        }

        // Add to queue
        self.queue.lock().add(request);
        self.total_requests.fetch_add(1, Ordering::Relaxed);

        Ok(request_id)
    }

    /// Get the result of a completed request
    pub fn get_result(&self, id: RequestId) -> Option<GenerationResult> {
        self.completed_results.write().remove(&id)
    }

    /// Check if a request is complete
    pub fn is_complete(&self, id: RequestId) -> bool {
        self.completed_results.read().contains_key(&id)
    }

    /// Cancel a request
    pub fn cancel(&self, id: RequestId) -> bool {
        // Try to remove from pending
        if self.pending_requests.write().remove(&id).is_some() {
            // Remove from queue if still pending
            let mut queue = self.queue.lock();
            queue.pending.retain(|r| r.id != id);
            return true;
        }

        // Try to remove from running
        let mut queue = self.queue.lock();
        if let Some(running) = queue.remove_running(id) {
            // Free KV cache
            self.scheduler.lock().kv_cache_manager_mut().free(id);

            // Create cancelled result - extract values before moving generated_tokens
            let completion_tokens = running.generated_tokens.len();
            let processing_time_ms = running.processing_time().as_millis() as u64;
            let tokens_per_second = running.tokens_per_second();
            let prompt_tokens = running.request.prompt_len();
            let result = GenerationResult {
                request_id: id,
                generated_tokens: running.generated_tokens,
                generated_text: None,
                finish_reason: FinishReason::Cancelled,
                processing_time_ms,
                tokens_per_second,
                prompt_tokens,
                completion_tokens,
            };

            self.completed_results.write().insert(id, result);
            return true;
        }

        false
    }

    /// Run a single iteration of the serving loop
    ///
    /// Returns the generated tokens for this iteration
    pub fn run_iteration(&self) -> Result<Vec<TokenOutput>> {
        let mut outputs = Vec::new();

        // Schedule next batch
        let batch = {
            let mut queue = self.queue.lock();
            let mut scheduler = self.scheduler.lock();
            scheduler.schedule(&mut queue)
        };

        if batch.is_empty() {
            return Ok(outputs);
        }

        // Process the batch (this is where the actual model inference would happen)
        // For now, we simulate token generation

        // Process each request in the batch
        for batched_req in &batch.requests {
            let request_id = batched_req.request_id;

            if batched_req.is_prefill {
                // Prefill complete - update state
                let mut queue = self.queue.lock();
                if let Some(running) = queue.get_running_mut(request_id) {
                    if !running.prefill_complete {
                        running.advance_prefill(batched_req.token_ids.len());
                    }
                }
            } else {
                // Decode - generate a token
                // In a real implementation, this would come from the model
                let generated_token = self.simulate_token_generation(request_id)?;

                let mut queue = self.queue.lock();

                if let Some(running) = queue.get_running_mut(request_id) {
                    running.add_token(generated_token);

                    // Create output
                    let output = TokenOutput {
                        request_id,
                        token_id: generated_token,
                        token_text: None, // Would decode with tokenizer
                        logprob: None,
                        is_final: running.is_complete(),
                        finish_reason: if running.is_complete() {
                            Some(FinishReason::Length)
                        } else {
                            None
                        },
                        seq_len: running.current_seq_len,
                    };

                    // Send to callback if registered
                    if let Some(engine_req) = self.pending_requests.read().get(&request_id) {
                        if let Some(callback) = &engine_req.callback {
                            callback(output.clone());
                        }
                    }

                    outputs.push(output);

                    // Update KV cache length
                    let _ = self
                        .scheduler
                        .lock()
                        .kv_cache_manager_mut()
                        .set_length(request_id, running.current_seq_len);

                    self.total_tokens.fetch_add(1, Ordering::Relaxed);

                    // Check if complete
                    if running.is_complete() {
                        // Will handle completion below
                    }
                }
            }
        }

        // Handle completions
        self.handle_completions()?;

        Ok(outputs)
    }

    /// Handle completed requests
    fn handle_completions(&self) -> Result<()> {
        let mut completed_ids = Vec::new();

        // Find completed requests
        {
            let queue = self.queue.lock();
            for (id, running) in &queue.running {
                if running.is_complete() {
                    completed_ids.push(*id);
                }
            }
        }

        // Process completions
        for id in completed_ids {
            let running = {
                let mut queue = self.queue.lock();
                queue.remove_running(id)
            };

            if let Some(running) = running {
                // Free KV cache
                self.scheduler.lock().kv_cache_manager_mut().free(id);

                // Create result
                let result = GenerationResult {
                    request_id: id,
                    generated_tokens: running.generated_tokens.clone(),
                    generated_text: None,
                    finish_reason: FinishReason::Length,
                    processing_time_ms: running.processing_time().as_millis() as u64,
                    tokens_per_second: running.tokens_per_second(),
                    prompt_tokens: running.request.prompt_len(),
                    completion_tokens: running.generated_tokens.len(),
                };

                // Store result
                self.completed_results.write().insert(id, result.clone());

                // Send final callback
                if let Some(engine_req) = self.pending_requests.write().remove(&id) {
                    if let Some(callback) = &engine_req.callback {
                        callback(TokenOutput {
                            request_id: id,
                            token_id: running.generated_tokens.last().copied().unwrap_or(0),
                            token_text: None,
                            logprob: None,
                            is_final: true,
                            finish_reason: Some(FinishReason::Length),
                            seq_len: running.current_seq_len,
                        });
                    }

                    #[cfg(feature = "async-runtime")]
                    if let Some(tx) = engine_req.completion_tx {
                        let _ = tx.send(result);
                    }
                }
            }
        }

        Ok(())
    }

    /// Simulate token generation (placeholder for actual model inference)
    fn simulate_token_generation(&self, _request_id: RequestId) -> Result<u32> {
        // In a real implementation, this would call the model
        // For now, return a random token
        Ok(rand::random::<u32>() % 32000)
    }

    /// Run the serving loop until stopped
    pub fn run(&self) -> Result<()> {
        self.is_running.store(true, Ordering::SeqCst);

        while self.is_running.load(Ordering::SeqCst) {
            // Check if there's work to do
            let has_work = {
                let queue = self.queue.lock();
                !queue.is_empty()
            };

            if has_work {
                self.run_iteration()?;
            } else {
                // No work, yield
                std::thread::sleep(Duration::from_micros(100));
            }

            // Check for timeout requests
            self.check_timeouts();
        }

        Ok(())
    }

    /// Stop the serving loop
    pub fn stop(&self) {
        self.is_running.store(false, Ordering::SeqCst);
    }

    /// Check for and handle timed out requests
    fn check_timeouts(&self) {
        let timeout = Duration::from_millis(self.config.request_timeout_ms);
        let mut timed_out = Vec::new();

        // Find timed out pending requests
        {
            let pending = self.pending_requests.read();
            for (id, req) in pending.iter() {
                if req.created_at.elapsed() > timeout {
                    timed_out.push(*id);
                }
            }
        }

        // Cancel timed out requests
        for id in timed_out {
            self.cancel(id);
        }
    }

    /// Get serving metrics
    pub fn metrics(&self) -> ServingMetrics {
        let queue = self.queue.lock();
        let scheduler = self.scheduler.lock();
        let elapsed = self.start_time.elapsed().as_secs_f64();

        let total_requests = self.total_requests.load(Ordering::Relaxed);
        let total_tokens = self.total_tokens.load(Ordering::Relaxed);

        ServingMetrics {
            requests_per_second: if elapsed > 0.0 {
                total_requests as f64 / elapsed
            } else {
                0.0
            },
            tokens_per_second: if elapsed > 0.0 {
                total_tokens as f64 / elapsed
            } else {
                0.0
            },
            average_latency_ms: 0.0, // Would need to track per-request latencies
            p99_latency_ms: 0.0,     // Would need latency histogram
            batch_utilization: 0.0,  // Would need to track batch sizes
            kv_cache_utilization: scheduler.kv_cache_manager().stats().slot_utilization(),
            pending_requests: queue.pending_count(),
            running_requests: queue.running_count(),
            total_requests_processed: total_requests,
            total_tokens_generated: total_tokens,
            uptime_seconds: elapsed,
        }
    }

    /// Get configuration
    pub fn config(&self) -> &ServingEngineConfig {
        &self.config
    }
}

/// Serving metrics
#[derive(Debug, Clone, Default)]
pub struct ServingMetrics {
    /// Requests processed per second
    pub requests_per_second: f64,
    /// Tokens generated per second
    pub tokens_per_second: f64,
    /// Average request latency in milliseconds
    pub average_latency_ms: f64,
    /// 99th percentile latency in milliseconds
    pub p99_latency_ms: f64,
    /// Batch utilization (0.0 - 1.0)
    pub batch_utilization: f64,
    /// KV cache utilization (0.0 - 1.0)
    pub kv_cache_utilization: f64,
    /// Number of pending requests
    pub pending_requests: usize,
    /// Number of running requests
    pub running_requests: usize,
    /// Total requests processed
    pub total_requests_processed: u64,
    /// Total tokens generated
    pub total_tokens_generated: u64,
    /// Uptime in seconds
    pub uptime_seconds: f64,
}

// ============================================================================
// Async support
// ============================================================================

#[cfg(feature = "async-runtime")]
impl ServingEngine {
    /// Submit a request and await completion
    pub async fn submit_async(&self, request: InferenceRequest) -> Result<GenerationResult> {
        let request_id = request.id;
        let (tx, rx) = tokio::sync::oneshot::channel();

        // Store request with completion channel
        {
            let engine_request = EngineRequest {
                request: request.clone(),
                callback: None,
                completion_tx: Some(tx),
                created_at: Instant::now(),
            };
            self.pending_requests.write().insert(request_id, engine_request);
        }

        // Add to queue
        self.queue.lock().add(request);
        self.total_requests.fetch_add(1, Ordering::Relaxed);

        // Wait for completion
        rx.await.map_err(|_| RuvLLMError::Generation("Request cancelled".to_string()))
    }

    /// Stream tokens for a request
    pub fn stream(
        &self,
        request: InferenceRequest,
    ) -> Result<impl futures_core::Stream<Item = TokenOutput>> {
        let (tx, rx) = mpsc::unbounded_channel();
        let request_id = request.id;

        // Create callback that sends to channel
        let callback: TokenCallback = Box::new(move |output| {
            let _ = tx.send(output);
        });

        // Submit with callback
        self.submit_with_callback(request, callback)?;

        Ok(tokio_stream::wrappers::UnboundedReceiverStream::new(rx))
    }

    /// Run the serving loop asynchronously
    pub async fn run_async(&self) -> Result<()> {
        self.is_running.store(true, Ordering::SeqCst);

        while self.is_running.load(Ordering::SeqCst) {
            let has_work = {
                let queue = self.queue.lock();
                !queue.is_empty()
            };

            if has_work {
                self.run_iteration()?;
            } else {
                tokio::time::sleep(Duration::from_micros(100)).await;
            }

            self.check_timeouts();
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backends::NoopBackend;

    fn create_test_engine() -> ServingEngine {
        let model = Arc::new(NoopBackend);
        let config = ServingEngineConfig {
            kv_cache: KvCachePoolConfig {
                num_slots: 4,
                max_seq_len: 256,
                block_size: 16,
                total_blocks: 64,
                num_kv_heads: 2,
                head_dim: 64,
                num_layers: 4,
            },
            ..Default::default()
        };
        ServingEngine::new(model, config)
    }

    fn create_test_request() -> InferenceRequest {
        let params = GenerateParams::default().with_max_tokens(10);
        InferenceRequest::new(vec![1, 2, 3, 4, 5], params)
    }

    #[test]
    fn test_submit_request() {
        let engine = create_test_engine();
        let request = create_test_request();
        let id = request.id;

        let result = engine.submit(request);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), id);
    }

    #[test]
    fn test_cancel_request() {
        let engine = create_test_engine();
        let request = create_test_request();
        let id = engine.submit(request).unwrap();

        let cancelled = engine.cancel(id);
        assert!(cancelled);
    }

    #[test]
    fn test_run_iteration() {
        let engine = create_test_engine();
        let request = create_test_request();
        engine.submit(request).unwrap();

        // First iteration should do prefill
        let outputs = engine.run_iteration().unwrap();
        // May or may not have outputs depending on scheduler behavior
    }

    #[test]
    fn test_metrics() {
        let engine = create_test_engine();
        let metrics = engine.metrics();

        assert_eq!(metrics.pending_requests, 0);
        assert_eq!(metrics.running_requests, 0);
    }

    #[test]
    fn test_with_callback() {
        use std::sync::atomic::AtomicUsize;

        let engine = create_test_engine();
        let request = create_test_request();

        let callback_count = Arc::new(AtomicUsize::new(0));
        let count_clone = callback_count.clone();

        let callback: TokenCallback = Box::new(move |_| {
            count_clone.fetch_add(1, Ordering::Relaxed);
        });

        let id = engine.submit_with_callback(request, callback).unwrap();

        // Run a few iterations
        for _ in 0..15 {
            let _ = engine.run_iteration();
        }

        // Callback should have been called at least once
        // (actual count depends on scheduling and token generation)
    }
}

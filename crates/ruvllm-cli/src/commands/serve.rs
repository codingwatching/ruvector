//! Inference server command implementation
//!
//! Starts an OpenAI-compatible HTTP server for model inference,
//! providing endpoints for chat completions, health checks, and metrics.

use anyhow::{Context, Result};
use axum::{
    extract::{Json, State},
    http::StatusCode,
    response::IntoResponse,
    routing::{get, post},
    Router,
};
use colored::Colorize;
use console::style;
use serde::{Deserialize, Serialize};
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::RwLock;
use tower_http::cors::{Any, CorsLayer};
use tower_http::trace::TraceLayer;

use crate::models::{resolve_model_id, QuantPreset};

/// Server state
struct ServerState {
    model_id: String,
    backend: Option<Box<dyn ruvllm::LlmBackend>>,
    request_count: u64,
    total_tokens: u64,
    start_time: Instant,
}

type SharedState = Arc<RwLock<ServerState>>;

/// Run the serve command
pub async fn run(
    model: &str,
    host: &str,
    port: u16,
    max_concurrent: usize,
    max_context: usize,
    quantization: &str,
    cache_dir: &str,
) -> Result<()> {
    let model_id = resolve_model_id(model);
    let quant = QuantPreset::from_str(quantization)
        .ok_or_else(|| anyhow::anyhow!("Invalid quantization format: {}", quantization))?;

    println!();
    println!("{}", style("RuvLLM Inference Server").bold().cyan());
    println!();
    println!("  {} {}", "Model:".dimmed(), model_id);
    println!("  {} {}", "Quantization:".dimmed(), quant);
    println!("  {} {}", "Max Concurrent:".dimmed(), max_concurrent);
    println!("  {} {}", "Max Context:".dimmed(), max_context);
    println!();

    // Initialize backend
    println!("{}", "Loading model...".yellow());

    let mut backend = ruvllm::create_backend();
    let config = ruvllm::ModelConfig {
        architecture: detect_architecture(&model_id),
        quantization: Some(map_quantization(quant)),
        max_sequence_length: max_context,
        ..Default::default()
    };

    // Try to load from cache first, then from HuggingFace
    let model_path = PathBuf::from(cache_dir).join("models").join(&model_id);
    let load_result = if model_path.exists() {
        backend.load_model(model_path.to_str().unwrap(), config.clone())
    } else {
        backend.load_model(&model_id, config)
    };

    match load_result {
        Ok(_) => {
            if let Some(info) = backend.model_info() {
                println!(
                    "{} Loaded {} ({:.1}B params, {} memory)",
                    style("Success!").green().bold(),
                    info.name,
                    info.num_parameters as f64 / 1e9,
                    bytesize::ByteSize(info.memory_usage as u64)
                );
            } else {
                println!("{} Model loaded", style("Success!").green().bold());
            }
        }
        Err(e) => {
            // Create a mock server for development/testing
            println!(
                "{} Model loading failed: {}. Running in mock mode.",
                style("Warning:").yellow().bold(),
                e
            );
        }
    }

    // Create server state
    let state = Arc::new(RwLock::new(ServerState {
        model_id: model_id.clone(),
        backend: Some(backend),
        request_count: 0,
        total_tokens: 0,
        start_time: Instant::now(),
    }));

    // Build router
    let app = Router::new()
        // OpenAI-compatible endpoints
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/models", get(list_models))
        // Health and metrics
        .route("/health", get(health_check))
        .route("/metrics", get(metrics))
        .route("/", get(root))
        // State and middleware
        .with_state(state)
        .layer(CorsLayer::new().allow_origin(Any).allow_methods(Any).allow_headers(Any))
        .layer(TraceLayer::new_for_http());

    // Start server
    let addr = format!("{}:{}", host, port)
        .parse::<SocketAddr>()
        .context("Invalid address")?;

    println!();
    println!("{}", style("Server ready!").bold().green());
    println!();
    println!("  {} http://{}/v1/chat/completions", "API:".cyan(), addr);
    println!("  {} http://{}/health", "Health:".cyan(), addr);
    println!("  {} http://{}/metrics", "Metrics:".cyan(), addr);
    println!();
    println!("{}", "Example curl:".dimmed());
    println!(
        r#"  curl http://{}/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{{"model": "{}", "messages": [{{"role": "user", "content": "Hello!"}}]}}'"#,
        addr, model_id
    );
    println!();
    println!("Press Ctrl+C to stop the server.");
    println!();

    // Set up graceful shutdown
    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await
        .context("Server error")?;

    println!();
    println!("{}", "Server stopped.".dimmed());

    Ok(())
}

/// OpenAI-compatible chat completion request
#[derive(Debug, Deserialize)]
struct ChatCompletionRequest {
    model: String,
    messages: Vec<ChatMessage>,
    #[serde(default = "default_max_tokens")]
    max_tokens: usize,
    #[serde(default = "default_temperature")]
    temperature: f32,
    #[serde(default)]
    top_p: Option<f32>,
    #[serde(default)]
    stream: bool,
    #[serde(default)]
    stop: Option<Vec<String>>,
}

fn default_max_tokens() -> usize {
    512
}

fn default_temperature() -> f32 {
    0.7
}

#[derive(Debug, Serialize, Deserialize)]
struct ChatMessage {
    role: String,
    content: String,
}

/// OpenAI-compatible chat completion response
#[derive(Debug, Serialize)]
struct ChatCompletionResponse {
    id: String,
    object: String,
    created: u64,
    model: String,
    choices: Vec<ChatChoice>,
    usage: Usage,
}

#[derive(Debug, Serialize)]
struct ChatChoice {
    index: usize,
    message: ChatMessage,
    finish_reason: String,
}

#[derive(Debug, Serialize)]
struct Usage {
    prompt_tokens: usize,
    completion_tokens: usize,
    total_tokens: usize,
}

/// Chat completions endpoint
async fn chat_completions(
    State(state): State<SharedState>,
    Json(request): Json<ChatCompletionRequest>,
) -> impl IntoResponse {
    let start = Instant::now();

    // Build prompt from messages
    let prompt = build_prompt(&request.messages);

    // Get state for generation
    let mut state_lock = state.write().await;
    state_lock.request_count += 1;

    // Generate response
    let response_text = if let Some(backend) = &state_lock.backend {
        if backend.is_model_loaded() {
            let params = ruvllm::GenerateParams {
                max_tokens: request.max_tokens,
                temperature: request.temperature,
                top_p: request.top_p.unwrap_or(0.9),
                stop_sequences: request.stop.unwrap_or_default(),
                ..Default::default()
            };

            match backend.generate(&prompt, params) {
                Ok(text) => text,
                Err(e) => format!("Generation error: {}", e),
            }
        } else {
            // Mock response
            mock_response(&prompt)
        }
    } else {
        mock_response(&prompt)
    };

    // Calculate tokens (rough estimate)
    let prompt_tokens = prompt.split_whitespace().count();
    let completion_tokens = response_text.split_whitespace().count();
    state_lock.total_tokens += (prompt_tokens + completion_tokens) as u64;

    drop(state_lock);

    // Build response
    let response = ChatCompletionResponse {
        id: format!("chatcmpl-{}", uuid::Uuid::new_v4()),
        object: "chat.completion".to_string(),
        created: chrono::Utc::now().timestamp() as u64,
        model: request.model,
        choices: vec![ChatChoice {
            index: 0,
            message: ChatMessage {
                role: "assistant".to_string(),
                content: response_text,
            },
            finish_reason: "stop".to_string(),
        }],
        usage: Usage {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens + completion_tokens,
        },
    };

    tracing::info!(
        "Chat completion: {} tokens in {:.2}ms",
        response.usage.total_tokens,
        start.elapsed().as_secs_f64() * 1000.0
    );

    Json(response)
}

/// Build prompt from chat messages
fn build_prompt(messages: &[ChatMessage]) -> String {
    let mut prompt = String::new();

    for msg in messages {
        match msg.role.as_str() {
            "system" => {
                prompt.push_str(&format!("<|system|>\n{}\n", msg.content));
            }
            "user" => {
                prompt.push_str(&format!("<|user|>\n{}\n", msg.content));
            }
            "assistant" => {
                prompt.push_str(&format!("<|assistant|>\n{}\n", msg.content));
            }
            _ => {
                prompt.push_str(&format!("{}: {}\n", msg.role, msg.content));
            }
        }
    }

    prompt.push_str("<|assistant|>\n");
    prompt
}

/// Mock response for development/testing
fn mock_response(prompt: &str) -> String {
    let prompt_lower = prompt.to_lowercase();

    if prompt_lower.contains("hello") || prompt_lower.contains("hi") {
        "Hello! I'm RuvLLM, a local AI assistant running on your Mac. How can I help you today?".to_string()
    } else if prompt_lower.contains("code") || prompt_lower.contains("function") {
        "Here's an example function:\n\n```rust\nfn hello() {\n    println!(\"Hello, world!\");\n}\n```\n\nWould you like me to explain this code?".to_string()
    } else {
        "I understand your request. To provide real responses, please ensure the model is properly loaded. Currently running in mock mode for development.".to_string()
    }
}

/// List available models
async fn list_models(State(state): State<SharedState>) -> impl IntoResponse {
    let state_lock = state.read().await;

    let models = serde_json::json!({
        "object": "list",
        "data": [{
            "id": state_lock.model_id,
            "object": "model",
            "owned_by": "ruvllm",
            "permission": []
        }]
    });

    Json(models)
}

/// Health check endpoint
async fn health_check(State(state): State<SharedState>) -> impl IntoResponse {
    let state_lock = state.read().await;

    let status = if state_lock.backend.as_ref().map(|b| b.is_model_loaded()).unwrap_or(false) {
        "healthy"
    } else {
        "degraded"
    };

    let health = serde_json::json!({
        "status": status,
        "model": state_lock.model_id,
        "uptime_seconds": state_lock.start_time.elapsed().as_secs()
    });

    Json(health)
}

/// Metrics endpoint
async fn metrics(State(state): State<SharedState>) -> impl IntoResponse {
    let state_lock = state.read().await;
    let uptime = state_lock.start_time.elapsed();

    let metrics = serde_json::json!({
        "model": state_lock.model_id,
        "requests_total": state_lock.request_count,
        "tokens_total": state_lock.total_tokens,
        "uptime_seconds": uptime.as_secs(),
        "requests_per_second": if uptime.as_secs() > 0 {
            state_lock.request_count as f64 / uptime.as_secs() as f64
        } else {
            0.0
        },
        "tokens_per_second": if uptime.as_secs() > 0 {
            state_lock.total_tokens as f64 / uptime.as_secs() as f64
        } else {
            0.0
        }
    });

    Json(metrics)
}

/// Root endpoint
async fn root() -> impl IntoResponse {
    let info = serde_json::json!({
        "name": "RuvLLM Inference Server",
        "version": env!("CARGO_PKG_VERSION"),
        "endpoints": {
            "chat": "/v1/chat/completions",
            "models": "/v1/models",
            "health": "/health",
            "metrics": "/metrics"
        }
    });

    Json(info)
}

/// Graceful shutdown signal handler
async fn shutdown_signal() {
    let ctrl_c = async {
        tokio::signal::ctrl_c()
            .await
            .expect("Failed to install Ctrl+C handler");
    };

    #[cfg(unix)]
    let terminate = async {
        tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            .expect("Failed to install signal handler")
            .recv()
            .await;
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {},
        _ = terminate => {},
    }

    println!();
    println!("{}", "Shutting down...".yellow());
}

/// Detect model architecture from model ID
fn detect_architecture(model_id: &str) -> ruvllm::ModelArchitecture {
    let lower = model_id.to_lowercase();
    if lower.contains("mistral") {
        ruvllm::ModelArchitecture::Mistral
    } else if lower.contains("llama") {
        ruvllm::ModelArchitecture::Llama
    } else if lower.contains("phi") {
        ruvllm::ModelArchitecture::Phi
    } else if lower.contains("qwen") {
        ruvllm::ModelArchitecture::Qwen
    } else if lower.contains("gemma") {
        ruvllm::ModelArchitecture::Gemma
    } else {
        ruvllm::ModelArchitecture::Llama // Default
    }
}

/// Map our quantization preset to ruvllm quantization
fn map_quantization(quant: QuantPreset) -> ruvllm::Quantization {
    match quant {
        QuantPreset::Q4K => ruvllm::Quantization::Q4K,
        QuantPreset::Q8 => ruvllm::Quantization::Q8,
        QuantPreset::F16 => ruvllm::Quantization::F16,
        QuantPreset::None => ruvllm::Quantization::None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_prompt() {
        let messages = vec![
            ChatMessage {
                role: "system".to_string(),
                content: "You are helpful.".to_string(),
            },
            ChatMessage {
                role: "user".to_string(),
                content: "Hello!".to_string(),
            },
        ];

        let prompt = build_prompt(&messages);
        assert!(prompt.contains("You are helpful"));
        assert!(prompt.contains("Hello"));
        assert!(prompt.ends_with("<|assistant|>\n"));
    }

    #[test]
    fn test_detect_architecture() {
        assert_eq!(
            detect_architecture("mistralai/Mistral-7B"),
            ruvllm::ModelArchitecture::Mistral
        );
        assert_eq!(
            detect_architecture("Qwen/Qwen2.5-14B"),
            ruvllm::ModelArchitecture::Qwen
        );
    }
}

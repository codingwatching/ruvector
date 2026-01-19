//! RuvLLM CLI - Model Management and Inference for Apple Silicon
//!
//! A command-line interface for downloading, managing, and running LLM models
//! optimized for Mac M4 Pro and other Apple Silicon devices.
//!
//! ## Commands
//!
//! - `ruvllm download <model>` - Download model from HuggingFace Hub
//! - `ruvllm list` - List available/downloaded models
//! - `ruvllm info <model>` - Show model information
//! - `ruvllm serve <model>` - Start inference server
//! - `ruvllm chat <model>` - Interactive chat mode
//! - `ruvllm benchmark <model>` - Run performance benchmarks

use clap::{Parser, Subcommand};
use colored::Colorize;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

mod commands;
mod models;

use commands::{benchmark, chat, download, info, list, serve};

/// RuvLLM - High-performance LLM inference for Apple Silicon
#[derive(Parser)]
#[command(name = "ruvllm")]
#[command(author, version, about, long_about = None)]
#[command(propagate_version = true)]
struct Cli {
    /// Enable verbose logging
    #[arg(short, long, global = true)]
    verbose: bool,

    /// Disable colored output
    #[arg(long, global = true)]
    no_color: bool,

    /// Custom cache directory for models
    #[arg(long, global = true, env = "RUVLLM_CACHE_DIR")]
    cache_dir: Option<String>,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Download a model from HuggingFace Hub
    #[command(alias = "dl")]
    Download {
        /// Model identifier (HuggingFace model ID or alias)
        ///
        /// Aliases: qwen, mistral, phi, llama
        model: String,

        /// Quantization format (q4k, q8, f16, none)
        #[arg(short, long, default_value = "q4k")]
        quantization: String,

        /// Force re-download even if model exists
        #[arg(short, long)]
        force: bool,

        /// Specific revision/branch to download
        #[arg(long)]
        revision: Option<String>,
    },

    /// List available and downloaded models
    #[command(alias = "ls")]
    List {
        /// Show only downloaded models
        #[arg(short, long)]
        downloaded: bool,

        /// Show detailed information
        #[arg(short, long)]
        long: bool,
    },

    /// Show detailed model information
    Info {
        /// Model identifier or alias
        model: String,
    },

    /// Start an OpenAI-compatible inference server
    Serve {
        /// Model to serve
        model: String,

        /// Host to bind to
        #[arg(long, default_value = "127.0.0.1")]
        host: String,

        /// Port to bind to
        #[arg(short, long, default_value = "8080")]
        port: u16,

        /// Maximum concurrent requests
        #[arg(long, default_value = "4")]
        max_concurrent: usize,

        /// Maximum context length
        #[arg(long, default_value = "4096")]
        max_context: usize,

        /// Quantization format
        #[arg(short, long, default_value = "q4k")]
        quantization: String,
    },

    /// Interactive chat mode
    Chat {
        /// Model to use for chat
        model: String,

        /// System prompt
        #[arg(short, long)]
        system: Option<String>,

        /// Maximum tokens to generate per response
        #[arg(long, default_value = "512")]
        max_tokens: usize,

        /// Temperature for sampling (0.0 = deterministic)
        #[arg(short, long, default_value = "0.7")]
        temperature: f32,

        /// Quantization format
        #[arg(short, long, default_value = "q4k")]
        quantization: String,
    },

    /// Run performance benchmarks
    #[command(alias = "bench")]
    Benchmark {
        /// Model to benchmark
        model: String,

        /// Number of warmup iterations
        #[arg(long, default_value = "3")]
        warmup: usize,

        /// Number of benchmark iterations
        #[arg(short, long, default_value = "10")]
        iterations: usize,

        /// Prompt length for benchmarking
        #[arg(long, default_value = "128")]
        prompt_length: usize,

        /// Generation length for benchmarking
        #[arg(long, default_value = "64")]
        gen_length: usize,

        /// Quantization format
        #[arg(short, long, default_value = "q4k")]
        quantization: String,

        /// Output format (text, json, csv)
        #[arg(long, default_value = "text")]
        format: String,
    },
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    // Initialize logging
    let log_level = if cli.verbose { "debug" } else { "info" };
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| log_level.into()),
        )
        .with(tracing_subscriber::fmt::layer().with_target(false))
        .init();

    // Set up colored output
    if cli.no_color {
        colored::control::set_override(false);
    }

    // Get cache directory
    let cache_dir = cli.cache_dir.unwrap_or_else(|| {
        dirs::cache_dir()
            .unwrap_or_else(|| std::path::PathBuf::from("."))
            .join("ruvllm")
            .to_string_lossy()
            .to_string()
    });

    // Execute command
    let result = match cli.command {
        Commands::Download {
            model,
            quantization,
            force,
            revision,
        } => {
            download::run(&model, &quantization, force, revision.as_deref(), &cache_dir).await
        }

        Commands::List { downloaded, long } => {
            list::run(downloaded, long, &cache_dir).await
        }

        Commands::Info { model } => {
            info::run(&model, &cache_dir).await
        }

        Commands::Serve {
            model,
            host,
            port,
            max_concurrent,
            max_context,
            quantization,
        } => {
            serve::run(
                &model,
                &host,
                port,
                max_concurrent,
                max_context,
                &quantization,
                &cache_dir,
            )
            .await
        }

        Commands::Chat {
            model,
            system,
            max_tokens,
            temperature,
            quantization,
        } => {
            chat::run(
                &model,
                system.as_deref(),
                max_tokens,
                temperature,
                &quantization,
                &cache_dir,
            )
            .await
        }

        Commands::Benchmark {
            model,
            warmup,
            iterations,
            prompt_length,
            gen_length,
            quantization,
            format,
        } => {
            benchmark::run(
                &model,
                warmup,
                iterations,
                prompt_length,
                gen_length,
                &quantization,
                &format,
                &cache_dir,
            )
            .await
        }
    };

    if let Err(e) = result {
        eprintln!("{} {}", "Error:".red().bold(), e);
        std::process::exit(1);
    }

    Ok(())
}

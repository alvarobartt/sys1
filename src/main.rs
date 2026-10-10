use anyhow::Context;
use candle_core::DType;
use clap::{ArgGroup, Parser, ValueEnum};
use std::{
    net::{IpAddr, SocketAddr},
    path::PathBuf,
    sync::Arc,
    time::{Duration, Instant},
};
use sys1::{
    api,
    batching::{Batcher, BatcherConfig},
    models,
};
use tracing_subscriber::EnvFilter;

#[derive(Debug, Parser)]
#[command(
    version,
    about,
    group(
        ArgGroup::new("model_source")
            .required(true)
            .multiple(false)
            .args(["model_id", "model_path"])
    )
)]
struct Args {
    /// Hugging Face model repository to load.
    #[arg(short = 'm', long, env, help_heading = "Model options")]
    model_id: Option<String>,
    /// Local model directory to load instead of a Hugging Face repository.
    #[arg(short = 'M', long, env, help_heading = "Model options")]
    model_path: Option<PathBuf>,
    /// Hugging Face model revision (branch, tag, or commit).
    #[arg(
        short,
        long,
        env,
        default_value = "main",
        help_heading = "Model options"
    )]
    revision: String,
    /// Model name returned by the API.
    #[arg(
        short = 'n',
        long,
        env,
        value_parser = clap::builder::NonEmptyStringValueParser::new(),
        help_heading = "Model options"
    )]
    served_model_name: Option<String>,
    /// Address on which the HTTP server listens.
    #[arg(
        short = 'H',
        long,
        env,
        default_value = "0.0.0.0",
        help_heading = "Server options"
    )]
    host: IpAddr,
    /// Port on which the HTTP server listens.
    #[arg(
        short,
        long,
        env,
        default_value_t = 3000,
        help_heading = "Server options"
    )]
    port: u16,
    /// Maximum number of requests processed in one batch.
    #[arg(
        short = 'b',
        long,
        env,
        default_value_t = 32,
        help_heading = "Batching options"
    )]
    max_batch_size: usize,
    /// Maximum total number of questions processed in one batch.
    #[arg(long, env, default_value_t = 128, help_heading = "Batching options")]
    max_batch_questions: usize,
    /// Maximum number of questions accepted in one request.
    #[arg(long, env, default_value_t = 64, help_heading = "Batching options")]
    max_questions_per_request: usize,
    /// Maximum number of requests waiting to be processed.
    #[arg(long, env, default_value_t = 256, help_heading = "Batching options")]
    max_queue_size: usize,
    /// Time to wait for more requests before processing a batch, in milliseconds.
    #[arg(long, env, default_value_t = 0, help_heading = "Batching options")]
    batch_wait_ms: u64,
    /// Per-request timeout in milliseconds; 0 disables the timeout.
    #[arg(
        short = 't',
        long,
        env,
        default_value_t = 30_000,
        help_heading = "Batching options"
    )]
    request_timeout_ms: u64,
    /// Maximum request body size in bytes, including base64 images and videos.
    #[arg(
        long,
        env,
        default_value_t = 16_777_216,
        help_heading = "Server options"
    )]
    max_request_bytes: usize,
    /// Maximum model context length; uses the model configuration when omitted.
    #[arg(long, env, help_heading = "Model options")]
    max_model_len: Option<usize>,
    /// Floating-point precision; omitted or auto uses the model's inference policy.
    #[arg(short = 'd', long, env, value_enum, help_heading = "Model options")]
    dtype: Option<Precision>,
    /// Attention implementation used for inference.
    #[arg(
        short = 'a',
        long,
        env,
        value_enum,
        default_value_t = models::AttentionImplementation::Auto,
        help_heading = "Model options"
    )]
    attention: models::AttentionImplementation,
}

impl Args {
    fn attention(
        requested: models::AttentionImplementation,
        architecture: models::Architecture,
        dtype: DType,
    ) -> models::AttentionImplementation {
        if architecture != models::Architecture::Laya
            || requested != models::AttentionImplementation::Auto
        {
            return requested;
        }
        if matches!(dtype, DType::F16 | DType::BF16) && cfg!(feature = "flash-attn-2") {
            return models::AttentionImplementation::FlashAttention2;
        }
        models::AttentionImplementation::Eager
    }

    fn validate(&self) -> anyhow::Result<()> {
        anyhow::ensure!(self.max_batch_size > 0, "--max-batch-size must be positive");
        anyhow::ensure!(
            self.max_batch_questions > 0,
            "--max-batch-questions must be positive"
        );
        anyhow::ensure!(
            self.max_questions_per_request > 0,
            "--max-questions-per-request must be positive"
        );
        anyhow::ensure!(
            self.max_questions_per_request <= self.max_batch_questions,
            "--max-questions-per-request cannot exceed --max-batch-questions"
        );
        anyhow::ensure!(self.max_queue_size > 0, "--max-queue-size must be positive");
        anyhow::ensure!(
            self.max_request_bytes > 0,
            "--max-request-bytes must be positive"
        );
        anyhow::ensure!(
            self.max_model_len != Some(0),
            "--max-model-len must be positive"
        );
        if let Some(dtype) = self.dtype.filter(|dtype| *dtype != Precision::Auto) {
            self.attention.validate(dtype.resolve_explicit())?;
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
enum Precision {
    Auto,
    F32,
    F16,
    Bf16,
}

impl Precision {
    fn resolve_explicit(self) -> DType {
        match self {
            Self::Auto => unreachable!("auto is resolved from model files"),
            Self::F32 => DType::F32,
            Self::F16 => DType::F16,
            Self::Bf16 => DType::BF16,
        }
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("sys1=info")),
        )
        .with_target(false)
        .compact()
        .init();
    let args = Args::parse();
    args.validate()?;
    sys1::validate_backend()?;
    let backend = if cfg!(feature = "cuda") {
        "cuda"
    } else if cfg!(feature = "metal") {
        "metal"
    } else {
        "cpu"
    };
    tracing::info!(
        version = env!("CARGO_PKG_VERSION"),
        description = env!("CARGO_PKG_DESCRIPTION"),
        backend,
        ?args,
        "Starting sys1"
    );
    let address = SocketAddr::new(args.host, args.port);
    let (model_path, source_name, architecture, source) = match (args.model_path, args.model_id) {
        (Some(path), None) => {
            if args.served_model_name.is_none() {
                tracing::warn!("Set --served-model-name when using --model-path");
            }
            let name = path.display().to_string();
            let architecture = models::Architecture::from_path(&path)?;
            (path, name, architecture, "path")
        }
        (None, Some(model_id)) => {
            let started = Instant::now();
            tracing::info!(%model_id, revision = %args.revision, "Resolving model snapshot");
            let outcome = sys1::hub::download_with_source(&model_id, &args.revision).await?;
            let elapsed_ms = started.elapsed().as_millis();
            tracing::info!(
                %model_id,
                path = %outcome.path.display(),
                source = outcome.source.as_str(),
                elapsed_ms,
                total_files = outcome.total_files,
                downloaded_files = outcome.downloaded_files,
                cached_files = outcome.cached_files,
                total_bytes = outcome.total_bytes,
                "Model files ready"
            );
            let architecture = models::Architecture::from_path(&outcome.path)?;
            (
                outcome.path,
                model_id,
                architecture,
                outcome.source.as_str(),
            )
        }
        _ => unreachable!("model source is validated by clap"),
    };
    let requested_dtype = args
        .dtype
        .filter(|dtype| *dtype != Precision::Auto)
        .map(Precision::resolve_explicit);
    let dtype = match requested_dtype {
        Some(dtype) => dtype,
        None => models::model_default_dtype(&model_path, architecture)?,
    };
    let attention = Args::attention(args.attention, architecture, dtype);
    attention.validate(dtype)?;
    let served_model_name = args.served_model_name.unwrap_or(source_name);
    let started = Instant::now();
    tracing::info!(model = %served_model_name, source, ?architecture, ?dtype, ?attention, path = %model_path.display(), "Loading model");
    #[cfg(feature = "metal")]
    sys1::report_dtype_support(dtype);
    let model = models::load(
        &model_path,
        architecture,
        requested_dtype,
        args.max_model_len,
        attention,
    )
    .with_context(|| format!("failed to load model from {}", model_path.display()))?;
    let model_load_ms = started.elapsed().as_millis();
    tracing::info!(model_load_ms, "Model loaded");
    let batcher = Batcher::new(
        Arc::new(model),
        served_model_name.clone(),
        BatcherConfig {
            max_batch_size: args.max_batch_size,
            max_batch_questions: args.max_batch_questions,
            max_questions_per_request: args.max_questions_per_request,
            wait: Duration::from_millis(args.batch_wait_ms),
            queue_capacity: args.max_queue_size,
            response_timeout: (args.request_timeout_ms > 0)
                .then(|| Duration::from_millis(args.request_timeout_ms)),
        },
    );
    let started = Instant::now();
    batcher
        .warmup()
        .await
        .map_err(|error| anyhow::anyhow!(error.error))
        .context("model warmup failed")?;
    let warmup_ms = started.elapsed().as_millis();
    tracing::info!(warmup_ms, "Model warmup complete");
    let app = api::router(batcher, args.max_request_bytes);
    let listener = tokio::net::TcpListener::bind(address).await?;
    let address = listener.local_addr()?;
    tracing::info!("Available API routes:");
    for &(method, route) in api::PUBLIC_ROUTES {
        tracing::info!("[{method:>4}] {route}");
    }
    tracing::info!(%address, "Server running");
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown())
        .await?;
    tracing::info!("Server stopped");
    Ok(())
}

#[cfg(unix)]
async fn shutdown() {
    use tokio::signal::unix::{SignalKind, signal};

    let mut terminate = signal(SignalKind::terminate()).expect("install SIGTERM handler");
    tokio::select! {
        _ = tokio::signal::ctrl_c() => {}
        _ = terminate.recv() => {}
    }
    tracing::info!("Shutdown requested");
}

#[cfg(not(unix))]
async fn shutdown() {
    let _ = tokio::signal::ctrl_c().await;
    tracing::info!("Shutdown requested");
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(options: &[&str]) -> Args {
        let mut command = vec!["sys1", "--model-id", "owner/model"];
        command.extend_from_slice(options);
        Args::try_parse_from(command).unwrap()
    }

    #[test]
    fn validates_model_source() {
        assert!(Args::try_parse_from(["sys1"]).is_err());
        assert!(
            Args::try_parse_from([
                "sys1",
                "--model-id",
                "owner/model",
                "--model-path",
                "/models/laya",
            ])
            .is_err()
        );
    }

    #[test]
    fn selects_attention_for_laya() {
        assert_eq!(
            Args::attention(
                models::AttentionImplementation::Auto,
                models::Architecture::Laya,
                DType::F32,
            ),
            models::AttentionImplementation::Eager
        );
        let low_precision = if cfg!(feature = "flash-attn-2") {
            models::AttentionImplementation::FlashAttention2
        } else {
            models::AttentionImplementation::Eager
        };
        assert_eq!(
            Args::attention(
                models::AttentionImplementation::Auto,
                models::Architecture::Laya,
                DType::F16,
            ),
            low_precision
        );
        assert_eq!(
            Args::attention(
                models::AttentionImplementation::Auto,
                models::Architecture::Qwen35,
                DType::BF16,
            ),
            models::AttentionImplementation::Auto
        );
    }

    #[test]
    fn rejects_invalid_configuration() {
        let invalid_batch = args(&[
            "--max-batch-questions",
            "8",
            "--max-questions-per-request",
            "9",
        ]);
        assert!(invalid_batch.validate().is_err());
        assert!(args(&["--max-model-len", "0"]).validate().is_err());
        assert!(args(&["--max-queue-size", "0"]).validate().is_err());

        let invalid_attention = args(&["--attention", "flash-attn-2", "--dtype", "f32"]);
        assert!(invalid_attention.validate().is_err());
    }
}

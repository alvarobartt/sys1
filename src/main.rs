use anyhow::Context;
use candle_core::DType;
use clap::{Parser, ValueEnum};
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
#[command(version, about)]
struct Args {
    /// Hugging Face model repository to load.
    #[arg(
        short = 'm',
        long,
        env,
        default_value = "convaiinnovations/laya",
        conflicts_with = "model_path",
        help_heading = "Model options"
    )]
    model_id: String,
    /// Local model directory to load instead of a Hugging Face repository.
    #[arg(
        short = 'M',
        long,
        env,
        conflicts_with = "model_id",
        help_heading = "Model options"
    )]
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
    /// Maximum request body size in bytes.
    #[arg(
        long,
        env,
        default_value_t = 1_048_576,
        help_heading = "Server options"
    )]
    max_request_bytes: usize,
    /// Maximum model context length; uses the model configuration when omitted.
    #[arg(long, env, help_heading = "Model options")]
    max_model_len: Option<usize>,
    /// Floating-point precision used for inference.
    #[arg(
        short = 'd',
        long,
        env,
        value_enum,
        default_value_t = Precision::Auto,
        help_heading = "Model options"
    )]
    dtype: Precision,
    /// Attention implementation used for inference.
    #[arg(
        short = 'a',
        long,
        env,
        value_enum,
        default_value_t = models::AttentionImplementation::Eager,
        help_heading = "Model options"
    )]
    attention: models::AttentionImplementation,
}

impl Args {
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
        self.attention.validate(self.dtype.resolve())?;
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
    fn resolve(self) -> DType {
        match self {
            Self::Auto | Self::F32 => DType::F32,
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
    let dtype = args.dtype.resolve();
    tracing::info!(
        version = env!("CARGO_PKG_VERSION"),
        backend = backend(),
        ?args,
        ?dtype,
        "Starting sys1"
    );
    let address = SocketAddr::new(args.host, args.port);
    let (model_path, source_name, architecture, source) = match args.model_path {
        Some(path) => {
            if args.served_model_name.is_none() {
                tracing::warn!("Set --served-model-name when using --model-path");
            }
            let name = path.display().to_string();
            let architecture = models::Architecture::from_path(&path)?;
            (path, name, architecture, "path")
        }
        None => {
            let architecture = models::Architecture::from_model_id(&args.model_id)?;
            let started = Instant::now();
            let outcome = sys1::hub::download_with_source(&args.model_id, &args.revision).await?;
            let elapsed_ms = started.elapsed().as_millis();
            tracing::info!(
                elapsed_ms,
                total_files = outcome.total_files,
                downloaded_files = outcome.downloaded_files,
                cached_files = outcome.cached_files,
                total_bytes = outcome.total_bytes,
                "Model files ready"
            );
            (
                outcome.path,
                args.model_id,
                architecture,
                outcome.source.as_str(),
            )
        }
    };
    let served_model_name = args.served_model_name.unwrap_or(source_name);
    let started = Instant::now();
    tracing::info!(source, ?architecture, "Loading model");
    let model = models::load(
        &model_path,
        architecture,
        dtype,
        args.max_model_len,
        args.attention,
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

fn backend() -> &'static str {
    if cfg!(feature = "cuda") {
        "cuda"
    } else if cfg!(feature = "metal") {
        "metal"
    } else {
        "cpu"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::CommandFactory;

    #[test]
    fn defaults_to_laya_on_hugging_face() {
        let args = Args::try_parse_from(["sys1"]).unwrap();
        assert_eq!(args.model_id, "convaiinnovations/laya");
        assert_eq!(args.revision, "main");
        assert_eq!(args.model_path, None);
        assert_eq!(args.served_model_name, None);
        assert_eq!(args.host, "0.0.0.0".parse::<IpAddr>().unwrap());
        assert_eq!(args.batch_wait_ms, 0);
        assert_eq!(args.max_batch_size, 32);
        assert_eq!(args.max_batch_questions, 128);
        assert_eq!(args.max_questions_per_request, 64);
        assert_eq!(args.max_queue_size, 256);
        assert_eq!(args.request_timeout_ms, 30_000);
        assert_eq!(args.max_request_bytes, 1_048_576);
        assert_eq!(args.max_model_len, None);
        assert_eq!(args.dtype, Precision::Auto);
        assert_eq!(args.attention, models::AttentionImplementation::Eager);
    }

    #[test]
    fn accepts_a_local_model_path() {
        let args = Args::try_parse_from(["sys1", "--model-path", "/models/laya"]).unwrap();
        assert_eq!(args.model_path, Some(PathBuf::from("/models/laya")));
    }

    #[test]
    fn accepts_short_options() {
        let args = Args::try_parse_from([
            "sys1",
            "-m",
            "owner/model",
            "-r",
            "release",
            "-n",
            "public-name",
            "-H",
            "127.0.0.1",
            "-p",
            "8080",
            "-b",
            "4",
            "-t",
            "100",
            "-d",
            "f16",
            "-a",
            "eager",
        ])
        .unwrap();

        assert_eq!(args.model_id, "owner/model");
        assert_eq!(args.revision, "release");
        assert_eq!(args.served_model_name.as_deref(), Some("public-name"));
        assert_eq!(args.host, "127.0.0.1".parse::<IpAddr>().unwrap());
        assert_eq!(args.port, 8080);
        assert_eq!(args.max_batch_size, 4);
        assert_eq!(args.request_timeout_ms, 100);
        assert_eq!(args.dtype, Precision::F16);
        assert_eq!(args.attention, models::AttentionImplementation::Eager);
    }

    #[test]
    fn exposes_environment_variables_and_help_metadata() {
        let mut command = Args::command();
        assert_eq!(
            command.get_about().map(ToString::to_string).as_deref(),
            Some(env!("CARGO_PKG_DESCRIPTION"))
        );
        let environment_variables = command
            .get_arguments()
            .filter_map(|argument| {
                argument.get_env().map(|environment| {
                    (
                        argument.get_id().as_str(),
                        environment.to_str().expect("environment names are UTF-8"),
                    )
                })
            })
            .collect::<Vec<_>>();

        assert_eq!(
            environment_variables,
            [
                ("model_id", "MODEL_ID"),
                ("model_path", "MODEL_PATH"),
                ("revision", "REVISION"),
                ("served_model_name", "SERVED_MODEL_NAME"),
                ("host", "HOST"),
                ("port", "PORT"),
                ("max_batch_size", "MAX_BATCH_SIZE"),
                ("max_batch_questions", "MAX_BATCH_QUESTIONS"),
                ("max_questions_per_request", "MAX_QUESTIONS_PER_REQUEST"),
                ("max_queue_size", "MAX_QUEUE_SIZE"),
                ("batch_wait_ms", "BATCH_WAIT_MS"),
                ("request_timeout_ms", "REQUEST_TIMEOUT_MS"),
                ("max_request_bytes", "MAX_REQUEST_BYTES"),
                ("max_model_len", "MAX_MODEL_LEN"),
                ("dtype", "DTYPE"),
                ("attention", "ATTENTION"),
            ]
        );

        let help = command.render_help().to_string();
        assert!(help.contains("Model options:"));
        assert!(help.contains("Server options:"));
        assert!(help.contains("Batching options:"));
        assert!(!help.contains("Inference options:"));
        assert!(help.contains("[env: MODEL_ID=]"));
        assert!(help.contains("[default: convaiinnovations/laya]"));
    }

    #[test]
    fn resolves_requested_and_backend_default_dtypes() {
        assert_eq!(Precision::F32.resolve(), DType::F32);
        assert_eq!(Precision::F16.resolve(), DType::F16);
        assert_eq!(Precision::Bf16.resolve(), DType::BF16);
        assert_eq!(Precision::Auto.resolve(), DType::F32);

        let args = Args::try_parse_from(["sys1", "--dtype", "bf16"]).unwrap();
        assert_eq!(args.dtype, Precision::Bf16);

        let args = Args::try_parse_from(["sys1", "--attention", "flash-attn-2"]).unwrap();
        assert_eq!(
            args.attention,
            models::AttentionImplementation::FlashAttention2
        );
    }

    #[test]
    fn accepts_a_model_length_override() {
        let args = Args::try_parse_from(["sys1", "--max-model-len", "8192"]).unwrap();
        assert_eq!(args.max_model_len, Some(8192));
        assert!(args.validate().is_ok());

        let args = Args::try_parse_from(["sys1", "--max-model-len", "0"]).unwrap();
        assert!(args.validate().is_err());
    }

    #[test]
    fn rejects_two_model_sources() {
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
    fn rejects_invalid_capacity_configuration() {
        let args = Args::try_parse_from([
            "sys1",
            "--max-batch-questions",
            "8",
            "--max-questions-per-request",
            "9",
        ])
        .unwrap();
        assert!(args.validate().is_err());

        let args = Args::try_parse_from(["sys1", "--max-queue-size", "0"]).unwrap();
        assert!(args.validate().is_err());
    }

    #[test]
    fn rejects_flash_attention_with_f32() {
        let args = Args::try_parse_from(["sys1", "--attention", "flash-attn-3", "--dtype", "f32"])
            .unwrap();
        assert!(args.validate().is_err());
    }
}

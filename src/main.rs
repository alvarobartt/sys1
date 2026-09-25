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
use tracing::{info, warn};
use tracing_subscriber::EnvFilter;

#[derive(Debug, Parser)]
#[command(version, about)]
struct Args {
    #[arg(
        long,
        default_value = "convaiinnovations/laya",
        conflicts_with = "model_path"
    )]
    model_id: String,
    #[arg(long, conflicts_with = "model_id")]
    model_path: Option<PathBuf>,
    #[arg(long, default_value = "main")]
    revision: String,
    #[arg(long, value_parser = clap::builder::NonEmptyStringValueParser::new())]
    served_model_name: Option<String>,
    #[arg(long, default_value = "0.0.0.0")]
    host: IpAddr,
    #[arg(long, default_value_t = 3000)]
    port: u16,
    #[arg(long, default_value_t = 32)]
    max_batch_size: usize,
    #[arg(long, default_value_t = 128)]
    max_batch_questions: usize,
    #[arg(long, default_value_t = 64)]
    max_questions_per_request: usize,
    #[arg(long, default_value_t = 256)]
    max_queue_size: usize,
    #[arg(long, default_value_t = 0)]
    batch_wait_ms: u64,
    #[arg(long, default_value_t = 30_000)]
    request_timeout_ms: u64,
    #[arg(long, default_value_t = 1_048_576)]
    max_request_bytes: usize,
    #[arg(long)]
    max_model_len: Option<usize>,
    #[arg(long, value_enum, default_value_t = Precision::Auto)]
    dtype: Precision,
    #[arg(long, value_enum, default_value_t = models::AttentionImplementation::Eager)]
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
    info!(
        version = env!("CARGO_PKG_VERSION"),
        description = env!("CARGO_PKG_DESCRIPTION"),
        backend = backend(),
        ?args,
        ?dtype,
        "sys1 starting"
    );
    let address = SocketAddr::new(args.host, args.port);
    let (model_path, source_name, architecture) = match args.model_path {
        Some(path) => {
            if args.served_model_name.is_none() {
                warn!("--served-model-name is recommended when using --model-path");
            }
            let name = path.display().to_string();
            let architecture = models::Architecture::from_path(&path)?;
            (path, name, architecture)
        }
        None => {
            let architecture = models::Architecture::from_model_id(&args.model_id)?;
            let started = Instant::now();
            info!(model_id = %args.model_id, revision = %args.revision, "resolving model snapshot");
            let path = sys1::hub::download(&args.model_id, &args.revision).await?;
            info!(
                model_id = %args.model_id,
                path = %path.display(),
                elapsed_ms = started.elapsed().as_millis(),
                "model snapshot ready"
            );
            (path, args.model_id, architecture)
        }
    };
    let served_model_name = args.served_model_name.unwrap_or(source_name);
    let started = Instant::now();
    info!(
        model = %served_model_name,
        ?architecture,
        path = %model_path.display(),
        "loading model"
    );
    let model = models::load(
        &model_path,
        architecture,
        dtype,
        args.max_model_len,
        args.attention,
    )
    .with_context(|| format!("failed to load model from {}", model_path.display()))?;
    info!(
        model = %served_model_name,
        elapsed_ms = started.elapsed().as_millis(),
        "model loaded"
    );
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
    info!(model = %served_model_name, "model warmup started");
    batcher
        .warmup()
        .await
        .map_err(|error| anyhow::anyhow!(error.error))
        .context("model warmup failed")?;
    info!(
        model = %served_model_name,
        elapsed_ms = started.elapsed().as_millis(),
        "model warmup completed"
    );
    batcher.reset_stats();
    let app = api::router(batcher, args.max_request_bytes);
    let listener = tokio::net::TcpListener::bind(address).await?;
    info!(%address, model = %served_model_name, "sys1 ready");
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown())
        .await?;
    info!("sys1 stopped");
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
    info!("shutdown requested");
}

#[cfg(not(unix))]
async fn shutdown() {
    let _ = tokio::signal::ctrl_c().await;
    info!("shutdown requested");
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

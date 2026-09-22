use anyhow::Context;
use clap::Parser;
use std::{
    net::{IpAddr, SocketAddr},
    path::PathBuf,
    sync::Arc,
    time::{Duration, Instant},
};
use sys1::{api, batching::Batcher, models};
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
    #[arg(long, default_value_t = 5)]
    batch_wait_ms: u64,
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
    info!(
        version = env!("CARGO_PKG_VERSION"),
        description = env!("CARGO_PKG_DESCRIPTION"),
        backend = backend(),
        ?args,
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
    let model = models::load(&model_path, architecture)
        .with_context(|| format!("failed to load model from {}", model_path.display()))?;
    info!(
        model = %served_model_name,
        elapsed_ms = started.elapsed().as_millis(),
        "model loaded"
    );
    let batcher = Batcher::new(
        Arc::new(model),
        served_model_name.clone(),
        args.max_batch_size,
        Duration::from_millis(args.batch_wait_ms),
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
    let app = api::router(batcher);
    let listener = tokio::net::TcpListener::bind(address).await?;
    info!(%address, model = %served_model_name, "sys1 ready");
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown())
        .await?;
    info!("sys1 stopped");
    Ok(())
}

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
    }

    #[test]
    fn accepts_a_local_model_path() {
        let args = Args::try_parse_from(["sys1", "--model-path", "/models/laya"]).unwrap();
        assert_eq!(args.model_path, Some(PathBuf::from("/models/laya")));
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
}

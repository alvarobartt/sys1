use anyhow::Context;
use clap::Parser;
use std::{
    net::{IpAddr, SocketAddr},
    path::PathBuf,
    sync::Arc,
    time::Duration,
};
use sys1::{api, batching::Batcher, models};

#[derive(Parser)]
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
    #[arg(long, default_value = "127.0.0.1")]
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
    let args = Args::parse();
    let address = SocketAddr::new(args.host, args.port);
    let (model_path, source_name, architecture) = match args.model_path {
        Some(path) => {
            if args.served_model_name.is_none() {
                eprintln!("warning: --served-model-name is recommended when using --model-path");
            }
            let name = path.display().to_string();
            let architecture = models::Architecture::from_path(&path)?;
            (path, name, architecture)
        }
        None => {
            let architecture = models::Architecture::from_model_id(&args.model_id)?;
            (
                sys1::hub::download(&args.model_id, &args.revision).await?,
                args.model_id,
                architecture,
            )
        }
    };
    let served_model_name = args.served_model_name.unwrap_or(source_name);
    let model = models::load(&model_path, architecture)
        .with_context(|| format!("failed to load model from {}", model_path.display()))?;
    let app = api::router(Batcher::new(
        Arc::new(model),
        served_model_name,
        args.max_batch_size,
        Duration::from_millis(args.batch_wait_ms),
    ));
    let listener = tokio::net::TcpListener::bind(address).await?;
    println!("listening on http://{address}");
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown())
        .await?;
    Ok(())
}

async fn shutdown() {
    let _ = tokio::signal::ctrl_c().await;
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

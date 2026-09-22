use anyhow::{Context, bail};
use hf_hub::HFClient;
use std::path::PathBuf;

const MODEL_FILES: &[&str] = &[
    "encoder/config.json",
    "model.safetensors",
    "rl_agent_config.json",
    "tokenizer/tokenizer.json",
];

pub async fn download(model_id: &str, revision: &str) -> anyhow::Result<PathBuf> {
    let Some((owner, name)) = model_id.split_once('/') else {
        bail!("model id must use the owner/name format")
    };
    if owner.is_empty() || name.is_empty() || name.contains('/') {
        bail!("model id must use the owner/name format")
    }

    HFClient::new()
        .context("failed to create Hugging Face client")?
        .model(owner, name)
        .snapshot_download()
        .revision(revision)
        .allow_patterns(MODEL_FILES.iter().map(|path| (*path).to_owned()).collect())
        .send()
        .await
        .with_context(|| format!("failed to download {model_id} at revision {revision}"))
}

use anyhow::{Context, bail};
use hf_hub::{
    HFClient,
    progress::{DownloadEvent, FileStatus, ProgressEvent, ProgressHandler},
};
use std::{collections::HashMap, path::PathBuf, sync::Mutex};
use tracing::info;

const MODEL_FILES: &[&str] = &[
    "encoder/config.json",
    "model.safetensors",
    "rl_agent_config.json",
    "tokenizer/tokenizer.json",
];

#[derive(Default)]
struct DownloadProgress(Mutex<DownloadState>);

#[derive(Default)]
struct DownloadState {
    total_files: usize,
    total_bytes: u64,
    files: HashMap<String, (u64, bool)>,
    aggregate_bytes: u64,
    last_percent: u64,
    last_files: usize,
    started: bool,
}

impl ProgressHandler for DownloadProgress {
    fn on_progress(&self, event: &ProgressEvent) {
        let ProgressEvent::Download(event) = event else {
            return;
        };
        let update = {
            let mut state = self.0.lock().unwrap();
            let first = match event {
                DownloadEvent::Start {
                    total_files,
                    total_bytes,
                } if !state.started => {
                    state.total_files = *total_files;
                    state.total_bytes = *total_bytes;
                    state.started = true;
                    true
                }
                DownloadEvent::Progress { files } => {
                    for file in files {
                        let entry = state.files.entry(file.filename.clone()).or_default();
                        entry.0 = entry.0.max(file.bytes_completed);
                        entry.1 |= file.status == FileStatus::Complete;
                    }
                    false
                }
                DownloadEvent::AggregateProgress {
                    bytes_completed, ..
                } => {
                    state.aggregate_bytes = state.aggregate_bytes.max(*bytes_completed);
                    false
                }
                DownloadEvent::Complete => false,
                DownloadEvent::Start { .. } => return,
            };
            let completed_files = state.files.values().filter(|(_, done)| *done).count();
            let file_bytes = state
                .files
                .values()
                .fold(0u64, |sum, (bytes, _)| sum.saturating_add(*bytes));
            let downloaded = file_bytes.max(state.aggregate_bytes).min(state.total_bytes);
            let complete = matches!(event, DownloadEvent::Complete);
            let percent = if complete {
                100
            } else if state.total_bytes > 0 {
                ((downloaded as u128 * 100 / state.total_bytes as u128) as u64).min(99)
            } else if state.total_files > 0 {
                (completed_files as u64 * 100 / state.total_files as u64).min(99)
            } else {
                0
            }
            .max(state.last_percent);
            if state.total_bytes == 0 && !complete {
                None
            } else if first
                || complete
                || percent / 5 > state.last_percent / 5
                || completed_files > state.last_files
            {
                state.last_percent = percent;
                state.last_files = completed_files;
                Some((
                    percent,
                    downloaded,
                    state.total_bytes,
                    completed_files,
                    state.total_files,
                ))
            } else {
                None
            }
        };
        if let Some((percent, downloaded, total_bytes, completed_files, total_files)) = update {
            if total_bytes == 0 {
                info!(files = completed_files, "model snapshot cached");
                return;
            }
            let filled = (percent / 5) as usize;
            let bar = format!("[{}{}]", "=".repeat(filled), " ".repeat(20 - filled));
            info!(
                progress = %bar,
                percent,
                bytes = downloaded,
                total_bytes,
                files = completed_files,
                total_files,
                "model download progress"
            );
        }
    }
}

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
        .progress(DownloadProgress::default())
        .send()
        .await
        .with_context(|| format!("failed to download {model_id} at revision {revision}"))
}

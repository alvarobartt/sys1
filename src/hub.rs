use anyhow::{Context, bail};
use hf_hub::{
    HFClient,
    progress::{DownloadEvent, FileStatus, ProgressEvent, ProgressHandler},
};
use indicatif::{ProgressBar, ProgressStyle};
use std::{collections::HashMap, path::PathBuf, sync::Mutex, time::Duration};
use tracing::info;

const MODEL_FILES: &[&str] = &[
    "encoder/config.json",
    "model.safetensors",
    "rl_agent_config.json",
    "tokenizer/tokenizer.json",
];

struct DownloadProgress {
    bar: ProgressBar,
    interactive: bool,
    state: Mutex<DownloadState>,
}

#[derive(Default)]
struct DownloadState {
    total_files: usize,
    total_bytes: u64,
    files: HashMap<String, (u64, bool)>,
    aggregate_bytes: u64,
    logged_quarter: u64,
    started: bool,
}

impl ProgressHandler for DownloadProgress {
    fn on_progress(&self, event: &ProgressEvent) {
        let ProgressEvent::Download(event) = event else {
            return;
        };
        let (
            started,
            complete,
            downloaded,
            total_bytes,
            completed_files,
            total_files,
            percent,
            log,
        ) = {
            let Ok(mut state) = self.state.lock() else {
                return;
            };
            let started = match event {
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
                // Snapshot downloads also emit Start for individual files.
                DownloadEvent::Start { .. } => return,
            };
            let complete = matches!(event, DownloadEvent::Complete);
            let completed_files = state.files.values().filter(|(_, done)| *done).count();
            let file_bytes = state
                .files
                .values()
                .fold(0u64, |sum, (bytes, _)| sum.saturating_add(*bytes));
            // Xet batch totals and per-file updates overlap, so do not add them.
            let downloaded = file_bytes.max(state.aggregate_bytes).min(state.total_bytes);
            let percent = if state.total_bytes > 0 {
                (downloaded as u128 * 100 / state.total_bytes as u128) as u64
            } else {
                0
            };
            let quarter = percent / 25;
            let log = !self.interactive
                && state.total_bytes > 0
                && (started || (quarter > state.logged_quarter && !complete));
            state.logged_quarter = state.logged_quarter.max(quarter);
            (
                started,
                complete,
                downloaded,
                state.total_bytes,
                completed_files,
                state.total_files,
                percent,
                log,
            )
        };
        if complete {
            if total_bytes == 0 {
                self.bar.finish_and_clear();
                info!(files = completed_files, "model snapshot cached");
            } else {
                self.bar.set_position(total_bytes);
            }
            return;
        }
        if total_bytes == 0 {
            return;
        }
        if started && self.interactive {
            self.bar.set_length(total_bytes);
            self.bar.set_style(
                ProgressStyle::with_template(
                    "{spinner:.cyan} {msg} [{wide_bar:.cyan/blue}] {bytes}/{total_bytes} ({percent}%) {elapsed_precise}",
                )
                .expect("Valid model download progress template"),
            );
            self.bar.set_message("Downloading model");
        }
        if self.interactive {
            self.bar.set_position(downloaded);
        } else if log {
            info!(
                bytes = downloaded,
                total_bytes,
                percent,
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

    let client = HFClient::new().context("failed to create Hugging Face client")?;
    let bar = ProgressBar::new_spinner();
    let interactive = !bar.is_hidden();
    if interactive {
        bar.set_style(
            ProgressStyle::with_template("{spinner:.cyan} Resolving model files {elapsed_precise}")
                .expect("Valid model resolution progress template"),
        );
        bar.enable_steady_tick(Duration::from_millis(120));
    }
    let result = client
        .model(owner, name)
        .snapshot_download()
        .revision(revision)
        .allow_patterns(MODEL_FILES.iter().map(|path| (*path).to_owned()).collect())
        .progress(DownloadProgress {
            bar: bar.clone(),
            interactive,
            state: Mutex::new(DownloadState::default()),
        })
        .send()
        .await;
    if interactive && result.is_ok() && bar.length().is_some_and(|length| length > 0) {
        bar.finish_with_message("Model downloaded");
        eprintln!();
    } else {
        bar.finish_and_clear();
    }
    result.with_context(|| format!("failed to download {model_id} at revision {revision}"))
}

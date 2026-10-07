use anyhow::{Context, bail};
use hf_hub::{
    HFClient,
    progress::{DownloadEvent, FileStatus, ProgressEvent, ProgressHandler},
};
use indicatif::{ProgressBar, ProgressStyle};
use std::{
    collections::HashMap,
    path::PathBuf,
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};

const STALL_TIMEOUT: Duration = Duration::from_secs(600);
const MODEL_FILES: &[&str] = &[
    "encoder/config.json",
    "model.safetensors",
    "rl_agent_config.json",
    "tokenizer/tokenizer.json",
];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DownloadSource {
    Cache,
    HuggingFace,
}

impl DownloadSource {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Cache => "cache",
            Self::HuggingFace => "huggingface",
        }
    }
}

pub struct DownloadOutcome {
    pub path: PathBuf,
    pub source: DownloadSource,
}

struct DownloadProgress {
    bar: ProgressBar,
    state: Arc<Mutex<DownloadState>>,
}

struct DownloadState {
    started: bool,
    total_bytes: u64,
    files: HashMap<String, u64>,
    aggregate_bytes: u64,
    transfer_started: bool,
    last_change: Instant,
}

impl Default for DownloadState {
    fn default() -> Self {
        Self {
            started: false,
            total_bytes: 0,
            files: HashMap::new(),
            aggregate_bytes: 0,
            transfer_started: false,
            last_change: Instant::now(),
        }
    }
}

impl DownloadState {
    fn source(&self) -> DownloadSource {
        if self.transfer_started {
            DownloadSource::HuggingFace
        } else {
            DownloadSource::Cache
        }
    }

    fn downloaded(&self) -> u64 {
        let file_bytes = self
            .files
            .values()
            .fold(0u64, |sum, bytes| sum.saturating_add(*bytes));
        // Xet reports the same transfer through aggregate and per-file events.
        file_bytes.max(self.aggregate_bytes).min(self.total_bytes)
    }
}

impl ProgressHandler for DownloadProgress {
    fn on_progress(&self, event: &ProgressEvent) {
        let ProgressEvent::Download(event) = event else {
            return;
        };
        let Ok(mut state) = self.state.lock() else {
            return;
        };
        match event {
            DownloadEvent::Start { total_bytes, .. } if !state.started => {
                state.started = true;
                state.total_bytes = *total_bytes;
                state.last_change = Instant::now();
                if *total_bytes > 0 {
                    self.bar.set_length(*total_bytes);
                    self.bar.set_style(
                        ProgressStyle::with_template("Downloading [{bar:30}] {percent}%")
                            .expect("Valid download progress template")
                            .progress_chars("=> "),
                    );
                }
            }
            DownloadEvent::Progress { files } => {
                for file in files {
                    if file.status != FileStatus::Complete {
                        state.transfer_started = true;
                    }
                    let entry = state.files.entry(file.filename.clone()).or_default();
                    if file.bytes_completed > *entry {
                        *entry = file.bytes_completed;
                        state.last_change = Instant::now();
                    }
                }
            }
            DownloadEvent::AggregateProgress {
                bytes_completed, ..
            } => {
                if *bytes_completed > state.aggregate_bytes {
                    state.aggregate_bytes = *bytes_completed;
                    state.transfer_started = true;
                    state.last_change = Instant::now();
                }
            }
            DownloadEvent::Complete => return,
            // Snapshot downloads may also emit Start for individual files.
            DownloadEvent::Start { .. } => return,
        }
        if state.total_bytes > 0 {
            self.bar.set_position(state.downloaded());
        }
    }
}

pub async fn download(model_id: &str, revision: &str) -> anyhow::Result<PathBuf> {
    Ok(download_with_source(model_id, revision).await?.path)
}

pub async fn download_with_source(
    model_id: &str,
    revision: &str,
) -> anyhow::Result<DownloadOutcome> {
    let Some((owner, name)) = model_id.split_once('/') else {
        bail!("model id must use the owner/name format")
    };
    if owner.is_empty() || name.is_empty() || name.contains('/') {
        bail!("model id must use the owner/name format")
    }

    let client = HFClient::new().context("failed to create Hugging Face client")?;
    let bar = ProgressBar::new(0);
    let state = Arc::new(Mutex::new(DownloadState::default()));
    let repository = client.model(owner, name);
    let download = repository
        .snapshot_download()
        .revision(revision)
        .allow_patterns(MODEL_FILES.iter().map(|path| (*path).to_owned()).collect())
        .progress(DownloadProgress {
            bar: bar.clone(),
            state: Arc::clone(&state),
        })
        .send();
    tokio::pin!(download);
    let mut check = tokio::time::interval(Duration::from_secs(5));
    let result = loop {
        tokio::select! {
            result = &mut download => break result.context("model download failed"),
            _ = check.tick() => {
                if state.lock().is_ok_and(|state| state.last_change.elapsed() >= STALL_TIMEOUT) {
                    break Err(anyhow::anyhow!("model download made no progress for {} seconds; check the network connection and retry", STALL_TIMEOUT.as_secs()));
                }
            }
        }
    };
    let path =
        result.with_context(|| format!("failed to download {model_id} at revision {revision}"));
    if let (true, Some(total)) = (path.is_ok(), bar.length().filter(|length| *length > 0)) {
        bar.set_position(total);
        bar.finish();
    } else {
        bar.finish_and_clear();
    }
    let path = path?;
    let source = state
        .lock()
        .map_err(|_| anyhow::anyhow!("model download progress state poisoned"))?
        .source();
    Ok(DownloadOutcome { path, source })
}

#[cfg(test)]
mod tests {
    use super::*;
    use hf_hub::progress::FileProgress;

    fn handler() -> (DownloadProgress, Arc<Mutex<DownloadState>>) {
        let state = Arc::new(Mutex::new(DownloadState::default()));
        (
            DownloadProgress {
                bar: ProgressBar::hidden(),
                state: Arc::clone(&state),
            },
            state,
        )
    }

    #[test]
    fn source_distinguishes_cached_files_from_transfers() {
        let (handler, state) = handler();
        handler.on_progress(&ProgressEvent::Download(DownloadEvent::Start {
            total_files: 2,
            total_bytes: 100,
        }));
        handler.on_progress(&ProgressEvent::Download(DownloadEvent::Progress {
            files: vec![FileProgress {
                filename: "cached.bin".into(),
                bytes_completed: 100,
                total_bytes: 100,
                status: FileStatus::Complete,
            }],
        }));
        assert_eq!(state.lock().unwrap().source(), DownloadSource::Cache);
        handler.on_progress(&ProgressEvent::Download(DownloadEvent::Progress {
            files: vec![FileProgress {
                filename: "remote.bin".into(),
                bytes_completed: 0,
                total_bytes: 100,
                status: FileStatus::Started,
            }],
        }));
        assert_eq!(state.lock().unwrap().source(), DownloadSource::HuggingFace);
    }

    #[test]
    fn aggregate_progress_does_not_double_count_file_bytes() {
        let (handler, state) = handler();
        handler.on_progress(&ProgressEvent::Download(DownloadEvent::Start {
            total_files: 1,
            total_bytes: 100,
        }));
        handler.on_progress(&ProgressEvent::Download(DownloadEvent::Progress {
            files: vec![FileProgress {
                filename: "model.safetensors".into(),
                bytes_completed: 40,
                total_bytes: 100,
                status: FileStatus::InProgress,
            }],
        }));
        handler.on_progress(&ProgressEvent::Download(DownloadEvent::AggregateProgress {
            bytes_completed: 50,
            total_bytes: 100,
            bytes_per_sec: None,
        }));
        assert_eq!(state.lock().unwrap().downloaded(), 50);
    }
}

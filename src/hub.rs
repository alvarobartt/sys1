use anyhow::{Context, bail};
use hf_hub::{
    HFClient,
    progress::{DownloadEvent, FileStatus, ProgressEvent, ProgressHandler},
};
use indicatif::{ProgressBar, ProgressStyle};
use std::{
    collections::HashMap,
    ffi::OsString,
    path::PathBuf,
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};

const STALL_TIMEOUT: Duration = Duration::from_secs(600);
const DOWNLOAD_WORKERS: usize = 8;
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
    total_files: usize,
    total_bytes: u64,
    files: HashMap<String, FileState>,
    aggregate_bytes: u64,
    transfer_started: bool,
    last_change: Instant,
}

#[derive(Default)]
struct FileState {
    bytes: u64,
    complete: bool,
    active: bool,
}

impl Default for DownloadState {
    fn default() -> Self {
        Self {
            started: false,
            total_files: 0,
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
            .fold(0u64, |sum, file| sum.saturating_add(file.bytes));
        // Xet reports the same transfer through aggregate and per-file events.
        file_bytes.max(self.aggregate_bytes).min(self.total_bytes)
    }

    fn status(&self) -> String {
        let completed = self.files.values().filter(|file| file.complete).count();
        let mut active = self
            .files
            .iter()
            .filter(|(_, file)| file.active && !file.complete)
            .map(|(name, _)| name.as_str());
        let file = active.next();
        let active_count = usize::from(file.is_some()) + active.count();
        let detail = match (file, active_count) {
            (Some(name), 1) => format!("; {name}"),
            (_, count) if count > 1 => format!("; {count} active"),
            _ => String::new(),
        };
        let waiting =
            if self.transfer_started && self.last_change.elapsed() >= Duration::from_secs(5) {
                format!(" (idle {}s)", self.last_change.elapsed().as_secs())
            } else {
                String::new()
            };
        format!("{completed}/{} files{detail}{waiting}", self.total_files)
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
            DownloadEvent::Start {
                total_files,
                total_bytes,
            } if !state.started => {
                state.started = true;
                state.total_files = *total_files;
                state.total_bytes = *total_bytes;
                state.last_change = Instant::now();
                if *total_bytes > 0 {
                    self.bar.set_length(*total_bytes);
                    self.bar.set_style(
                        ProgressStyle::with_template("Downloading [{bar:20}] {percent}% {msg}")
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
                    entry.active |= file.status != FileStatus::Complete;
                    entry.complete |= file.status == FileStatus::Complete;
                    if file.bytes_completed > entry.bytes {
                        entry.bytes = file.bytes_completed;
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
            self.bar.set_message(state.status());
            self.bar.set_position(state.downloaded());
        }
    }
}

fn token_from_file_or_env(
    hf_home: Option<OsString>,
    home: Option<OsString>,
    env_token: Option<String>,
) -> Option<String> {
    let root = hf_home
        .filter(|path| !path.is_empty())
        .map(PathBuf::from)
        .or_else(|| {
            home.filter(|path| !path.is_empty())
                .map(|path| PathBuf::from(path).join(".cache/huggingface"))
        });
    root.and_then(|path| std::fs::read_to_string(path.join("token")).ok())
        .map(|token| token.trim().to_owned())
        .filter(|token| !token.is_empty())
        .or_else(|| {
            env_token
                .map(|token| token.trim().to_owned())
                .filter(|token| !token.is_empty())
        })
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

    let token = token_from_file_or_env(
        std::env::var_os("HF_HOME"),
        std::env::var_os("HOME"),
        std::env::var("HF_TOKEN").ok(),
    );
    let token_available = token.is_some();
    let mut client_builder = HFClient::builder();
    if let Some(token) = token {
        client_builder = client_builder.token(token);
    }
    let client = client_builder
        .build()
        .context("failed to create Hugging Face client")?;
    tracing::info!(
        token_found = token_available,
        workers = DOWNLOAD_WORKERS,
        "Fetching model files"
    );
    let bar = ProgressBar::new(0);
    bar.set_style(
        ProgressStyle::with_template("Resolving model files...")
            .expect("Valid model resolution template"),
    );
    bar.tick();
    let state = Arc::new(Mutex::new(DownloadState::default()));
    let repository = client.model(owner, name);
    let download = repository
        .snapshot_download()
        .revision(revision)
        .allow_patterns(MODEL_FILES.iter().map(|path| (*path).to_owned()).collect())
        .max_workers(DOWNLOAD_WORKERS)
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
                if let Ok(state) = state.lock() {
                    if state.started && state.total_bytes > 0 {
                        bar.set_message(state.status());
                        bar.tick();
                    }
                    if state.last_change.elapsed() >= STALL_TIMEOUT {
                        break Err(anyhow::anyhow!("model download made no progress for {} seconds; check the network connection and retry", STALL_TIMEOUT.as_secs()));
                    }
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
    use std::sync::atomic::{AtomicUsize, Ordering};

    static TOKEN_TEST_ID: AtomicUsize = AtomicUsize::new(0);

    #[test]
    fn token_file_precedes_env_and_hf_home_precedes_default_home() {
        let root = std::env::temp_dir().join(format!(
            "sys1-hub-token-{}-{}",
            std::process::id(),
            TOKEN_TEST_ID.fetch_add(1, Ordering::Relaxed)
        ));
        let default_home = root.join("default");
        let hf_home = root.join("hf");
        std::fs::create_dir_all(default_home.join(".cache/huggingface")).unwrap();
        std::fs::create_dir_all(&hf_home).unwrap();
        std::fs::write(
            default_home.join(".cache/huggingface/token"),
            "default-token\n",
        )
        .unwrap();
        std::fs::write(hf_home.join("token"), "hf-home-token\n").unwrap();

        assert_eq!(
            token_from_file_or_env(
                Some(hf_home.clone().into_os_string()),
                Some(default_home.clone().into_os_string()),
                Some("env-token".into())
            )
            .as_deref(),
            Some("hf-home-token")
        );
        assert_eq!(
            token_from_file_or_env(
                None,
                Some(default_home.into_os_string()),
                Some("env-token".into())
            )
            .as_deref(),
            Some("default-token")
        );
        std::fs::remove_file(hf_home.join("token")).unwrap();
        assert_eq!(
            token_from_file_or_env(
                Some(hf_home.into_os_string()),
                None,
                Some("env-token".into())
            )
            .as_deref(),
            Some("env-token")
        );
        std::fs::remove_dir_all(root).unwrap();
    }

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
        let state = state.lock().unwrap();
        assert_eq!(state.downloaded(), 50);
        assert_eq!(state.status(), "0/1 files; model.safetensors");
    }
}

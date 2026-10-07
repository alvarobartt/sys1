use anyhow::{Context, bail};
use hf_hub::{
    HFClient,
    progress::{DownloadEvent, FileStatus, ProgressEvent, ProgressHandler},
};
use indicatif::{HumanBytes, HumanDuration, MultiProgress, ProgressBar, ProgressStyle};
use std::{
    collections::HashMap,
    path::PathBuf,
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};
use tracing::info;

// Xet reports reconstructed bytes in large batches, so a healthy slow transfer
// can go several minutes without a new byte count.
const STALL_TIMEOUT: Duration = Duration::from_secs(600);

fn rate_and_eta(bytes_per_sec: f64, remaining: u64) -> String {
    if bytes_per_sec <= 0.0 || !bytes_per_sec.is_finite() {
        return "speed -- ETA --".to_owned();
    }
    let eta = Duration::from_secs_f64((remaining as f64 / bytes_per_sec).min(u32::MAX as f64));
    format!(
        "{}/s ETA {:#}",
        HumanBytes(bytes_per_sec.round() as u64),
        HumanDuration(eta)
    )
}

const MODEL_FILES: &[&str] = &[
    "encoder/config.json",
    "model.safetensors",
    "rl_agent_config.json",
    "tokenizer/tokenizer.json",
];

struct DownloadProgress {
    multi: MultiProgress,
    bar: ProgressBar,
    interactive: bool,
    state: Arc<Mutex<DownloadState>>,
    file_bars: Arc<Mutex<HashMap<String, ProgressBar>>>,
}

struct DownloadState {
    total_files: usize,
    total_bytes: u64,
    files: HashMap<String, (u64, u64, bool)>,
    aggregate_bytes: u64,
    logged_quarter: u64,
    started: bool,
    last_change: Instant,
    active_file: String,
    last_log: Instant,
    last_rate_sample: Instant,
    last_rate_bytes: u64,
    bytes_per_sec: f64,
}

impl Default for DownloadState {
    fn default() -> Self {
        Self {
            total_files: 0,
            total_bytes: 0,
            files: HashMap::new(),
            aggregate_bytes: 0,
            logged_quarter: 0,
            started: false,
            last_change: Instant::now(),
            active_file: String::new(),
            last_log: Instant::now(),
            last_rate_sample: Instant::now(),
            last_rate_bytes: 0,
            bytes_per_sec: 0.0,
        }
    }
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
            active_file,
            bytes_per_sec,
            rate_status,
            active_files,
            active_count,
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
                    state.last_change = Instant::now();
                    true
                }
                DownloadEvent::Progress { files } => {
                    for file in files {
                        let entry = state.files.entry(file.filename.clone()).or_default();
                        let changed = file.bytes_completed > entry.0
                            || (file.status == FileStatus::Complete && !entry.2);
                        entry.0 = entry.0.max(file.bytes_completed);
                        entry.1 = entry.1.max(file.total_bytes);
                        entry.2 |= file.status == FileStatus::Complete;
                        if changed {
                            state.last_change = Instant::now();
                        }
                        if file.status != FileStatus::Complete {
                            state.active_file = file.filename.clone();
                        }
                    }
                    false
                }
                DownloadEvent::AggregateProgress {
                    bytes_completed,
                    bytes_per_sec,
                    ..
                } => {
                    if *bytes_completed > state.aggregate_bytes {
                        state.last_change = Instant::now();
                    }
                    state.aggregate_bytes = state.aggregate_bytes.max(*bytes_completed);
                    if let Some(rate) =
                        bytes_per_sec.filter(|rate| rate.is_finite() && *rate >= 0.0)
                    {
                        state.bytes_per_sec = rate;
                    }
                    false
                }
                DownloadEvent::Complete => false,
                // Snapshot downloads also emit Start for individual files.
                DownloadEvent::Start { .. } => return,
            };
            let complete = matches!(event, DownloadEvent::Complete);
            let completed_files = state.files.values().filter(|(_, _, done)| *done).count();
            let file_bytes = state
                .files
                .values()
                .fold(0u64, |sum, (bytes, _, _)| sum.saturating_add(*bytes));
            // Xet batch totals and per-file updates overlap, so do not add them.
            let downloaded = file_bytes.max(state.aggregate_bytes).min(state.total_bytes);
            let elapsed = state.last_rate_sample.elapsed();
            if downloaded > state.last_rate_bytes && elapsed >= Duration::from_secs(1) {
                state.bytes_per_sec =
                    (downloaded - state.last_rate_bytes) as f64 / elapsed.as_secs_f64();
                state.last_rate_bytes = downloaded;
                state.last_rate_sample = Instant::now();
            }
            let percent = if state.total_bytes > 0 {
                (downloaded as u128 * 100 / state.total_bytes as u128) as u64
            } else {
                0
            };
            let quarter = percent / 25;
            let log = !self.interactive
                && state.total_bytes > 0
                && (started
                    || quarter > state.logged_quarter
                    || state.last_log.elapsed() >= Duration::from_secs(30));
            if log {
                state.last_log = Instant::now();
            }
            state.logged_quarter = state.logged_quarter.max(quarter);
            let active_file = if state.active_file.is_empty() {
                "waiting for transfer".to_owned()
            } else {
                state.active_file.clone()
            };
            let bytes_per_sec = if state.last_change.elapsed() >= Duration::from_secs(5) {
                0.0
            } else {
                state.bytes_per_sec
            };
            let rate_status = if state.last_change.elapsed() >= Duration::from_secs(5) {
                "waiting for next update".to_owned()
            } else {
                rate_and_eta(bytes_per_sec, state.total_bytes.saturating_sub(downloaded))
            };
            let active_files = state
                .files
                .iter()
                .filter(|(_, (_, total, done))| !done && *total > 0)
                .map(|(name, (bytes, total, _))| (name.clone(), *bytes, *total))
                .collect::<Vec<_>>();
            let active_count = active_files.len();
            (
                started,
                complete,
                downloaded,
                state.total_bytes,
                completed_files,
                state.total_files,
                percent,
                log,
                active_file,
                bytes_per_sec,
                rate_status,
                active_files,
                active_count,
            )
        };
        if self.interactive {
            self.update_file_bars(active_files);
        }
        if complete {
            if total_bytes == 0 {
                self.bar.finish_and_clear();
                info!(files = completed_files, "model snapshot cached");
            } else {
                self.bar.set_prefix("done");
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
                    "{msg}\n[{bar:20}] {bytes}/{total_bytes} ({percent}%) {prefix}",
                )
                .expect("Valid model download progress template")
                .progress_chars("#>-"),
            );
        }
        if self.interactive {
            let name = if active_count > 1 {
                format!("{active_count} files")
            } else {
                active_file
            };
            self.bar.set_message(format!(
                "Downloading {name} ({completed_files}/{total_files} files)"
            ));
            self.bar.set_prefix(rate_status);
            self.bar.set_position(downloaded);
        } else if log {
            info!(
                bytes = downloaded,
                total_bytes,
                percent,
                files = completed_files,
                total_files,
                file = %active_file,
                bytes_per_sec,
                "model download progress"
            );
        }
    }
}

impl DownloadProgress {
    fn update_file_bars(&self, mut active: Vec<(String, u64, u64)>) {
        let Ok(mut bars) = self.file_bars.lock() else {
            return;
        };
        if active.len() < 2 {
            for (_, bar) in bars.drain() {
                bar.finish_and_clear();
            }
            return;
        }
        active.sort_by(|a, b| a.0.cmp(&b.0));
        bars.retain(|name, bar| {
            if active.iter().any(|(file, _, _)| file == name) {
                true
            } else {
                bar.finish_and_clear();
                false
            }
        });
        for (name, bytes, total) in active {
            let bar = bars.entry(name.clone()).or_insert_with(|| {
                let bar = self.multi.add(ProgressBar::new(total));
                bar.set_style(
                    ProgressStyle::with_template("  {msg} [{bar:20}] {bytes}/{total_bytes}")
                        .expect("Valid per-file download progress template")
                        .progress_chars("#>-"),
                );
                bar.set_message(name);
                bar
            });
            bar.set_length(total);
            if bar.position() != bytes {
                bar.set_position(bytes);
            }
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
    let multi = MultiProgress::new();
    let bar = multi.add(ProgressBar::new_spinner());
    let interactive = !multi.is_hidden();
    if interactive {
        bar.set_style(
            ProgressStyle::with_template("Resolving model files {elapsed_precise}")
                .expect("Valid model resolution progress template"),
        );
        bar.enable_steady_tick(Duration::from_millis(120));
    }
    let state = Arc::new(Mutex::new(DownloadState::default()));
    let file_bars = Arc::new(Mutex::new(HashMap::new()));
    let repository = client.model(owner, name);
    let download = repository
        .snapshot_download()
        .revision(revision)
        .allow_patterns(MODEL_FILES.iter().map(|path| (*path).to_owned()).collect())
        .progress(DownloadProgress {
            multi,
            bar: bar.clone(),
            interactive,
            state: Arc::clone(&state),
            file_bars: Arc::clone(&file_bars),
        })
        .send();
    tokio::pin!(download);
    let mut check = tokio::time::interval(Duration::from_secs(5));
    let result = loop {
        tokio::select! {
            result = &mut download => break result.context("model download failed"),
            _ = check.tick() => {
                if let Ok(state) = state.lock() {
                    let idle = state.last_change.elapsed();
                    if interactive && state.started && idle >= Duration::from_secs(5) {
                        bar.set_prefix("waiting for next update");
                    }
                    if idle >= STALL_TIMEOUT {
                        break Err(anyhow::anyhow!("model download made no progress for {} seconds while downloading {}; check the network connection and retry", STALL_TIMEOUT.as_secs(), if state.active_file.is_empty() { "model files" } else { &state.active_file }));
                    }
                }
            }
        }
    };
    if let Ok(mut bars) = file_bars.lock() {
        for (_, child) in bars.drain() {
            child.finish_and_clear();
        }
    }
    if interactive && result.is_ok() && bar.length().is_some_and(|length| length > 0) {
        bar.finish_with_message("Model downloaded");
    } else {
        bar.finish_and_clear();
    }
    result.with_context(|| format!("failed to download {model_id} at revision {revision}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use hf_hub::progress::FileProgress;

    #[test]
    fn progress_tracks_active_file_and_only_resets_idle_time_when_bytes_advance() {
        let state = Arc::new(Mutex::new(DownloadState::default()));
        let handler = DownloadProgress {
            multi: MultiProgress::new(),
            bar: ProgressBar::hidden(),
            interactive: false,
            state: Arc::clone(&state),
            file_bars: Arc::new(Mutex::new(HashMap::new())),
        };
        handler.on_progress(&ProgressEvent::Download(DownloadEvent::Start {
            total_files: 1,
            total_bytes: 100,
        }));
        let file = FileProgress {
            filename: "model.safetensors".to_owned(),
            bytes_completed: 0,
            total_bytes: 100,
            status: FileStatus::Started,
        };
        handler.on_progress(&ProgressEvent::Download(DownloadEvent::Progress {
            files: vec![file.clone()],
        }));
        let old = Instant::now() - STALL_TIMEOUT;
        state.lock().unwrap().last_change = old;
        handler.on_progress(&ProgressEvent::Download(DownloadEvent::Progress {
            files: vec![file],
        }));
        assert_eq!(state.lock().unwrap().last_change, old);

        handler.on_progress(&ProgressEvent::Download(DownloadEvent::AggregateProgress {
            bytes_completed: 50,
            total_bytes: 100,
            bytes_per_sec: Some(25.0),
        }));
        let state = state.lock().unwrap();
        assert_eq!(state.active_file, "model.safetensors");
        assert_eq!(state.aggregate_bytes, 50);
        assert!(state.last_change.elapsed() < Duration::from_secs(1));
    }

    #[test]
    fn separate_bars_only_when_files_transfer_together() {
        let handler = DownloadProgress {
            multi: MultiProgress::new(),
            bar: ProgressBar::hidden(),
            interactive: true,
            state: Arc::new(Mutex::new(DownloadState::default())),
            file_bars: Arc::new(Mutex::new(HashMap::new())),
        };
        handler.on_progress(&ProgressEvent::Download(DownloadEvent::Start {
            total_files: 2,
            total_bytes: 200,
        }));
        let files = ["a.bin", "b.bin"].map(|name| FileProgress {
            filename: name.to_owned(),
            bytes_completed: 10,
            total_bytes: 100,
            status: FileStatus::InProgress,
        });
        handler.on_progress(&ProgressEvent::Download(DownloadEvent::Progress {
            files: files.to_vec(),
        }));
        assert_eq!(handler.file_bars.lock().unwrap().len(), 2);

        handler.on_progress(&ProgressEvent::Download(DownloadEvent::Progress {
            files: vec![FileProgress {
                bytes_completed: 100,
                status: FileStatus::Complete,
                ..files[0].clone()
            }],
        }));
        assert!(handler.file_bars.lock().unwrap().is_empty());
    }
}

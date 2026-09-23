use crate::{
    models::DecisionModel,
    schema::{ApiError, DecisionRequest, DecisionResponse},
};

use serde_json::{Map, json};
use std::{
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
    time::Duration,
};
use tokio::sync::{mpsc, oneshot};
use tracing::{debug, error};

struct Job {
    request: DecisionRequest,
    response: oneshot::Sender<Result<DecisionResponse, ApiError>>,
}

#[derive(Clone, Copy, Debug)]
pub struct BatcherConfig {
    pub max_batch_size: usize,
    pub max_batch_questions: usize,
    pub max_questions_per_request: usize,
    pub wait: Duration,
    pub queue_capacity: usize,
    pub response_timeout: Option<Duration>,
}

#[derive(Debug)]
pub struct BatcherStats {
    pub accepted_requests: u64,
    pub rejected_requests: u64,
    pub timed_out_requests: u64,
    pub queued_requests: u64,
    pub in_flight_requests: u64,
    pub batches: u64,
    pub batch_questions: u64,
    pub model_failures: u64,
    pub inference_microseconds: u64,
}

#[derive(Default)]
struct BatcherMetrics {
    accepted_requests: AtomicU64,
    rejected_requests: AtomicU64,
    timed_out_requests: AtomicU64,
    queued_requests: AtomicU64,
    in_flight_requests: AtomicU64,
    batches: AtomicU64,
    batch_questions: AtomicU64,
    model_failures: AtomicU64,
    inference_microseconds: AtomicU64,
}

#[derive(Clone)]
pub struct Batcher {
    sender: mpsc::Sender<Job>,
    served_model_name: Arc<str>,
    response_timeout: Option<Duration>,
    max_questions_per_request: usize,
    metrics: Arc<BatcherMetrics>,
}

impl Batcher {
    pub fn new(
        model: Arc<dyn DecisionModel>,
        served_model_name: String,
        config: BatcherConfig,
    ) -> Self {
        let max_batch_size = config.max_batch_size.max(1);
        let max_batch_questions = config.max_batch_questions.max(1);
        let max_questions_per_request = config
            .max_questions_per_request
            .max(1)
            .min(max_batch_questions);
        let wait = config.wait;
        let (sender, mut receiver) = mpsc::channel::<Job>(config.queue_capacity.max(1));
        let metrics = Arc::new(BatcherMetrics::default());
        let worker_metrics = metrics.clone();
        tokio::spawn(async move {
            let mut pending = None;
            loop {
                let first = match pending.take() {
                    Some(job) => job,
                    None => match receiver.recv().await {
                        Some(job) => {
                            worker_metrics
                                .queued_requests
                                .fetch_sub(1, Ordering::Relaxed);
                            job
                        }
                        None => break,
                    },
                };
                let mut question_count = first.request.questions.len();
                let mut jobs = vec![first];
                let deadline = tokio::time::Instant::now() + wait;
                while jobs.len() < max_batch_size {
                    match tokio::time::timeout_at(deadline, receiver.recv()).await {
                        Ok(Some(job)) => {
                            worker_metrics
                                .queued_requests
                                .fetch_sub(1, Ordering::Relaxed);
                            let next_questions = job.request.questions.len();
                            if question_count + next_questions > max_batch_questions {
                                pending = Some(job);
                                break;
                            }
                            question_count += next_questions;
                            jobs.push(job);
                        }
                        _ => break,
                    }
                }
                let batch_size = jobs.len();
                let started = std::time::Instant::now();
                worker_metrics.batches.fetch_add(1, Ordering::Relaxed);
                worker_metrics
                    .batch_questions
                    .fetch_add(question_count as u64, Ordering::Relaxed);
                worker_metrics
                    .in_flight_requests
                    .fetch_add(batch_size as u64, Ordering::Relaxed);
                debug!(batch_size, question_count, "inference batch started");
                let (requests, channels): (Vec<_>, Vec<_>) = jobs
                    .into_iter()
                    .map(|job| (job.request, job.response))
                    .unzip();
                let model = model.clone();
                let responses = tokio::task::spawn_blocking(move || model.predict_batch(requests))
                    .await
                    .unwrap_or_else(|error| {
                        error!(%error, "inference worker failed");
                        (0..batch_size)
                            .map(|_| Err(ApiError::internal(error.to_string())))
                            .collect()
                    });
                let failures = responses
                    .iter()
                    .filter(|response| response.is_err())
                    .count();
                worker_metrics
                    .model_failures
                    .fetch_add(failures as u64, Ordering::Relaxed);
                worker_metrics
                    .inference_microseconds
                    .fetch_add(started.elapsed().as_micros() as u64, Ordering::Relaxed);
                worker_metrics
                    .in_flight_requests
                    .fetch_sub(batch_size as u64, Ordering::Relaxed);
                debug!(
                    batch_size,
                    failures,
                    elapsed_ms = started.elapsed().as_millis(),
                    "inference batch completed"
                );
                let response_count = responses.len();
                let mut responses = responses.into_iter();
                for channel in channels {
                    let response = responses.next().unwrap_or_else(|| {
                        Err(ApiError::internal(
                            "inference worker returned fewer responses than requests",
                        ))
                    });
                    let _ = channel.send(response);
                }
                if response_count != batch_size {
                    error!(
                        batch_size,
                        response_count, "inference response count mismatch"
                    );
                }
            }
        });
        Self {
            sender,
            served_model_name: served_model_name.into(),
            response_timeout: config.response_timeout,
            max_questions_per_request,
            metrics,
        }
    }

    pub fn served_model_name(&self) -> &str {
        &self.served_model_name
    }

    pub fn is_ready(&self) -> bool {
        !self.sender.is_closed()
    }

    pub fn stats(&self) -> BatcherStats {
        BatcherStats {
            accepted_requests: self.metrics.accepted_requests.load(Ordering::Relaxed),
            rejected_requests: self.metrics.rejected_requests.load(Ordering::Relaxed),
            timed_out_requests: self.metrics.timed_out_requests.load(Ordering::Relaxed),
            queued_requests: self.metrics.queued_requests.load(Ordering::Relaxed),
            in_flight_requests: self.metrics.in_flight_requests.load(Ordering::Relaxed),
            batches: self.metrics.batches.load(Ordering::Relaxed),
            batch_questions: self.metrics.batch_questions.load(Ordering::Relaxed),
            model_failures: self.metrics.model_failures.load(Ordering::Relaxed),
            inference_microseconds: self.metrics.inference_microseconds.load(Ordering::Relaxed),
        }
    }

    pub fn reset_stats(&self) {
        self.metrics.accepted_requests.store(0, Ordering::Relaxed);
        self.metrics.rejected_requests.store(0, Ordering::Relaxed);
        self.metrics.timed_out_requests.store(0, Ordering::Relaxed);
        self.metrics.queued_requests.store(0, Ordering::Relaxed);
        self.metrics.in_flight_requests.store(0, Ordering::Relaxed);
        self.metrics.batches.store(0, Ordering::Relaxed);
        self.metrics.batch_questions.store(0, Ordering::Relaxed);
        self.metrics.model_failures.store(0, Ordering::Relaxed);
        self.metrics
            .inference_microseconds
            .store(0, Ordering::Relaxed);
    }

    pub async fn warmup(&self) -> Result<(), ApiError> {
        let mut questions = Map::new();
        questions.insert(
            "warmup".to_owned(),
            json!({
                "type": "choice",
                "instructions": "Choose the matching option.",
                "criteria": {
                    "first": "The first option.",
                    "second": "The second option."
                }
            }),
        );
        self.predict_inner(
            DecisionRequest {
                model: None,
                state: json!("Warm up the decision model before serving requests."),
                questions,
            },
            None,
        )
        .await
        .map(|_| ())
    }

    pub async fn predict(&self, request: DecisionRequest) -> Result<DecisionResponse, ApiError> {
        self.predict_inner(request, self.response_timeout).await
    }

    async fn predict_inner(
        &self,
        request: DecisionRequest,
        response_timeout: Option<Duration>,
    ) -> Result<DecisionResponse, ApiError> {
        if !matches_model(request.model.as_deref(), &self.served_model_name) {
            let model = request.model.as_deref().unwrap();
            return Err(ApiError::new(format!("unknown model: {model}")));
        }
        if request.questions.is_empty() {
            return Err(ApiError::new("questions must not be empty"));
        }
        if request.questions.len() > self.max_questions_per_request {
            return Err(ApiError::new(format!(
                "request has {} questions; maximum is {}",
                request.questions.len(),
                self.max_questions_per_request
            )));
        }
        let (response, receiver) = oneshot::channel();
        self.metrics.queued_requests.fetch_add(1, Ordering::Relaxed);
        if let Err(error) = self.sender.try_send(Job { request, response }) {
            self.metrics.queued_requests.fetch_sub(1, Ordering::Relaxed);
            self.metrics
                .rejected_requests
                .fetch_add(1, Ordering::Relaxed);
            return Err(match error {
                mpsc::error::TrySendError::Full(_) => {
                    ApiError::unavailable("inference queue is full")
                }
                mpsc::error::TrySendError::Closed(_) => {
                    ApiError::unavailable("inference worker stopped")
                }
            });
        }
        self.metrics
            .accepted_requests
            .fetch_add(1, Ordering::Relaxed);
        let received = if let Some(timeout) = response_timeout {
            tokio::time::timeout(timeout, receiver).await.map_err(|_| {
                self.metrics
                    .timed_out_requests
                    .fetch_add(1, Ordering::Relaxed);
                ApiError::timeout("inference request timed out")
            })?
        } else {
            receiver.await
        };
        let mut response =
            received.map_err(|_| ApiError::unavailable("inference worker stopped"))??;
        response.model = self.served_model_name.to_string();
        Ok(response)
    }
}

fn matches_model(requested: Option<&str>, served: &str) -> bool {
    requested.is_none_or(|model| model.is_empty() || model == served)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::Usage;
    use std::{sync::Mutex, thread, time::Duration};

    struct SlowModel;

    impl DecisionModel for SlowModel {
        fn predict_batch(
            &self,
            requests: Vec<DecisionRequest>,
        ) -> Vec<Result<DecisionResponse, ApiError>> {
            thread::sleep(Duration::from_millis(20));
            requests
                .into_iter()
                .map(|_| {
                    Ok(DecisionResponse {
                        model: String::new(),
                        answers: Map::new(),
                        usage: Usage {
                            input_tokens: 0,
                            output_tokens: 0,
                        },
                    })
                })
                .collect()
        }
    }

    #[derive(Default)]
    struct RecordingModel {
        batch_question_counts: Mutex<Vec<usize>>,
    }

    impl DecisionModel for RecordingModel {
        fn predict_batch(
            &self,
            requests: Vec<DecisionRequest>,
        ) -> Vec<Result<DecisionResponse, ApiError>> {
            self.batch_question_counts
                .lock()
                .unwrap()
                .push(requests.iter().map(|request| request.questions.len()).sum());
            requests
                .into_iter()
                .map(|_| {
                    Ok(DecisionResponse {
                        model: String::new(),
                        answers: Map::new(),
                        usage: Usage {
                            input_tokens: 0,
                            output_tokens: 0,
                        },
                    })
                })
                .collect()
        }
    }

    fn one_question_request() -> DecisionRequest {
        let mut questions = Map::new();
        questions.insert("q0".to_owned(), json!({}));
        DecisionRequest {
            model: None,
            state: json!("test"),
            questions,
        }
    }

    #[test]
    fn accepts_optional_or_matching_model_names() {
        let served = "convaiinnovations/laya";
        assert!(matches_model(None, served));
        assert!(matches_model(Some(""), served));
        assert!(matches_model(Some(served), served));
        assert!(!matches_model(Some("other/model"), served));
    }

    #[tokio::test]
    async fn times_out_slow_inference() {
        let batcher = Batcher::new(
            Arc::new(SlowModel),
            "test-model".to_owned(),
            BatcherConfig {
                max_batch_size: 1,
                max_batch_questions: 1,
                max_questions_per_request: 1,
                wait: Duration::ZERO,
                queue_capacity: 1,
                response_timeout: Some(Duration::from_millis(1)),
            },
        );
        let mut questions = Map::new();
        questions.insert("q0".to_owned(), json!({}));
        let error = batcher
            .predict(DecisionRequest {
                model: None,
                state: json!("test"),
                questions,
            })
            .await
            .unwrap_err();
        assert_eq!(error.status(), 504);
    }

    #[tokio::test]
    async fn rejects_empty_and_oversized_question_sets() {
        let batcher = Batcher::new(
            Arc::new(SlowModel),
            "test-model".to_owned(),
            BatcherConfig {
                max_batch_size: 1,
                max_batch_questions: 2,
                max_questions_per_request: 1,
                wait: Duration::ZERO,
                queue_capacity: 1,
                response_timeout: None,
            },
        );
        let empty = batcher
            .predict(DecisionRequest {
                model: None,
                state: json!("test"),
                questions: Map::new(),
            })
            .await
            .unwrap_err();
        assert_eq!(empty.status(), 400);

        let mut questions = Map::new();
        questions.insert("q0".to_owned(), json!({}));
        questions.insert("q1".to_owned(), json!({}));
        let oversized = batcher
            .predict(DecisionRequest {
                model: None,
                state: json!("test"),
                questions,
            })
            .await
            .unwrap_err();
        assert_eq!(oversized.status(), 400);
    }

    #[tokio::test]
    async fn caps_total_questions_in_dynamic_batches() {
        let model = Arc::new(RecordingModel::default());
        let batcher = Batcher::new(
            model.clone(),
            "test-model".to_owned(),
            BatcherConfig {
                max_batch_size: 8,
                max_batch_questions: 2,
                max_questions_per_request: 1,
                wait: Duration::from_millis(10),
                queue_capacity: 8,
                response_timeout: Some(Duration::from_secs(1)),
            },
        );
        let (first, second, third) = tokio::join!(
            batcher.predict(one_question_request()),
            batcher.predict(one_question_request()),
            batcher.predict(one_question_request()),
        );
        assert!(first.is_ok() && second.is_ok() && third.is_ok());

        let counts = model.batch_question_counts.lock().unwrap().clone();
        assert_eq!(counts.iter().sum::<usize>(), 3);
        assert_eq!(counts.len(), 2);
        assert!(counts.into_iter().all(|count| count <= 2));
    }
}

use crate::{
    metrics::{BatcherMetrics, EncodedMetrics},
    models::DecisionModel,
    schema::{ApiError, DecisionRequest, DecisionResponse},
};

use serde_json::{Map, json};
use std::{sync::Arc, time::Duration};
use tokio::sync::{mpsc, oneshot};
use tracing::{debug, error};

struct Job {
    request: DecisionRequest,
    response: oneshot::Sender<Result<DecisionResponse, ApiError>>,
    kind: RequestKind,
}

#[derive(Clone, Copy)]
enum RequestKind {
    Warmup,
    Inference,
}

impl RequestKind {
    fn records_metrics(self) -> bool {
        matches!(self, Self::Inference)
    }
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
        let metrics = Arc::new(BatcherMetrics::new());
        let worker_metrics = metrics.clone();
        tokio::spawn(async move {
            let mut pending = None;
            loop {
                let first = match pending.take() {
                    Some(job) => job,
                    None => match receiver.recv().await {
                        Some(job) => {
                            if job.kind.records_metrics() {
                                worker_metrics.request_dequeued();
                            }
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
                            if job.kind.records_metrics() {
                                worker_metrics.request_dequeued();
                            }
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
                let metric_batch_size =
                    jobs.iter().filter(|job| job.kind.records_metrics()).count();
                let metric_question_count = jobs
                    .iter()
                    .filter(|job| job.kind.records_metrics())
                    .map(|job| job.request.questions.len())
                    .sum::<usize>();
                let started = std::time::Instant::now();
                let batch_guard = (metric_batch_size > 0).then(|| {
                    worker_metrics.batch_started(metric_batch_size, metric_question_count)
                });
                debug!(batch_size, question_count, "inference batch started");
                let (requests, deliveries): (Vec<_>, Vec<_>) = jobs
                    .into_iter()
                    .map(|job| (job.request, (job.response, job.kind)))
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
                    .zip(&deliveries)
                    .filter(|(response, (_, kind))| kind.records_metrics() && response.is_err())
                    .count()
                    + deliveries
                        .iter()
                        .skip(responses.len())
                        .filter(|(_, kind)| kind.records_metrics())
                        .count();
                worker_metrics.model_requests_failed(failures);
                drop(batch_guard);
                debug!(
                    batch_size,
                    failures,
                    elapsed_ms = started.elapsed().as_millis(),
                    "inference batch completed"
                );
                let response_count = responses.len();
                let mut responses = responses.into_iter();
                for (channel, _) in deliveries {
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

    pub(crate) fn encode_metrics(&self) -> prometheus::Result<EncodedMetrics> {
        self.metrics.encode()
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
            RequestKind::Warmup,
        )
        .await
        .map(|_| ())
    }

    pub async fn predict(&self, request: DecisionRequest) -> Result<DecisionResponse, ApiError> {
        self.predict_inner(request, self.response_timeout, RequestKind::Inference)
            .await
    }

    async fn predict_inner(
        &self,
        request: DecisionRequest,
        response_timeout: Option<Duration>,
        kind: RequestKind,
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
        if kind.records_metrics() {
            self.metrics.request_queued();
        }
        if let Err(error) = self.sender.try_send(Job {
            request,
            response,
            kind,
        }) {
            if kind.records_metrics() {
                self.metrics.request_dequeued();
                self.metrics.request_rejected();
            }
            return Err(match error {
                mpsc::error::TrySendError::Full(_) => {
                    ApiError::unavailable("inference queue is full")
                }
                mpsc::error::TrySendError::Closed(_) => {
                    ApiError::unavailable("inference worker stopped")
                }
            });
        }
        if kind.records_metrics() {
            self.metrics.request_accepted();
        }
        let received = if let Some(timeout) = response_timeout {
            tokio::time::timeout(timeout, receiver).await.map_err(|_| {
                if kind.records_metrics() {
                    self.metrics.request_timed_out();
                }
                ApiError::timeout("inference request timed out")
            })?
        } else {
            receiver.await
        };
        let response = received.map_err(|_| {
            if kind.records_metrics() {
                self.metrics.request_failed();
            }
            ApiError::unavailable("inference worker stopped")
        })?;
        let mut response = match response {
            Ok(response) => {
                if kind.records_metrics() {
                    self.metrics.request_succeeded();
                }
                response
            }
            Err(error) => {
                if kind.records_metrics() {
                    self.metrics.request_failed();
                }
                return Err(error);
            }
        };
        response.model = self.served_model_name.to_string();
        Ok(response)
    }
}

fn matches_model(requested: Option<&str>, served: &str) -> bool {
    requested.is_none_or(|model| model.is_empty() || model == served || model == "jev-latest")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::Usage;
    use std::{
        sync::{
            Arc, Barrier, Mutex,
            atomic::{AtomicBool, Ordering},
        },
        thread,
        time::Duration,
    };

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

    struct FailingModel;

    impl DecisionModel for FailingModel {
        fn predict_batch(
            &self,
            requests: Vec<DecisionRequest>,
        ) -> Vec<Result<DecisionResponse, ApiError>> {
            requests
                .into_iter()
                .map(|_| Err(ApiError::internal("model failed")))
                .collect()
        }
    }

    struct BlockOnceModel {
        block: AtomicBool,
        started: Arc<Barrier>,
        release: Arc<Barrier>,
    }

    impl DecisionModel for BlockOnceModel {
        fn predict_batch(
            &self,
            requests: Vec<DecisionRequest>,
        ) -> Vec<Result<DecisionResponse, ApiError>> {
            if self.block.swap(false, Ordering::SeqCst) {
                self.started.wait();
                self.release.wait();
            }
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

    fn metric_value(batcher: &Batcher, name: &str) -> f64 {
        let encoded = batcher.encode_metrics().expect("encode metrics");
        let text = std::str::from_utf8(&encoded.body).expect("metrics are UTF-8");
        text.lines()
            .find_map(|line| {
                let (metric, value) = line.split_once(' ')?;
                (metric == name).then(|| value.parse().expect("metric value is numeric"))
            })
            .unwrap_or_else(|| panic!("missing metric {name}"))
    }

    async fn wait_for_metric(batcher: &Batcher, name: &str, expected: f64) {
        tokio::time::timeout(Duration::from_secs(1), async {
            loop {
                if metric_value(batcher, name) == expected {
                    break;
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap_or_else(|_| panic!("{name} did not reach {expected} in time"));
    }

    #[test]
    fn accepts_optional_or_matching_model_names() {
        let served = "convaiinnovations/laya";
        assert!(matches_model(None, served));
        assert!(matches_model(Some(""), served));
        assert!(matches_model(Some(served), served));
        assert!(matches_model(Some("jev-latest"), served));
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
        wait_for_metric(&batcher, "sys1_inference_duration_seconds_count", 1.0).await;
        assert_eq!(metric_value(&batcher, "sys1_requests_accepted_total"), 1.0);
        assert_eq!(metric_value(&batcher, "sys1_requests_timed_out_total"), 1.0);
        assert_eq!(
            metric_value(&batcher, "sys1_requests_successful_total"),
            0.0
        );
        assert_eq!(metric_value(&batcher, "sys1_requests_in_flight"), 0.0);
    }

    #[tokio::test]
    async fn tracks_failed_model_requests() {
        let batcher = Batcher::new(
            Arc::new(FailingModel),
            "test-model".to_owned(),
            BatcherConfig {
                max_batch_size: 1,
                max_batch_questions: 1,
                max_questions_per_request: 1,
                wait: Duration::ZERO,
                queue_capacity: 1,
                response_timeout: None,
            },
        );

        let error = batcher.predict(one_question_request()).await.unwrap_err();
        assert_eq!(error.status(), 500);
        assert_eq!(metric_value(&batcher, "sys1_requests_failed_total"), 1.0);
        assert_eq!(metric_value(&batcher, "sys1_model_failures_total"), 1.0);
        assert_eq!(
            metric_value(&batcher, "sys1_requests_successful_total"),
            0.0
        );
    }

    #[tokio::test]
    async fn excludes_warmup_from_metrics() {
        let batcher = Batcher::new(
            Arc::new(RecordingModel::default()),
            "test-model".to_owned(),
            BatcherConfig {
                max_batch_size: 1,
                max_batch_questions: 1,
                max_questions_per_request: 1,
                wait: Duration::ZERO,
                queue_capacity: 1,
                response_timeout: None,
            },
        );

        batcher.warmup().await.unwrap();
        assert_eq!(metric_value(&batcher, "sys1_requests_accepted_total"), 0.0);
        assert_eq!(
            metric_value(&batcher, "sys1_requests_successful_total"),
            0.0
        );
        assert_eq!(metric_value(&batcher, "sys1_batches_total"), 0.0);
        assert_eq!(metric_value(&batcher, "sys1_batch_size_count"), 0.0);
    }

    #[tokio::test]
    async fn tracks_requests_rejected_by_a_full_queue() {
        let started = Arc::new(Barrier::new(2));
        let release = Arc::new(Barrier::new(2));
        let model = Arc::new(BlockOnceModel {
            block: AtomicBool::new(true),
            started: started.clone(),
            release: release.clone(),
        });
        let batcher = Batcher::new(
            model,
            "test-model".to_owned(),
            BatcherConfig {
                max_batch_size: 1,
                max_batch_questions: 1,
                max_questions_per_request: 1,
                wait: Duration::ZERO,
                queue_capacity: 1,
                response_timeout: None,
            },
        );

        let first_batcher = batcher.clone();
        let first =
            tokio::spawn(async move { first_batcher.predict(one_question_request()).await });
        tokio::time::timeout(
            Duration::from_secs(1),
            tokio::task::spawn_blocking(move || started.wait()),
        )
        .await
        .expect("model started in time")
        .unwrap();

        let second_batcher = batcher.clone();
        let second =
            tokio::spawn(async move { second_batcher.predict(one_question_request()).await });
        wait_for_metric(&batcher, "sys1_requests_queued", 1.0).await;
        assert_eq!(metric_value(&batcher, "sys1_requests_queued"), 1.0);

        let error = batcher.predict(one_question_request()).await.unwrap_err();
        assert_eq!(error.status(), 503);
        assert_eq!(metric_value(&batcher, "sys1_requests_rejected_total"), 1.0);

        tokio::time::timeout(
            Duration::from_secs(1),
            tokio::task::spawn_blocking(move || release.wait()),
        )
        .await
        .expect("model released in time")
        .unwrap();
        assert!(first.await.unwrap().is_ok());
        assert!(second.await.unwrap().is_ok());
        assert_eq!(metric_value(&batcher, "sys1_requests_accepted_total"), 2.0);
        assert_eq!(
            metric_value(&batcher, "sys1_requests_successful_total"),
            2.0
        );
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
        assert_eq!(metric_value(&batcher, "sys1_batches_total"), 2.0);
        assert_eq!(metric_value(&batcher, "sys1_batch_size_count"), 2.0);
        assert_eq!(metric_value(&batcher, "sys1_batch_size_sum"), 3.0);
        assert_eq!(metric_value(&batcher, "sys1_batch_questions_sum"), 3.0);
    }
}

use prometheus::{
    Encoder, Histogram, HistogramOpts, IntCounter, IntGauge, Registry, TEXT_FORMAT, TextEncoder,
    core::Collector,
};
use std::time::Instant;

pub(crate) struct EncodedMetrics {
    pub body: Vec<u8>,
    pub content_type: &'static str,
}

pub(crate) struct BatcherMetrics {
    registry: Registry,
    accepted_requests: IntCounter,
    successful_requests: IntCounter,
    rejected_requests: IntCounter,
    timed_out_requests: IntCounter,
    failed_requests: IntCounter,
    queued_requests: IntGauge,
    in_flight_requests: IntGauge,
    batches: IntCounter,
    batch_size: Histogram,
    batch_questions: Histogram,
    model_failures: IntCounter,
    inference_duration: Histogram,
}

impl BatcherMetrics {
    pub fn new() -> Self {
        let registry = Registry::new();
        let accepted_requests = IntCounter::new(
            "sys1_requests_accepted_total",
            "Inference requests accepted into the processing queue.",
        )
        .expect("accepted request metric must be valid");
        let successful_requests = IntCounter::new(
            "sys1_requests_successful_total",
            "Inference requests that returned a successful response.",
        )
        .expect("successful request metric must be valid");
        let rejected_requests = IntCounter::new(
            "sys1_requests_rejected_total",
            "Inference requests rejected because the queue was unavailable.",
        )
        .expect("rejected request metric must be valid");
        let timed_out_requests = IntCounter::new(
            "sys1_requests_timed_out_total",
            "Inference requests that timed out while waiting for a response.",
        )
        .expect("timed out request metric must be valid");
        let failed_requests = IntCounter::new(
            "sys1_requests_failed_total",
            "Accepted inference requests that returned an error.",
        )
        .expect("failed request metric must be valid");
        let queued_requests = IntGauge::new(
            "sys1_requests_queued",
            "Inference requests currently waiting in the queue.",
        )
        .expect("queued request metric must be valid");
        let in_flight_requests = IntGauge::new(
            "sys1_requests_in_flight",
            "Inference requests currently being processed.",
        )
        .expect("in-flight request metric must be valid");
        let batches = IntCounter::new(
            "sys1_batches_total",
            "Inference batches processed by the model.",
        )
        .expect("batch counter metric must be valid");
        let batch_size = Histogram::with_opts(
            HistogramOpts::new(
                "sys1_batch_size",
                "Number of inference requests in each batch.",
            )
            .buckets(vec![1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0]),
        )
        .expect("batch size metric must be valid");
        let batch_questions = Histogram::with_opts(
            HistogramOpts::new(
                "sys1_batch_questions",
                "Number of questions in each inference batch.",
            )
            .buckets(vec![1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0]),
        )
        .expect("batch question metric must be valid");
        let model_failures = IntCounter::new(
            "sys1_model_failures_total",
            "Inference requests for which model execution returned an error.",
        )
        .expect("model failure metric must be valid");
        let inference_duration = Histogram::with_opts(
            HistogramOpts::new(
                "sys1_inference_duration_seconds",
                "Time spent executing each inference batch.",
            )
            .buckets(vec![
                0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0,
            ]),
        )
        .expect("inference duration metric must be valid");

        register(&registry, accepted_requests.clone());
        register(&registry, successful_requests.clone());
        register(&registry, rejected_requests.clone());
        register(&registry, timed_out_requests.clone());
        register(&registry, failed_requests.clone());
        register(&registry, queued_requests.clone());
        register(&registry, in_flight_requests.clone());
        register(&registry, batches.clone());
        register(&registry, batch_size.clone());
        register(&registry, batch_questions.clone());
        register(&registry, model_failures.clone());
        register(&registry, inference_duration.clone());

        Self {
            registry,
            accepted_requests,
            successful_requests,
            rejected_requests,
            timed_out_requests,
            failed_requests,
            queued_requests,
            in_flight_requests,
            batches,
            batch_size,
            batch_questions,
            model_failures,
            inference_duration,
        }
    }

    pub fn request_queued(&self) {
        self.queued_requests.inc();
    }

    pub fn request_dequeued(&self) {
        self.queued_requests.dec();
    }

    pub fn request_accepted(&self) {
        self.accepted_requests.inc();
    }

    pub fn request_rejected(&self) {
        self.rejected_requests.inc();
    }

    pub fn request_timed_out(&self) {
        self.timed_out_requests.inc();
    }

    pub fn request_failed(&self) {
        self.failed_requests.inc();
    }

    pub fn request_succeeded(&self) {
        self.successful_requests.inc();
    }

    pub fn batch_started(&self, requests: usize, questions: usize) -> BatchGuard<'_> {
        self.batches.inc();
        self.batch_size.observe(requests as f64);
        self.batch_questions.observe(questions as f64);
        self.in_flight_requests.add(requests as i64);
        BatchGuard {
            metrics: self,
            requests,
            started: Instant::now(),
        }
    }

    pub fn model_requests_failed(&self, failures: usize) {
        self.model_failures.inc_by(failures as u64);
    }

    pub fn encode(&self) -> prometheus::Result<EncodedMetrics> {
        let mut body = Vec::new();
        TextEncoder::new().encode(&self.registry.gather(), &mut body)?;
        Ok(EncodedMetrics {
            body,
            content_type: TEXT_FORMAT,
        })
    }
}

pub(crate) struct BatchGuard<'a> {
    metrics: &'a BatcherMetrics,
    requests: usize,
    started: Instant,
}

impl Drop for BatchGuard<'_> {
    fn drop(&mut self) {
        self.metrics
            .inference_duration
            .observe(self.started.elapsed().as_secs_f64());
        self.metrics.in_flight_requests.sub(self.requests as i64);
    }
}

fn register<C>(registry: &Registry, collector: C)
where
    C: Collector + 'static,
{
    registry
        .register(Box::new(collector))
        .expect("metric names must be unique");
}

use crate::{
    models::DecisionModel,
    schema::{ApiError, DecisionRequest, DecisionResponse},
};

use std::{sync::Arc, time::Duration};
use tokio::sync::{mpsc, oneshot};

struct Job {
    request: DecisionRequest,
    response: oneshot::Sender<Result<DecisionResponse, ApiError>>,
}

#[derive(Clone)]
pub struct Batcher {
    sender: mpsc::Sender<Job>,
    served_model_name: Arc<str>,
}

impl Batcher {
    pub fn new(
        model: Arc<dyn DecisionModel>,
        served_model_name: String,
        max_batch_size: usize,
        wait: Duration,
    ) -> Self {
        let max_batch_size = max_batch_size.max(1);
        let (sender, mut receiver) = mpsc::channel::<Job>(max_batch_size * 8);
        tokio::spawn(async move {
            while let Some(first) = receiver.recv().await {
                let mut jobs = vec![first];
                let deadline = tokio::time::Instant::now() + wait;
                while jobs.len() < max_batch_size {
                    match tokio::time::timeout_at(deadline, receiver.recv()).await {
                        Ok(Some(job)) => jobs.push(job),
                        _ => break,
                    }
                }
                let requests = jobs.iter().map(|job| job.request.clone()).collect();
                let model = model.clone();
                let responses = tokio::task::spawn_blocking(move || model.predict_batch(requests))
                    .await
                    .unwrap_or_else(|error| {
                        (0..jobs.len())
                            .map(|_| Err(ApiError::new(error.to_string())))
                            .collect()
                    });
                for (job, response) in jobs.into_iter().zip(responses) {
                    let _ = job.response.send(response);
                }
            }
        });
        Self {
            sender,
            served_model_name: served_model_name.into(),
        }
    }

    pub fn served_model_name(&self) -> &str {
        &self.served_model_name
    }

    pub async fn predict(&self, request: DecisionRequest) -> Result<DecisionResponse, ApiError> {
        if !matches_model(request.model.as_deref(), &self.served_model_name) {
            let model = request.model.as_deref().unwrap();
            return Err(ApiError::new(format!("unknown model: {model}")));
        }
        let (response, receiver) = oneshot::channel();
        self.sender
            .send(Job { request, response })
            .await
            .map_err(|_| ApiError::new("inference worker stopped"))?;
        let mut response = receiver
            .await
            .map_err(|_| ApiError::new("inference worker stopped"))??;
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

    #[test]
    fn accepts_optional_or_matching_model_names() {
        let served = "convaiinnovations/laya";
        assert!(matches_model(None, served));
        assert!(matches_model(Some(""), served));
        assert!(matches_model(Some(served), served));
        assert!(!matches_model(Some("other/model"), served));
    }
}

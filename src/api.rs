use crate::{
    batching::Batcher,
    schema::{ApiError, DecisionRequest},
};

use axum::{
    Json, Router,
    extract::{Request, State},
    http::StatusCode,
    middleware::{self, Next},
    response::{IntoResponse, Response},
    routing::{get, post},
};
use serde_json::json;
use std::time::Instant;
use tracing::Instrument;

pub fn router(batcher: Batcher) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/v1/models", get(models))
        .route("/v1/systemone", post(decide))
        .route("/v1/decide", post(decide))
        .layer(middleware::from_fn(trace_request))
        .with_state(batcher)
}

async fn trace_request(request: Request, next: Next) -> Response {
    let method = request.method().clone();
    let path = request.uri().path().to_owned();
    let span = tracing::info_span!("request", %method, %path);
    async move {
        let started = Instant::now();
        let response = next.run(request).await;
        let status = response.status();
        let elapsed_ms = started.elapsed().as_millis() as u64;
        if status.is_server_error() {
            tracing::error!(status = status.as_u16(), elapsed_ms, "request completed");
        } else if status.is_client_error() {
            tracing::warn!(status = status.as_u16(), elapsed_ms, "request completed");
        } else {
            tracing::info!(status = status.as_u16(), elapsed_ms, "request completed");
        }
        response
    }
    .instrument(span)
    .await
}

async fn health() -> Json<serde_json::Value> {
    Json(json!({"status": "ok"}))
}

async fn models(State(batcher): State<Batcher>) -> Json<serde_json::Value> {
    Json(json!({
        "data": [{
            "id": batcher.served_model_name(),
            "object": "model",
            "owned_by": "sys1"
        }]
    }))
}

async fn decide(
    State(batcher): State<Batcher>,
    Json(request): Json<DecisionRequest>,
) -> Result<Json<crate::schema::DecisionResponse>, ApiError> {
    batcher.predict(request).await.map(Json)
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        (StatusCode::BAD_REQUEST, Json(self)).into_response()
    }
}

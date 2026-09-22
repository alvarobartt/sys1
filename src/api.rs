use crate::{
    batching::Batcher,
    schema::{ApiError, DecisionRequest},
};

use axum::{
    Json, Router,
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{get, post},
};
use serde_json::json;

pub fn router(batcher: Batcher) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/v1/models", get(models))
        .route("/v1/systemone", post(decide))
        .route("/v1/decide", post(decide))
        .with_state(batcher)
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

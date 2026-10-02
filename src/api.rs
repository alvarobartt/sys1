use crate::{
    batching::Batcher,
    schema::{ApiError, DecisionRequest},
};

use axum::{
    Json, Router,
    extract::{DefaultBodyLimit, Extension, Request, State, rejection::JsonRejection},
    http::StatusCode,
    middleware::{self, Next},
    response::{IntoResponse, Response},
    routing::{get, post},
};
use serde::Serialize;
use std::time::Instant;
use tracing::Instrument;
use utoipa::{OpenApi, ToSchema};
use utoipa_swagger_ui::SwaggerUi;

#[derive(OpenApi)]
#[openapi(
    info(
        title = "sys1",
        version = env!("CARGO_PKG_VERSION"),
        description = env!("CARGO_PKG_DESCRIPTION")
    ),
    paths(health, metrics, models, decide, systemone),
    components(schemas(
        HealthResponse,
        ModelsResponse,
        Model,
        DecisionRequest,
        crate::schema::DecisionResponse,
        crate::schema::Usage,
        ApiError
    )),
    tags((name = "sys1", description = "System One decision API"))
)]
pub struct ApiDoc;

#[derive(Serialize, ToSchema)]
struct HealthResponse {
    status: &'static str,
}

#[derive(Serialize, ToSchema)]
struct ModelsResponse {
    data: Vec<Model>,
}

#[derive(Serialize, ToSchema)]
struct Model {
    id: String,
    object: &'static str,
    owned_by: &'static str,
}

pub fn router(batcher: Batcher, max_request_bytes: usize) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/metrics", get(metrics))
        .route("/v1/models", get(models))
        .route("/v1/systemone", post(systemone))
        .route("/v1/decide", post(decide))
        .merge(SwaggerUi::new("/docs").url("/openapi.json", ApiDoc::openapi()))
        .layer(DefaultBodyLimit::max(max_request_bytes.max(1)))
        .layer(Extension(RequestLimit(max_request_bytes.max(1))))
        .layer(middleware::from_fn(trace_request))
        .with_state(batcher)
}

#[derive(Clone, Copy)]
struct RequestLimit(usize);

fn parse_decision_request(
    request: Result<Json<DecisionRequest>, JsonRejection>,
    max_request_bytes: usize,
) -> Result<DecisionRequest, ApiError> {
    match request {
        Ok(Json(request)) => Ok(request),
        Err(error) if error.status() == StatusCode::PAYLOAD_TOO_LARGE => {
            Err(ApiError::payload_too_large(format!(
                "request body exceeds {max_request_bytes} bytes; base64 images and videos count toward this limit. Resize or compress the media, use a URL, or increase --max-request-bytes"
            )))
        }
        Err(error) => Err(ApiError::new(format!("invalid request JSON: {error}"))),
    }
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
            tracing::debug!(status = status.as_u16(), elapsed_ms, "request completed");
        }
        response
    }
    .instrument(span)
    .await
}

/// Check whether the service is running.
#[utoipa::path(
    get,
    path = "/health",
    tag = "sys1",
    responses((status = OK, description = "Service is healthy", body = HealthResponse))
)]
async fn health(State(batcher): State<Batcher>) -> (StatusCode, Json<HealthResponse>) {
    if batcher.is_ready() {
        (StatusCode::OK, Json(HealthResponse { status: "ok" }))
    } else {
        (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(HealthResponse {
                status: "unavailable",
            }),
        )
    }
}

/// Return inference and admission-control metrics in Prometheus text format.
#[utoipa::path(
    get,
    path = "/metrics",
    tag = "sys1",
    responses((
        status = OK,
        description = "Prometheus metrics",
        content_type = "text/plain",
        body = String
    ), (status = INTERNAL_SERVER_ERROR, description = "Metrics encoding failed"))
)]
async fn metrics(State(batcher): State<Batcher>) -> Result<Response, StatusCode> {
    let metrics = batcher.encode_metrics().map_err(|error| {
        tracing::error!(%error, "failed to encode Prometheus metrics");
        StatusCode::INTERNAL_SERVER_ERROR
    })?;
    Ok((
        [(axum::http::header::CONTENT_TYPE, metrics.content_type)],
        metrics.body,
    )
        .into_response())
}

/// List the model served by this instance.
#[utoipa::path(
    get,
    path = "/v1/models",
    tag = "sys1",
    responses((status = OK, description = "Available models", body = ModelsResponse))
)]
async fn models(State(batcher): State<Batcher>) -> Json<ModelsResponse> {
    Json(ModelsResponse {
        data: vec![Model {
            id: batcher.served_model_name().to_owned(),
            object: "model",
            owned_by: "sys1",
        }],
    })
}

/// Make a decision.
#[utoipa::path(
    post,
    path = "/v1/decide",
    tag = "sys1",
    request_body = DecisionRequest,
    responses(
        (status = OK, description = "Decision generated", body = crate::schema::DecisionResponse),
        (status = BAD_REQUEST, description = "Invalid request", body = ApiError),
        (status = PAYLOAD_TOO_LARGE, description = "Request exceeds --max-request-bytes", body = ApiError)
    )
)]
async fn decide(
    State(batcher): State<Batcher>,
    Extension(limit): Extension<RequestLimit>,
    request: Result<Json<DecisionRequest>, JsonRejection>,
) -> Result<Json<crate::schema::DecisionResponse>, ApiError> {
    let request = parse_decision_request(request, limit.0)?;
    batcher.predict(request).await.map(Json)
}

/// Make a decision using the Jev-compatible route.
#[utoipa::path(
    post,
    path = "/v1/systemone",
    tag = "sys1",
    request_body = DecisionRequest,
    responses(
        (status = OK, description = "Decision generated", body = crate::schema::DecisionResponse),
        (status = BAD_REQUEST, description = "Invalid request", body = ApiError),
        (status = PAYLOAD_TOO_LARGE, description = "Request exceeds --max-request-bytes", body = ApiError)
    )
)]
async fn systemone(
    State(batcher): State<Batcher>,
    Extension(limit): Extension<RequestLimit>,
    request: Result<Json<DecisionRequest>, JsonRejection>,
) -> Result<Json<crate::schema::DecisionResponse>, ApiError> {
    let request = parse_decision_request(request, limit.0)?;
    batcher.predict(request).await.map(Json)
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let status =
            StatusCode::from_u16(self.status()).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
        (status, Json(self)).into_response()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        batching::BatcherConfig,
        models::DecisionModel,
        schema::{DecisionResponse, Usage},
    };
    use axum::{
        body::{Body, to_bytes},
        http::Request,
    };
    use serde_json::Map;
    use std::{sync::Arc, time::Duration};
    use tower::ServiceExt;

    struct TestModel;

    impl DecisionModel for TestModel {
        fn supports_images(&self) -> bool {
            true
        }

        fn supports_videos(&self) -> bool {
            true
        }

        fn predict_batch(
            &self,
            requests: Vec<DecisionRequest>,
        ) -> Vec<Result<DecisionResponse, ApiError>> {
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

    fn test_router() -> Router {
        router(
            Batcher::new(
                Arc::new(TestModel),
                "test-model".to_owned(),
                BatcherConfig {
                    max_batch_size: 1,
                    max_batch_questions: 8,
                    max_questions_per_request: 8,
                    wait: Duration::ZERO,
                    queue_capacity: 8,
                    response_timeout: Some(Duration::from_secs(1)),
                },
            ),
            1024,
        )
    }

    #[test]
    fn openapi_contains_all_public_routes() {
        let document = ApiDoc::openapi();
        let json = serde_json::to_value(document).unwrap();
        let paths = json["paths"].as_object().unwrap();

        for path in [
            "/health",
            "/metrics",
            "/v1/models",
            "/v1/decide",
            "/v1/systemone",
        ] {
            assert!(paths.contains_key(path), "missing OpenAPI path {path}");
        }
        for field in ["images", "videos"] {
            assert!(
                json["components"]["schemas"]["DecisionRequest"]["properties"][field]
                    ["description"]
                    .as_str()
                    .unwrap()
                    .contains("Experimental")
            );
        }
    }

    #[tokio::test]
    async fn serves_openapi_document() {
        let response = test_router()
            .oneshot(Request::get("/openapi.json").body(Body::empty()).unwrap())
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(response.headers()["content-type"], "application/json");
    }

    #[tokio::test]
    async fn reports_ready_after_the_worker_starts() {
        let response = test_router()
            .oneshot(Request::get("/health").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn serves_inference_metrics() {
        let app = test_router();
        let inference_response = app
            .clone()
            .oneshot(
                Request::post("/v1/decide")
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{"state":"test","questions":{"q0":{}}}"#))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(inference_response.status(), StatusCode::OK);

        let response = app
            .oneshot(Request::get("/metrics").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(
            response.headers()["content-type"],
            "text/plain; version=0.0.4"
        );
        let body = to_bytes(response.into_body(), 32 * 1024).await.unwrap();
        let body = std::str::from_utf8(&body).unwrap();
        assert!(body.contains("# TYPE sys1_requests_accepted_total counter"));
        assert!(
            body.lines()
                .any(|line| line == "sys1_requests_accepted_total 1")
        );
        assert!(
            body.lines()
                .any(|line| line == "sys1_requests_successful_total 1")
        );
        assert!(body.contains("# TYPE sys1_requests_queued gauge"));
        assert!(body.lines().any(|line| line == "sys1_requests_queued 0"));
        assert!(body.contains("# TYPE sys1_batch_size histogram"));
        assert!(body.lines().any(|line| line == "sys1_batch_size_count 1"));
        assert!(body.lines().any(|line| line == "sys1_batches_total 1"));
    }

    #[tokio::test]
    async fn renders_swagger_ui() {
        let response = test_router()
            .oneshot(Request::get("/docs/").body(Body::empty()).unwrap())
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        assert!(
            response.headers()["content-type"]
                .to_str()
                .unwrap()
                .starts_with("text/html")
        );
    }

    #[tokio::test]
    async fn rejects_oversized_request_bodies() {
        let response = test_router()
            .oneshot(
                Request::post("/v1/decide")
                    .header("content-type", "application/json")
                    .body(Body::from(vec![b' '; 2048]))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::PAYLOAD_TOO_LARGE);
        let body = to_bytes(response.into_body(), 4096).await.unwrap();
        let error: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert!(
            error["error"]
                .as_str()
                .unwrap()
                .contains("--max-request-bytes")
        );
        assert!(error["error"].as_str().unwrap().contains("base64 images"));
    }

    #[tokio::test]
    async fn accepts_image_and_video_sources_in_json_requests() {
        let response = test_router()
            .oneshot(
                Request::post("/v1/systemone")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        serde_json::to_vec(&serde_json::json!({
                            "state": "test",
                            "images": [{"bytes": [1, 2, 3]}],
                            "videos": [{"base64": "AA=="}],
                            "questions": {"q0": {}}
                        }))
                        .unwrap(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
    }

    #[test]
    fn returns_service_error_status_codes() {
        assert_eq!(
            ApiError::unavailable("busy").into_response().status(),
            StatusCode::SERVICE_UNAVAILABLE
        );
        assert_eq!(
            ApiError::timeout("slow").into_response().status(),
            StatusCode::GATEWAY_TIMEOUT
        );
    }
}

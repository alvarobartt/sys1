use crate::{
    batching::Batcher,
    schema::{ApiError, DecisionRequest, DecisionResponse, HTTPValidationError, SystemOneRequest},
};

use axum::{
    Json, Router,
    extract::{DefaultBodyLimit, Extension, Request, State, rejection::JsonRejection},
    http::{HeaderMap, StatusCode},
    middleware::{self, Next},
    response::{IntoResponse, Response},
    routing::{get, post},
};
use serde::Serialize;
use serde_json::Value;
use std::time::Instant;
use utoipa::{OpenApi, ToSchema};
use utoipa_swagger_ui::SwaggerUi;

pub const PUBLIC_ROUTES: &[(&str, &str)] = &[
    ("GET", "/health"),
    ("GET", "/metrics"),
    ("GET", "/v1/models"),
    ("POST", "/v1/systemone"),
    ("POST", "/v1/decide"),
    ("GET", "/docs/"),
    ("GET", "/openapi.json"),
];

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
        ModelMetadataList,
        ModelMetadata,
        DecisionRequest,
        crate::schema::SystemOneRequest,
        crate::schema::DecisionResponse,
        crate::schema::Usage,
        crate::schema::ValidationError,
        HTTPValidationError,
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
struct ModelMetadataList {
    models: Vec<ModelMetadata>,
}

#[derive(Serialize, ToSchema)]
struct ModelMetadata {
    name: String,
    description: &'static str,
    release_date: &'static str,
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
    let started = Instant::now();
    let response = next.run(request).await;
    let status = response.status().as_u16();
    let elapsed_ms = started.elapsed().as_millis() as u64;
    if response.status().is_server_error() {
        tracing::error!(%method, %path, status, elapsed_ms, "Request completed");
    } else if response.status().is_client_error() {
        tracing::warn!(%method, %path, status, elapsed_ms, "Request completed");
    } else {
        tracing::info!(%method, %path, status, elapsed_ms, "Request completed");
    }
    response
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
        tracing::error!(%error, "Failed to encode Prometheus metrics");
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
    responses((status = OK, description = "Available models and aliases", body = ModelMetadataList))
)]
async fn models(State(batcher): State<Batcher>) -> Json<ModelMetadataList> {
    let served = batcher.served_model_name();
    let mut models = vec![ModelMetadata {
        name: served.to_owned(),
        description: "Self-hosted Laya decision model served by sys1.",
        release_date: "2026-09-18",
    }];
    if served != "jev-latest" {
        models.push(ModelMetadata {
            name: "jev-latest".to_owned(),
            description: "TypeSafe SDK compatibility alias for the model served by this instance.",
            release_date: "2026-09-18",
        });
    }
    Json(ModelMetadataList { models })
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
    request_body = crate::schema::SystemOneRequest,
    responses(
        (status = OK, description = "Decision generated", body = crate::schema::DecisionResponse),
        (status = BAD_REQUEST, description = "Unknown model", body = ApiError),
        (status = UNPROCESSABLE_ENTITY, description = "Validation error", body = HTTPValidationError),
        (status = PAYLOAD_TOO_LARGE, description = "Request exceeds --max-request-bytes", body = ApiError)
    )
)]
async fn systemone(
    State(batcher): State<Batcher>,
    Extension(limit): Extension<RequestLimit>,
    headers: HeaderMap,
    payload: Result<Json<Value>, JsonRejection>,
) -> Response {
    let Json(value) = match payload {
        Ok(value) => value,
        Err(error) if error.status() == StatusCode::PAYLOAD_TOO_LARGE => {
            return ApiError::payload_too_large(format!(
                "request body exceeds {} bytes; base64 images and videos count toward this limit. Resize or compress the media, use a URL, or increase --max-request-bytes",
                limit.0
            ))
            .into_response();
        }
        Err(error) => {
            return validation_response(HTTPValidationError::single(
                vec!["body".into()],
                error.body_text(),
                "json_invalid",
            ));
        }
    };
    let request = match SystemOneRequest::try_from(value) {
        Ok(request) => request,
        Err(error) => return validation_response(error),
    };
    let request = DecisionRequest::from(request);
    if sdk_uses_default_alias(
        &headers,
        request.model.as_deref(),
        batcher.served_model_name(),
    ) {
        tracing::warn!(
            alias = "jev-latest",
            served_model = batcher.served_model_name(),
            "TypeSafe SDK used its default model alias; routing to the served model"
        );
    }
    match batcher.predict(request).await {
        Ok(response) => Json(typesafe_schema_response(response)).into_response(),
        Err(error)
            if error.status() == StatusCode::BAD_REQUEST.as_u16() && !error.is_unknown_model() =>
        {
            validation_response(HTTPValidationError::single(
                vec!["body".into()],
                error.error,
                "value_error",
            ))
        }
        Err(error) => error.into_response(),
    }
}

fn typesafe_schema_response(mut response: DecisionResponse) -> DecisionResponse {
    for answer in response.answers.values_mut() {
        let Some(fields) = answer.as_object_mut() else {
            continue;
        };
        let allowed: &[&str] = match fields.get("type").and_then(Value::as_str) {
            Some("choice") => &["type", "choice", "probabilities", "confidence"],
            Some("score") => &["type", "score", "probabilities", "legend", "confidence"],
            Some("noul") => &["type", "noul"],
            _ => continue,
        };
        fields.retain(|name, _| allowed.contains(&name.as_str()));
    }
    response
}

fn sdk_uses_default_alias(headers: &HeaderMap, requested: Option<&str>, served: &str) -> bool {
    requested == Some("jev-latest")
        && served != "jev-latest"
        && headers
            .get("x-typesafe-sdk")
            .and_then(|value| value.to_str().ok())
            .is_some_and(|value| value.starts_with("typesafe-sdk/"))
}

fn validation_response(error: HTTPValidationError) -> Response {
    (StatusCode::UNPROCESSABLE_ENTITY, Json(error)).into_response()
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
                        answers: serde_json::from_value(serde_json::json!({
                            "route": {
                                "type": "choice",
                                "choice": "billing",
                                "probabilities": {"billing": 0.6, "bug": 0.3, "account": 0.1},
                                "confidence": 0.1829,
                                "answer_confidence": 0.6,
                                "action": {"act_probability": 0.8}
                            },
                            "urgency": {
                                "type": "score",
                                "score": 1.5,
                                "probabilities": {"0": 0.0, "1": 0.5, "2": 0.5},
                                "legend": {"0": "low", "1": "medium", "2": "high"},
                                "confidence": 0.3691,
                                "answer_confidence": 0.5
                            },
                            "escalate": {
                                "type": "noul",
                                "noul": 0.7,
                                "confidence": 0.7,
                                "answer_confidence": 0.7
                            }
                        }))
                        .unwrap(),
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

        for &(method, path) in PUBLIC_ROUTES {
            if matches!(path, "/docs/" | "/openapi.json") {
                continue;
            }
            assert!(
                paths[path]
                    .get(method.to_ascii_lowercase().as_str())
                    .is_some(),
                "missing OpenAPI route [{method}] {path}"
            );
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

        let state = &json["components"]["schemas"]["SystemOneRequest"]["properties"]["state"];
        assert!(state.is_object());
        assert!(state.get("type").is_none());
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
    async fn lists_sdk_compatible_models_and_accepts_default_alias() {
        let app = test_router();
        let response = app
            .clone()
            .oneshot(Request::get("/v1/models").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body = to_bytes(response.into_body(), 32 * 1024).await.unwrap();
        let listing: serde_json::Value = serde_json::from_slice(&body).unwrap();
        let models = listing["models"].as_array().unwrap();
        assert_eq!(models.len(), 2);
        assert_eq!(models[0]["name"], "test-model");
        assert_eq!(models[1]["name"], "jev-latest");
        for model in models {
            assert!(model["description"].as_str().is_some_and(|s| !s.is_empty()));
            assert_eq!(model["release_date"], "2026-09-18");
        }

        let response = app
            .oneshot(
                Request::post("/v1/systemone")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        r#"{"model":"jev-latest","state":"test","questions":{"q0":{"type":"noul"}}}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body = to_bytes(response.into_body(), 32 * 1024).await.unwrap();
        let decision: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(decision["model"], "test-model");
    }

    #[tokio::test]
    async fn systemone_defaults_missing_null_and_empty_models_to_served_model() {
        let app = test_router();
        for model in [
            None,
            Some("null"),
            Some("\"\""),
            Some("\"jev-latest\""),
            Some("\"test-model\""),
        ] {
            let model_field = model.map_or(String::new(), |value| format!("\"model\":{value},"));
            let body = format!(
                "{{{model_field}\"state\":\"test\",\"questions\":{{\"q\":{{\"type\":\"noul\"}}}}}}"
            );
            let response = app
                .clone()
                .oneshot(
                    Request::post("/v1/systemone")
                        .header("content-type", "application/json")
                        .body(Body::from(body))
                        .unwrap(),
                )
                .await
                .unwrap();
            assert_eq!(response.status(), StatusCode::OK, "model: {model:?}");
            let bytes = to_bytes(response.into_body(), 32 * 1024).await.unwrap();
            let body: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
            assert_eq!(body["model"], "test-model");
        }
    }

    #[tokio::test]
    async fn systemone_keeps_native_confidence_with_typesafe_response_shape() {
        let app = test_router();
        let mut responses = Vec::new();
        for path in ["/v1/decide", "/v1/systemone"] {
            let response = app
                .clone()
                .oneshot(
                    Request::post(path)
                        .header("content-type", "application/json")
                        .body(Body::from(
                            r#"{"state":"test","questions":{"route":{"type":"choice","criteria":{"billing":null,"bug":null,"account":null}},"urgency":{"type":"score","criteria":["low","medium","high"]},"escalate":{"type":"noul"}}}"#,
                        ))
                        .unwrap(),
                )
                .await
                .unwrap();
            assert_eq!(response.status(), StatusCode::OK);
            let bytes = to_bytes(response.into_body(), 32 * 1024).await.unwrap();
            responses.push(serde_json::from_slice::<Value>(&bytes).unwrap());
        }

        let native = &responses[0]["answers"];
        let typesafe = &responses[1]["answers"];
        assert_eq!(native["route"]["confidence"], 0.1829);
        assert_eq!(native["route"]["answer_confidence"], 0.6);
        assert!(native["route"].get("action").is_some());
        assert_eq!(native["urgency"]["confidence"], 0.3691);
        assert_eq!(native["escalate"]["confidence"], 0.7);
        assert_eq!(
            typesafe["route"]["confidence"],
            native["route"]["confidence"]
        );
        assert_eq!(
            typesafe["urgency"]["confidence"],
            native["urgency"]["confidence"]
        );
        assert!(typesafe["escalate"].get("confidence").is_none());
        for key in ["route", "urgency", "escalate"] {
            assert!(typesafe[key].get("answer_confidence").is_none());
        }
        assert!(typesafe["route"].get("action").is_none());
        assert_eq!(
            typesafe["route"]["probabilities"],
            native["route"]["probabilities"]
        );
        assert_eq!(
            typesafe["urgency"]["probabilities"],
            native["urgency"]["probabilities"]
        );
        assert_eq!(typesafe["escalate"]["noul"], native["escalate"]["noul"]);
    }

    #[test]
    fn identifies_sdk_default_alias_for_warning() {
        let mut headers = HeaderMap::new();
        assert!(!sdk_uses_default_alias(
            &headers,
            Some("jev-latest"),
            "test-model"
        ));
        headers.insert("x-typesafe-sdk", "typesafe-sdk/0.7.2".parse().unwrap());
        assert!(sdk_uses_default_alias(
            &headers,
            Some("jev-latest"),
            "test-model"
        ));
        assert!(!sdk_uses_default_alias(&headers, None, "test-model"));
        assert!(!sdk_uses_default_alias(&headers, Some(""), "test-model"));
        assert!(!sdk_uses_default_alias(
            &headers,
            Some("test-model"),
            "test-model"
        ));
        assert!(!sdk_uses_default_alias(
            &headers,
            Some("jev-latest"),
            "jev-latest"
        ));
    }

    #[tokio::test]
    async fn systemone_rejects_invalid_requests_with_typesafe_validation_shape() {
        let app = test_router();
        for (body, location, kind) in [
            (
                r#"{"model":12,"state":"test","questions":{"q":{"type":"noul"}}}"#,
                "model",
                "value_error",
            ),
            (
                r#"{"model":"test-model","state":"test","questions":{"q":{"type":"score"}}}"#,
                "criteria",
                "missing",
            ),
            (
                r#"{"model":"test-model","state":"test","questions":{"q":{"type":"other"}}}"#,
                "type",
                "value_error",
            ),
        ] {
            let response = app
                .clone()
                .oneshot(
                    Request::post("/v1/systemone")
                        .header("content-type", "application/json")
                        .body(Body::from(body))
                        .unwrap(),
                )
                .await
                .unwrap();
            assert_eq!(response.status(), StatusCode::UNPROCESSABLE_ENTITY);
            let bytes = to_bytes(response.into_body(), 32 * 1024).await.unwrap();
            let body: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
            let issue = &body["detail"][0];
            assert_eq!(issue["loc"].as_array().unwrap().last().unwrap(), location);
            assert_eq!(issue["type"], kind);
            assert!(issue["msg"].as_str().is_some_and(|value| !value.is_empty()));
        }
    }

    #[tokio::test]
    async fn unknown_model_returns_bad_request() {
        let app = test_router();
        let body = r#"{"model":"wrong-model","state":"test","questions":{"q":{"type":"noul"}}}"#;
        for path in ["/v1/systemone", "/v1/decide"] {
            let response = app
                .clone()
                .oneshot(
                    Request::post(path)
                        .header("content-type", "application/json")
                        .body(Body::from(body))
                        .unwrap(),
                )
                .await
                .unwrap();
            assert_eq!(response.status(), StatusCode::BAD_REQUEST, "{path}");
            let bytes = to_bytes(response.into_body(), 32 * 1024).await.unwrap();
            let error: Value = serde_json::from_slice(&bytes).unwrap();
            assert_eq!(error["error"], "unknown model: wrong-model");
        }
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
        for path in ["/v1/decide", "/v1/systemone"] {
            let response = test_router()
                .oneshot(
                    Request::post(path)
                        .header("content-type", "application/json")
                        .body(Body::from(vec![b' '; 2048]))
                        .unwrap(),
                )
                .await
                .unwrap();

            assert_eq!(response.status(), StatusCode::PAYLOAD_TOO_LARGE, "{path}");
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
                            "questions": {"q0": {"type": "noul"}}
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

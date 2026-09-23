use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use utoipa::ToSchema;

#[derive(Clone, Debug, Deserialize, ToSchema)]
pub struct DecisionRequest {
    #[serde(default)]
    pub model: Option<String>,
    pub state: Value,
    #[schema(value_type = Object)]
    pub questions: Map<String, Value>,
}

#[derive(Clone, Debug, Serialize, ToSchema)]
pub struct DecisionResponse {
    pub model: String,
    #[schema(value_type = Object)]
    pub answers: Map<String, Value>,
    pub usage: Usage,
}

#[derive(Clone, Debug, Serialize, ToSchema)]
pub struct Usage {
    pub input_tokens: usize,
    pub output_tokens: usize,
}

#[derive(Clone, Debug, Serialize, ToSchema)]
pub struct ApiError {
    pub error: String,
    #[serde(skip)]
    #[schema(ignore)]
    status: u16,
}

impl ApiError {
    pub fn new(message: impl Into<String>) -> Self {
        Self::with_status(message, 400)
    }

    pub fn unavailable(message: impl Into<String>) -> Self {
        Self::with_status(message, 503)
    }

    pub fn timeout(message: impl Into<String>) -> Self {
        Self::with_status(message, 504)
    }

    pub fn internal(message: impl Into<String>) -> Self {
        Self::with_status(message, 500)
    }

    pub fn status(&self) -> u16 {
        self.status
    }

    fn with_status(message: impl Into<String>, status: u16) -> Self {
        Self {
            error: message.into(),
            status,
        }
    }
}

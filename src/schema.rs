use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

#[derive(Clone, Debug, Deserialize)]
pub struct DecisionRequest {
    #[serde(default)]
    pub model: Option<String>,
    pub state: Value,
    pub questions: Map<String, Value>,
}

#[derive(Clone, Debug, Serialize)]
pub struct DecisionResponse {
    pub model: String,
    pub answers: Map<String, Value>,
    pub usage: Usage,
}

#[derive(Clone, Debug, Serialize)]
pub struct Usage {
    pub input_tokens: usize,
    pub output_tokens: usize,
}

#[derive(Clone, Debug, Serialize)]
pub struct ApiError {
    pub error: String,
}

impl ApiError {
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            error: message.into(),
        }
    }
}

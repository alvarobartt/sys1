use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use utoipa::ToSchema;

#[derive(Clone, Debug, Deserialize, ToSchema)]
pub struct DecisionRequest {
    #[serde(default)]
    pub model: Option<String>,
    pub state: Value,
    /// Experimental image inputs, accepted only by models with image support. Each item may be a public HTTP(S) URL, base64 string or data URL, JSON byte array, or object containing url, base64, or bytes.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub images: Vec<MediaInput>,
    /// Experimental video inputs, accepted only by models with video support. Each item may be a public HTTP(S) URL, base64 string or data URL, JSON byte array, or object containing url, base64, or bytes.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub videos: Vec<MediaInput>,
    #[schema(value_type = Object)]
    pub questions: Map<String, Value>,
}

#[derive(Clone, Debug, Deserialize, ToSchema)]
#[serde(untagged)]
pub enum MediaInput {
    Text(String),
    Bytes(Vec<u8>),
    Url(MediaUrl),
    Base64(MediaBase64),
    ByteObject(MediaBytes),
}

impl From<&str> for MediaInput {
    fn from(value: &str) -> Self {
        Self::Text(value.to_owned())
    }
}

impl From<String> for MediaInput {
    fn from(value: String) -> Self {
        Self::Text(value)
    }
}

#[derive(Clone, Debug, Deserialize, ToSchema)]
#[serde(deny_unknown_fields)]
pub struct MediaUrl {
    pub url: String,
}

#[derive(Clone, Debug, Deserialize, ToSchema)]
#[serde(deny_unknown_fields)]
pub struct MediaBase64 {
    pub base64: String,
}

#[derive(Clone, Debug, Deserialize, ToSchema)]
#[serde(deny_unknown_fields)]
pub struct MediaBytes {
    pub bytes: Vec<u8>,
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

    pub fn payload_too_large(message: impl Into<String>) -> Self {
        Self::with_status(message, 413)
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

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn accepts_each_json_media_representation() {
        assert!(matches!(
            serde_json::from_value::<MediaInput>(json!("data:image/png;base64,AA==")).unwrap(),
            MediaInput::Text(_)
        ));
        assert!(matches!(
            serde_json::from_value::<MediaInput>(json!([0, 1, 2])).unwrap(),
            MediaInput::Bytes(_)
        ));
        assert!(matches!(
            serde_json::from_value::<MediaInput>(json!({"bytes": [0, 1, 2]})).unwrap(),
            MediaInput::ByteObject(_)
        ));
        assert!(matches!(
            serde_json::from_value::<MediaInput>(json!({"base64": "AA=="})).unwrap(),
            MediaInput::Base64(_)
        ));
        assert!(matches!(
            serde_json::from_value::<MediaInput>(json!({"url": "https://example.com/a.mp4"}))
                .unwrap(),
            MediaInput::Url(_)
        ));
    }
}

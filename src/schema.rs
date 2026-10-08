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

#[derive(ToSchema)]
pub struct SystemOneRequest {
    pub model: Option<String>,
    pub state: Value,
    /// Experimental image inputs, accepted only by models with image support.
    #[schema(required = false)]
    pub images: Vec<MediaInput>,
    /// Experimental video inputs, accepted only by models with video support.
    #[schema(required = false)]
    pub videos: Vec<MediaInput>,
    #[schema(value_type = Object)]
    pub questions: Map<String, Value>,
}

#[derive(Debug, Serialize, ToSchema)]
pub struct ValidationError {
    pub loc: Vec<String>,
    pub msg: String,
    #[serde(rename = "type")]
    pub kind: &'static str,
}

#[derive(Debug, Serialize, ToSchema)]
pub struct HTTPValidationError {
    pub detail: Vec<ValidationError>,
}

impl HTTPValidationError {
    pub fn single(loc: Vec<String>, msg: impl Into<String>, kind: &'static str) -> Self {
        Self {
            detail: vec![ValidationError {
                loc,
                msg: msg.into(),
                kind,
            }],
        }
    }
}

fn invalid(path: Vec<String>, message: impl Into<String>) -> HTTPValidationError {
    HTTPValidationError::single(path, message, "value_error")
}

fn content(value: &Value, nullable: bool) -> bool {
    matches!(value, Value::String(_) | Value::Object(_) | Value::Array(_))
        || (nullable && value.is_null())
}

impl TryFrom<Value> for SystemOneRequest {
    type Error = HTTPValidationError;

    fn try_from(input: Value) -> Result<Self, Self::Error> {
        let body = input
            .as_object()
            .ok_or_else(|| invalid(vec!["body".into()], "Input should be an object"))?;
        let required = |name: &str| {
            body.get(name).ok_or_else(|| {
                HTTPValidationError::single(
                    vec!["body".into(), name.into()],
                    "Field required",
                    "missing",
                )
            })
        };
        let model = match body.get("model") {
            None | Some(Value::Null) => None,
            Some(Value::String(model)) => Some(model.clone()),
            Some(_) => {
                return Err(invalid(
                    vec!["body".into(), "model".into()],
                    "Input should be a string or null",
                ));
            }
        };
        let state = required("state")?;
        if !content(state, false) {
            return Err(invalid(
                vec!["body".into(), "state".into()],
                "Input should be a string, object, or array",
            ));
        }
        let media = |name: &str| -> Result<Vec<MediaInput>, HTTPValidationError> {
            match body.get(name) {
                None => Ok(Vec::new()),
                Some(value) => serde_json::from_value(value.clone())
                    .map_err(|error| invalid(vec!["body".into(), name.into()], error.to_string())),
            }
        };
        let images = media("images")?;
        let videos = media("videos")?;
        let questions = required("questions")?.as_object().ok_or_else(|| {
            invalid(
                vec!["body".into(), "questions".into()],
                "Input should be an object",
            )
        })?;
        if questions.is_empty() {
            return Err(invalid(
                vec!["body".into(), "questions".into()],
                "Object should have at least one property",
            ));
        }
        for (name, question) in questions {
            let path = vec!["body".into(), "questions".into(), name.clone()];
            let object = question
                .as_object()
                .ok_or_else(|| invalid(path.clone(), "Input should be an object"))?;
            let kind = object.get("type").ok_or_else(|| {
                HTTPValidationError::single(
                    [path.clone(), vec!["type".into()]].concat(),
                    "Field required",
                    "missing",
                )
            })?;
            let kind = kind.as_str().ok_or_else(|| {
                invalid(
                    [path.clone(), vec!["type".into()]].concat(),
                    "Input should be a string",
                )
            })?;
            if !matches!(kind, "noul" | "choice" | "score") {
                return Err(invalid(
                    [path.clone(), vec!["type".into()]].concat(),
                    "Input should be 'noul', 'choice', or 'score'",
                ));
            }
            if let Some(instructions) = object.get("instructions")
                && !content(instructions, true)
            {
                return Err(invalid(
                    [path.clone(), vec!["instructions".into()]].concat(),
                    "Input should be a string, object, array, or null",
                ));
            }
            let criteria = object.get("criteria");
            let criteria_path = [path, vec!["criteria".into()]].concat();
            match kind {
                "noul" => {
                    if let Some(criteria) = criteria
                        && !criteria.is_null()
                    {
                        let object = criteria.as_object().ok_or_else(|| {
                            invalid(criteria_path.clone(), "Input should be an object or null")
                        })?;
                        for (name, value) in object {
                            if matches!(name.as_str(), "true" | "false") && !content(value, true) {
                                return Err(invalid(
                                    [criteria_path.clone(), vec![name.clone()]].concat(),
                                    "Input should be a string, object, array, or null",
                                ));
                            }
                        }
                    }
                }
                "choice" => {
                    let criteria = criteria.ok_or_else(|| {
                        HTTPValidationError::single(
                            criteria_path.clone(),
                            "Field required",
                            "missing",
                        )
                    })?;
                    let object = criteria.as_object().ok_or_else(|| {
                        invalid(criteria_path.clone(), "Input should be an object")
                    })?;
                    if object.is_empty() || object.len() > 255 {
                        return Err(invalid(criteria_path, "Choice requires 1 to 255 options"));
                    }
                    for (name, value) in object {
                        if !content(value, true) {
                            return Err(invalid(
                                [criteria_path.clone(), vec![name.clone()]].concat(),
                                "Input should be a string, object, array, or null",
                            ));
                        }
                    }
                }
                "score" => {
                    let criteria = criteria.ok_or_else(|| {
                        HTTPValidationError::single(
                            criteria_path.clone(),
                            "Field required",
                            "missing",
                        )
                    })?;
                    let values = criteria.as_array().ok_or_else(|| {
                        invalid(criteria_path.clone(), "Input should be an array")
                    })?;
                    if values.is_empty() || values.len() > 10 {
                        return Err(invalid(criteria_path, "Score requires 1 to 10 levels"));
                    }
                    for (index, value) in values.iter().enumerate() {
                        if !content(value, false) {
                            return Err(invalid(
                                [criteria_path.clone(), vec![index.to_string()]].concat(),
                                "Input should be a string, object, or array",
                            ));
                        }
                    }
                }
                _ => unreachable!(),
            }
        }
        Ok(Self {
            model,
            state: state.clone(),
            images,
            videos,
            questions: questions.clone(),
        })
    }
}

impl From<SystemOneRequest> for DecisionRequest {
    fn from(request: SystemOneRequest) -> Self {
        Self {
            model: request.model,
            state: request.state,
            images: request.images,
            videos: request.videos,
            questions: request.questions,
        }
    }
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
    #[serde(skip)]
    #[schema(ignore)]
    unknown_model: bool,
}

impl ApiError {
    pub fn new(message: impl Into<String>) -> Self {
        Self::with_status(message, 400)
    }

    pub fn unknown_model(model: &str) -> Self {
        Self {
            error: format!("unknown model: {model}"),
            status: 400,
            unknown_model: true,
        }
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

    pub fn is_unknown_model(&self) -> bool {
        self.unknown_model
    }

    fn with_status(message: impl Into<String>, status: u16) -> Self {
        Self {
            error: message.into(),
            status,
            unknown_model: false,
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

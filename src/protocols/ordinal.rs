use crate::{
    schema::{ApiError, DecisionRequest, DecisionResponse, Usage},
    tokenizer,
};
use serde::Deserialize;
use serde_json::{Map, Value, json};
use std::collections::HashMap;
const TYPES: [&str; 3] = ["choice", "score", "noul"];
#[derive(Deserialize)]
pub(crate) struct Config {
    pub(crate) max_len: usize,
    pub(crate) head_max_len: usize,
    pub(crate) temperature: Vec<f32>,
    #[serde(default)]
    pub(crate) temperature_by_options: HashMap<String, f32>,
}

pub(crate) struct Question {
    pub(crate) id: String,
    pub(crate) kind: usize,
    pub(crate) instructions: String,
    pub(crate) criteria: Value,
    pub(crate) labels: Vec<String>,
    pub(crate) options: Vec<String>,
}

pub(crate) struct Item {
    pub(crate) question: Question,
    pub(crate) ids: Vec<u32>,
    pub(crate) markers: Vec<usize>,
}

pub(crate) struct RequestItems {
    pub(crate) items: Vec<Item>,
}

pub(crate) struct OrdinalProtocol {
    pub(crate) config: Config,
    pub(crate) tokenizer: tokenizer::Tokenizer,
    pub(crate) pad_id: u32,
    pub(crate) cls_id: u32,
    pub(crate) sep_id: u32,
    pub(crate) mask_id: u32,
}
impl OrdinalProtocol {
    fn encode(&self, text: &str) -> Result<Vec<u32>, ApiError> {
        self.tokenizer
            .encode(text)
            .map_err(|error| ApiError::new(error.to_string()))
    }

    pub(crate) fn prepare(&self, request: DecisionRequest) -> Result<RequestItems, ApiError> {
        if !request.images.is_empty() {
            return Err(ApiError::new("images are not supported by this model"));
        }
        if !request.videos.is_empty() {
            return Err(ApiError::new("videos are not supported by this model"));
        }
        if !matches!(
            request.state,
            Value::String(_) | Value::Array(_) | Value::Object(_)
        ) {
            return Err(ApiError::new("state must be a string, object, or array"));
        }
        if request.questions.is_empty() {
            return Err(ApiError::new("questions must not be empty"));
        }
        let state = render_value(&request.state);
        let sanitized_state = sanitize_masks(&state);
        let state_ids = self.encode(&sanitized_state)?;
        let mut items = Vec::with_capacity(request.questions.len());
        for (id, value) in request.questions {
            let object = value
                .as_object()
                .ok_or_else(|| ApiError::new(format!("question {id:?} must be an object")))?;
            let kind_name = object
                .get("type")
                .and_then(Value::as_str)
                .ok_or_else(|| ApiError::new(format!("question {id:?} requires type")))?;
            let kind = TYPES
                .iter()
                .position(|value| *value == kind_name)
                .ok_or_else(|| ApiError::new(format!("question {id:?} has invalid type")))?;
            let instructions = object
                .get("instructions")
                .filter(|value| !value.is_null())
                .map(render_value)
                .unwrap_or_else(|| id.clone());
            let criteria = object.get("criteria").cloned().unwrap_or(Value::Null);
            let (labels, options) = render_options(kind, &criteria, &id)?;
            let question = Question {
                id: id.to_string(),
                kind,
                instructions,
                criteria,
                labels,
                options,
            };
            let (ids, markers) = self.build_sequence(&state_ids, &question)?;
            items.push(Item {
                question,
                ids,
                markers,
            });
        }
        Ok(RequestItems { items })
    }

    fn build_sequence(
        &self,
        state_ids: &[u32],
        question: &Question,
    ) -> Result<(Vec<u32>, Vec<usize>), ApiError> {
        let instructions = sanitize_masks(&question.instructions);
        let mut head = self.encode(&format!(
            "{} question: {instructions}",
            TYPES[question.kind]
        ))?;
        let mut options = Vec::with_capacity(question.options.len());
        for option in &question.options {
            let mut ids = vec![self.mask_id];
            ids.extend(
                self.encode(&format!(" {}", sanitize_masks(option)))?
                    .into_iter()
                    .take(48),
            );
            options.push(ids);
        }
        let mut budget = self.config.head_max_len as isize
            - options.iter().map(Vec::len).sum::<usize>() as isize;
        if budget < 16 {
            let per = ((self.config.head_max_len - 16) / options.len().max(1)).max(4);
            options.iter_mut().for_each(|option| option.truncate(per));
            budget = self.config.head_max_len as isize
                - options.iter().map(Vec::len).sum::<usize>() as isize;
        }
        head.truncate((budget.max(8)) as usize);
        let mut ids = vec![self.cls_id];
        ids.extend(head);
        ids.push(self.sep_id);
        let mut markers = Vec::with_capacity(options.len());
        for option in options {
            markers.push(ids.len());
            ids.extend(option);
        }
        ids.push(self.sep_id);
        let room = self.config.max_len.saturating_sub(ids.len() + 1);
        ids.extend(state_ids.iter().copied().take(room));
        ids.push(self.sep_id);
        ids.truncate(self.config.max_len);
        markers.retain(|marker| *marker < self.config.max_len);
        if markers.len() != question.options.len() {
            return Err(ApiError::new(format!(
                "question {:?} options exceed head_max_len={}",
                question.id, self.config.head_max_len
            )));
        }
        Ok((ids, markers))
    }

    pub(crate) fn response(
        &self,
        request: RequestItems,
        logits: &mut impl Iterator<Item = Vec<f32>>,
    ) -> DecisionResponse {
        let input_tokens = request.items.iter().map(|item| item.ids.len()).sum();
        let mut answers = Map::new();
        for item in request.items {
            let values = logits.next().unwrap();
            let bucket = bucket(item.question.kind, values.len());
            let temperature = self
                .config
                .temperature_by_options
                .get(&bucket)
                .copied()
                .unwrap_or(self.config.temperature[item.question.kind])
                .clamp(0.5, 5.0);
            let probabilities = probability(&values, temperature);
            let answer = match item.question.kind {
                0 => {
                    let index = argmax(&probabilities);
                    let confidence = round4(confidence_from_probs(&probabilities));
                    let answer_confidence = round4(probabilities[index]);
                    let values = item
                        .question
                        .labels
                        .iter()
                        .cloned()
                        .zip(
                            probabilities
                                .iter()
                                .copied()
                                .map(|value| json!(round4(value))),
                        )
                        .collect::<Map<_, _>>();
                    json!({
                        "type": "choice",
                        "choice": item.question.labels[index],
                        "probabilities": values,
                        "confidence": confidence,
                        "answer_confidence": answer_confidence
                    })
                }
                1 => {
                    let confidence = round4(confidence_from_probs(&probabilities));
                    let answer_confidence = round4(probabilities[argmax(&probabilities)]);
                    let score = probabilities
                        .iter()
                        .enumerate()
                        .map(|(index, value)| index as f32 * value)
                        .sum();
                    let probabilities = probabilities
                        .iter()
                        .enumerate()
                        .map(|(index, value)| (index.to_string(), json!(round4(*value))))
                        .collect::<Map<_, _>>();
                    let legend = item
                        .question
                        .criteria
                        .as_array()
                        .unwrap()
                        .iter()
                        .enumerate()
                        .map(|(index, value)| (index.to_string(), value.clone()))
                        .collect::<Map<_, _>>();
                    json!({
                        "type": "score",
                        "score": round4(score),
                        "probabilities": probabilities,
                        "legend": legend,
                        "confidence": confidence,
                        "answer_confidence": answer_confidence
                    })
                }
                _ => json!({
                    "type": "noul",
                    "noul": round4(probabilities[1]),
                    "confidence": round4(probabilities[0].max(probabilities[1])),
                    "answer_confidence": round4(probabilities[0].max(probabilities[1]))
                }),
            };
            answers.insert(item.question.id, answer);
        }
        DecisionResponse {
            model: String::new(),
            answers,
            usage: Usage {
                input_tokens,
                output_tokens: 0,
            },
        }
    }
}
fn render_options(
    kind: usize,
    criteria: &Value,
    id: &str,
) -> Result<(Vec<String>, Vec<String>), ApiError> {
    match kind {
        0 => {
            let criteria = criteria.as_object().ok_or_else(|| {
                ApiError::new(format!("choice question {id:?} requires object criteria"))
            })?;
            if criteria.is_empty() || criteria.len() > 255 {
                return Err(ApiError::new(format!(
                    "choice question {id:?} requires 1 to 255 criteria"
                )));
            }
            let labels: Vec<_> = criteria.keys().cloned().collect();
            let options = criteria
                .iter()
                .map(|(label, value)| {
                    if value.is_null() || value.as_str() == Some("") {
                        label.clone()
                    } else {
                        format!("{label}: {}", render_value(value))
                    }
                })
                .collect();
            Ok((labels, options))
        }
        1 => {
            let criteria = criteria.as_array().ok_or_else(|| {
                ApiError::new(format!("score question {id:?} requires array criteria"))
            })?;
            if !(1..=10).contains(&criteria.len()) {
                return Err(ApiError::new(format!(
                    "score question {id:?} requires 1 to 10 criteria"
                )));
            }
            Ok((
                (0..criteria.len()).map(|value| value.to_string()).collect(),
                criteria
                    .iter()
                    .enumerate()
                    .map(|(index, value)| format!("level {index}: {}", render_value(value)))
                    .collect(),
            ))
        }
        _ => {
            let criteria = criteria.as_object();
            let false_value = criteria.and_then(|value| value.get("false"));
            let true_value = criteria.and_then(|value| value.get("true"));
            Ok((
                vec!["false".to_owned(), "true".to_owned()],
                vec![
                    format!(
                        "false: {}",
                        criterion(false_value, "no, the statement does not hold")
                    ),
                    format!(
                        "true: {}",
                        criterion(true_value, "yes, the statement holds")
                    ),
                ],
            ))
        }
    }
}

fn criterion(value: Option<&Value>, default: &str) -> String {
    match value {
        Some(value) if !value.is_null() && value.as_str() != Some("") => render_value(value),
        _ => default.to_owned(),
    }
}

fn render_value(value: &Value) -> String {
    match value {
        Value::String(value) => value.clone(),
        Value::Array(values) => format!(
            "[{}]",
            values
                .iter()
                .map(render_json)
                .collect::<Vec<_>>()
                .join(", ")
        ),
        Value::Object(values) => format!(
            "{{{}}}",
            values
                .iter()
                .map(|(key, value)| format!(
                    "{}: {}",
                    serde_json::to_string(key).unwrap(),
                    render_json(value)
                ))
                .collect::<Vec<_>>()
                .join(", ")
        ),
        value => render_json(value),
    }
}

fn render_json(value: &Value) -> String {
    match value {
        Value::Array(values) => format!(
            "[{}]",
            values
                .iter()
                .map(render_json)
                .collect::<Vec<_>>()
                .join(", ")
        ),
        Value::Object(values) => format!(
            "{{{}}}",
            values
                .iter()
                .map(|(key, value)| format!(
                    "{}: {}",
                    serde_json::to_string(key).unwrap(),
                    render_json(value)
                ))
                .collect::<Vec<_>>()
                .join(", ")
        ),
        value => serde_json::to_string(value).unwrap(),
    }
}

pub(crate) fn bucket(kind: usize, count: usize) -> String {
    let size = if count <= 2 {
        "2"
    } else if count <= 5 {
        "3-5"
    } else if count <= 10 {
        "6-10"
    } else {
        "11+"
    };
    format!("{}:{size}", TYPES[kind])
}

pub(crate) fn probability(logits: &[f32], temperature: f32) -> Vec<f32> {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max) / temperature;
    let mut values: Vec<_> = logits
        .iter()
        .map(|value| (value / temperature - max).exp())
        .collect();
    let total: f32 = values.iter().sum();
    values.iter_mut().for_each(|value| *value /= total);
    values
}

pub(crate) fn confidence_from_probs(probabilities: &[f32]) -> f32 {
    if probabilities.len() < 2 {
        return 1.0;
    }
    let entropy: f32 = probabilities
        .iter()
        .map(|value| -value * value.max(1e-12).ln())
        .sum();
    (1.0 - entropy / (probabilities.len() as f32).ln()).clamp(0.0, 1.0)
}

pub(crate) fn argmax(values: &[f32]) -> usize {
    values
        .iter()
        .enumerate()
        .max_by(|(left, a), (right, b)| a.total_cmp(b).then_with(|| right.cmp(left)))
        .map(|(index, _)| index)
        .unwrap_or(0)
}

fn round4(value: f32) -> f64 {
    ((value as f64) * 10_000.0).round() / 10_000.0
}

fn sanitize_masks(value: &str) -> String {
    value.replace("[MASK]", " ").replace("<mask>", " ")
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn confidence_matches_laya_formulas() {
        let probabilities: [f32; 3] = [0.6, 0.3, 0.1];
        let entropy = -probabilities
            .iter()
            .map(|value| value * value.ln())
            .sum::<f32>();
        assert!(
            (confidence_from_probs(&probabilities) - (1.0 - entropy / 3.0_f32.ln())).abs() < 1e-6
        );
        assert_eq!(confidence_from_probs(&[0.5, 0.5]), 0.0);
        assert_eq!(confidence_from_probs(&[1.0]), 1.0);
        assert_eq!(argmax(&[0.5, 0.5]), 0);
    }
}

use super::DecisionModel;
use super::modernbert::{Config as ModernBertConfig, Encoder as ModernBertEncoder};
use crate::{
    device,
    schema::{ApiError, DecisionRequest, DecisionResponse, Usage},
    tokenizer,
};

use candle_core::{D, DType, Device, IndexOp, Tensor};
use candle_nn::{
    Embedding, LayerNorm, Linear, VarBuilder, embedding, layer_norm, linear, ops::softmax,
};
use serde::Deserialize;
use serde_json::{Map, Value, json};
use std::{collections::HashMap, fs, path::Path};

const TYPES: [&str; 3] = ["choice", "score", "noul"];

#[derive(Deserialize)]
struct LayaConfig {
    max_len: usize,
    head_max_len: usize,
    temperature: Vec<f32>,
    #[serde(default)]
    temperature_by_options: HashMap<String, f32>,
}

struct Question {
    id: String,
    kind: usize,
    instructions: String,
    criteria: Value,
    labels: Vec<String>,
    options: Vec<String>,
}

struct Item {
    question: Question,
    ids: Vec<u32>,
    markers: Vec<usize>,
}

struct RequestItems {
    items: Vec<Item>,
}

struct HeadLayer {
    qkv: Linear,
    projection: Linear,
    norm1: LayerNorm,
    norm2: LayerNorm,
    linear1: Linear,
    linear2: Linear,
    heads: usize,
}

impl HeadLayer {
    fn load(vb: VarBuilder, hidden: usize) -> candle_core::Result<Self> {
        Ok(Self {
            qkv: Linear::new(
                vb.get((hidden * 3, hidden), "self_attn.in_proj_weight")?,
                Some(vb.get(hidden * 3, "self_attn.in_proj_bias")?),
            ),
            projection: linear(hidden, hidden, vb.pp("self_attn.out_proj"))?,
            norm1: layer_norm(hidden, 1e-5, vb.pp("norm1"))?,
            norm2: layer_norm(hidden, 1e-5, vb.pp("norm2"))?,
            linear1: linear(hidden, hidden * 4, vb.pp("linear1"))?,
            linear2: linear(hidden * 4, hidden, vb.pp("linear2"))?,
            heads: hidden / 64,
        })
    }

    fn forward(&self, xs: &Tensor, mask: &Tensor) -> candle_core::Result<Tensor> {
        let (batch, length, hidden) = xs.dims3()?;
        let size = hidden / self.heads;
        let qkv = xs
            .apply(&self.norm1)?
            .apply(&self.qkv)?
            .reshape((batch, length, 3, self.heads, size))?
            .permute((2, 0, 3, 1, 4))?;
        let q = (qkv.get(0)? * (size as f64).powf(-0.5))?;
        let k = qkv.get(1)?;
        let v = qkv.get(2)?;
        let attention = q
            .matmul(&k.transpose(D::Minus2, D::Minus1)?)?
            .broadcast_add(mask)?;
        let attention = softmax(&attention, D::Minus1)?;
        let attention = attention
            .matmul(&v)?
            .transpose(1, 2)?
            .reshape((batch, length, hidden))?
            .apply(&self.projection)?;
        let xs = (xs + attention)?;
        let feed_forward = xs
            .apply(&self.norm2)?
            .apply(&self.linear1)?
            .relu()?
            .apply(&self.linear2)?;
        xs + feed_forward
    }
}

pub struct Laya {
    tokenizer: tokenizer::Tokenizer,
    encoder: ModernBertEncoder,
    head: Vec<HeadLayer>,
    type_embedding: Embedding,
    scorer_norm: LayerNorm,
    scorer_in: Linear,
    scorer_out: Linear,
    config: LayaConfig,
    device: Device,
    pad_id: u32,
    cls_id: u32,
    sep_id: u32,
    mask_id: u32,
}

impl Laya {
    pub fn load(path: &Path) -> anyhow::Result<Self> {
        let device = device::load()?;
        let config: LayaConfig =
            serde_json::from_slice(&fs::read(path.join("rl_agent_config.json"))?)?;
        let encoder_config = ModernBertConfig::load(&path.join("encoder/config.json"))?;
        let weights = path.join("model.safetensors");
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[weights], DType::F32, &device)? };
        let encoder_vb = vb.clone().rename_f(|name| {
            name.strip_prefix("model.")
                .map(|name| format!("encoder.{name}"))
                .unwrap_or_else(|| name.to_owned())
        });
        let encoder = ModernBertEncoder::load(encoder_vb, &encoder_config)?;
        let head = (0..2)
            .map(|index| {
                HeadLayer::load(
                    vb.pp(format!("head.layers.{index}")),
                    encoder_config.hidden_size(),
                )
            })
            .collect::<candle_core::Result<Vec<_>>>()?;
        let tokenizer_path = path.join("tokenizer/tokenizer.v1.json");
        let legacy_tokenizer_path = path.join("tokenizer/tokenizer.json");
        let tokenizer_path = if tokenizer_path.exists() {
            &tokenizer_path
        } else {
            &legacy_tokenizer_path
        };
        let tokenizer_json = fs::read(tokenizer_path)?;
        let tokenizer = tokenizer::from_json(&tokenizer_json)?;
        let token = |value: &str| {
            tokenizer
                .token_to_id(value)
                .ok_or_else(|| anyhow::anyhow!("tokenizer is missing {value}"))
        };
        let pad_id = token("[PAD]")?;
        let cls_id = token("[CLS]")?;
        let sep_id = token("[SEP]")?;
        let mask_id = token("[MASK]")?;
        Ok(Self {
            tokenizer,
            encoder,
            head,
            type_embedding: embedding(3, encoder_config.hidden_size(), vb.pp("type_emb"))?,
            scorer_norm: layer_norm(encoder_config.hidden_size(), 1e-5, vb.pp("scorer.0"))?,
            scorer_in: linear(
                encoder_config.hidden_size(),
                encoder_config.hidden_size(),
                vb.pp("scorer.1"),
            )?,
            scorer_out: linear(encoder_config.hidden_size(), 1, vb.pp("scorer.3"))?,
            config,
            device,
            pad_id,
            cls_id,
            sep_id,
            mask_id,
        })
    }

    fn encode(&self, text: &str) -> Result<Vec<u32>, ApiError> {
        self.tokenizer
            .encode(text)
            .map_err(|error| ApiError::new(error.to_string()))
    }

    fn prepare(&self, request: DecisionRequest) -> Result<RequestItems, ApiError> {
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
                .and_then(Value::as_str)
                .ok_or_else(|| {
                    ApiError::new(format!("question {id:?} requires string instructions"))
                })?
                .to_owned();
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
            let (ids, markers) = self.build_sequence(&state, &question)?;
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
        state: &str,
        question: &Question,
    ) -> Result<(Vec<u32>, Vec<usize>), ApiError> {
        let instructions = question.instructions.replace("[MASK]", " ");
        let mut head = self.encode(&format!(
            "{} question: {instructions}",
            TYPES[question.kind]
        ))?;
        let mut options = Vec::with_capacity(question.options.len());
        for option in &question.options {
            let mut ids = vec![self.mask_id];
            ids.extend(
                self.encode(&format!(" {}", option.replace("[MASK]", " ")))?
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
        ids.extend(
            self.encode(&state.replace("[MASK]", " "))?
                .into_iter()
                .take(room),
        );
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

    fn forward(&self, prepared: &[RequestItems]) -> anyhow::Result<Vec<Vec<f32>>> {
        let items: Vec<_> = prepared.iter().flat_map(|request| &request.items).collect();
        let batch = items.len();
        let length = items.iter().map(|item| item.ids.len()).max().unwrap_or(0);
        let mut ids = vec![self.pad_id; batch * length];
        let mut mask = vec![0f32; batch * length];
        let mut kinds = Vec::with_capacity(batch);
        for (row, item) in items.iter().enumerate() {
            let start = row * length;
            ids[start..start + item.ids.len()].copy_from_slice(&item.ids);
            mask[start..start + item.ids.len()].fill(1.0);
            kinds.push(item.question.kind as u32);
        }
        let ids = Tensor::from_vec(ids, (batch, length), &self.device)?;
        let attention_mask = Tensor::from_vec(mask.clone(), (batch, length), &self.device)?;
        let head_mask: Vec<_> = mask
            .into_iter()
            .map(|value| if value == 0.0 { f32::NEG_INFINITY } else { 0.0 })
            .collect();
        let head_mask = Tensor::from_vec(head_mask, (batch, 1, 1, length), &self.device)?;
        let type_ids = Tensor::from_vec(kinds, batch, &self.device)?;
        let mut hidden = self.encoder.forward(&ids, &attention_mask)?;
        hidden = hidden.broadcast_add(&type_ids.apply(&self.type_embedding)?.unsqueeze(1)?)?;
        for layer in &self.head {
            hidden = layer.forward(&hidden, &head_mask)?;
        }
        let mut output = Vec::with_capacity(batch);
        for (row, item) in items.iter().enumerate() {
            let markers = Tensor::from_vec(
                item.markers
                    .iter()
                    .map(|value| *value as u32)
                    .collect::<Vec<_>>(),
                item.markers.len(),
                &self.device,
            )?;
            let selected = hidden.i(row)?.index_select(&markers, 0)?;
            let logits = selected
                .apply(&self.scorer_norm)?
                .apply(&self.scorer_in)?
                .gelu_erf()?
                .apply(&self.scorer_out)?
                .squeeze(1)?
                .to_vec1::<f32>()?;
            output.push(logits);
        }
        Ok(output)
    }

    fn response(
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
            let confidence = round4(confidence(&probabilities));
            let answer = match item.question.kind {
                0 => {
                    let index = argmax(&probabilities);
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
                        "confidence": confidence
                    })
                }
                1 => {
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
                        "confidence": confidence
                    })
                }
                _ => json!({
                    "type": "noul",
                    "noul": round4(probabilities[1]),
                    "confidence": round4(probabilities[1].max(1.0 - probabilities[1]))
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

impl DecisionModel for Laya {
    fn predict_batch(
        &self,
        requests: Vec<DecisionRequest>,
    ) -> Vec<Result<DecisionResponse, ApiError>> {
        let mut slots = Vec::with_capacity(requests.len());
        let mut prepared = Vec::new();
        for request in requests {
            match self.prepare(request) {
                Ok(request) => {
                    slots.push(Ok(prepared.len()));
                    prepared.push(request);
                }
                Err(error) => slots.push(Err(error)),
            }
        }
        let outputs = match self.forward(&prepared) {
            Ok(outputs) => outputs,
            Err(error) => {
                let message = error.to_string();
                return slots
                    .into_iter()
                    .map(|slot| slot.and_then(|_| Err(ApiError::new(message.clone()))))
                    .collect();
            }
        };
        let mut outputs = outputs.into_iter();
        let responses: Vec<_> = prepared
            .into_iter()
            .map(|request| Ok(self.response(request, &mut outputs)))
            .collect();
        slots
            .into_iter()
            .map(|slot| slot.and_then(|index| responses[index].clone()))
            .collect()
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
            if !(2..=10).contains(&criteria.len()) {
                return Err(ApiError::new(format!(
                    "score question {id:?} requires 2 to 10 criteria"
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

fn bucket(kind: usize, count: usize) -> String {
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

fn probability(logits: &[f32], temperature: f32) -> Vec<f32> {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max) / temperature;
    let mut values: Vec<_> = logits
        .iter()
        .map(|value| (value / temperature - max).exp())
        .collect();
    let total: f32 = values.iter().sum();
    values.iter_mut().for_each(|value| *value /= total);
    values
}

fn confidence(probabilities: &[f32]) -> f32 {
    if probabilities.len() < 2 {
        return 1.0;
    }
    let entropy: f32 = probabilities
        .iter()
        .map(|value| -value * value.max(1e-12).ln())
        .sum();
    (1.0 - entropy / (probabilities.len() as f32).ln()).clamp(0.0, 1.0)
}

fn argmax(values: &[f32]) -> usize {
    values
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
        .map(|(index, _)| index)
        .unwrap_or(0)
}

fn round4(value: f32) -> f64 {
    ((value as f64) * 10_000.0).round() / 10_000.0
}

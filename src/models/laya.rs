use super::AttentionImplementation;
use super::DecisionModel;
use super::modernbert::{
    AttentionOptions, Config as ModernBertConfig, Encoder as ModernBertEncoder,
};
use crate::{
    device,
    schema::{ApiError, DecisionRequest, DecisionResponse, Usage},
    tokenizer,
};

use anyhow::Context;
use candle_core::{D, DType, Device, IndexOp, Tensor};
use candle_nn::{Embedding, LayerNorm, Linear, VarBuilder, embedding, layer_norm};
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

#[derive(Eq, Hash, PartialEq)]
struct ItemKey<'a> {
    kind: usize,
    ids: &'a [u32],
    markers: &'a [usize],
}

struct HeadLayer {
    qkv: Linear,
    projection: Linear,
    norm1: LayerNorm,
    norm2: LayerNorm,
    linear1: Linear,
    linear2: Linear,
    heads: usize,
    compute_dtype: DType,
    attention: AttentionImplementation,
}

impl HeadLayer {
    fn load(
        vb: VarBuilder,
        hidden: usize,
        compute_dtype: DType,
        attention: AttentionImplementation,
    ) -> candle_core::Result<Self> {
        Ok(Self {
            qkv: Linear::new(
                vb.get((hidden * 3, hidden), "self_attn.in_proj_weight")?
                    .to_dtype(compute_dtype)?,
                Some(
                    vb.get(hidden * 3, "self_attn.in_proj_bias")?
                        .to_dtype(compute_dtype)?,
                ),
            ),
            projection: linear_dtype(hidden, hidden, vb.pp("self_attn.out_proj"), compute_dtype)?,
            norm1: layer_norm(hidden, 1e-5, vb.pp("norm1"))?,
            norm2: layer_norm(hidden, 1e-5, vb.pp("norm2"))?,
            linear1: linear_dtype(hidden, hidden * 4, vb.pp("linear1"), compute_dtype)?,
            linear2: linear_dtype(hidden * 4, hidden, vb.pp("linear2"), compute_dtype)?,
            heads: hidden / 64,
            compute_dtype,
            attention,
        })
    }

    fn forward(
        &self,
        xs: &Tensor,
        mask: &Tensor,
        lengths: &[usize],
    ) -> candle_core::Result<Tensor> {
        let (batch, length, hidden) = xs.dims3()?;
        let size = hidden / self.heads;
        let qkv = xs
            .apply(&self.norm1)?
            .to_dtype(self.compute_dtype)?
            .apply(&self.qkv)?
            .reshape((batch, length, 3, self.heads, size))?
            .permute((2, 0, 3, 1, 4))?;
        let q = qkv.get(0)?;
        let k = qkv.get(1)?;
        let v = qkv.get(2)?;
        let scale = (size as f64).powf(-0.5);

        #[cfg(feature = "metal")]
        let attended = if xs.device().is_metal() {
            let mask = mask
                .broadcast_as((batch, self.heads, length, length))?
                .contiguous()?;
            candle_nn::ops::sdpa(&q, &k, &v, Some(&mask), false, scale as f32, 1.0)?
        } else {
            super::modernbert::scaled_dot_product_attention(
                &q,
                &k,
                &v,
                scale,
                AttentionOptions {
                    mask: Some(mask),
                    implementation: self.attention,
                    lengths,
                    window: None,
                },
            )?
        };
        #[cfg(not(feature = "metal"))]
        let attended = super::modernbert::scaled_dot_product_attention(
            &q,
            &k,
            &v,
            scale,
            AttentionOptions {
                mask: Some(mask),
                implementation: self.attention,
                lengths,
                window: None,
            },
        )?;
        let attention = attended
            .transpose(1, 2)?
            .reshape((batch, length, hidden))?
            .apply(&self.projection)?
            .to_dtype(xs.dtype())?;
        let xs = (xs + attention)?;
        let feed_forward = xs
            .apply(&self.norm2)?
            .to_dtype(self.compute_dtype)?
            .apply(&self.linear1)?
            .relu()?
            .apply(&self.linear2)?
            .to_dtype(xs.dtype())?;
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
    dtype: DType,
    compute_dtype: DType,
    pad_id: u32,
    cls_id: u32,
    sep_id: u32,
    mask_id: u32,
}

impl Laya {
    pub fn load(
        path: &Path,
        dtype: DType,
        max_model_len: Option<usize>,
        attention: AttentionImplementation,
    ) -> anyhow::Result<Self> {
        let device = device::load()?;
        let (model_dtype, compute_dtype) = execution_dtypes(dtype, device.is_cuda());
        validate_attention(attention, &device, compute_dtype)?;
        let mut config: LayaConfig =
            serde_json::from_slice(&fs::read(path.join("rl_agent_config.json"))?)?;
        let encoder_config = ModernBertConfig::load(&path.join("encoder/config.json"))?;
        if let Some(max_model_len) = max_model_len {
            anyhow::ensure!(
                max_model_len <= encoder_config.max_position_embeddings(),
                "--max-model-len {max_model_len} exceeds the encoder limit of {}",
                encoder_config.max_position_embeddings()
            );
            config.max_len = max_model_len;
        }
        let weights = path.join("model.safetensors");
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[weights], model_dtype, &device)? };
        let encoder_vb = vb.clone().rename_f(|name| {
            name.strip_prefix("model.")
                .map(|name| format!("encoder.{name}"))
                .unwrap_or_else(|| name.to_owned())
        });
        let encoder =
            ModernBertEncoder::load(encoder_vb, &encoder_config, compute_dtype, attention)?;
        let head = (0..2)
            .map(|index| {
                HeadLayer::load(
                    vb.pp(format!("head.layers.{index}")),
                    encoder_config.hidden_size(),
                    compute_dtype,
                    attention,
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
        let token = |values: &[&str]| {
            values
                .iter()
                .find_map(|value| tokenizer.token_to_id(value))
                .ok_or_else(|| anyhow::anyhow!("tokenizer is missing one of {values:?}"))
        };
        let pad_id = token(&["[PAD]", "<pad>"])?;
        let cls_id = token(&["[CLS]", "<bos>"])?;
        let sep_id = token(&["[SEP]", "<eos>"])?;
        let mask_id = token(&["[MASK]", "<mask>"])?;
        Ok(Self {
            tokenizer,
            encoder,
            head,
            type_embedding: embedding(3, encoder_config.hidden_size(), vb.pp("type_emb"))?,
            scorer_norm: layer_norm(encoder_config.hidden_size(), 1e-5, vb.pp("scorer.0"))?,
            scorer_in: linear_dtype(
                encoder_config.hidden_size(),
                encoder_config.hidden_size(),
                vb.pp("scorer.1"),
                compute_dtype,
            )?,
            scorer_out: linear_dtype(
                encoder_config.hidden_size(),
                1,
                vb.pp("scorer.3"),
                compute_dtype,
            )?,
            config,
            device,
            dtype: model_dtype,
            compute_dtype,
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

    fn forward(&self, prepared: &[RequestItems]) -> anyhow::Result<Vec<Vec<f32>>> {
        let items: Vec<_> = prepared.iter().flat_map(|request| &request.items).collect();
        let (unique, indices) = deduplicate(&items);
        let batch = unique.len();
        let length = unique.iter().map(|item| item.ids.len()).max().unwrap_or(0);
        let mut ids = vec![self.pad_id; batch * length];
        let mut mask = vec![0f32; batch * length];
        let mut kinds = Vec::with_capacity(batch);
        for (row, item) in unique.iter().enumerate() {
            let start = row * length;
            ids[start..start + item.ids.len()].copy_from_slice(&item.ids);
            mask[start..start + item.ids.len()].fill(1.0);
            kinds.push(item.question.kind as u32);
        }
        let ids = Tensor::from_vec(ids, (batch, length), &self.device)?;
        let head_mask: Vec<_> = mask
            .iter()
            .map(|value| {
                if *value == 0.0 {
                    f32::NEG_INFINITY
                } else {
                    0.0
                }
            })
            .collect();
        let attention_mask =
            Tensor::from_vec(mask, (batch, length), &self.device)?.to_dtype(self.dtype)?;
        let head_mask = Tensor::from_vec(head_mask, (batch, 1, 1, length), &self.device)?
            .to_dtype(self.compute_dtype)?;
        let type_ids = Tensor::from_vec(kinds, batch, &self.device)?;
        let has_padding = unique.iter().any(|item| item.ids.len() != length);
        let lengths: Vec<_> = unique.iter().map(|item| item.ids.len()).collect();
        let mut hidden = self
            .encoder
            .forward(&ids, &attention_mask, &lengths, has_padding)
            .context("encoder forward")?;
        hidden = hidden.broadcast_add(&type_ids.apply(&self.type_embedding)?.unsqueeze(1)?)?;
        for (index, layer) in self.head.iter().enumerate() {
            hidden = layer
                .forward(&hidden, &head_mask, &lengths)
                .with_context(|| format!("decision head layer {index}"))?;
        }
        let output = if batch >= 8 && !self.device.is_cpu() {
            self.score_batched(&hidden, &unique, length)?
        } else {
            self.score_rows(&hidden, &unique)?
        };
        Ok(indices
            .into_iter()
            .map(|index| output[index].clone())
            .collect())
    }

    fn score_rows(&self, hidden: &Tensor, items: &[&Item]) -> anyhow::Result<Vec<Vec<f32>>> {
        let mut output = Vec::with_capacity(items.len());
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
            output.push(self.score(&selected)?);
        }
        Ok(output)
    }

    fn score_batched(
        &self,
        hidden: &Tensor,
        items: &[&Item],
        length: usize,
    ) -> anyhow::Result<Vec<Vec<f32>>> {
        let hidden_size = hidden.dim(D::Minus1)?;
        let counts: Vec<_> = items.iter().map(|item| item.markers.len()).collect();
        let marker_count = counts.iter().sum();
        let mut markers = Vec::with_capacity(marker_count);
        for (row, item) in items.iter().enumerate() {
            markers.extend(
                item.markers
                    .iter()
                    .map(|position| (row * length + position) as u32),
            );
        }
        let markers = Tensor::from_vec(markers, marker_count, &self.device)?;
        let selected = hidden
            .reshape((items.len() * length, hidden_size))?
            .index_select(&markers, 0)?;
        let logits = self.score(&selected)?;
        let mut offset = 0;
        Ok(counts
            .into_iter()
            .map(|count| {
                let values = logits[offset..offset + count].to_vec();
                offset += count;
                values
            })
            .collect())
    }

    fn score(&self, selected: &Tensor) -> anyhow::Result<Vec<f32>> {
        Ok(selected
            .apply(&self.scorer_norm)?
            .to_dtype(self.compute_dtype)?
            .apply(&self.scorer_in)?
            .gelu_erf()?
            .apply(&self.scorer_out)?
            .squeeze(1)?
            .to_dtype(DType::F32)?
            .to_vec1::<f32>()?)
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

fn sanitize_masks(value: &str) -> String {
    value.replace("[MASK]", " ").replace("<mask>", " ")
}

fn linear_dtype(
    input: usize,
    output: usize,
    vb: VarBuilder,
    dtype: DType,
) -> candle_core::Result<Linear> {
    Ok(Linear::new(
        vb.get((output, input), "weight")?.to_dtype(dtype)?,
        Some(vb.get(output, "bias")?.to_dtype(dtype)?),
    ))
}

fn execution_dtypes(requested: DType, is_cuda: bool) -> (DType, DType) {
    let model = if requested == DType::F16 && is_cuda {
        DType::F32
    } else {
        requested
    };
    (model, requested)
}

fn validate_attention(
    attention: AttentionImplementation,
    _device: &Device,
    compute_dtype: DType,
) -> anyhow::Result<()> {
    attention.validate(compute_dtype)?;
    #[cfg(any(feature = "flash-attn-2", feature = "flash-attn-3"))]
    if attention != AttentionImplementation::Eager {
        anyhow::ensure!(
            _device.is_cuda(),
            "{} requires a CUDA device",
            attention.cli_name()
        );
        let (major, minor) = match _device {
            Device::Cuda(cuda) => cuda
                .cuda_stream()
                .context()
                .compute_capability()
                .context("failed to query CUDA compute capability")?,
            _ => unreachable!(),
        };
        validate_flash_capability(attention, major, minor)?;
    }
    Ok(())
}

#[cfg(any(feature = "flash-attn-2", feature = "flash-attn-3", test))]
fn validate_flash_capability(
    attention: AttentionImplementation,
    major: i32,
    minor: i32,
) -> anyhow::Result<()> {
    let supported = match attention {
        AttentionImplementation::Eager => true,
        AttentionImplementation::FlashAttention2 => (8..=9).contains(&major),
        AttentionImplementation::FlashAttention3 => (major, minor) == (9, 0),
    };
    anyhow::ensure!(
        supported,
        "{} does not support CUDA compute capability {major}.{minor}",
        attention.cli_name()
    );
    Ok(())
}

fn deduplicate<'a>(items: &[&'a Item]) -> (Vec<&'a Item>, Vec<usize>) {
    let mut unique = Vec::with_capacity(items.len());
    let mut indices = Vec::with_capacity(items.len());
    let mut seen = HashMap::with_capacity(items.len());
    for item in items {
        let key = ItemKey {
            kind: item.question.kind,
            ids: &item.ids,
            markers: &item.markers,
        };
        let index = *seen.entry(key).or_insert_with(|| {
            let index = unique.len();
            unique.push(*item);
            index
        });
        indices.push(index);
    }
    (unique, indices)
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

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn fp32_logits_and_probabilities() -> anyhow::Result<()> {
        let models = [
            (
                "laya",
                "convaiinnovations/laya",
                "aa8c91ca088ec597df95a0d1c76b3063cb2ae5e8",
            ),
            (
                "laya_typed_decisions",
                "convaiinnovations/laya-typed-decisions",
                "1a793eb568e6718f15941d08f85432581df534e3",
            ),
            (
                "laya_multilingual",
                "convaiinnovations/laya-multilingual",
                "e4e9ddf21a7b1903b7acffd8814ad4307bf63a67",
            ),
        ];

        for (snapshot, model_id, revision) in models {
            let path = crate::hub::download(model_id, revision).await?;
            let model = Laya::load(&path, DType::F32, None, AttentionImplementation::Eager)?;
            let request: DecisionRequest = serde_json::from_value(json!({
                "state": {
                    "message": "I was charged twice for invoice 4411. Please refund me today.",
                    "account_tier": "enterprise"
                },
                "questions": {
                    "route": {
                        "type": "choice",
                        "instructions": "Where should this ticket go?",
                        "criteria": {
                            "billing": "payments, refunds, invoices",
                            "bug": "the product is broken",
                            "account": "login or access"
                        }
                    },
                    "urgency": {
                        "type": "score",
                        "instructions": "How urgent is this message?",
                        "criteria": [
                            "routine, no rush",
                            "today",
                            "urgent",
                            "critical, about to churn"
                        ]
                    },
                    "escalate": {
                        "type": "noul",
                        "instructions": "Escalate to a human immediately?"
                    }
                }
            }))?;

            let prepared = model.prepare(request).expect("prepare the test request");
            let logits = model.forward(std::slice::from_ref(&prepared))?;
            let probabilities: Vec<_> = prepared
                .items
                .iter()
                .zip(&logits)
                .map(|(item, logits)| {
                    let bucket = bucket(item.question.kind, logits.len());
                    let temperature = model
                        .config
                        .temperature_by_options
                        .get(&bucket)
                        .copied()
                        .unwrap_or(model.config.temperature[item.question.kind])
                        .clamp(0.5, 5.0);
                    probability(logits, temperature)
                })
                .collect();

            insta::assert_yaml_snapshot!(format!("{snapshot}_fp32_logits"), logits, {
                "[][]" => insta::rounded_redaction(3),
            });
            insta::assert_yaml_snapshot!(
                format!("{snapshot}_fp32_probabilities"),
                probabilities,
                {
                    "[][]" => insta::rounded_redaction(4),
                }
            );
        }

        Ok(())
    }

    fn item(id: &str, kind: usize, ids: &[u32]) -> Item {
        Item {
            question: Question {
                id: id.to_owned(),
                kind,
                instructions: String::new(),
                criteria: Value::Null,
                labels: Vec::new(),
                options: Vec::new(),
            },
            ids: ids.to_vec(),
            markers: vec![1],
        }
    }

    #[test]
    fn deduplicates_identical_model_inputs_without_losing_order() {
        let first = item("first", 0, &[1, 2, 3]);
        let duplicate = item("duplicate", 0, &[1, 2, 3]);
        let other_kind = item("other-kind", 1, &[1, 2, 3]);
        let items = vec![&first, &duplicate, &other_kind, &first];

        let (unique, indices) = deduplicate(&items);

        assert_eq!(unique.len(), 2);
        assert_eq!(indices, vec![0, 0, 1, 0]);
    }

    #[test]
    fn cuda_f16_uses_autocast_style_storage_and_compute_dtypes() {
        assert_eq!(execution_dtypes(DType::F16, true), (DType::F32, DType::F16));
        assert_eq!(
            execution_dtypes(DType::F16, false),
            (DType::F16, DType::F16)
        );
        assert_eq!(execution_dtypes(DType::F32, true), (DType::F32, DType::F32));
    }

    #[test]
    fn rejects_flash_attention_without_a_compatible_backend() {
        assert!(
            validate_attention(
                AttentionImplementation::FlashAttention2,
                &Device::Cpu,
                DType::BF16,
            )
            .is_err()
        );
        validate_attention(AttentionImplementation::Eager, &Device::Cpu, DType::F32).unwrap();
    }

    #[test]
    fn validates_flash_attention_compute_capabilities() {
        validate_flash_capability(AttentionImplementation::FlashAttention2, 8, 9).unwrap();
        validate_flash_capability(AttentionImplementation::FlashAttention2, 9, 0).unwrap();
        validate_flash_capability(AttentionImplementation::FlashAttention3, 9, 0).unwrap();

        assert!(validate_flash_capability(AttentionImplementation::FlashAttention2, 7, 5).is_err());
        assert!(
            validate_flash_capability(AttentionImplementation::FlashAttention2, 10, 0).is_err()
        );
        assert!(validate_flash_capability(AttentionImplementation::FlashAttention3, 8, 9).is_err());
        assert!(validate_flash_capability(AttentionImplementation::FlashAttention3, 9, 1).is_err());
        assert!(
            validate_flash_capability(AttentionImplementation::FlashAttention3, 12, 0).is_err()
        );
    }
}

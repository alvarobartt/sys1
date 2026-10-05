use super::{AttentionImplementation, DecisionModel};
use crate::{
    archs::{qwen35_text as text, qwen35_vision as vision},
    device,
    schema::{ApiError, DecisionRequest, DecisionResponse, Usage},
    tokenizer,
};
use anyhow::Context;
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use serde_json::{Map, Value, json};
use std::{fs, path::Path};

const SYSTEM_PROMPT: &str = "Read the complete state and schema. Decide every field jointly. Each answer must be exactly one of that field's allowed options.";
const PREFIX: &str = "<|im_start|>system\n";
const USER_PREFIX: &str = "<|im_end|>\n<|im_start|>user\nSTATE:\n";
const SUFFIX: &str =
    "\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nJOINT SCHEMA DECISIONS:";

fn cuda_capability(_device: &Device) -> anyhow::Result<Option<(i32, i32)>> {
    #[cfg(feature = "cuda")]
    if let Device::Cuda(cuda) = _device {
        return Ok(Some(
            cuda.cuda_stream()
                .context()
                .compute_capability()
                .context("failed to query CUDA compute capability")?,
        ));
    }
    Ok(None)
}

fn select_attention(
    requested: AttentionImplementation,
    dtype: DType,
    capability: Option<(i32, i32)>,
) -> AttentionImplementation {
    if requested != AttentionImplementation::Auto {
        return requested;
    }
    if !matches!(dtype, DType::F16 | DType::BF16) {
        return AttentionImplementation::Eager;
    }
    #[cfg(feature = "flash-attn-3")]
    if capability == Some((9, 0)) {
        return AttentionImplementation::FlashAttention3;
    }
    #[cfg(feature = "flash-attn-2")]
    if capability.is_some_and(|(major, _)| (8..=9).contains(&major)) {
        return AttentionImplementation::FlashAttention2;
    }
    let _ = capability;
    AttentionImplementation::Eager
}

fn validate_flash_device(
    attention: AttentionImplementation,
    capability: Option<(i32, i32)>,
) -> anyhow::Result<()> {
    let supported = match attention {
        AttentionImplementation::Eager => return Ok(()),
        AttentionImplementation::FlashAttention2 => {
            capability.is_some_and(|(major, _)| (8..=9).contains(&major))
        }
        AttentionImplementation::FlashAttention3 => capability == Some((9, 0)),
        AttentionImplementation::Auto => false,
    };
    anyhow::ensure!(
        capability.is_some(),
        "{} requires a CUDA device",
        attention.cli_name()
    );
    let (major, minor) = capability.unwrap();
    anyhow::ensure!(
        supported,
        "{} does not support CUDA compute capability {major}.{minor}",
        attention.cli_name()
    );
    Ok(())
}

struct Encoded {
    ids: Vec<u32>,
    positions: Vec<[u32; 3]>,
    images: Option<vision::images::Images>,
    videos: Option<vision::video::Videos>,
    media_spans: Vec<(usize, usize, bool)>,
    questions: Vec<clef_head::Question>,
    criteria: Vec<Value>,
}

fn validate_declared_dtypes(
    declared: Option<&str>,
    text_dtype: Option<&str>,
    vision_dtype: Option<&str>,
    requested: DType,
) -> anyhow::Result<()> {
    fn parse(value: &str) -> anyhow::Result<DType> {
        match value {
            "float32" | "f32" => Ok(DType::F32),
            "float16" | "f16" => Ok(DType::F16),
            "bfloat16" | "bf16" => Ok(DType::BF16),
            _ => anyhow::bail!("unsupported Clef checkpoint dtype {value:?} in config.json"),
        }
    }

    let declared = declared.context("Clef config.json is missing dtype")?;
    let source = parse(declared)?;
    for (section, value) in [("text_config", text_dtype), ("vision_config", vision_dtype)] {
        if let Some(value) = value {
            anyhow::ensure!(
                parse(value)? == source,
                "Clef {section}.dtype {value:?} differs from config.json dtype {declared:?}"
            );
        }
    }
    anyhow::ensure!(
        requested != DType::F16 || source == DType::F16,
        "unsafe Clef dtype conversion from {declared} to f16: FP16 has a narrower exponent range; use --dtype bf16 or --dtype f32"
    );
    tracing::info!(checkpoint_dtype = declared, inference_dtype = ?requested, "Clef dtype validated");
    Ok(())
}

pub struct Clef {
    tokenizer: tokenizer::Tokenizer,
    decoder: text::Decoder,
    vision: vision::VisionTower,
    vision_config: text::VisionConfig,
    image_token_id: u32,
    video_token_id: u32,
    head: clef_head::JointHead,
    output_embedding: Tensor,
    device: Device,
    dtype: DType,
    max_length: usize,
}

impl Clef {
    pub fn load(
        path: &Path,
        dtype: DType,
        max_model_len: Option<usize>,
        attention: AttentionImplementation,
    ) -> anyhow::Result<Self> {
        let config = text::Config::load(&path.join("config.json"))?;
        validate_declared_dtypes(
            config.dtype.as_deref(),
            config.text_config.dtype.as_deref(),
            config.vision_config.dtype.as_deref(),
            dtype,
        )?;
        let head_config = clef_head::Config::load(&path.join("joint_head_config.json"))?;
        vision::images::validate_config(
            &path.join("processor_config.json"),
            &config.vision_config,
        )?;
        vision::video::validate_config(&path.join("processor_config.json"), &config.vision_config)?;
        anyhow::ensure!(
            head_config.hidden_size == config.text_config.hidden_size,
            "Clef head hidden size differs from the decoder"
        );
        let device = device::load()?;
        anyhow::ensure!(
            !device.is_cpu() || dtype == DType::F32,
            "Clef on CPU requires f32; use --dtype f32 or --dtype auto"
        );
        let capability = if attention == AttentionImplementation::Eager
            || (attention == AttentionImplementation::Auto && dtype == DType::F32)
        {
            None
        } else {
            cuda_capability(&device)?
        };
        let attention = select_attention(attention, dtype, capability);
        attention.validate(dtype)?;
        validate_flash_device(attention, capability)?;
        tracing::info!(attention = attention.cli_name(), "Clef attention selected");
        let max_length = max_model_len.unwrap_or(16384);
        anyhow::ensure!(
            max_length > 0 && max_length <= 262144,
            "invalid Qwen3.5 model length"
        );
        let tokenizer = tokenizer::from_qwen_json(&fs::read(path.join("tokenizer.json"))?)
            .context("Qwen tokenizer")?;
        let mut shards: Vec<_> = fs::read_dir(path)?
            .filter_map(|entry| entry.ok().map(|entry| entry.path()))
            .filter(|file| {
                file.file_name()
                    .and_then(|name| name.to_str())
                    .is_some_and(|name| {
                        name == "model.safetensors"
                            || (name.starts_with("model-") && name.ends_with(".safetensors"))
                    })
            })
            .collect();
        shards.sort();
        anyhow::ensure!(!shards.is_empty(), "no Qwen3.5 model safetensors found");
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&shards, dtype, &device)? };
        let decoder = text::Decoder::load(
            &config.text_config,
            vb.pp("model.language_model"),
            attention,
        )
        .context("Qwen3.5 decoder")?;
        let vision =
            vision::VisionTower::load(&config.vision_config, vb.pp("model.visual"), attention)
                .context("Qwen3.5 vision tower")?;
        let output_embedding = vb.get(
            (
                config.text_config.vocab_size,
                config.text_config.hidden_size,
            ),
            "lm_head.weight",
        )?;
        let head_path = path.join("joint_head.safetensors");
        let head_vb = unsafe { VarBuilder::from_mmaped_safetensors(&[head_path], dtype, &device)? };
        let head = clef_head::JointHead::load(&head_config, head_vb, attention)
            .context("Clef joint head")?;
        Ok(Self {
            tokenizer,
            decoder,
            vision,
            vision_config: config.vision_config,
            image_token_id: config.image_token_id,
            video_token_id: config.video_token_id,
            head,
            output_embedding,
            device,
            dtype,
            max_length,
        })
    }

    fn tokens(&self, value: &str) -> Result<Vec<u32>, ApiError> {
        self.tokenizer
            .encode(value)
            .map_err(|error| ApiError::new(error.to_string()))
    }

    fn render(value: &Value) -> String {
        if let Some(value) = value.as_str() {
            return value.to_owned();
        }
        let mut value = value.clone();
        fn sort(value: &mut Value) {
            match value {
                Value::Object(object) => {
                    for child in object.values_mut() {
                        sort(child);
                    }
                    object.sort_keys();
                }
                Value::Array(array) => {
                    for child in array {
                        sort(child);
                    }
                }
                _ => {}
            }
        }
        sort(&mut value);
        value.to_string()
    }

    fn options(kind: &str, criteria: &Value, id: &str) -> Result<Vec<(String, Value)>, ApiError> {
        match kind {
            "noul" => {
                let mut options = vec![
                    (
                        "true".to_owned(),
                        json!("The proposition is true or the answer is yes."),
                    ),
                    (
                        "false".to_owned(),
                        json!("The proposition is false or the answer is no."),
                    ),
                ];
                if !criteria.is_null() {
                    let supplied = criteria.as_object().ok_or_else(|| {
                        ApiError::new(format!("question {id:?} noul criteria must be an object"))
                    })?;
                    for (key, value) in &mut options {
                        if let Some(replacement) = supplied.get(key) {
                            *value = replacement.clone();
                        }
                    }
                }
                Ok(options)
            }
            "choice" => {
                let supplied = criteria.as_object().ok_or_else(|| {
                    ApiError::new(format!("question {id:?} choice criteria must be an object"))
                })?;
                if supplied.is_empty() {
                    return Err(ApiError::new(format!(
                        "question {id:?} criteria must not be empty"
                    )));
                }
                let mut options: Vec<_> = supplied
                    .iter()
                    .map(|(key, value)| (key.clone(), value.clone()))
                    .collect();
                options.sort_by(|left, right| left.0.cmp(&right.0));
                Ok(options)
            }
            "score" => {
                let supplied = criteria.as_array().ok_or_else(|| {
                    ApiError::new(format!("question {id:?} score criteria must be an array"))
                })?;
                if supplied.is_empty() {
                    return Err(ApiError::new(format!(
                        "question {id:?} criteria must not be empty"
                    )));
                }
                Ok(supplied
                    .iter()
                    .enumerate()
                    .map(|(index, value)| (index.to_string(), value.clone()))
                    .collect())
            }
            _ => Err(ApiError::new(format!(
                "question {id:?} type must be noul, choice, or score"
            ))),
        }
    }

    fn encode(&self, request: &DecisionRequest) -> Result<Encoded, ApiError> {
        if request.questions.is_empty() {
            return Err(ApiError::new("questions must not be empty"));
        }
        let mut schema = self.tokens("\n\nSCHEMA FIELDS:\n")?;
        let mut questions = Vec::with_capacity(request.questions.len());
        let mut criteria = Vec::with_capacity(request.questions.len());
        for (index, (id, value)) in request.questions.iter().enumerate() {
            let object = value
                .as_object()
                .ok_or_else(|| ApiError::new(format!("question {id:?} must be an object")))?;
            let kind_name = object
                .get("type")
                .and_then(Value::as_str)
                .ok_or_else(|| ApiError::new(format!("question {id:?} requires type")))?;
            let kind = match kind_name {
                "noul" => 0,
                "choice" => 1,
                "score" => 2,
                _ => return Err(ApiError::new(format!("question {id:?} has invalid type"))),
            };
            let criterion = object.get("criteria").cloned().unwrap_or(Value::Null);
            let options = Self::options(kind_name, &criterion, id)?;
            schema.extend(self.tokens(&format!(
                "\nFIELD {}\nID: {id}\nTYPE: {kind_name}\nINSTRUCTION: ",
                index + 1
            ))?);
            let question_start = schema.len();
            let instruction = object
                .get("instructions")
                .and_then(Value::as_str)
                .filter(|value| !value.is_empty())
                .unwrap_or(id);
            schema.extend(self.tokens(instruction)?);
            let question_span = (question_start, schema.len());
            schema.extend(self.tokens("\nALLOWED OPTIONS:\n")?);
            let mut option_spans = Vec::with_capacity(options.len());
            let mut option_ids = Vec::with_capacity(options.len());
            for (option_index, (option_id, description)) in options.into_iter().enumerate() {
                schema.extend(self.tokens(&format!("OPTION {}: ", option_index + 1))?);
                let start = schema.len();
                let mut semantics = Map::new();
                semantics.insert("option_id".into(), Value::String(option_id.clone()));
                if !description.is_null() {
                    semantics.insert("description".into(), description);
                }
                schema.extend(self.tokens(&Self::render(&Value::Object(semantics)))?);
                option_spans.push((start, schema.len()));
                option_ids.push(option_id);
                schema.extend(self.tokens("\n")?);
            }
            schema.extend(self.tokens("END FIELD\n")?);
            questions.push(clef_head::Question {
                id: id.clone(),
                kind,
                question_span,
                option_spans,
                option_ids,
            });
            criteria.push(criterion);
        }
        let mut prefix = self.tokens(&format!("{PREFIX}{SYSTEM_PROMPT}{USER_PREFIX}"))?;
        let suffix = self.tokens(SUFFIX)?;
        let baseline = prefix.len() + schema.len() + suffix.len();
        if baseline > self.max_length {
            return Err(ApiError::new(format!(
                "schema requires {baseline} tokens before state; maximum is {}. Reduce schema size or increase --max-model-len",
                self.max_length
            )));
        }
        let media_budget = self.max_length - baseline;
        let patch_width = self.vision_config.patch_size * self.vision_config.spatial_merge_size;
        let min_image_tokens = (65_536 / patch_width.pow(2)).max(1);
        if request.images.len() > media_budget / min_image_tokens {
            return Err(ApiError::new(format!(
                "image count exceeds the {media_budget}-token media budget; use fewer images or increase --max-model-len"
            )));
        }
        let images = if request.images.is_empty() {
            None
        } else {
            Some(
                vision::images::decode_images(&request.images, &self.vision_config, media_budget)
                    .map_err(|error| ApiError::new(error.to_string()))?,
            )
        };
        let video_budget = media_budget - images.as_ref().map_or(0, |images| images.token_count);
        if request.videos.len() > video_budget {
            return Err(ApiError::new(format!(
                "video count exceeds the {video_budget}-token media budget; use fewer videos or increase --max-model-len"
            )));
        }
        let videos = if request.videos.is_empty() {
            None
        } else {
            Some(
                vision::video::decode_videos(&request.videos, &self.vision_config, video_budget)
                    .map_err(|error| ApiError::new(error.to_string()))?,
            )
        };
        let mut media_text = String::new();
        if let Some(images) = &images {
            for &[t, h, w] in &images.grids {
                media_text.push_str("<|vision_start|>");
                for _ in 0..t * h * w / self.vision_config.spatial_merge_size.pow(2) {
                    media_text.push_str("<|image_pad|>");
                }
                media_text.push_str("<|vision_end|>");
            }
        }
        if let Some(videos) = &videos {
            for (grid, timestamps) in videos.grids.iter().zip(&videos.timestamps) {
                let &[t, h, w] = grid;
                let count = h * w / self.vision_config.spatial_merge_size.pow(2);
                for timestamp in &timestamps[..t] {
                    media_text.push_str(&format!("<{timestamp:.1} seconds>"));
                    media_text.push_str("<|vision_start|>");
                    for _ in 0..count {
                        media_text.push_str("<|video_pad|>");
                    }
                    media_text.push_str("<|vision_end|>");
                }
            }
        }
        if !media_text.is_empty() {
            media_text.push('\n');
            prefix.extend(self.tokens(&media_text)?);
        }
        let mut state = self.tokens(&Self::render(&request.state))?;
        let fixed = prefix.len() + schema.len() + suffix.len();
        if fixed > self.max_length {
            return Err(ApiError::new(format!(
                "schema and media require {fixed} tokens before state; maximum is {}. Reduce media resolution or schema size, or increase --max-model-len",
                self.max_length
            )));
        }
        state.truncate(self.max_length - fixed);
        let offset = prefix.len() + state.len();
        for question in &mut questions {
            question.question_span.0 += offset;
            question.question_span.1 += offset;
            for span in &mut question.option_spans {
                span.0 += offset;
                span.1 += offset;
            }
        }
        prefix.extend(state);
        prefix.extend(schema);
        prefix.extend(suffix);
        let mut positions = Vec::with_capacity(prefix.len());
        let mut media_spans = Vec::new();
        let mut current_position = 0u32;
        let mut cursor = 0usize;
        let mut image_index = 0usize;
        let mut video_index = 0usize;
        let video_frames: Vec<_> = videos
            .as_ref()
            .map(|videos| {
                videos
                    .grids
                    .iter()
                    .flat_map(|&[t, h, w]| (0..t).map(move |_| [1, h, w]))
                    .collect()
            })
            .unwrap_or_default();
        while cursor < prefix.len() {
            if prefix[cursor] == self.image_token_id || prefix[cursor] == self.video_token_id {
                let is_video = prefix[cursor] == self.video_token_id;
                let &[t, h, w] = if is_video {
                    let grid = video_frames
                        .get(video_index)
                        .ok_or_else(|| ApiError::new("too many video tokens"))?;
                    video_index += 1;
                    grid
                } else {
                    let grid = images
                        .as_ref()
                        .and_then(|images| images.grids.get(image_index))
                        .ok_or_else(|| ApiError::new("too many image tokens"))?;
                    image_index += 1;
                    grid
                };
                let merge = self.vision_config.spatial_merge_size;
                let count = t * h * w / merge.pow(2);
                let token_id = if is_video {
                    self.video_token_id
                } else {
                    self.image_token_id
                };
                if prefix
                    .get(cursor..cursor + count)
                    .is_none_or(|ids| ids.iter().any(|id| *id != token_id))
                {
                    return Err(ApiError::new(
                        "media token count does not match visual grid",
                    ));
                }
                media_spans.push((cursor, count, is_video));
                for frame in 0..t {
                    for row in 0..h / merge {
                        for col in 0..w / merge {
                            let _ = frame;
                            positions.push([
                                current_position,
                                current_position + row as u32,
                                current_position + col as u32,
                            ]);
                        }
                    }
                }
                current_position += (h.max(w) / merge) as u32;
                cursor += count;
            } else {
                positions.push([current_position; 3]);
                current_position += 1;
                cursor += 1;
            }
        }
        Ok(Encoded {
            ids: prefix,
            positions,
            images,
            videos,
            media_spans,
            questions,
            criteria,
        })
    }

    fn visual_features(&self, patches: &[f32], grids: &[[usize; 3]]) -> Result<Tensor, ApiError> {
        let patch_width = self.vision_config.in_channels
            * self.vision_config.temporal_patch_size
            * self.vision_config.patch_size.pow(2);
        let patches = Tensor::from_vec(
            patches.to_vec(),
            (patches.len() / patch_width, patch_width),
            &self.device,
        )
        .and_then(|tensor| tensor.to_dtype(self.dtype))
        .map_err(|error| ApiError::internal(error.to_string()))?;
        self.vision
            .forward(&patches, grids)
            .map_err(|error| ApiError::internal(error.to_string()))
    }

    fn predict_one(&self, request: DecisionRequest) -> Result<DecisionResponse, ApiError> {
        let encoded = self.encode(&request)?;
        let ids = Tensor::from_vec(encoded.ids.clone(), (1, encoded.ids.len()), &self.device)
            .map_err(|error| ApiError::internal(error.to_string()))?;
        let hidden = if encoded.images.is_some() || encoded.videos.is_some() {
            let image_features = encoded
                .images
                .as_ref()
                .map(|images| self.visual_features(&images.patches, &images.grids))
                .transpose()?;
            let video_features = encoded
                .videos
                .as_ref()
                .map(|videos| self.visual_features(&videos.patches, &videos.grids))
                .transpose()?;
            let embeddings = self
                .decoder
                .embed(&ids)
                .and_then(|tensor| tensor.squeeze(0))
                .map_err(|error| ApiError::internal(error.to_string()))?;
            let mut chunks = Vec::new();
            let mut token_offset = 0;
            let mut image_offset = 0;
            let mut video_offset = 0;
            for &(start, count, is_video) in &encoded.media_spans {
                if start > token_offset {
                    chunks.push(
                        embeddings
                            .narrow(0, token_offset, start - token_offset)
                            .map_err(|error| ApiError::internal(error.to_string()))?,
                    );
                }
                let (features, offset) = if is_video {
                    (video_features.as_ref().unwrap(), &mut video_offset)
                } else {
                    (image_features.as_ref().unwrap(), &mut image_offset)
                };
                chunks.push(
                    features
                        .narrow(0, *offset, count)
                        .map_err(|error| ApiError::internal(error.to_string()))?,
                );
                token_offset = start + count;
                *offset += count;
            }
            if token_offset < encoded.ids.len() {
                chunks.push(
                    embeddings
                        .narrow(0, token_offset, encoded.ids.len() - token_offset)
                        .map_err(|error| ApiError::internal(error.to_string()))?,
                );
            }
            let embeddings = Tensor::cat(&chunks.iter().collect::<Vec<_>>(), 0)
                .and_then(|tensor| tensor.unsqueeze(0))
                .map_err(|error| ApiError::internal(error.to_string()))?;
            self.decoder
                .forward_embeds(&embeddings, &encoded.positions)
                .map_err(|error| ApiError::internal(error.to_string()))?
        } else {
            self.decoder
                .forward(&ids)
                .map_err(|error| ApiError::internal(error.to_string()))?
        };
        let logits = self
            .head
            .forward(
                &hidden,
                &ids.squeeze(0)
                    .map_err(|error| ApiError::internal(error.to_string()))?,
                &encoded.questions,
                &self.output_embedding,
            )
            .map_err(|error| ApiError::internal(error.to_string()))?;
        let mut answers = Map::new();
        for ((question, criterion), scores) in
            encoded.questions.iter().zip(&encoded.criteria).zip(logits)
        {
            if scores.iter().any(|score| !score.is_finite()) {
                return Err(ApiError::internal(format!(
                    "non-finite logits for question {:?}",
                    question.id
                )));
            }
            let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut probs: Vec<_> = scores.iter().map(|value| (value - max).exp()).collect();
            let sum: f32 = probs.iter().sum();
            for probability in &mut probs {
                *probability /= sum;
            }
            let rounded = |value: f32| (value as f64 * 10000.).round() / 10000.;
            let answer = match question.kind {
                0 => json!({"type":"noul", "noul": rounded(probs[0])}),
                1 => {
                    let winner = probs
                        .iter()
                        .enumerate()
                        .max_by(|left, right| left.1.total_cmp(right.1))
                        .unwrap()
                        .0;
                    let values = question
                        .option_ids
                        .iter()
                        .cloned()
                        .zip(probs.iter().map(|value| json!(rounded(*value))))
                        .collect::<Map<_, _>>();
                    json!({"type":"choice", "choice":question.option_ids[winner], "confidence":rounded(probs[winner]), "probabilities":values})
                }
                _ => {
                    let values = question
                        .option_ids
                        .iter()
                        .cloned()
                        .zip(probs.iter().map(|value| json!(rounded(*value))))
                        .collect::<Map<_, _>>();
                    let legend = criterion
                        .as_array()
                        .unwrap()
                        .iter()
                        .enumerate()
                        .map(|(index, value)| (index.to_string(), value.clone()))
                        .collect::<Map<_, _>>();
                    let score: f32 = probs
                        .iter()
                        .enumerate()
                        .map(|(index, probability)| index as f32 * probability)
                        .sum();
                    let confidence = probs.iter().copied().fold(0., f32::max);
                    json!({"type":"score", "score":rounded(score), "confidence":rounded(confidence), "legend":legend, "probabilities":values})
                }
            };
            answers.insert(question.id.clone(), answer);
        }
        Ok(DecisionResponse {
            model: String::new(),
            answers,
            usage: Usage {
                input_tokens: encoded.ids.len(),
                output_tokens: 0,
            },
        })
    }
}

impl DecisionModel for Clef {
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
            .map(|request| self.predict_one(request))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::batching::{Batcher, BatcherConfig};
    use axum::{
        body::{Body, to_bytes},
        http::{Request, StatusCode},
    };
    use base64::{Engine, engine::general_purpose::STANDARD};
    use hf_hub::HFClient;
    use std::{sync::Arc, time::Duration};
    use tower::ServiceExt;

    const CLEF_FLASH_MODEL_ID: &str = "Cloudflare/clef-flash";
    const CLEF_FLASH_REVISION: &str = "17f0b0ad64efb65d273590632833508766b2aae6";

    async fn clef_flash_snapshot(include_weights: bool) -> anyhow::Result<std::path::PathBuf> {
        let (owner, name) = CLEF_FLASH_MODEL_ID.split_once('/').unwrap();
        let cached = hf_hub::resolve_cache_dir()
            .join(format!("models--{owner}--{name}"))
            .join("snapshots")
            .join(CLEF_FLASH_REVISION);
        let metadata = [
            "config.json",
            "joint_head_config.json",
            "model.safetensors.index.json",
            "processor_config.json",
            "tokenizer.json",
        ];
        if metadata.iter().all(|name| cached.join(name).is_file()) {
            if !include_weights {
                return Ok(cached);
            }
            let index: Value =
                serde_json::from_slice(&fs::read(cached.join("model.safetensors.index.json"))?)?;
            let shards = index["weight_map"]
                .as_object()
                .context("Clef-Flash has no safetensors weight map")?;
            if cached.join("joint_head.safetensors").is_file()
                && shards.values().all(|name| {
                    name.as_str()
                        .is_some_and(|name| cached.join(name).is_file())
                })
            {
                return Ok(cached);
            }
        }
        if include_weights {
            crate::hub::download(CLEF_FLASH_MODEL_ID, CLEF_FLASH_REVISION).await
        } else {
            HFClient::new()?
                .model(owner, name)
                .snapshot_download()
                .revision(CLEF_FLASH_REVISION)
                .allow_patterns(metadata.into_iter().map(str::to_owned).collect())
                .send()
                .await
                .map_err(Into::into)
        }
    }

    #[tokio::test]
    async fn public_clef_flash_release_matches_loader_contract() -> anyhow::Result<()> {
        let path = clef_flash_snapshot(false).await?;
        assert_eq!(
            crate::models::Architecture::from_path(&path)?,
            crate::models::Architecture::Qwen35
        );
        let config = text::Config::load(&path.join("config.json"))?;
        let head = clef_head::Config::load(&path.join("joint_head_config.json"))?;
        assert_eq!(head.hidden_size, config.text_config.hidden_size);
        vision::images::validate_config(
            &path.join("processor_config.json"),
            &config.vision_config,
        )?;
        vision::video::validate_config(&path.join("processor_config.json"), &config.vision_config)?;
        let tokenizer = tokenizer::from_qwen_json(&fs::read(path.join("tokenizer.json"))?)?;
        assert!(!tokenizer.encode("Clef-Flash")?.is_empty());
        let index: Value =
            serde_json::from_slice(&fs::read(path.join("model.safetensors.index.json"))?)?;
        let weights = index["weight_map"]
            .as_object()
            .context("Clef-Flash has no safetensors weight map")?;
        for name in [
            "lm_head.weight",
            "model.language_model.embed_tokens.weight",
            "model.visual.patch_embed.proj.weight",
        ] {
            assert!(
                weights.contains_key(name),
                "missing Clef-Flash weight {name}"
            );
        }
        Ok(())
    }

    #[test]
    fn checkpoint_dtype_rejects_unsafe_fp16_conversion() {
        for source in ["bfloat16", "float32"] {
            let error =
                validate_declared_dtypes(Some(source), Some(source), Some(source), DType::F16)
                    .unwrap_err();
            assert!(error.to_string().contains("unsafe Clef dtype conversion"));
        }
        for requested in [DType::BF16, DType::F32] {
            validate_declared_dtypes(
                Some("bfloat16"),
                Some("bfloat16"),
                Some("bfloat16"),
                requested,
            )
            .unwrap();
        }
        validate_declared_dtypes(
            Some("float16"),
            Some("float16"),
            Some("float16"),
            DType::F16,
        )
        .unwrap();
        assert!(validate_declared_dtypes(None, None, None, DType::F16).is_err());
        assert!(
            validate_declared_dtypes(
                Some("bfloat16"),
                Some("float16"),
                Some("bfloat16"),
                DType::BF16,
            )
            .is_err()
        );
    }

    #[test]
    fn auto_attention_respects_dtype_and_cuda_capability() {
        assert_eq!(
            select_attention(AttentionImplementation::Auto, DType::F32, Some((8, 9))),
            AttentionImplementation::Eager
        );
        assert_eq!(
            select_attention(AttentionImplementation::Auto, DType::BF16, None),
            AttentionImplementation::Eager
        );
        #[cfg(feature = "flash-attn-2")]
        {
            assert_eq!(
                select_attention(AttentionImplementation::Auto, DType::BF16, Some((8, 9))),
                AttentionImplementation::FlashAttention2
            );
            assert_eq!(
                select_attention(AttentionImplementation::Auto, DType::BF16, Some((7, 5))),
                AttentionImplementation::Eager
            );
        }
        #[cfg(feature = "flash-attn-3")]
        {
            assert_eq!(
                select_attention(AttentionImplementation::Auto, DType::BF16, Some((9, 0))),
                AttentionImplementation::FlashAttention3
            );
            assert_eq!(
                select_attention(AttentionImplementation::Auto, DType::BF16, Some((8, 9))),
                AttentionImplementation::Eager
            );
        }
        assert!(
            validate_flash_device(AttentionImplementation::FlashAttention3, Some((8, 9))).is_err()
        );
    }

    #[test]
    fn public_clef_flash_answers_image_and_video_requests() {
        let path = tokio::runtime::Runtime::new()
            .unwrap()
            .block_on(clef_flash_snapshot(true))
            .unwrap();
        #[cfg(feature = "cpu")]
        {
            let error = Clef::load(
                &path,
                DType::BF16,
                Some(512),
                AttentionImplementation::Eager,
            )
            .err()
            .unwrap();
            assert!(error.to_string().contains("Clef on CPU requires f32"));
        }
        let dtype = if cfg!(feature = "cpu") {
            DType::F32
        } else if cfg!(feature = "cuda") {
            DType::BF16
        } else {
            DType::F16
        };
        let model = Clef::load(&path, dtype, Some(512), AttentionImplementation::Auto).unwrap();
        let mut image_cases = Map::new();
        let mut video_cases = Map::new();
        let mut text_cases = Map::new();
        let too_many_images: DecisionRequest = serde_json::from_value(json!({
            "state": "test",
            "images": vec!["not-base64"; 8],
            "questions": {
                "color": {
                    "type": "choice",
                    "instructions": "Choose the color.",
                    "criteria": {"red": "Red.", "blue": "Blue."}
                }
            }
        }))
        .unwrap();
        let error = model.encode(&too_many_images).err().unwrap();
        assert!(error.error.contains("image count exceeds"));
        for (name, rgb) in [("red", [255, 0, 0]), ("blue", [0, 0, 255])] {
            let mut ppm = b"P6\n2 2\n255\n".to_vec();
            for _ in 0..4 {
                ppm.extend_from_slice(&rgb);
            }
            let image = format!(
                "data:image/x-portable-pixmap;base64,{}",
                STANDARD.encode(&ppm)
            );
            let source = if name == "red" {
                json!(image)
            } else {
                json!({"bytes": ppm})
            };
            let request: DecisionRequest = serde_json::from_value(json!({
                "state": "What color is the square in the image?",
                "images": [source],
                "questions": {
                    "color": {
                        "type": "choice",
                        "instructions": "Choose the color visible in the image.",
                        "criteria": {"red": "A red square.", "blue": "A blue square."}
                    }
                }
            }))
            .unwrap();
            let encoded = model.encode(&request).unwrap();
            assert_eq!(encoded.ids.len(), 207);
            assert_eq!(encoded.questions[0].question_span, (136, 144));
            assert_eq!(
                encoded.questions[0].option_spans,
                vec![(155, 167), (173, 185)]
            );
            let response = model.predict_one(request).unwrap();
            println!("{name} image: {}", response.answers["color"]);
            image_cases.insert(
                name.to_owned(),
                json!({
                    "input_tokens": response.usage.input_tokens,
                    "media_spans": encoded.media_spans,
                    "grids": encoded.images.as_ref().unwrap().grids,
                    "choice": response.answers["color"]["choice"]
                }),
            );
            assert!(response.usage.input_tokens > 64);
            assert_eq!(response.answers["color"]["choice"], name);
            if name == "red" {
                let probability = response.answers["color"]["probabilities"]["red"]
                    .as_f64()
                    .unwrap();
                assert!((probability - 0.9916).abs() < 0.01);
            }
            let probabilities = &response.answers["color"]["probabilities"];
            for color in ["red", "blue"] {
                let probability = probabilities[color].as_f64().unwrap();
                assert!(probability.is_finite() && (0.0..=1.0).contains(&probability));
            }
        }
        let mut large_ppm = b"P6\n640 480\n255\n".to_vec();
        for _ in 0..640 * 480 {
            large_ppm.extend_from_slice(&[255, 0, 0]);
        }
        let large_request: DecisionRequest = serde_json::from_value(json!({
            "state": "What color is the image?",
            "images": [{"bytes": large_ppm}],
            "questions": {
                "color": {
                    "type": "choice",
                    "instructions": "Choose the image color.",
                    "criteria": {"red": "Red.", "blue": "Blue."}
                }
            }
        }))
        .unwrap();
        let large_encoded = model.encode(&large_request).unwrap();
        assert_eq!(large_encoded.ids.len(), 433);
        let large_response = model.predict_one(large_request).unwrap();
        println!("large image: {}", large_response.answers["color"]);
        image_cases.insert(
            "large_red".to_owned(),
            json!({
                "input_tokens": large_response.usage.input_tokens,
                "media_spans": large_encoded.media_spans,
                "grids": large_encoded.images.as_ref().unwrap().grids,
                "choice": large_response.answers["color"]["choice"]
            }),
        );
        assert_eq!(large_response.answers["color"]["choice"], "red");
        assert!(
            large_response.answers["color"]["confidence"]
                .as_f64()
                .unwrap()
                > 0.9
        );
        let images = [[255, 0, 0], [0, 0, 255]].map(|rgb| {
            let mut ppm = b"P6\n2 2\n255\n".to_vec();
            for _ in 0..4 {
                ppm.extend_from_slice(&rgb);
            }
            json!({"bytes": ppm})
        });
        let request: DecisionRequest = serde_json::from_value(json!({
            "state": "There are two images. What color is the second image?",
            "images": images,
            "questions": {
                "color": {
                    "type": "choice",
                    "instructions": "Choose the color in the second image.",
                    "criteria": {"red": "A red square.", "blue": "A blue square."}
                }
            }
        }))
        .unwrap();
        let encoded = model.encode(&request).unwrap();
        assert_eq!(encoded.ids.len(), 276);
        assert_eq!(encoded.media_spans.len(), 2);
        let response = model.predict_one(request).unwrap();
        println!("two images: {}", response.answers["color"]);
        image_cases.insert(
            "red_then_blue".to_owned(),
            json!({
                "input_tokens": response.usage.input_tokens,
                "media_spans": encoded.media_spans,
                "grids": encoded.images.as_ref().unwrap().grids,
                "choice": response.answers["color"]["choice"]
            }),
        );
        assert_eq!(response.answers["color"]["choice"], "blue");
        assert!(
            (response.answers["color"]["probabilities"]["blue"]
                .as_f64()
                .unwrap()
                - 0.9879)
                .abs()
                < 0.01
        );
        let video = vision::video::synthetic_test_video();
        let request: DecisionRequest = serde_json::from_value(json!({
            "state": "The clip changes color halfway through.",
            "videos": [{"bytes": video}],
            "questions": {
                "color": {
                    "type": "choice",
                    "instructions": "What color is visible in the second half of the video?",
                    "criteria": {"red": "Red frames.", "blue": "Blue frames."}
                }
            }
        }))
        .unwrap();
        let encoded = model.encode(&request).unwrap();
        assert_eq!(encoded.videos.as_ref().unwrap().grids, vec![[2, 4, 4]]);
        assert_eq!(encoded.ids.len(), 165);
        assert_eq!(encoded.questions[0].question_span, (92, 104));
        assert_eq!(
            encoded.questions[0].option_spans,
            vec![(115, 126), (132, 143)]
        );
        let response = model.predict_one(request).unwrap();
        println!("video: {}", response.answers["color"]);
        video_cases.insert(
            "red_then_blue".to_owned(),
            json!({
                "input_tokens": response.usage.input_tokens,
                "media_spans": encoded.media_spans,
                "grids": encoded.videos.as_ref().unwrap().grids,
                "timestamps": encoded.videos.as_ref().unwrap().timestamps,
                "choice": response.answers["color"]["choice"]
            }),
        );
        assert_eq!(response.answers["color"]["choice"], "blue");
        assert!(
            (response.answers["color"]["probabilities"]["blue"]
                .as_f64()
                .unwrap()
                - 0.9782)
                .abs()
                < 0.01
        );
        let probabilities = &response.answers["color"]["probabilities"];
        assert!(probabilities["red"].as_f64().unwrap().is_finite());
        assert!(probabilities["blue"].as_f64().unwrap().is_finite());

        let mut ppm = b"P6\n2 2\n255\n".to_vec();
        for _ in 0..4 {
            ppm.extend_from_slice(&[255, 0, 0]);
        }
        let request: DecisionRequest = serde_json::from_value(json!({
            "state": "The image is a reference; use the later video frames for the answer.",
            "images": [{"bytes": ppm}],
            "videos": [{"bytes": vision::video::synthetic_test_video()}],
            "questions": {
                "color": {
                    "type": "choice",
                    "instructions": "What color appears in the final video frames?",
                    "criteria": {"red": "Red final frames.", "blue": "Blue final frames."}
                }
            }
        }))
        .unwrap();
        let encoded = model.encode(&request).unwrap();
        assert_eq!(encoded.media_spans.len(), 3);
        let response = model.predict_one(request).unwrap();
        println!("mixed media: {}", response.answers["color"]);
        video_cases.insert(
            "image_then_video".to_owned(),
            json!({
                "input_tokens": response.usage.input_tokens,
                "media_spans": encoded.media_spans,
                "image_grids": encoded.images.as_ref().unwrap().grids,
                "video_grids": encoded.videos.as_ref().unwrap().grids,
                "choice": response.answers["color"]["choice"]
            }),
        );
        assert_eq!(response.answers["color"]["choice"], "blue");
        assert!(
            response.answers["color"]["confidence"]
                .as_f64()
                .unwrap()
                .is_finite()
        );

        let request: DecisionRequest = serde_json::from_value(json!({
            "state": "The sky is blue.",
            "questions": {
                "color": {
                    "type": "choice",
                    "instructions": "What color is the sky?",
                    "criteria": {"blue": "Blue sky.", "red": "Red sky."}
                }
            }
        }))
        .unwrap();
        let response = model.predict_one(request).unwrap();
        println!("text reference: {}", response.answers["color"]);
        text_cases.insert(
            "sky".to_owned(),
            json!({
                "input_tokens": response.usage.input_tokens,
                "choice": response.answers["color"]["choice"]
            }),
        );
        assert_eq!(response.answers["color"]["choice"], "blue");
        assert!(
            (response.answers["color"]["probabilities"]["blue"]
                .as_f64()
                .unwrap()
                - 0.9707)
                .abs()
                < 0.01
        );

        let mut ppm = b"P6\n2 2\n255\n".to_vec();
        for _ in 0..4 {
            ppm.extend_from_slice(&[255, 0, 0]);
        }
        let request = json!({
            "state": "The image is a reference; use the later video frames for the answer.",
            "images": [{"bytes": ppm}],
            "videos": [{"base64": STANDARD.encode(vision::video::synthetic_test_video())}],
            "questions": {
                "color": {
                    "type": "choice",
                    "instructions": "What color appears in the final video frames?",
                    "criteria": {"red": "Red final frames.", "blue": "Blue final frames."}
                }
            }
        });
        let readme_request: DecisionRequest = serde_json::from_value(json!({
            "model": "clef-flash",
            "state": "Our checkout started returning errors and orders are blocked.",
            "questions": {
                "department": {
                    "type": "choice",
                    "instructions": "Which team should handle the message?",
                    "criteria": {"billing": "Payments or invoices", "technical": "Bugs or outages"}
                },
                "urgency": {"type": "score", "criteria": ["Can wait", "This week", "Today"]},
                "outage": {"type": "noul", "instructions": "Is a service down?"}
            }
        }))
        .unwrap();
        let readme_response = model.predict_one(readme_request).unwrap();
        println!(
            "README example: {}",
            serde_json::to_string(&readme_response).unwrap()
        );
        assert_eq!(readme_response.answers.len(), 3);
        assert_eq!(readme_response.answers["department"]["choice"], "technical");
        assert!(
            (readme_response.answers["urgency"]["score"]
                .as_f64()
                .unwrap()
                - 1.7879)
                .abs()
                < 0.02
        );
        assert!(
            (readme_response.answers["outage"]["noul"].as_f64().unwrap() - 0.8415).abs() < 0.02
        );
        text_cases.insert(
            "checkout".to_owned(),
            json!({
                "input_tokens": readme_response.usage.input_tokens,
                "department": readme_response.answers["department"]["choice"],
                "urgency_type": readme_response.answers["urgency"]["type"],
                "urgency_legend": readme_response.answers["urgency"]["legend"],
                "outage_type": readme_response.answers["outage"]["type"]
            }),
        );
        let invoice_request: DecisionRequest = serde_json::from_value(json!({
            "state": {"invoice": {"vendor": "Acme", "total": 1250.0, "currency": "USD", "status": "overdue"}},
            "questions": {
                "status": {
                    "type": "choice",
                    "instructions": "What is the invoice status?",
                    "criteria": {"paid": "Invoice is paid.", "overdue": "Invoice is past due.", "draft": "Not sent."}
                },
                "large": {"type": "noul", "instructions": "Is the total above 1000 USD?"}
            }
        }))
        .unwrap();
        let invoice_response = model.predict_one(invoice_request).unwrap();
        assert_eq!(invoice_response.answers["status"]["choice"], "overdue");
        assert!(invoice_response.answers["large"]["noul"].as_f64().unwrap() > 0.5);
        text_cases.insert(
            "invoice".to_owned(),
            json!({
                "input_tokens": invoice_response.usage.input_tokens,
                "status": invoice_response.answers["status"]["choice"],
                "large_type": invoice_response.answers["large"]["type"],
                "large_positive": invoice_response.answers["large"]["noul"].as_f64().unwrap() > 0.5
            }),
        );
        insta::assert_yaml_snapshot!("clef_flash_text", text_cases);
        insta::assert_yaml_snapshot!("clef_flash_images", image_cases);
        insta::assert_yaml_snapshot!("clef_flash_videos", video_cases);
        let runtime = tokio::runtime::Runtime::new().unwrap();
        runtime.block_on(async move {
            let batcher = Batcher::new(
                Arc::new(model),
                "clef-test".to_owned(),
                BatcherConfig {
                    max_batch_size: 1,
                    max_batch_questions: 1,
                    max_questions_per_request: 1,
                    wait: Duration::ZERO,
                    queue_capacity: 1,
                    response_timeout: None,
                },
            );
            let response = crate::api::router(batcher, 16_777_216)
                .oneshot(
                    Request::post("/v1/systemone")
                        .header("content-type", "application/json")
                        .body(Body::from(serde_json::to_vec(&request).unwrap()))
                        .unwrap(),
                )
                .await
                .unwrap();
            assert_eq!(response.status(), StatusCode::OK);
            let body = to_bytes(response.into_body(), 1 << 20).await.unwrap();
            let response: Value = serde_json::from_slice(&body).unwrap();
            assert_eq!(response["model"], "clef-test");
            assert_eq!(response["answers"]["color"]["choice"], "blue");
        });
    }
}

mod clef_head {
    use super::AttentionImplementation;
    use candle_core::{D, DType, Result, Tensor};
    use candle_nn::{
        Embedding, LayerNorm, Linear, VarBuilder, embedding, layer_norm,
        ops::{sigmoid, softmax_last_dim},
    };
    use serde::Deserialize;
    use std::{fs, path::Path};

    #[derive(Deserialize)]
    pub struct Config {
        pub hidden_size: usize,
        pub width: usize,
        pub routing_layers: usize,
        pub layers: usize,
        pub heads: usize,
        pub feedforward: usize,
    }

    impl Config {
        pub fn load(path: &Path) -> anyhow::Result<Self> {
            Ok(serde_json::from_slice(&fs::read(path)?)?)
        }
    }

    #[derive(Clone, Debug)]
    pub struct Question {
        pub id: String,
        pub kind: usize,
        pub question_span: (usize, usize),
        pub option_spans: Vec<(usize, usize)>,
        pub option_ids: Vec<String>,
    }

    fn projection(input: usize, output: usize, vb: VarBuilder) -> Result<Linear> {
        Ok(Linear::new(vb.get((output, input), "weight")?, None))
    }

    fn dense(input: usize, output: usize, vb: VarBuilder) -> Result<Linear> {
        Ok(Linear::new(
            vb.get((output, input), "weight")?,
            Some(vb.get(output, "bias")?),
        ))
    }

    struct Attention {
        in_proj: Linear,
        out_proj: Linear,
        heads: usize,
        width: usize,
        implementation: AttentionImplementation,
    }

    impl Attention {
        fn load(
            width: usize,
            heads: usize,
            vb: VarBuilder,
            implementation: AttentionImplementation,
        ) -> Result<Self> {
            Ok(Self {
                in_proj: Linear::new(
                    vb.get((width * 3, width), "in_proj_weight")?,
                    Some(vb.get(width * 3, "in_proj_bias")?),
                ),
                out_proj: dense(width, width, vb.pp("out_proj"))?,
                heads,
                width,
                implementation,
            })
        }

        fn forward(&self, query: &Tensor, memory: &Tensor) -> Result<Tensor> {
            let (batch, q_len, _) = query.dims3()?;
            let kv_len = memory.dim(1)?;
            let head_dim = self.width / self.heads;
            let q = query
                .apply(&self.in_proj)?
                .narrow(D::Minus1, 0, self.width)?
                .reshape((batch, q_len, self.heads, head_dim))?;
            let kv = memory.apply(&self.in_proj)?;
            let k = kv
                .narrow(D::Minus1, self.width, self.width)?
                .reshape((batch, kv_len, self.heads, head_dim))?;
            let v = kv
                .narrow(D::Minus1, self.width * 2, self.width)?
                .reshape((batch, kv_len, self.heads, head_dim))?;
            let scale = (head_dim as f32).powf(-0.5);
            #[cfg(feature = "metal")]
            if query.device().is_metal() {
                return candle_nn::ops::sdpa(
                    &q.transpose(1, 2)?.contiguous()?,
                    &k.transpose(1, 2)?.contiguous()?,
                    &v.transpose(1, 2)?.contiguous()?,
                    None,
                    false,
                    scale,
                    1.0,
                )?
                .transpose(1, 2)?
                .reshape((batch, q_len, self.width))?
                .apply(&self.out_proj);
            }
            let attended = match self.implementation {
                AttentionImplementation::Eager => {
                    let q = q.transpose(1, 2)?;
                    let k = k.transpose(1, 2)?;
                    let v = v.transpose(1, 2)?;
                    let scores = (q
                        .contiguous()?
                        .matmul(&k.transpose(D::Minus2, D::Minus1)?.contiguous()?)?
                        * scale as f64)?;
                    softmax_last_dim(&scores.to_dtype(DType::F32)?)?
                        .to_dtype(query.dtype())?
                        .contiguous()?
                        .matmul(&v.contiguous()?)?
                        .transpose(1, 2)?
                }
                #[cfg(feature = "flash-attn-2")]
                AttentionImplementation::FlashAttention2 => candle_flash_attn::flash_attn(
                    &q.contiguous()?,
                    &k.contiguous()?,
                    &v.contiguous()?,
                    scale,
                    false,
                )?,
                #[cfg(feature = "flash-attn-3")]
                AttentionImplementation::FlashAttention3 => candle_flash_attn_v3::flash_attn(
                    &q.contiguous()?,
                    &k.contiguous()?,
                    &v.contiguous()?,
                    scale,
                    false,
                    false,
                )?,
                #[allow(unreachable_patterns)]
                other => candle_core::bail!("{} support is not compiled in", other.cli_name()),
            };
            attended
                .reshape((batch, q_len, self.width))?
                .apply(&self.out_proj)
        }
    }

    struct EvidenceLayer {
        query_norm: LayerNorm,
        memory_norm: LayerNorm,
        attention: Attention,
        feedforward_norm: LayerNorm,
        fc1: Linear,
        fc2: Linear,
    }

    impl EvidenceLayer {
        fn load(
            config: &Config,
            vb: VarBuilder,
            implementation: AttentionImplementation,
        ) -> Result<Self> {
            Ok(Self {
                query_norm: layer_norm(config.width, 1e-5, vb.pp("query_norm"))?,
                memory_norm: layer_norm(config.width, 1e-5, vb.pp("memory_norm"))?,
                attention: Attention::load(
                    config.width,
                    config.heads,
                    vb.pp("attention"),
                    implementation,
                )?,
                feedforward_norm: layer_norm(config.width, 1e-5, vb.pp("feedforward_norm"))?,
                fc1: dense(config.width, config.feedforward, vb.pp("feedforward.0"))?,
                fc2: dense(config.feedforward, config.width, vb.pp("feedforward.3"))?,
            })
        }

        fn forward(&self, queries: &Tensor, memory: &Tensor) -> Result<Tensor> {
            let routed = self.attention.forward(
                &queries.apply(&self.query_norm)?,
                &memory.apply(&self.memory_norm)?,
            )?;
            let queries = (queries + routed)?;
            &queries
                + queries
                    .apply(&self.feedforward_norm)?
                    .apply(&self.fc1)?
                    .gelu_erf()?
                    .apply(&self.fc2)?
        }
    }

    struct DecoderLayer {
        self_attention: Attention,
        cross_attention: Attention,
        norm1: LayerNorm,
        norm2: LayerNorm,
        norm3: LayerNorm,
        fc1: Linear,
        fc2: Linear,
    }

    impl DecoderLayer {
        fn load(
            config: &Config,
            vb: VarBuilder,
            implementation: AttentionImplementation,
        ) -> Result<Self> {
            Ok(Self {
                self_attention: Attention::load(
                    config.width,
                    config.heads,
                    vb.pp("self_attn"),
                    implementation,
                )?,
                cross_attention: Attention::load(
                    config.width,
                    config.heads,
                    vb.pp("multihead_attn"),
                    implementation,
                )?,
                norm1: layer_norm(config.width, 1e-5, vb.pp("norm1"))?,
                norm2: layer_norm(config.width, 1e-5, vb.pp("norm2"))?,
                norm3: layer_norm(config.width, 1e-5, vb.pp("norm3"))?,
                fc1: dense(config.width, config.feedforward, vb.pp("linear1"))?,
                fc2: dense(config.feedforward, config.width, vb.pp("linear2"))?,
            })
        }

        fn forward(&self, fields: &Tensor, memory: &Tensor) -> Result<Tensor> {
            let normalized = fields.apply(&self.norm1)?;
            let fields = (fields + self.self_attention.forward(&normalized, &normalized)?)?;
            let fields = (&fields
                + self
                    .cross_attention
                    .forward(&fields.apply(&self.norm2)?, memory)?)?;
            &fields
                + fields
                    .apply(&self.norm3)?
                    .apply(&self.fc1)?
                    .gelu_erf()?
                    .apply(&self.fc2)?
        }
    }

    pub struct JointHead {
        hidden_norm: LayerNorm,
        memory_projection: Linear,
        question_projection: Linear,
        option_question_projection: Linear,
        global_projection: Linear,
        option_context_projection: Linear,
        option_lexical_projection: Linear,
        type_embedding: Embedding,
        evidence_layers: Vec<EvidenceLayer>,
        option_summary_norm: LayerNorm,
        layers: Vec<DecoderLayer>,
        field_norm: LayerNorm,
        option_norm: LayerNorm,
        scorer1: Linear,
        scorer2: Linear,
        prior_scale: Tensor,
        joint_scale: Tensor,
        residual_gate: Tensor,
        width: usize,
    }

    impl JointHead {
        pub fn load(
            config: &Config,
            vb: VarBuilder,
            implementation: AttentionImplementation,
        ) -> Result<Self> {
            let h = config.hidden_size;
            let w = config.width;
            Ok(Self {
                hidden_norm: layer_norm(h, 1e-5, vb.pp("hidden_norm"))?,
                memory_projection: projection(h, w, vb.pp("memory_projection"))?,
                question_projection: projection(h, w, vb.pp("question_projection"))?,
                option_question_projection: projection(h, w, vb.pp("option_question_projection"))?,
                global_projection: projection(h, w, vb.pp("global_projection"))?,
                option_context_projection: projection(h, w, vb.pp("option_context_projection"))?,
                option_lexical_projection: projection(h, w, vb.pp("option_lexical_projection"))?,
                type_embedding: embedding(3, w, vb.pp("type_embedding"))?,
                evidence_layers: (0..config.routing_layers)
                    .map(|i| {
                        EvidenceLayer::load(
                            config,
                            vb.pp(format!("evidence_layers.{i}")),
                            implementation,
                        )
                    })
                    .collect::<Result<Vec<_>>>()?,
                option_summary_norm: layer_norm(w, 1e-5, vb.pp("option_summary_norm"))?,
                layers: (0..config.layers)
                    .map(|i| {
                        DecoderLayer::load(config, vb.pp(format!("layers.{i}")), implementation)
                    })
                    .collect::<Result<Vec<_>>>()?,
                field_norm: layer_norm(w, 1e-5, vb.pp("field_norm"))?,
                option_norm: layer_norm(w, 1e-5, vb.pp("option_norm"))?,
                scorer1: dense(w * 4, w, vb.pp("residual_scorer.0"))?,
                scorer2: dense(w, 1, vb.pp("residual_scorer.3"))?,
                prior_scale: vb.get((), "prior_logit_scale")?,
                joint_scale: vb.get((), "joint_logit_scale")?,
                residual_gate: vb.get((), "residual_gate")?,
                width: w,
            })
        }

        fn mean_span(values: &Tensor, (start, end): (usize, usize)) -> Result<Tensor> {
            Self::mean_rows(&values.narrow(0, start, end - start)?)
        }

        fn mean_rows(values: &Tensor) -> Result<Tensor> {
            let rows = values.dim(0)?;
            let weights =
                (Tensor::ones((1, rows), values.dtype(), values.device())? * (1. / rows as f64))?;
            weights.matmul(&values.contiguous()?)?.squeeze(0)
        }

        fn normalize(values: &Tensor) -> Result<Tensor> {
            let values = values.to_dtype(DType::F32)?;
            let norm = (values.sqr()?.sum_keepdim(D::Minus1)? + 1e-12)?.sqrt()?;
            values.broadcast_div(&norm)
        }

        pub fn forward(
            &self,
            hidden: &Tensor,
            ids: &Tensor,
            questions: &[Question],
            output_embedding: &Tensor,
        ) -> Result<Vec<Vec<f32>>> {
            let hidden = hidden.apply(&self.hidden_norm)?;
            let hidden = if hidden.rank() == 3 {
                hidden.squeeze(0)?
            } else {
                hidden
            };
            let length = hidden.dim(0)?;
            let memory = hidden.apply(&self.memory_projection)?.unsqueeze(0)?;
            let global = hidden.narrow(0, length - 1, 1)?.squeeze(0)?;
            let question_vectors = Tensor::stack(
                &questions
                    .iter()
                    .map(|q| Self::mean_span(&hidden, q.question_span))
                    .collect::<Result<Vec<_>>>()?
                    .iter()
                    .collect::<Vec<_>>(),
                0,
            )?;
            let type_ids = Tensor::from_vec(
                questions.iter().map(|q| q.kind as u32).collect::<Vec<_>>(),
                questions.len(),
                hidden.device(),
            )?;
            let mut lexical = Vec::with_capacity(questions.len());
            let mut routed = Vec::with_capacity(questions.len());
            let mut counts = Vec::with_capacity(questions.len());
            for (index, question) in questions.iter().enumerate() {
                let context = Tensor::stack(
                    &question
                        .option_spans
                        .iter()
                        .map(|&span| Self::mean_span(&hidden, span))
                        .collect::<Result<Vec<_>>>()?
                        .iter()
                        .collect::<Vec<_>>(),
                    0,
                )?;
                let lex = Tensor::stack(
                    &question
                        .option_spans
                        .iter()
                        .map(|&(start, end)| {
                            let selected = ids.narrow(0, start, end - start)?;
                            Self::mean_rows(&output_embedding.index_select(&selected, 0)?)
                        })
                        .collect::<Result<Vec<_>>>()?
                        .iter()
                        .collect::<Vec<_>>(),
                    0,
                )?;
                let question_vector = question_vectors.narrow(0, index, 1)?;
                let query = (context.apply(&self.option_context_projection)?
                    + lex.apply(&self.option_lexical_projection)?)?
                .broadcast_add(&question_vector.apply(&self.option_question_projection)?)?;
                counts.push(question.option_spans.len());
                lexical.push(lex);
                routed.push(query);
            }
            let mut options = Tensor::cat(&routed.iter().collect::<Vec<_>>(), 0)?.unsqueeze(0)?;
            for layer in &self.evidence_layers {
                options = layer.forward(&options, &memory)?;
            }
            let options = options.squeeze(0)?;
            let mut split = Vec::with_capacity(questions.len());
            let mut start = 0;
            for count in counts {
                split.push(options.narrow(0, start, count)?);
                start += count;
            }
            let base_fields = question_vectors.apply(&self.question_projection)?;
            let mut summaries = Vec::with_capacity(questions.len());
            for (index, values) in split.iter().enumerate() {
                let field = base_fields.narrow(0, index, 1)?.squeeze(0)?;
                let scores = (values.matmul(&field.unsqueeze(1)?)? / (self.width as f64).sqrt())?
                    .transpose(0, 1)?;
                let weights =
                    softmax_last_dim(&scores.to_dtype(DType::F32)?)?.to_dtype(values.dtype())?;
                summaries.push(
                    weights
                        .contiguous()?
                        .matmul(&values.contiguous()?)?
                        .squeeze(0)?,
                );
            }
            let fields = ((&base_fields
                + Tensor::stack(&summaries.iter().collect::<Vec<_>>(), 0)?
                    .apply(&self.option_summary_norm)?)?
            .broadcast_add(&global.unsqueeze(0)?.apply(&self.global_projection)?)?
                + type_ids.apply(&self.type_embedding)?)?;
            let mut fields = fields.unsqueeze(0)?;
            for layer in &self.layers {
                fields = layer.forward(&fields, &memory)?;
            }
            let fields = fields.squeeze(0)?.apply(&self.field_norm)?;
            let prior_scale = self
                .prior_scale
                .to_dtype(DType::F32)?
                .clamp(f64::NEG_INFINITY, 100f64.ln())?
                .exp()?;
            let joint_scale = self
                .joint_scale
                .to_dtype(DType::F32)?
                .clamp(f64::NEG_INFINITY, 100f64.ln())?
                .exp()?;
            let gate = sigmoid(&self.residual_gate.to_dtype(DType::F32)?)?;
            let mut result = Vec::with_capacity(questions.len());
            for (index, values) in split.iter().enumerate() {
                let field = fields.narrow(0, index, 1)?;
                let normalized_anchor = Self::normalize(
                    &(question_vectors.narrow(0, index, 1)? + global.unsqueeze(0)?)?,
                )?;
                let prior = Self::normalize(&lexical[index])?
                    .matmul(&normalized_anchor.t()?)?
                    .squeeze(1)?
                    .broadcast_mul(&prior_scale)?;
                let values = values.apply(&self.option_norm)?;
                let repeated = field.broadcast_as(values.shape())?;
                let field_cos = Self::normalize(&repeated)?
                    .broadcast_mul(&Self::normalize(&values)?)?
                    .sum(D::Minus1)?;
                let features = Tensor::cat(
                    &[
                        &repeated,
                        &values,
                        &repeated.broadcast_mul(&values)?,
                        &(&repeated - &values)?.abs()?,
                    ],
                    D::Minus1,
                )?;
                let residual = features
                    .apply(&self.scorer1)?
                    .gelu_erf()?
                    .apply(&self.scorer2)?
                    .squeeze(1)?
                    .to_dtype(DType::F32)?;
                let joint = (field_cos.broadcast_mul(&joint_scale)? + residual)?;
                result.push((prior + joint.broadcast_mul(&gate)?)?.to_vec1::<f32>()?);
            }
            Ok(result)
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn joint_head_scores_a_field_on_cpu() {
            let config = Config {
                hidden_size: 8,
                width: 8,
                routing_layers: 1,
                layers: 1,
                heads: 2,
                feedforward: 16,
            };
            let device = candle_core::Device::Cpu;
            let vb = VarBuilder::zeros(DType::F32, &device);
            let head = JointHead::load(&config, vb, AttentionImplementation::Eager).unwrap();
            let hidden = Tensor::zeros((1, 6, 8), DType::F32, &device).unwrap();
            let ids = Tensor::from_vec(vec![0u32, 1, 2, 3, 4, 5], 6, &device).unwrap();
            let embedding = Tensor::zeros((6, 8), DType::F32, &device).unwrap();
            let questions = vec![Question {
                id: "test".into(),
                kind: 1,
                question_span: (1, 2),
                option_spans: vec![(2, 3), (4, 5)],
                option_ids: vec!["first".into(), "second".into()],
            }];
            let scores = head.forward(&hidden, &ids, &questions, &embedding).unwrap();
            assert_eq!(scores.len(), 1);
            assert_eq!(scores[0].len(), 2);
            assert!(scores[0].iter().all(|value| value.is_finite()));
        }
    }
}

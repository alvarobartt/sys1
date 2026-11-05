#[cfg(all(test, feature = "cuda"))]
static COMPOSITE_HEAD_STRIDED_CALLS: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

#[cfg(all(test, feature = "cuda"))]
static COMPOSITE_HEAD_PACKED_STRIDED_CALLS: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);
#[cfg(all(test, feature = "cuda"))]
static COMPOSITE_HEAD_MULTI_SEGMENT_STRIDED_CALLS: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

#[cfg(feature = "cuda")]
fn grouped_final_scores(
    raw_priors: &[Tensor],
    field_cos: &Tensor,
    residual_output: &Tensor,
    prior_scale: &Tensor,
    joint_scale: &Tensor,
    gate: &Tensor,
) -> candle_core::Result<(Tensor, Tensor)> {
    if raw_priors.is_empty()
        || raw_priors
            .iter()
            .any(|p| p.rank() != 1 || p.elem_count() == 0 || p.dtype() != DType::F32)
        || field_cos.dtype() != DType::F32
        || [prior_scale, joint_scale, gate]
            .iter()
            .any(|s| s.dtype() != DType::F32 || s.elem_count() != 1)
    {
        candle_core::bail!("nonempty FP32 prior vectors, field cosines and scalar scales required")
    }
    let rows = raw_priors.iter().try_fold(0usize, |sum, p| {
        sum.checked_add(p.elem_count())
            .ok_or_else(|| candle_core::Error::Msg("prior length overflow".into()))
    })?;
    if field_cos.dims() != [rows] || residual_output.dims() != [rows, 1] {
        candle_core::bail!("final-head field/residual population mismatch")
    }
    let raw_prior = if raw_priors.len() == 1 {
        raw_priors[0].clone()
    } else {
        Tensor::cat(&raw_priors.iter().collect::<Vec<_>>(), 0)?
    };
    let prior = raw_prior.broadcast_mul(prior_scale)?;
    let residual = residual_output.squeeze(1)?.to_dtype(DType::F32)?;
    let joint = (field_cos.broadcast_mul(joint_scale)? + residual)?;
    let scores = (&prior + joint.broadcast_mul(gate)?)?;

    Ok((scores, prior))
}

use crate::models::AttentionImplementation;
use candle_core::{D, DType, Result, Tensor};
use candle_nn::{
    Embedding, LayerNorm, VarBuilder, embedding, layer_norm,
    ops::{sigmoid, softmax_last_dim},
};
use serde::Deserialize;
use std::{fs, path::Path};

struct Linear(candle_nn::Linear);

impl Linear {
    fn new(weight: Tensor, bias: Option<Tensor>) -> Self {
        Self(candle_nn::Linear::new(weight, bias))
    }

    fn weight(&self) -> &Tensor {
        self.0.weight()
    }

    fn bias(&self) -> Option<&Tensor> {
        self.0.bias()
    }
}

impl candle_nn::Module for Linear {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        #[cfg(feature = "metal")]
        if input.device().is_metal() {
            return input.apply(&self.0);
        }

        #[cfg(feature = "cuda")]
        if input.device().is_cuda() && input.dtype() == DType::BF16 {
            if crate::kernels::cuda::head_blaslt::eligible(input, &self.0) {
                return crate::kernels::cuda::head_blaslt::project(input, &self.0);
            }
            return crate::architectures::qwen35_text::compensated_bf16_projection(input, &self.0);
        }
        crate::kernels::linear::forward(input, &self.0)
    }
}

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

#[cfg(feature = "cuda")]
use crate::protocols::Encoded;
use crate::protocols::Question;

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
    query_proj: Linear,
    kv_proj: Linear,
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
        Self::from_packed(
            Linear::new(
                vb.get((width * 3, width), "in_proj_weight")?,
                Some(vb.get(width * 3, "in_proj_bias")?),
            ),
            dense(width, width, vb.pp("out_proj"))?,
            heads,
            width,
            implementation,
        )
    }

    fn from_packed(
        in_proj: Linear,
        out_proj: Linear,
        heads: usize,
        width: usize,
        implementation: AttentionImplementation,
    ) -> Result<Self> {
        let projection = |start, count| -> Result<Linear> {
            Ok(Linear::new(
                in_proj.weight().narrow(0, start, count)?,
                in_proj
                    .bias()
                    .map(|bias| bias.narrow(0, start, count))
                    .transpose()?,
            ))
        };
        let query_proj = projection(0, width)?;
        let kv_proj = projection(width, width * 2)?;
        Ok(Self {
            in_proj,
            query_proj,
            kv_proj,
            out_proj,
            heads,
            width,
            implementation,
        })
    }

    fn project(&self, query: &Tensor, memory: &Tensor) -> Result<(Tensor, Tensor, Tensor)> {
        if query.id() == memory.id() {
            let packed = query.apply(&self.in_proj)?;
            Ok((
                packed.narrow(D::Minus1, 0, self.width)?,
                packed.narrow(D::Minus1, self.width, self.width)?,
                packed.narrow(D::Minus1, self.width * 2, self.width)?,
            ))
        } else {
            let q = query.apply(&self.query_proj)?;
            let kv = memory.apply(&self.kv_proj)?;
            Ok((
                q,
                kv.narrow(D::Minus1, 0, self.width)?,
                kv.narrow(D::Minus1, self.width, self.width)?,
            ))
        }
    }

    fn project_heads(
        &self,
        query: &Tensor,
        memory: &Tensor,
    ) -> Result<(Tensor, Tensor, Tensor, bool)> {
        let (batch, q_len, _) = query.dims3()?;
        let kv_len = memory.dim(1)?;
        let head_dim = self.width / self.heads;
        #[cfg(feature = "cuda")]
        if option_env!("SYS1_CUDA_COMPUTE_CAP") == Some("90")
            && query.device().is_cuda()
            && query.dtype() == DType::BF16
            && memory.dtype() == DType::BF16
            && query.device().same_device(memory.device())
            && self.heads == 16
            && head_dim == 64
        {
            let implementation = if self.implementation == AttentionImplementation::Auto {
                crate::models::attention_for_device(query.dtype(), query.device())
                    .map_err(|e| candle_core::Error::Msg(e.to_string()))?
            } else {
                self.implementation
            };
            if matches!(
                implementation,
                AttentionImplementation::FlashAttention2
                    | AttentionImplementation::FlashAttentionBlackwell
            ) {
                if query.id() == memory.id() {
                    let packed = query.apply(&self.in_proj)?;
                    if packed.is_contiguous() {
                        let packed = packed.reshape((batch, q_len, 3, self.heads, head_dim))?;
                        return Ok((
                            packed.narrow(2, 0, 1)?.squeeze(2)?,
                            packed.narrow(2, 1, 1)?.squeeze(2)?,
                            packed.narrow(2, 2, 1)?.squeeze(2)?,
                            true,
                        ));
                    }
                    return Ok((
                        packed
                            .narrow(D::Minus1, 0, self.width)?
                            .reshape((batch, q_len, self.heads, head_dim))?,
                        packed
                            .narrow(D::Minus1, self.width, self.width)?
                            .reshape((batch, q_len, self.heads, head_dim))?,
                        packed
                            .narrow(D::Minus1, 2 * self.width, self.width)?
                            .reshape((batch, q_len, self.heads, head_dim))?,
                        false,
                    ));
                }
                let q = query.apply(&self.query_proj)?;
                let packed = memory.apply(&self.kv_proj)?;
                if q.is_contiguous() && packed.is_contiguous() {
                    let q = q.reshape((batch, q_len, self.heads, head_dim))?;
                    let packed = packed.reshape((batch, kv_len, 2, self.heads, head_dim))?;
                    return Ok((
                        q,
                        packed.narrow(2, 0, 1)?.squeeze(2)?,
                        packed.narrow(2, 1, 1)?.squeeze(2)?,
                        true,
                    ));
                }
                return Ok((
                    q.reshape((batch, q_len, self.heads, head_dim))?,
                    packed
                        .narrow(D::Minus1, 0, self.width)?
                        .reshape((batch, kv_len, self.heads, head_dim))?,
                    packed
                        .narrow(D::Minus1, self.width, self.width)?
                        .reshape((batch, kv_len, self.heads, head_dim))?,
                    false,
                ));
            }
        }

        let (q, k, v) = self.project(query, memory)?;
        Ok((
            q.reshape((batch, q_len, self.heads, head_dim))?,
            k.reshape((batch, kv_len, self.heads, head_dim))?,
            v.reshape((batch, kv_len, self.heads, head_dim))?,
            false,
        ))
    }

    fn forward(&self, query: &Tensor, memory: &Tensor) -> Result<Tensor> {
        self.forward_with_capture(query, memory, None)
    }

    fn routing_capture(index: Option<usize>, stage: &str, tensor: &Tensor) -> Result<()> {
        if index == Some(0) && crate::numerics::enabled() {
            let captured = if tensor.rank() == 4
                && matches!(stage, "q-projected" | "k-projected" | "v-projected")
            {
                tensor.flatten_from(2)?
            } else {
                tensor.clone()
            };
            crate::numerics::capture(&format!("head-evidence-0-{stage}"), &captured)?;
        }
        Ok(())
    }

    #[cfg(feature = "metal")]
    fn forward_metal(&self, query: &Tensor, memory: &Tensor) -> Result<Tensor> {
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
        // BF16 Metal SDPA produced non-finite values for Clef's 256-wide heads.
        if query.device().is_metal()
            && self.implementation == AttentionImplementation::Auto
            && query.dtype() != DType::BF16
        {
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
            AttentionImplementation::Auto | AttentionImplementation::Eager => {
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
            #[cfg(feature = "cuda")]
            AttentionImplementation::FlashAttention2 => candle_flash_attn::flash_attn(
                &q.contiguous()?,
                &k.contiguous()?,
                &v.contiguous()?,
                scale,
                false,
            )?,
            #[allow(unreachable_patterns)]
            other => candle_core::bail!(
                "Support for attention {} is not compiled in",
                other.cli_name()
            ),
        };
        attended
            .reshape((batch, q_len, self.width))?
            .apply(&self.out_proj)
    }

    fn forward_with_capture(
        &self,
        query: &Tensor,
        memory: &Tensor,
        capture_index: Option<usize>,
    ) -> Result<Tensor> {
        #[cfg(feature = "metal")]
        if query.device().is_metal() {
            return self.forward_metal(query, memory);
        }

        let (batch, q_len, _) = query.dims3()?;

        let head_dim = self.width / self.heads;
        Self::routing_capture(capture_index, "query-input", query)?;
        Self::routing_capture(capture_index, "memory-input", memory)?;
        let (q, k, v, direct) = self.project_heads(query, memory)?;
        #[cfg(not(feature = "cuda"))]
        let _ = direct;
        Self::routing_capture(capture_index, "q-projected", &q)?;
        Self::routing_capture(capture_index, "k-projected", &k)?;
        Self::routing_capture(capture_index, "v-projected", &v)?;
        let scale = (head_dim as f32).powf(-0.5);
        #[cfg(feature = "metal")]
        // BF16 Metal SDPA produced non-finite values for Clef's 256-wide heads.
        if query.device().is_metal() && query.dtype() != DType::BF16 {
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
        #[cfg(feature = "cuda")]
        let attended = if direct {
            #[cfg(test)]
            COMPOSITE_HEAD_STRIDED_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            crate::kernels::cuda::attention::flash_attn(&q, &k, &v, scale, false)?
        } else {
            crate::architectures::qwen35_text::attention_forward(
                &q,
                &k,
                &v,
                scale,
                false,
                self.implementation,
            )?
        };
        #[cfg(not(feature = "cuda"))]
        let attended = crate::architectures::qwen35_text::attention_forward(
            &q,
            &k,
            &v,
            scale,
            false,
            self.implementation,
        )?;
        if capture_index == Some(0) && crate::numerics::enabled() {
            Self::routing_capture(capture_index, "attended", &attended.flatten_from(2)?)?;
        }
        let output = attended
            .reshape((batch, q_len, self.width))?
            .apply(&self.out_proj)?;
        Self::routing_capture(capture_index, "output-projected", &output)?;
        Ok(output)
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

    fn forward(&self, queries: &Tensor, memory: &Tensor, index: usize) -> Result<Tensor> {
        let routed = self.attention.forward_with_capture(
            &queries.apply(&self.query_norm)?,
            &memory.apply(&self.memory_norm)?,
            Some(index),
        )?;
        let queries = (queries + routed)?;
        let normalized = queries.apply(&self.feedforward_norm)?;
        Attention::routing_capture(Some(index), "feedforward-input", &normalized)?;
        let projected = normalized.apply(&self.fc1)?;
        Attention::routing_capture(Some(index), "feedforward-projected", &projected)?;
        let activated = projected.gelu_erf()?;
        Attention::routing_capture(Some(index), "feedforward-activated", &activated)?;
        let output = activated.apply(&self.fc2)?;
        Attention::routing_capture(Some(index), "feedforward-output", &output)?;
        &queries + output
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

#[cfg(feature = "cuda")]
struct PackedHeadSegments {
    cumulative: Tensor,
    maximum: usize,
    total: usize,
}

#[cfg(feature = "cuda")]
impl PackedHeadSegments {
    fn new(lengths: &[usize], device: &candle_core::Device) -> Result<Self> {
        if lengths.is_empty() || lengths.contains(&0) {
            candle_core::bail!("head packing requires nonempty request segments");
        }
        let mut cumulative = vec![0u32];
        for &length in lengths {
            let length = u32::try_from(length)
                .map_err(|_| candle_core::Error::Msg("head segment overflow".into()))?;
            cumulative.push(
                cumulative
                    .last()
                    .unwrap()
                    .checked_add(length)
                    .ok_or_else(|| candle_core::Error::Msg("head cumulative overflow".into()))?,
            );
        }
        let total = *cumulative.last().unwrap() as usize;
        let count = cumulative.len();
        Ok(Self {
            cumulative: Tensor::from_vec(cumulative, count, device)?,
            maximum: *lengths.iter().max().unwrap(),
            total,
        })
    }
}

#[cfg(feature = "cuda")]
impl Attention {
    fn forward_head_packed(
        &self,
        query: &Tensor,
        memory: &Tensor,
        queries: &PackedHeadSegments,
        keys: &PackedHeadSegments,
    ) -> Result<Tensor> {
        if crate::numerics::enabled() {
            candle_core::bail!("packed head is not a singleton tensor capture path");
        }
        if query.dims() != [1, queries.total, self.width]
            || memory.dims() != [1, keys.total, self.width]
        {
            candle_core::bail!("packed head attention segment shape mismatch");
        }

        let (q, k, v, direct) = self.project_heads(query, memory)?;
        let d = self.width / self.heads;
        let (q, k, v) = (q.squeeze(0)?, k.squeeze(0)?, v.squeeze(0)?);
        let (q, k, v) = if direct {
            #[cfg(test)]
            {
                COMPOSITE_HEAD_STRIDED_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                COMPOSITE_HEAD_PACKED_STRIDED_CALLS
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                if queries.cumulative.dim(0)? > 2 {
                    COMPOSITE_HEAD_MULTI_SEGMENT_STRIDED_CALLS
                        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                }
            }
            (q, k, v)
        } else {
            (q.contiguous()?, k.contiguous()?, v.contiguous()?)
        };
        let output = crate::kernels::cuda::attention::flash_attn_varlen(
            &q,
            &k,
            &v,
            &queries.cumulative,
            &keys.cumulative,
            queries.maximum,
            keys.maximum,
            (d as f32).powf(-0.5),
            false,
        )?;
        output
            .reshape((1, queries.total, self.width))?
            .apply(&self.out_proj)
    }
}

#[cfg(feature = "cuda")]
impl EvidenceLayer {
    fn forward_head_packed(
        &self,
        queries: &Tensor,
        memory: &Tensor,
        options: &PackedHeadSegments,
        tokens: &PackedHeadSegments,
    ) -> Result<Tensor> {
        let routed = self.attention.forward_head_packed(
            &queries.apply(&self.query_norm)?,
            &memory.apply(&self.memory_norm)?,
            options,
            tokens,
        )?;
        let queries = (queries + routed)?;
        let output = queries
            .apply(&self.feedforward_norm)?
            .apply(&self.fc1)?
            .gelu_erf()?
            .apply(&self.fc2)?;
        &queries + output
    }
}

#[cfg(feature = "cuda")]
impl DecoderLayer {
    fn forward_head_packed(
        &self,
        fields: &Tensor,
        memory: &Tensor,
        questions: &PackedHeadSegments,
        tokens: &PackedHeadSegments,
    ) -> Result<Tensor> {
        let normalized = fields.apply(&self.norm1)?;
        let fields = (fields
            + self.self_attention.forward_head_packed(
                &normalized,
                &normalized,
                questions,
                questions,
            )?)?;
        let fields = (&fields
            + self.cross_attention.forward_head_packed(
                &fields.apply(&self.norm2)?,
                memory,
                questions,
                tokens,
            )?)?;
        let output = fields
            .apply(&self.norm3)?
            .apply(&self.fc1)?
            .gelu_erf()?
            .apply(&self.fc2)?;
        &fields + output
    }
}

#[cfg(feature = "cuda")]
struct HeadFinalBatch {
    normalized_lexical: Tensor,
    normalized_anchor: Tensor,
    field_cos: Tensor,
    features: Tensor,
    projected: Tensor,
    activated: Tensor,
    output: Tensor,
}

#[cfg(feature = "cuda")]
impl JointHead {
    fn final_option_batch(
        &self,
        options: &Tensor,
        fields: &Tensor,
        questions: &Tensor,
        global: &Tensor,
        lexical: &[Tensor],
        counts: &[usize],
    ) -> Result<HeadFinalBatch> {
        if counts.len() != fields.dim(0)?
            || counts.len() != questions.dim(0)?
            || counts.len() != global.dim(0)?
            || counts.len() != lexical.len()
            || counts.contains(&0)
            || counts.iter().sum::<usize>() != options.dim(0)?
        {
            candle_core::bail!("invalid final option batch metadata");
        }
        let owner = counts
            .iter()
            .enumerate()
            .flat_map(|(i, &n)| std::iter::repeat_n(i as u32, n))
            .collect::<Vec<_>>();
        let owner = Tensor::from_vec(owner, options.dim(0)?, options.device())?;
        let repeated = fields.index_select(&owner, 0)?;
        let values = options.apply(&self.option_norm)?;

        let field_cos = Self::normalize_with_epsilon(&repeated, 1e-8)?
            .broadcast_mul(&Self::normalize_with_epsilon(&values, 1e-8)?)?
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
        let projected = features.apply(&self.scorer1)?;
        let activated = projected.gelu_erf()?;
        let output = activated.apply(&self.scorer2)?;
        let normalized_anchor = Self::normalize(&(questions + global)?)?;
        let lexical = Tensor::cat(&lexical.iter().collect::<Vec<_>>(), 0)?;
        let normalized_lexical = Self::normalize(&lexical)?;
        Ok(HeadFinalBatch {
            normalized_lexical,
            normalized_anchor,
            field_cos,
            features,
            projected,
            activated,
            output,
        })
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
                .map(|i| DecoderLayer::load(config, vb.pp(format!("layers.{i}")), implementation))
                .collect::<Result<Vec<_>>>()?,
            field_norm: layer_norm(w, 1e-5, vb.pp("field_norm"))?,
            option_norm: layer_norm(w, 1e-5, vb.pp("option_norm"))?,
            scorer1: dense(w * 4, w, vb.pp("residual_scorer.0"))?,
            scorer2: dense(w, 1, vb.pp("residual_scorer.3"))?,
            prior_scale: vb
                .get((), "prior_logit_scale")?
                .to_dtype(DType::F32)?
                .clamp(f64::NEG_INFINITY, 100f64.ln())?
                .exp()?,
            joint_scale: vb
                .get((), "joint_logit_scale")?
                .to_dtype(DType::F32)?
                .clamp(f64::NEG_INFINITY, 100f64.ln())?
                .exp()?,
            residual_gate: sigmoid(&vb.get((), "residual_gate")?.to_dtype(DType::F32)?)?,
            width: w,
        })
    }

    #[cfg(feature = "metal")]
    fn mean_rows_metal(values: &Tensor) -> Result<Tensor> {
        let rows = values.dim(0)?;
        let weights =
            (Tensor::ones((1, rows), values.dtype(), values.device())? * (1. / rows as f64))?;
        weights.matmul(&values.contiguous()?)?.squeeze(0)
    }

    #[cfg(feature = "metal")]
    fn normalize_metal(values: &Tensor) -> Result<Tensor> {
        let values = values.to_dtype(DType::F32)?;
        let norm = (values.sqr()?.sum_keepdim(D::Minus1)? + 1e-12)?.sqrt()?;
        values.broadcast_div(&norm)
    }

    fn mean_span(values: &Tensor, (start, end): (usize, usize)) -> Result<Tensor> {
        Self::mean_rows(&values.narrow(0, start, end - start)?)
    }

    fn mean_rows(values: &Tensor) -> Result<Tensor> {
        let rows = values.dim(0)?;
        if rows == 0 {
            candle_core::bail!("cannot average an empty schema span")
        }
        #[cfg(feature = "metal")]
        if values.device().is_metal() {
            return Self::mean_rows_metal(values);
        }

        (values.to_dtype(DType::F32)?.sum(0)? / rows as f64)?.to_dtype(values.dtype())
    }

    fn normalize(values: &Tensor) -> Result<Tensor> {
        Self::normalize_with_epsilon(values, 1e-12)
    }

    fn normalize_with_epsilon(values: &Tensor, epsilon: f64) -> Result<Tensor> {
        #[cfg(feature = "metal")]
        if values.device().is_metal() {
            return Self::normalize_metal(values);
        }

        let values = values.to_dtype(DType::F32)?;
        let norm = values
            .sqr()?
            .sum_keepdim(D::Minus1)?
            .sqrt()?
            .clamp(epsilon, f64::INFINITY)?;
        values.broadcast_div(&norm)
    }

    fn option_summary(values: &Tensor, weights: &Tensor) -> Result<Tensor> {
        #[cfg(feature = "metal")]
        if values.device().is_metal() {
            return weights
                .contiguous()?
                .matmul(&values.contiguous()?)?
                .squeeze(0);
        }
        #[cfg(feature = "cuda")]
        if values.device().is_cuda()
            && values.dtype() == DType::BF16
            && (1..=32).contains(&values.dim(0)?)
        {
            return crate::kernels::cuda::reduction::weighted_summary(values, weights);
        }

        values
            .broadcast_mul(&weights.t()?)?
            .to_dtype(DType::F32)?
            .sum(0)?
            .to_dtype(values.dtype())
    }

    fn option_scores(values: &Tensor, field: &Tensor) -> Result<Tensor> {
        #[cfg(feature = "cuda")]
        if values.device().is_cuda() && values.dtype() == DType::BF16 {
            let projection = candle_nn::Linear::new(field.unsqueeze(0)?, None);
            return crate::architectures::qwen35_text::compensated_bf16_projection(
                values,
                &projection,
            );
        }
        crate::kernels::linear::matmul_compatible(values, &field.unsqueeze(1)?)
    }

    pub fn forward(
        &self,
        hidden: &Tensor,
        ids: &Tensor,
        questions: &[Question],
        output_embedding: &Tensor,
    ) -> Result<Vec<Vec<f32>>> {
        #[cfg(feature = "cuda")]
        if matches!(option_env!("SYS1_CUDA_COMPUTE_CAP"), Some("80" | "120"))
            && hidden.device().is_cuda()
            && hidden.dtype() == DType::BF16
        {
            return self.forward_scoring_single(hidden, ids, questions, output_embedding);
        }
        let hidden = hidden.apply(&self.hidden_norm)?;
        crate::numerics::capture("head-normalized", &hidden)?;
        let hidden = if hidden.rank() == 3 {
            hidden.squeeze(0)?
        } else {
            hidden
        };
        let length = hidden.dim(0)?;
        let memory = hidden.apply(&self.memory_projection)?.unsqueeze(0)?;
        let global = hidden.narrow(0, length - 1, 1)?.squeeze(0)?;
        #[cfg(feature = "cuda")]
        let fused_means = if hidden.device().is_cuda()
            && hidden.dtype() == DType::BF16
            && output_embedding.dtype() == DType::BF16
            && ids.dtype() == DType::U32
        {
            let option_spans: Vec<_> = questions
                .iter()
                .flat_map(|q| q.option_spans.iter().copied())
                .collect();
            let mut spans: Vec<_> = questions.iter().map(|q| q.question_span).collect();
            spans.extend_from_slice(&option_spans);
            let means = crate::kernels::cuda::span_mean::span_means(&hidden, &spans)?;
            let lexical = crate::kernels::cuda::span_mean::lexical_means(
                output_embedding,
                ids,
                &option_spans,
            )?;
            Some((
                means.narrow(0, 0, questions.len())?,
                means.narrow(0, questions.len(), option_spans.len())?,
                lexical,
            ))
        } else {
            None
        };
        #[cfg(not(feature = "cuda"))]
        let fused_means: Option<(Tensor, Tensor, Tensor)> = None;
        let question_vectors = if let Some((questions, _, _)) = &fused_means {
            questions.clone()
        } else {
            Tensor::stack(
                &questions
                    .iter()
                    .map(|q| Self::mean_span(&hidden, q.question_span))
                    .collect::<Result<Vec<_>>>()?
                    .iter()
                    .collect::<Vec<_>>(),
                0,
            )?
        };
        let type_ids = Tensor::from_vec(
            questions.iter().map(|q| q.kind as u32).collect::<Vec<_>>(),
            questions.len(),
            hidden.device(),
        )?;

        let batched_option_projections = fused_means
            .as_ref()
            .filter(|(_, context, _)| {
                option_env!("SYS1_CUDA_COMPUTE_CAP") == Some("120")
                    || option_env!("SYS1_CUDA_COMPUTE_CAP") == Some("80")
                        && context.dim(0).is_ok_and(|rows| rows >= 128)
            })
            .map(|(_, context, lex)| {
                Ok::<_, candle_core::Error>((
                    context.apply(&self.option_context_projection)?,
                    lex.apply(&self.option_lexical_projection)?,
                    question_vectors.apply(&self.option_question_projection)?,
                ))
            })
            .transpose()?;
        let mut lexical = Vec::with_capacity(questions.len());
        let mut routed = Vec::with_capacity(questions.len());
        let mut counts = Vec::with_capacity(questions.len());
        for (index, question) in questions.iter().enumerate() {
            let start: usize = counts.iter().sum();
            let context = if let Some((_, context, _)) = &fused_means {
                context.narrow(0, start, question.option_spans.len())?
            } else {
                Tensor::stack(
                    &question
                        .option_spans
                        .iter()
                        .map(|&span| Self::mean_span(&hidden, span))
                        .collect::<Result<Vec<_>>>()?
                        .iter()
                        .collect::<Vec<_>>(),
                    0,
                )?
            };
            let lex = if let Some((_, _, lexical)) = &fused_means {
                lexical.narrow(0, start, question.option_spans.len())?
            } else {
                Tensor::stack(
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
                )?
            };
            let question_vector = question_vectors.narrow(0, index, 1)?;
            let query = if let Some((context_proj, lexical_proj, question_proj)) =
                &batched_option_projections
            {
                (context_proj.narrow(0, start, question.option_spans.len())?
                    + lexical_proj.narrow(0, start, question.option_spans.len())?)?
                .broadcast_add(&question_proj.narrow(0, index, 1)?)?
            } else {
                (context.apply(&self.option_context_projection)?
                    + lex.apply(&self.option_lexical_projection)?)?
                .broadcast_add(&question_vector.apply(&self.option_question_projection)?)?
            };
            counts.push(question.option_spans.len());
            lexical.push(lex);
            routed.push(query);
        }
        let mut options = Tensor::cat(&routed.iter().collect::<Vec<_>>(), 0)?.unsqueeze(0)?;
        for (index, layer) in self.evidence_layers.iter().enumerate() {
            options = layer.forward(&options, &memory, index)?;
            crate::numerics::capture(&format!("head-evidence-{index}"), &options)?;
        }
        let options = options.squeeze(0)?;
        let mut split = Vec::with_capacity(questions.len());
        let mut start = 0;
        for count in counts {
            split.push(options.narrow(0, start, count)?);
            start += count;
        }
        let base_fields = question_vectors.apply(&self.question_projection)?;
        crate::numerics::capture("head-initial-base-fields", &base_fields)?;
        let mut summaries = Vec::with_capacity(questions.len());
        for (index, values) in split.iter().enumerate() {
            let field = base_fields.narrow(0, index, 1)?.squeeze(0)?;
            let scores = (Self::option_scores(values, &field)? / (self.width as f64).sqrt())?
                .transpose(0, 1)?;
            let weights =
                softmax_last_dim(&scores.to_dtype(DType::F32)?)?.to_dtype(values.dtype())?;
            let summary = Self::option_summary(values, &weights)?;
            if crate::numerics::enabled() {
                for (stage, tensor) in [
                    ("values", values),
                    ("scores", &scores),
                    ("weights", &weights),
                    ("output", &summary),
                ] {
                    crate::numerics::capture(&format!("head-summary-{index}-{stage}"), tensor)?;
                }
            }
            summaries.push(summary);
        }
        let fields = ((&base_fields
            + Tensor::stack(&summaries.iter().collect::<Vec<_>>(), 0)?
                .apply(&self.option_summary_norm)?)?
        .broadcast_add(&global.unsqueeze(0)?.apply(&self.global_projection)?)?
            + type_ids.apply(&self.type_embedding)?)?;
        let mut fields = fields.unsqueeze(0)?;
        crate::numerics::capture("head-initial-fields", &fields)?;
        for (index, layer) in self.layers.iter().enumerate() {
            fields = layer.forward(&fields, &memory)?;
            crate::numerics::capture(&format!("head-fields-{index}"), &fields)?;
        }
        let fields = fields.squeeze(0)?.apply(&self.field_norm)?;
        let prior_scale = self.prior_scale.clone();
        let joint_scale = self.joint_scale.clone();
        let gate = self.residual_gate.clone();
        let mut pending_scores = Vec::with_capacity(questions.len());
        for (index, values) in split.iter().enumerate() {
            let field = fields.narrow(0, index, 1)?;
            let normalized_anchor =
                Self::normalize(&(question_vectors.narrow(0, index, 1)? + global.unsqueeze(0)?)?)?;
            let prior = Self::normalize(&lexical[index])?
                .matmul(&normalized_anchor.t()?)?
                .squeeze(1)?
                .broadcast_mul(&prior_scale)?;
            let values = values.apply(&self.option_norm)?;
            let repeated = field.broadcast_as(values.shape())?;

            let field_cos = Self::normalize_with_epsilon(&repeated, 1e-8)?
                .broadcast_mul(&Self::normalize_with_epsilon(&values, 1e-8)?)?
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
            let residual_projected = features.apply(&self.scorer1)?;
            let residual_activated = residual_projected.gelu_erf()?;
            let residual_output = residual_activated.apply(&self.scorer2)?;
            if crate::numerics::enabled() {
                for (stage, value) in [
                    ("features", &features),
                    ("projected", &residual_projected),
                    ("activated", &residual_activated),
                    ("output", &residual_output),
                    ("prior", &prior),
                    ("field-cos", &field_cos),
                ] {
                    crate::numerics::capture(&format!("head-residual-{index}-{stage}"), value)?;
                }
            }
            let residual = residual_output.squeeze(1)?.to_dtype(DType::F32)?;
            let joint = (field_cos.broadcast_mul(&joint_scale)? + residual)?;
            pending_scores.push((prior + joint.broadcast_mul(&gate)?)?);
        }
        if pending_scores.len() <= 1 || !hidden.device().is_cuda() {
            return pending_scores
                .into_iter()
                .map(|scores| scores.to_vec1::<f32>())
                .collect();
        }
        let lengths = pending_scores
            .iter()
            .map(Tensor::elem_count)
            .collect::<Vec<_>>();
        let scores =
            Tensor::cat(&pending_scores.iter().collect::<Vec<_>>(), 0)?.to_vec1::<f32>()?;
        let mut offset = 0;
        Ok(lengths
            .into_iter()
            .map(|length| {
                let slice = scores[offset..offset + length].to_vec();
                offset += length;
                slice
            })
            .collect())
    }

    #[cfg(feature = "cuda")]
    fn forward_scoring_single(
        &self,
        hidden: &Tensor,
        ids: &Tensor,
        questions: &[Question],
        output_embedding: &Tensor,
    ) -> Result<Vec<Vec<f32>>> {
        let hidden = hidden.apply(&self.hidden_norm)?;
        crate::numerics::capture("head-normalized", &hidden)?;
        let hidden = if hidden.rank() == 3 {
            hidden.squeeze(0)?
        } else {
            hidden
        };
        let length = hidden.dim(0)?;
        let memory = hidden.apply(&self.memory_projection)?.unsqueeze(0)?;
        let global = hidden.narrow(0, length - 1, 1)?.squeeze(0)?;
        #[cfg(feature = "cuda")]
        let fused_means = if hidden.device().is_cuda()
            && hidden.dtype() == DType::BF16
            && output_embedding.dtype() == DType::BF16
            && ids.dtype() == DType::U32
        {
            let option_spans: Vec<_> = questions
                .iter()
                .flat_map(|q| q.option_spans.iter().copied())
                .collect();
            let mut spans: Vec<_> = questions.iter().map(|q| q.question_span).collect();
            spans.extend_from_slice(&option_spans);
            let means = crate::kernels::cuda::span_mean::span_means(&hidden, &spans)?;
            let lexical = crate::kernels::cuda::span_mean::lexical_means(
                output_embedding,
                ids,
                &option_spans,
            )?;
            Some((
                means.narrow(0, 0, questions.len())?,
                means.narrow(0, questions.len(), option_spans.len())?,
                lexical,
            ))
        } else {
            None
        };
        #[cfg(not(feature = "cuda"))]
        let fused_means: Option<(Tensor, Tensor, Tensor)> = None;
        let question_vectors = if let Some((questions, _, _)) = &fused_means {
            questions.clone()
        } else {
            Tensor::stack(
                &questions
                    .iter()
                    .map(|q| Self::mean_span(&hidden, q.question_span))
                    .collect::<Result<Vec<_>>>()?
                    .iter()
                    .collect::<Vec<_>>(),
                0,
            )?
        };
        let type_ids = Tensor::from_vec(
            questions.iter().map(|q| q.kind as u32).collect::<Vec<_>>(),
            questions.len(),
            hidden.device(),
        )?;

        let batched_option_projections = fused_means
            .as_ref()
            .filter(|(_, context, _)| {
                option_env!("SYS1_CUDA_COMPUTE_CAP") == Some("120")
                    || option_env!("SYS1_CUDA_COMPUTE_CAP") == Some("80")
                        && context.dim(0).is_ok_and(|rows| rows >= 128)
            })
            .map(|(_, context, lex)| {
                Ok::<_, candle_core::Error>((
                    context.apply(&self.option_context_projection)?,
                    lex.apply(&self.option_lexical_projection)?,
                    question_vectors.apply(&self.option_question_projection)?,
                ))
            })
            .transpose()?;
        let mut lexical = Vec::with_capacity(questions.len());
        let mut routed = Vec::with_capacity(questions.len());
        let mut counts = Vec::with_capacity(questions.len());
        for (index, question) in questions.iter().enumerate() {
            let start: usize = counts.iter().sum();
            let context = if let Some((_, context, _)) = &fused_means {
                context.narrow(0, start, question.option_spans.len())?
            } else {
                Tensor::stack(
                    &question
                        .option_spans
                        .iter()
                        .map(|&span| Self::mean_span(&hidden, span))
                        .collect::<Result<Vec<_>>>()?
                        .iter()
                        .collect::<Vec<_>>(),
                    0,
                )?
            };
            let lex = if let Some((_, _, lexical)) = &fused_means {
                lexical.narrow(0, start, question.option_spans.len())?
            } else {
                Tensor::stack(
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
                )?
            };
            let question_vector = question_vectors.narrow(0, index, 1)?;
            let query = if let Some((context_proj, lexical_proj, question_proj)) =
                &batched_option_projections
            {
                (context_proj.narrow(0, start, question.option_spans.len())?
                    + lexical_proj.narrow(0, start, question.option_spans.len())?)?
                .broadcast_add(&question_proj.narrow(0, index, 1)?)?
            } else {
                (context.apply(&self.option_context_projection)?
                    + lex.apply(&self.option_lexical_projection)?)?
                .broadcast_add(&question_vector.apply(&self.option_question_projection)?)?
            };
            counts.push(question.option_spans.len());
            lexical.push(lex);
            routed.push(query);
        }
        let mut options = Tensor::cat(&routed.iter().collect::<Vec<_>>(), 0)?.unsqueeze(0)?;
        for (index, layer) in self.evidence_layers.iter().enumerate() {
            options = layer.forward(&options, &memory, index)?;
            crate::numerics::capture(&format!("head-evidence-{index}"), &options)?;
        }
        let options = options.squeeze(0)?;
        let mut split = Vec::with_capacity(questions.len());
        let mut start = 0;
        for count in counts {
            split.push(options.narrow(0, start, count)?);
            start += count;
        }
        let base_fields = question_vectors.apply(&self.question_projection)?;
        crate::numerics::capture("head-initial-base-fields", &base_fields)?;
        let mut summaries = Vec::with_capacity(questions.len());
        for (index, values) in split.iter().enumerate() {
            let field = base_fields.narrow(0, index, 1)?.squeeze(0)?;
            let scores = (Self::option_scores(values, &field)? / (self.width as f64).sqrt())?
                .transpose(0, 1)?;
            let weights =
                softmax_last_dim(&scores.to_dtype(DType::F32)?)?.to_dtype(values.dtype())?;
            let summary = Self::option_summary(values, &weights)?;
            if crate::numerics::enabled() {
                for (stage, tensor) in [
                    ("values", values),
                    ("scores", &scores),
                    ("weights", &weights),
                    ("output", &summary),
                ] {
                    crate::numerics::capture(&format!("head-summary-{index}-{stage}"), tensor)?;
                }
            }
            summaries.push(summary);
        }
        let fields = ((&base_fields
            + Tensor::stack(&summaries.iter().collect::<Vec<_>>(), 0)?
                .apply(&self.option_summary_norm)?)?
        .broadcast_add(&global.unsqueeze(0)?.apply(&self.global_projection)?)?
            + type_ids.apply(&self.type_embedding)?)?;
        let mut fields = fields.unsqueeze(0)?;
        crate::numerics::capture("head-initial-fields", &fields)?;
        for (index, layer) in self.layers.iter().enumerate() {
            fields = layer.forward(&fields, &memory)?;
            crate::numerics::capture(&format!("head-fields-{index}"), &fields)?;
        }
        let fields = fields.squeeze(0)?.apply(&self.field_norm)?;
        let prior_scale = self.prior_scale.clone();
        let joint_scale = self.joint_scale.clone();
        let gate = self.residual_gate.clone();

        let global = global
            .unsqueeze(0)?
            .broadcast_as(question_vectors.shape())?;
        let counts = split.iter().map(|v| v.dim(0)).collect::<Result<Vec<_>>>()?;
        let final_batch = self.final_option_batch(
            &options,
            &fields,
            &question_vectors,
            &global,
            &lexical,
            &counts,
        )?;
        let mut pending_scores = Vec::with_capacity(questions.len());
        let mut start = 0;
        for (index, &count) in counts.iter().enumerate() {
            let prior = final_batch
                .normalized_lexical
                .narrow(0, start, count)?
                .matmul(&final_batch.normalized_anchor.narrow(0, index, 1)?.t()?)?
                .squeeze(1)?
                .broadcast_mul(&prior_scale)?;
            let field_cos = final_batch.field_cos.narrow(0, start, count)?;
            let features = final_batch.features.narrow(0, start, count)?;
            let residual_projected = final_batch.projected.narrow(0, start, count)?;
            let residual_activated = final_batch.activated.narrow(0, start, count)?;
            let residual_output = final_batch.output.narrow(0, start, count)?;
            if crate::numerics::enabled() {
                for (stage, value) in [
                    ("features", &features),
                    ("projected", &residual_projected),
                    ("activated", &residual_activated),
                    ("output", &residual_output),
                    ("prior", &prior),
                    ("field-cos", &field_cos),
                ] {
                    crate::numerics::capture(&format!("head-residual-{index}-{stage}"), value)?;
                }
            }
            let residual = residual_output.squeeze(1)?.to_dtype(DType::F32)?;
            let joint = (field_cos.broadcast_mul(&joint_scale)? + residual)?;
            pending_scores.push((prior + joint.broadcast_mul(&gate)?)?);
            start += count;
        }
        if pending_scores.len() <= 1 || !hidden.device().is_cuda() {
            return pending_scores
                .into_iter()
                .map(|scores| scores.to_vec1::<f32>())
                .collect();
        }
        let lengths = pending_scores
            .iter()
            .map(Tensor::elem_count)
            .collect::<Vec<_>>();
        let scores =
            Tensor::cat(&pending_scores.iter().collect::<Vec<_>>(), 0)?.to_vec1::<f32>()?;
        let mut offset = 0;
        Ok(lengths
            .into_iter()
            .map(|length| {
                let slice = scores[offset..offset + length].to_vec();
                offset += length;
                slice
            })
            .collect())
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn forward_request_pack_scoring(
        &self,
        hidden: &Tensor,
        ids: &Tensor,
        questions: &[Question],
        output_embedding: &Tensor,
        request_lengths: &[usize],
        request_question_counts: &[usize],
    ) -> Result<Vec<Vec<f32>>> {
        let hidden = hidden.apply(&self.hidden_norm)?;
        crate::numerics::capture("head-normalized", &hidden)?;
        let hidden = if hidden.rank() == 3 {
            hidden.squeeze(0)?
        } else {
            hidden
        };
        let length = hidden.dim(0)?;
        if request_lengths.len() != request_question_counts.len()
            || request_lengths.iter().sum::<usize>() != length
            || request_question_counts.iter().sum::<usize>() != questions.len()
        {
            candle_core::bail!("packed head request metadata mismatch");
        }
        let tokens = PackedHeadSegments::new(request_lengths, hidden.device())?;
        let fields_plan = PackedHeadSegments::new(request_question_counts, hidden.device())?;
        let mut cursor = 0;
        let mut option_lengths = Vec::new();
        for &count in request_question_counts {
            option_lengths.push(
                questions[cursor..cursor + count]
                    .iter()
                    .map(|q| q.option_spans.len())
                    .sum(),
            );
            cursor += count;
        }
        let options_plan = PackedHeadSegments::new(&option_lengths, hidden.device())?;
        let memory = hidden.apply(&self.memory_projection)?.unsqueeze(0)?;
        let mut ends = Vec::new();
        let mut cursor = 0usize;
        let mut question_owners = Vec::new();
        for (request, (&length, &count)) in request_lengths
            .iter()
            .zip(request_question_counts)
            .enumerate()
        {
            cursor += length;
            ends.push((cursor - 1) as u32);
            question_owners.extend(std::iter::repeat_n(request as u32, count));
        }
        let ends = Tensor::from_vec(ends, request_lengths.len(), hidden.device())?;
        let owners = Tensor::from_vec(question_owners, questions.len(), hidden.device())?;
        let global = hidden.index_select(&ends, 0)?.index_select(&owners, 0)?;
        #[cfg(feature = "cuda")]
        let fused_means = if hidden.device().is_cuda()
            && hidden.dtype() == DType::BF16
            && output_embedding.dtype() == DType::BF16
            && ids.dtype() == DType::U32
        {
            let option_spans: Vec<_> = questions
                .iter()
                .flat_map(|q| q.option_spans.iter().copied())
                .collect();
            let mut spans: Vec<_> = questions.iter().map(|q| q.question_span).collect();
            spans.extend_from_slice(&option_spans);
            let means = crate::kernels::cuda::span_mean::span_means(&hidden, &spans)?;
            let lexical = crate::kernels::cuda::span_mean::lexical_means(
                output_embedding,
                ids,
                &option_spans,
            )?;
            Some((
                means.narrow(0, 0, questions.len())?,
                means.narrow(0, questions.len(), option_spans.len())?,
                lexical,
            ))
        } else {
            None
        };
        #[cfg(not(feature = "cuda"))]
        let fused_means: Option<(Tensor, Tensor, Tensor)> = None;
        let question_vectors = if let Some((questions, _, _)) = &fused_means {
            questions.clone()
        } else {
            Tensor::stack(
                &questions
                    .iter()
                    .map(|q| Self::mean_span(&hidden, q.question_span))
                    .collect::<Result<Vec<_>>>()?
                    .iter()
                    .collect::<Vec<_>>(),
                0,
            )?
        };
        let type_ids = Tensor::from_vec(
            questions.iter().map(|q| q.kind as u32).collect::<Vec<_>>(),
            questions.len(),
            hidden.device(),
        )?;

        let batched_option_projections = fused_means
            .as_ref()
            .filter(|(_, context, _)| {
                option_env!("SYS1_CUDA_COMPUTE_CAP") == Some("120")
                    || matches!(option_env!("SYS1_CUDA_COMPUTE_CAP"), Some("80" | "90"))
                        && context.dim(0).is_ok_and(|rows| rows >= 128)
            })
            .map(|(_, context, lex)| {
                #[cfg(test)]
                if option_env!("SYS1_CUDA_COMPUTE_CAP") == Some("90") {
                    eprintln!(
                        "HOPPER_OPTIONS171 options={} questions={}",
                        context.dim(0)?,
                        questions.len()
                    );
                }
                Ok::<_, candle_core::Error>((
                    context.apply(&self.option_context_projection)?,
                    lex.apply(&self.option_lexical_projection)?,
                    question_vectors.apply(&self.option_question_projection)?,
                ))
            })
            .transpose()?;
        let mut lexical = Vec::with_capacity(questions.len());
        let mut routed = Vec::with_capacity(questions.len());
        let mut counts = Vec::with_capacity(questions.len());
        for (index, question) in questions.iter().enumerate() {
            let start: usize = counts.iter().sum();
            let context = if let Some((_, context, _)) = &fused_means {
                context.narrow(0, start, question.option_spans.len())?
            } else {
                Tensor::stack(
                    &question
                        .option_spans
                        .iter()
                        .map(|&span| Self::mean_span(&hidden, span))
                        .collect::<Result<Vec<_>>>()?
                        .iter()
                        .collect::<Vec<_>>(),
                    0,
                )?
            };
            let lex = if let Some((_, _, lexical)) = &fused_means {
                lexical.narrow(0, start, question.option_spans.len())?
            } else {
                Tensor::stack(
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
                )?
            };
            let question_vector = question_vectors.narrow(0, index, 1)?;
            let query = if let Some((context_proj, lexical_proj, question_proj)) =
                &batched_option_projections
            {
                (context_proj.narrow(0, start, question.option_spans.len())?
                    + lexical_proj.narrow(0, start, question.option_spans.len())?)?
                .broadcast_add(&question_proj.narrow(0, index, 1)?)?
            } else {
                (context.apply(&self.option_context_projection)?
                    + lex.apply(&self.option_lexical_projection)?)?
                .broadcast_add(&question_vector.apply(&self.option_question_projection)?)?
            };
            counts.push(question.option_spans.len());
            lexical.push(lex);
            routed.push(query);
        }
        let mut options = Tensor::cat(&routed.iter().collect::<Vec<_>>(), 0)?.unsqueeze(0)?;
        for (index, layer) in self.evidence_layers.iter().enumerate() {
            options = layer.forward_head_packed(&options, &memory, &options_plan, &tokens)?;
            crate::numerics::capture(&format!("head-evidence-{index}"), &options)?;
        }
        let options = options.squeeze(0)?;
        let mut split = Vec::with_capacity(questions.len());
        let mut start = 0;
        for count in counts {
            split.push(options.narrow(0, start, count)?);
            start += count;
        }
        let base_fields = question_vectors.apply(&self.question_projection)?;
        crate::numerics::capture("head-initial-base-fields", &base_fields)?;
        let mut summaries = Vec::with_capacity(questions.len());
        for (index, values) in split.iter().enumerate() {
            let field = base_fields.narrow(0, index, 1)?.squeeze(0)?;
            let scores = (Self::option_scores(values, &field)? / (self.width as f64).sqrt())?
                .transpose(0, 1)?;
            let weights =
                softmax_last_dim(&scores.to_dtype(DType::F32)?)?.to_dtype(values.dtype())?;
            let summary = Self::option_summary(values, &weights)?;
            if crate::numerics::enabled() {
                for (stage, tensor) in [
                    ("values", values),
                    ("scores", &scores),
                    ("weights", &weights),
                    ("output", &summary),
                ] {
                    crate::numerics::capture(&format!("head-summary-{index}-{stage}"), tensor)?;
                }
            }
            summaries.push(summary);
        }
        let fields = ((&base_fields
            + Tensor::stack(&summaries.iter().collect::<Vec<_>>(), 0)?
                .apply(&self.option_summary_norm)?)?
        .broadcast_add(&global.apply(&self.global_projection)?)?
            + type_ids.apply(&self.type_embedding)?)?;
        let mut fields = fields.unsqueeze(0)?;
        crate::numerics::capture("head-initial-fields", &fields)?;
        for (index, layer) in self.layers.iter().enumerate() {
            fields = layer.forward_head_packed(&fields, &memory, &fields_plan, &tokens)?;
            crate::numerics::capture(&format!("head-fields-{index}"), &fields)?;
        }
        let fields = fields.squeeze(0)?.apply(&self.field_norm)?;
        let prior_scale = self.prior_scale.clone();
        let joint_scale = self.joint_scale.clone();
        let gate = self.residual_gate.clone();
        let counts = split.iter().map(|v| v.dim(0)).collect::<Result<Vec<_>>>()?;
        let final_batch = self.final_option_batch(
            &options,
            &fields,
            &question_vectors,
            &global,
            &lexical,
            &counts,
        )?;

        if option_env!("SYS1_CUDA_COMPUTE_CAP") == Some("90")
            && hidden.device().is_cuda()
            && counts.len() > 1
        {
            let mut raw_priors = Vec::with_capacity(counts.len());
            let mut start = 0;
            for (index, &count) in counts.iter().enumerate() {
                raw_priors.push(
                    final_batch
                        .normalized_lexical
                        .narrow(0, start, count)?
                        .matmul(&final_batch.normalized_anchor.narrow(0, index, 1)?.t()?)?
                        .squeeze(1)?,
                );
                start += count;
            }
            let (scores, prior) = grouped_final_scores(
                &raw_priors,
                &final_batch.field_cos,
                &final_batch.output,
                &prior_scale,
                &joint_scale,
                &gate,
            )?;
            if crate::numerics::enabled() {
                let mut start = 0;
                for (index, &count) in counts.iter().enumerate() {
                    let features = final_batch.features.narrow(0, start, count)?;
                    let projected = final_batch.projected.narrow(0, start, count)?;
                    let activated = final_batch.activated.narrow(0, start, count)?;
                    let output = final_batch.output.narrow(0, start, count)?;
                    let prior = prior.narrow(0, start, count)?;
                    let field_cos = final_batch.field_cos.narrow(0, start, count)?;
                    for (stage, value) in [
                        ("features", &features),
                        ("projected", &projected),
                        ("activated", &activated),
                        ("output", &output),
                        ("prior", &prior),
                        ("field-cos", &field_cos),
                    ] {
                        crate::numerics::capture(&format!("head-residual-{index}-{stage}"), value)?;
                    }
                    start += count;
                }
            }
            let scores = scores.to_vec1::<f32>()?;
            let mut start = 0;
            return Ok(counts
                .iter()
                .map(|&count| {
                    let result = scores[start..start + count].to_vec();
                    start += count;
                    result
                })
                .collect());
        }

        let mut pending_scores = Vec::with_capacity(questions.len());
        let mut start = 0;
        for (index, &count) in counts.iter().enumerate() {
            let prior = final_batch
                .normalized_lexical
                .narrow(0, start, count)?
                .matmul(&final_batch.normalized_anchor.narrow(0, index, 1)?.t()?)?
                .squeeze(1)?
                .broadcast_mul(&prior_scale)?;
            let field_cos = final_batch.field_cos.narrow(0, start, count)?;
            let features = final_batch.features.narrow(0, start, count)?;
            let residual_projected = final_batch.projected.narrow(0, start, count)?;
            let residual_activated = final_batch.activated.narrow(0, start, count)?;
            let residual_output = final_batch.output.narrow(0, start, count)?;
            if crate::numerics::enabled() {
                for (stage, value) in [
                    ("features", &features),
                    ("projected", &residual_projected),
                    ("activated", &residual_activated),
                    ("output", &residual_output),
                    ("prior", &prior),
                    ("field-cos", &field_cos),
                ] {
                    crate::numerics::capture(&format!("head-residual-{index}-{stage}"), value)?;
                }
            }
            let residual = residual_output.squeeze(1)?.to_dtype(DType::F32)?;
            let joint = (field_cos.broadcast_mul(&joint_scale)? + residual)?;
            pending_scores.push((prior + joint.broadcast_mul(&gate)?)?);
            start += count;
        }
        if pending_scores.len() <= 1 || !hidden.device().is_cuda() {
            return pending_scores
                .into_iter()
                .map(|scores| scores.to_vec1::<f32>())
                .collect();
        }
        let lengths = pending_scores
            .iter()
            .map(Tensor::elem_count)
            .collect::<Vec<_>>();
        let scores =
            Tensor::cat(&pending_scores.iter().collect::<Vec<_>>(), 0)?.to_vec1::<f32>()?;
        let mut offset = 0;
        Ok(lengths
            .into_iter()
            .map(|length| {
                let slice = scores[offset..offset + length].to_vec();
                offset += length;
                slice
            })
            .collect())
    }
}

impl JointHead {
    #[cfg(feature = "cuda")]
    pub(crate) fn forward_batch(
        &self,
        encoded: &[&Encoded],
        hidden: &Tensor,
        output_embedding: &Tensor,
    ) -> candle_core::Result<Vec<Vec<Vec<f32>>>> {
        if encoded.is_empty() {
            candle_core::bail!("cannot pack an empty decision head batch");
        }
        let lengths = encoded.iter().map(|r| r.ids.len()).collect::<Vec<_>>();
        let counts = encoded
            .iter()
            .map(|r| r.questions.len())
            .collect::<Vec<_>>();
        let flat = encoded
            .iter()
            .flat_map(|r| r.ids.iter().copied())
            .collect::<Vec<_>>();
        let total = flat.len();
        let ids = Tensor::from_vec(flat, total, hidden.device())?;
        let hidden = hidden.squeeze(0)?;
        if hidden.dim(0)? != total {
            candle_core::bail!("packed head hidden/request length mismatch");
        }
        let mut questions = Vec::new();
        let mut offset = 0usize;
        for item in encoded {
            for original in &item.questions {
                let mut question = original.clone();
                question.question_span.0 = question
                    .question_span
                    .0
                    .checked_add(offset)
                    .ok_or_else(|| {
                        candle_core::Error::Msg("packed question start overflow".into())
                    })?;
                question.question_span.1 = question
                    .question_span
                    .1
                    .checked_add(offset)
                    .ok_or_else(|| {
                        candle_core::Error::Msg("packed question end overflow".into())
                    })?;
                for span in &mut question.option_spans {
                    span.0 = span.0.checked_add(offset).ok_or_else(|| {
                        candle_core::Error::Msg("packed option start overflow".into())
                    })?;
                    span.1 = span.1.checked_add(offset).ok_or_else(|| {
                        candle_core::Error::Msg("packed option end overflow".into())
                    })?;
                }
                questions.push(question);
            }
            offset = offset
                .checked_add(item.ids.len())
                .ok_or_else(|| candle_core::Error::Msg("packed head offset overflow".into()))?;
        }
        let logits = self.forward_request_pack_scoring(
            &hidden,
            &ids,
            &questions,
            output_embedding,
            &lengths,
            &counts,
        )?;
        if logits.len() != counts.iter().sum::<usize>() {
            candle_core::bail!("packed head lost question outputs");
        }
        let mut iter = logits.into_iter();
        Ok(counts
            .into_iter()
            .map(|count| iter.by_ref().take(count).collect())
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[cfg(feature = "cuda")]
    #[test]
    fn batched_option_projection_matches_per_question_bits() -> Result<()> {
        let device = Device::new_cuda(0)?;

        if let Device::Cuda(cuda) = &device {
            let (major, _) = cuda
                .cuda_stream()
                .context()
                .compute_capability()
                .map_err(|error| candle_core::Error::Msg(error.to_string()))?;
            if major < 8 {
                return Ok(());
            }
        }
        let width = 64;
        let projection = |offset: usize| -> Result<Linear> {
            let weights = (0..width * width)
                .map(|i| ((i * 13 + offset) % 1009) as f32 * 0.003 - 1.5)
                .collect::<Vec<_>>();
            Ok(Linear::new(
                Tensor::from_vec(weights, (width, width), &device)?.to_dtype(DType::BF16)?,
                None,
            ))
        };
        let context_projection = projection(0)?;
        let lexical_projection = projection(137)?;
        let question_projection = projection(503)?;
        let bits = |tensor: Tensor| -> Result<Vec<u32>> {
            Ok(tensor
                .to_dtype(DType::F32)?
                .flatten_all()?
                .to_vec1::<f32>()?
                .into_iter()
                .map(f32::to_bits)
                .collect())
        };
        for question_count in [1, 43] {
            let counts = (0..question_count).map(|i| i % 4 + 1).collect::<Vec<_>>();
            let option_count: usize = counts.iter().sum();
            let input = |rows: usize, offset: usize| -> Result<Tensor> {
                Tensor::from_vec(
                    (0..rows * width)
                        .map(|i| ((i * 17 + offset) % 997) as f32 * 0.002 - 1.)
                        .collect::<Vec<_>>(),
                    (rows, width),
                    &device,
                )?
                .to_dtype(DType::BF16)
            };
            let context = input(option_count, 0)?;
            let lexical = input(option_count, 311)?;
            let questions = input(question_count, 719)?;
            let context_batched = context.apply(&context_projection)?;
            let lexical_batched = lexical.apply(&lexical_projection)?;
            let questions_batched = questions.apply(&question_projection)?;
            let mut offset = 0;
            let mut original = Vec::new();
            let mut batched = Vec::new();
            for (index, &count) in counts.iter().enumerate() {
                let c = context.narrow(0, offset, count)?;
                let l = lexical.narrow(0, offset, count)?;
                let q = questions.narrow(0, index, 1)?;
                original.push(
                    (c.apply(&context_projection)? + l.apply(&lexical_projection)?)?
                        .broadcast_add(&q.apply(&question_projection)?)?,
                );
                batched.push(
                    (context_batched.narrow(0, offset, count)?
                        + lexical_batched.narrow(0, offset, count)?)?
                    .broadcast_add(&questions_batched.narrow(0, index, 1)?)?,
                );
                offset += count;
            }
            assert_eq!(
                bits(Tensor::cat(&original.iter().collect::<Vec<_>>(), 0)?)?,
                bits(Tensor::cat(&batched.iter().collect::<Vec<_>>(), 0)?)?,
                "question_count={question_count}",
            );
        }
        Ok(())
    }

    #[test]
    fn head_normalization_matches_documented_torch_epsilon_semantics() -> Result<()> {
        #[allow(unused_mut)]
        let mut devices = vec![Device::Cpu];
        #[cfg(feature = "cuda")]
        devices.push(Device::new_cuda(0)?);
        for device in devices {
            let input =
                Tensor::from_vec(vec![0f32, 0., 1e-8, 0., 1e-14, 0., 3., 4.], (4, 2), &device)?;
            let observed = JointHead::normalize(&input)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            for (a, b) in observed.iter().zip([0f32, 0., 1., 0., 0.01, 0., 0.6, 0.8]) {
                assert!((*a - b).abs() <= 1e-5 + 1.3e-6 * b.abs());
            }
            let small = Tensor::from_vec(vec![1e-9f32, 0.], (1, 2), &device)?;
            let normalized = JointHead::normalize_with_epsilon(&small, 1e-8)?;
            let cosine = normalized.sqr()?.sum(D::Minus1)?.to_vec1::<f32>()?;
            assert!((cosine[0] - 0.01).abs() <= 1e-5);
        }
        Ok(())
    }

    #[cfg(feature = "metal")]
    #[test]
    fn metal_preserves_main_head_normalization_and_span_casts() -> Result<()> {
        let device = Device::new_metal(0)?;
        let pairs = [(0f32, 0f32), (1e-8, 0.), (1e-14, 0.), (3., 4.)];
        let input = Tensor::from_vec(
            pairs.iter().flat_map(|&(a, b)| [a, b]).collect::<Vec<_>>(),
            (pairs.len(), 2),
            &device,
        )?;
        let actual = JointHead::normalize(&input)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let expected: Vec<f32> = pairs
            .iter()
            .flat_map(|&(a, b)| {
                let norm = (f64::from(a).powi(2) + f64::from(b).powi(2) + 1e-12).sqrt();
                [(f64::from(a) / norm) as f32, (f64::from(b) / norm) as f32]
            })
            .collect();
        for (&a, &b) in actual.iter().zip(&expected) {
            assert!((a - b).abs() <= 1e-5 + 1.3e-6 * b.abs());
        }

        let small = Tensor::from_vec(vec![1e-9f32, 0.], (1, 2), &device)?;
        let actual = JointHead::normalize_with_epsilon(&small, 1e-8)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let expected = (1e-9f64 / (1e-18f64 + 1e-12).sqrt()) as f32;
        assert!((actual[0] - expected).abs() <= 1e-5 + 1.3e-6 * expected.abs());

        for dtype in [DType::F32, DType::F16] {
            let values = Tensor::from_vec(vec![1f32, 2., 4.], (3, 1), &device)?.to_dtype(dtype)?;
            let actual = JointHead::mean_rows(&values)?
                .to_dtype(DType::F32)?
                .to_vec1::<f32>()?;
            let coefficient = Tensor::from_vec(vec![1f32 / 3.], 1, &Device::Cpu)?
                .to_dtype(dtype)?
                .to_dtype(DType::F32)?
                .to_vec1::<f32>()?[0];
            let expected = Tensor::from_vec(vec![7. * coefficient], 1, &Device::Cpu)?
                .to_dtype(dtype)?
                .to_dtype(DType::F32)?
                .to_vec1::<f32>()?;
            if dtype == DType::F16 {
                assert_eq!(actual, expected, "main Metal rounds reciprocal before GEMM");
                let fp32_mean = Tensor::from_vec(vec![7f32 / 3.], 1, &Device::Cpu)?
                    .to_dtype(dtype)?
                    .to_dtype(DType::F32)?
                    .to_vec1::<f32>()?;
                assert_ne!(
                    actual, fp32_mean,
                    "fixture must distinguish backend contracts"
                );
            } else {
                assert!((actual[0] - expected[0]).abs() <= 1e-5 + 1.3e-6 * expected[0].abs());
            }
        }
        Ok(())
    }

    #[test]
    fn option_summary_rounds_products_before_accumulation() -> Result<()> {
        for dtype in [DType::F32, DType::F16, DType::BF16] {
            let values = Tensor::from_vec(vec![1f32, 2., 4., 3., -1., 7.], (3, 2), &Device::Cpu)?
                .to_dtype(dtype)?;
            let weights = Tensor::from_vec(vec![0.333f32, 0.333, 0.334], (1, 3), &Device::Cpu)?
                .to_dtype(dtype)?;
            let rounded_products = values
                .broadcast_mul(&weights.t()?)?
                .to_dtype(DType::F32)?
                .to_vec2::<f32>()?;
            let expected = Tensor::from_vec(
                (0..2)
                    .map(|column| rounded_products.iter().map(|row| row[column]).sum::<f32>())
                    .collect::<Vec<_>>(),
                2,
                &Device::Cpu,
            )?
            .to_dtype(dtype)?
            .to_dtype(DType::F32)?
            .to_vec1::<f32>()?;
            let actual = JointHead::option_summary(&values, &weights)?
                .to_dtype(DType::F32)?
                .to_vec1::<f32>()?;
            assert_eq!(actual, expected, "{dtype:?}");
        }
        Ok(())
    }

    #[test]
    fn cross_attention_projection_matches_independent_scalar_reference() -> Result<()> {
        let device =
            crate::device::load().map_err(|error| candle_core::Error::Msg(error.to_string()))?;
        let width = 128;
        let values = |count: usize, salt: usize| -> Vec<f32> {
            (0..count)
                .map(|index| (((index * 7 + salt) % 17) as f32 - 8.) / 32.)
                .collect()
        };
        let weights = values(3 * width * width, 3);
        let query_values = values(2 * 3 * width, 5);
        let memory_values = values(2 * 7 * width, 11);
        let mut dtypes = vec![DType::F32];
        if device.is_cuda() {
            dtypes.push(DType::F16);
            if crate::architectures::qwen35_text::inference::cuda_capability(&device)
                .map_err(|error| candle_core::Error::Msg(error.to_string()))?
                .is_some_and(|(major, _)| major >= 8)
            {
                dtypes.push(DType::BF16);
            }
        }
        for dtype in dtypes {
            for bias in [false, true] {
                let attention = Attention::from_packed(
                    Linear::new(
                        Tensor::from_vec(weights.clone(), (3 * width, width), &device)?
                            .to_dtype(dtype)?,
                        if bias {
                            Some(Tensor::zeros(3 * width, dtype, &device)?)
                        } else {
                            None
                        },
                    ),
                    Linear::new(Tensor::eye(width, dtype, &device)?, None),
                    2,
                    width,
                    AttentionImplementation::Eager,
                )?;
                let query = Tensor::from_vec(query_values.clone(), (2, 3, width), &device)?
                    .to_dtype(dtype)?;
                let memory = Tensor::from_vec(memory_values.clone(), (2, 7, width), &device)?
                    .to_dtype(dtype)?;
                let (q, k, v) = attention.project(&query, &memory)?;
                for (actual, inputs, offset) in [
                    (q, &query_values, 0),
                    (k, &memory_values, width),
                    (v, &memory_values, width * 2),
                ] {
                    let expected: Vec<f32> = inputs
                        .chunks_exact(width)
                        .flat_map(|row| {
                            let weights = &weights;
                            (offset..offset + width).map(move |column| {
                                row.iter()
                                    .zip(&weights[column * width..(column + 1) * width])
                                    .map(|(x, w)| x * w)
                                    .sum::<f32>()
                            })
                        })
                        .collect();
                    let expected = Tensor::from_vec(expected, actual.shape(), &Device::Cpu)?
                        .to_dtype(dtype)?
                        .to_dtype(DType::F32)?
                        .flatten_all()?
                        .to_vec1::<f32>()?;
                    let actual = actual
                        .to_dtype(DType::F32)?
                        .flatten_all()?
                        .to_vec1::<f32>()?;
                    assert_eq!(actual, expected, "{dtype:?} bias={bias} offset={offset}");
                }
            }
        }
        Ok(())
    }

    #[test]
    fn self_attention_projection_reuse_matches_two_packed_projections() -> Result<()> {
        let device =
            crate::device::load().map_err(|error| candle_core::Error::Msg(error.to_string()))?;

        let width = 128;
        let values = |count: usize, salt: usize| -> Vec<f32> {
            (0..count)
                .map(|index| (((index * 7 + salt) % 19) as f32 - 9.) * 0.025)
                .collect()
        };
        let attention = Attention::from_packed(
            Linear::new(
                Tensor::from_vec(values(3 * width * width, 1), (3 * width, width), &device)?,
                Some(Tensor::from_vec(values(3 * width, 2), 3 * width, &device)?),
            ),
            Linear::new(
                Tensor::from_vec(values(width * width, 3), (width, width), &device)?,
                Some(Tensor::from_vec(values(width, 4), width, &device)?),
            ),
            2,
            width,
            AttentionImplementation::Eager,
        )?;
        #[allow(unused_mut)]
        let mut dtypes = vec![DType::F32];
        #[cfg(feature = "cuda")]
        if let Device::Cuda(cuda) = &device {
            dtypes.push(DType::F16);
            let (major, _) = cuda
                .cuda_stream()
                .context()
                .compute_capability()
                .map_err(|error| candle_core::Error::Msg(error.to_string()))?;
            if major >= 8 {
                dtypes.push(DType::BF16);
            }
        }
        for dtype in dtypes {
            let automatic = crate::models::attention_for_device(dtype, &device)
                .map_err(|error| candle_core::Error::Msg(error.to_string()))?;
            let mut implementations = vec![AttentionImplementation::Eager];
            if automatic != AttentionImplementation::Eager {
                implementations.push(automatic);
            }
            for implementation in implementations {
                let cast_linear = |linear: &Linear| -> Result<Linear> {
                    Ok(Linear::new(
                        linear.weight().to_dtype(dtype)?,
                        linear.bias().map(|bias| bias.to_dtype(dtype)).transpose()?,
                    ))
                };
                let typed_attention = Attention::from_packed(
                    cast_linear(&attention.in_proj)?,
                    cast_linear(&attention.out_proj)?,
                    attention.heads,
                    width,
                    implementation,
                )?;
                let query = Tensor::from_vec(values(5 * width, 5), (1, 5, width), &device)?
                    .to_dtype(dtype)?;
                let independent_memory = query.copy()?;
                assert_ne!(query.id(), independent_memory.id());
                let (q, k, v) = typed_attention.project(&query, &query)?;
                let old_q = query
                    .apply(&typed_attention.in_proj)?
                    .narrow(D::Minus1, 0, width)?;
                let old_kv = independent_memory.apply(&typed_attention.in_proj)?;
                for (actual, expected) in [
                    (q, old_q),
                    (k, old_kv.narrow(D::Minus1, width, width)?),
                    (v, old_kv.narrow(D::Minus1, width * 2, width)?),
                ] {
                    let values = |tensor: &Tensor| {
                        tensor.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()
                    };
                    assert_eq!(
                        values(&actual)?,
                        values(&expected)?,
                        "{dtype:?} {implementation:?}"
                    );
                }

                let output = typed_attention
                    .forward(&query, &query)?
                    .to_dtype(DType::F32)?
                    .flatten_all()?
                    .to_vec1::<f32>()?;
                assert!(output.iter().all(|value| value.is_finite()));
                assert!(output.iter().any(|value| value.abs() > 0.));
            }
        }
        Ok(())
    }

    #[test]
    fn span_means_round_after_fp32_accumulation_not_before() {
        #[allow(unused_mut)]
        let mut devices = vec![candle_core::Device::Cpu];
        #[cfg(feature = "cuda")]
        devices.push(crate::device::load().unwrap());
        for device in devices {
            for dtype in [DType::F32, DType::F16, DType::BF16] {
                if dtype == DType::BF16
                    && (device.is_metal()
                        || crate::architectures::qwen35_text::inference::cuda_capability(&device)
                            .unwrap()
                            .is_some_and(|(major, _)| major < 8))
                {
                    continue;
                }
                let values = Tensor::from_vec(vec![1f32, 2., 4.], (3, 1), &device)
                    .unwrap()
                    .to_dtype(dtype)
                    .unwrap();
                let actual = JointHead::mean_rows(&values)
                    .unwrap()
                    .to_dtype(DType::F32)
                    .unwrap()
                    .to_vec1::<f32>()
                    .unwrap();
                let expected = Tensor::from_vec(vec![7f32 / 3.], 1, &device)
                    .unwrap()
                    .to_dtype(dtype)
                    .unwrap()
                    .to_dtype(DType::F32)
                    .unwrap()
                    .to_vec1::<f32>()
                    .unwrap();
                if dtype == DType::F32 {
                    assert!(
                        (actual[0] - expected[0]).abs() <= 1e-5 + 1.3e-6 * expected[0].abs(),
                        "span mean on {device:?}/{dtype:?}: actual={actual:?}, expected={expected:?}"
                    );
                } else {
                    assert_eq!(actual, expected, "span mean differs for {dtype:?}");
                }
                if dtype != DType::F32 {
                    let reciprocal = Tensor::from_vec(vec![1f32 / 3.], 1, &device)
                        .unwrap()
                        .to_dtype(dtype)
                        .unwrap()
                        .to_dtype(DType::F32)
                        .unwrap()
                        .to_vec1::<f32>()
                        .unwrap()[0];
                    let biased = Tensor::from_vec(vec![7. * reciprocal], 1, &device)
                        .unwrap()
                        .to_dtype(dtype)
                        .unwrap()
                        .to_dtype(DType::F32)
                        .unwrap()
                        .to_vec1::<f32>()
                        .unwrap();
                    assert_ne!(
                        biased, expected,
                        "regression must expose the old reciprocal bias"
                    );
                }
            }
        }
    }

    #[test]
    fn short_span_sums_match_fp32_reference_across_warp_boundaries() {
        #[allow(unused_mut)]
        let mut devices = vec![candle_core::Device::Cpu];
        #[cfg(feature = "cuda")]
        devices.push(crate::device::load().unwrap());
        for device in devices {
            for rows in [1, 2, 3, 7, 16, 17, 31, 32, 33, 63, 64, 65] {
                for width in [1, 3, 8] {
                    let data: Vec<f32> = (0..(rows + 2) * width)
                        .map(|i| ((i * 7 % 17) as f32 - 8.) * 0.25)
                        .collect();
                    let values = Tensor::from_vec(data.clone(), (rows + 2, width), &device)
                        .unwrap()
                        .narrow(0, 1, rows)
                        .unwrap();
                    let expected: Vec<f32> = (0..width)
                        .map(|column| (1..=rows).map(|row| data[row * width + column]).sum())
                        .collect();

                    for values in [
                        values.clone(),
                        values
                            .transpose(0, 1)
                            .unwrap()
                            .contiguous()
                            .unwrap()
                            .transpose(0, 1)
                            .unwrap(),
                    ] {
                        let actual = values.sum(0).unwrap().to_vec1::<f32>().unwrap();
                        assert_eq!(
                            actual, expected,
                            "sum differs on {device:?}, rows={rows}, width={width}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn span_mean_rejects_empty_spans() {
        let values = Tensor::zeros((0, 8), DType::F32, &candle_core::Device::Cpu).unwrap();
        assert!(JointHead::mean_rows(&values).is_err());
    }

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

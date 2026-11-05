use super::Config;
use crate::models::AttentionImplementation;

use candle_core::{D, DType, Device, Result, Tensor};
use candle_nn::{Embedding, Init, LayerNorm, Linear, Module, VarBuilder, embedding, ops::softmax};

#[cfg(test)]
use std::fs;
use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
};

const ATTENTION_MASK_VALUE: f32 = -10_000.0;

struct RotaryEmbedding {
    sin: Tensor,
    cos: Tensor,
}

impl RotaryEmbedding {
    fn new(config: &Config, rope_theta: f64, device: &Device) -> Result<Self> {
        let dim = config.hidden_size / config.num_attention_heads;
        let inv_freq: Vec<_> = (0..dim)
            .step_by(2)
            .map(|index| 1f32 / rope_theta.powf(index as f64 / dim as f64) as f32)
            .collect();
        let inv_freq_len = inv_freq.len();
        let inv_freq = Tensor::from_vec(inv_freq, (1, inv_freq_len), device)?;
        let positions = Tensor::arange(0u32, config.max_position_embeddings as u32, device)?
            .to_dtype(DType::F32)?
            .reshape((config.max_position_embeddings, 1))?;
        let frequencies = positions.matmul(&inv_freq)?;
        Ok(Self {
            sin: frequencies.sin()?,
            cos: frequencies.cos()?,
        })
    }

    fn apply(&self, q: &Tensor, k: &Tensor) -> Result<(Tensor, Tensor)> {
        let dtype = q.dtype();
        let rotary_dtype = self.cos.dtype();
        let q = candle_nn::rotary_emb::rope(
            &q.to_dtype(rotary_dtype)?.contiguous()?,
            &self.cos,
            &self.sin,
        )?
        .to_dtype(dtype)?;
        let k = candle_nn::rotary_emb::rope(
            &k.to_dtype(rotary_dtype)?.contiguous()?,
            &self.cos,
            &self.sin,
        )?
        .to_dtype(dtype)?;
        Ok((q, k))
    }

    fn apply_split(&self, qkv: &Tensor) -> Result<(Tensor, Tensor, Tensor)> {
        let qkv = qkv.permute((2, 0, 3, 1, 4))?;
        let q = qkv.get(0)?;
        let k = qkv.get(1)?;
        let v = qkv.get(2)?;
        let (q, k) = self.apply(&q, &k)?;
        Ok((q, k, v))
    }

    fn supports_fused_qkv(&self, dtype: DType) -> bool {
        self.cos.dtype() == DType::F32
            && self.cos.device().is_metal()
            && matches!(dtype, DType::F32 | DType::F16 | DType::BF16)
    }

    fn apply_qkv_metal(&self, qkv: &Tensor) -> Result<(Tensor, Tensor, Tensor)> {
        let output = crate::kernels::metal::rope_qkv(qkv, &self.cos, &self.sin)?;
        Ok((output.get(0)?, output.get(1)?, output.get(2)?))
    }
}

struct Attention {
    qkv: Linear,
    projection: Linear,
    heads: usize,
    head_size: usize,
    rotary: Arc<RotaryEmbedding>,
    compute_dtype: DType,
    implementation: AttentionImplementation,
}

impl Attention {
    fn load(
        vb: VarBuilder,
        config: &Config,
        rotary: Arc<RotaryEmbedding>,
        compute_dtype: DType,
        implementation: AttentionImplementation,
    ) -> Result<Self> {
        Ok(Self {
            qkv: linear_no_bias_dtype(
                config.hidden_size,
                config.hidden_size * 3,
                vb.pp("Wqkv"),
                compute_dtype,
            )?,
            projection: linear_no_bias_dtype(
                config.hidden_size,
                config.hidden_size,
                vb.pp("Wo"),
                compute_dtype,
            )?,
            heads: config.num_attention_heads,
            head_size: config.hidden_size / config.num_attention_heads,
            rotary,
            compute_dtype,
            implementation,
        })
    }

    fn forward(
        &self,
        xs: &Tensor,
        mask: Option<&Tensor>,
        lengths: &[usize],
        window: Option<usize>,
    ) -> Result<Tensor> {
        let (batch, length, hidden) = xs.dims3()?;
        let qkv = xs
            .to_dtype(self.compute_dtype)?
            .apply(&self.qkv)?
            .reshape((batch, length, 3, self.heads, self.head_size))?;
        let (q, k, v) = self.split_qkv(&qkv)?;
        let scale = (self.head_size as f64).powf(-0.5);
        let attention = self.attention_fn(&q, &k, &v, scale, mask, lengths, window)?;
        merge_heads(&attention, batch, length, hidden)?.apply(&self.projection)
    }

    fn split_qkv(&self, qkv: &Tensor) -> Result<(Tensor, Tensor, Tensor)> {
        if qkv.device().is_metal()
            && self.implementation == AttentionImplementation::Auto
            && self.rotary.supports_fused_qkv(qkv.dtype())
        {
            return self.rotary.apply_qkv_metal(qkv);
        }
        self.rotary.apply_split(qkv)
    }

    #[allow(clippy::too_many_arguments)]
    fn attention_fn(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        scale: f64,
        mask: Option<&Tensor>,
        lengths: &[usize],
        window: Option<usize>,
    ) -> Result<Tensor> {
        if q.device().is_metal() && self.implementation == AttentionImplementation::Auto {
            let (batch, _, length, _) = q.dims4()?;
            if let (Some(window), Some(mask)) = (window, mask)
                && length >= METAL_WINDOWED_MIN_LENGTH
            {
                return metal_windowed_sdpa(q, k, v, mask, scale as f32, window, self.heads);
            }
            let mask = mask
                .map(|mask| mask.broadcast_as((batch, self.heads, length, length)))
                .transpose()?;
            return candle_nn::ops::sdpa(q, k, v, mask.as_ref(), false, scale as f32, 1.0);
        }
        scaled_dot_product_attention(
            q,
            k,
            v,
            scale,
            AttentionOptions {
                mask,
                implementation: self.implementation,
                lengths,
                window,
            },
        )
    }
}

pub(crate) fn merge_heads(
    attention: &Tensor,
    batch: usize,
    length: usize,
    hidden: usize,
) -> Result<Tensor> {
    if attention.device().is_metal() {
        return crate::kernels::metal::merge_heads(attention);
    }
    attention.transpose(1, 2)?.reshape((batch, length, hidden))
}

const METAL_WINDOWED_MIN_LENGTH: usize = 768;

fn metal_windowed_sdpa(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    mask: &Tensor,
    scale: f32,
    window: usize,
    heads: usize,
) -> Result<Tensor> {
    const BLOCK: usize = 128;

    let (batch, _, length, _) = q.dims4()?;
    let mut blocks = Vec::with_capacity(length.div_ceil(BLOCK));
    for query_start in (0..length).step_by(BLOCK) {
        let query_length = BLOCK.min(length - query_start);
        let key_start = query_start.saturating_sub(window);
        let key_end = (query_start + query_length + window).min(length);
        let key_length = key_end - key_start;
        let block_mask = mask
            .narrow(D::Minus2, query_start, query_length)?
            .narrow(D::Minus1, key_start, key_length)?
            .contiguous()?
            .broadcast_as((batch, heads, query_length, key_length))?;
        blocks.push(candle_nn::ops::sdpa(
            &q.narrow(2, query_start, query_length)?,
            &k.narrow(2, key_start, key_length)?,
            &v.narrow(2, key_start, key_length)?,
            Some(&block_mask),
            false,
            scale,
            1.0,
        )?);
    }
    Tensor::cat(&blocks, 2)
}

pub(crate) struct AttentionOptions<'a> {
    pub mask: Option<&'a Tensor>,
    pub implementation: AttentionImplementation,
    pub lengths: &'a [usize],
    pub window: Option<usize>,
}

pub(crate) fn scaled_dot_product_attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    scale: f64,
    options: AttentionOptions<'_>,
) -> Result<Tensor> {
    match options.implementation {
        AttentionImplementation::Auto | AttentionImplementation::Eager
            if q.device().is_cpu() && options.window.is_some() =>
        {
            cpu_windowed_attention(q, k, v, scale, options.mask, options.window.unwrap())
        }
        AttentionImplementation::Auto | AttentionImplementation::Eager => {
            eager_attention(q, k, v, scale, options.mask)
        }
        implementation => flash_attention(
            q,
            k,
            v,
            options.lengths,
            scale as f32,
            options.window,
            implementation,
        ),
    }
}

fn eager_attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    scale: f64,
    mask: Option<&Tensor>,
) -> Result<Tensor> {
    let scores = (q * scale)?.matmul(&k.transpose(D::Minus2, D::Minus1)?)?;
    let scores = match mask {
        Some(mask) => scores.to_dtype(mask.dtype())?.broadcast_add(mask)?,
        None => scores,
    };
    let probabilities = attention_softmax(&scores)?;
    probabilities.to_dtype(v.dtype())?.matmul(v)
}

fn cpu_windowed_attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    scale: f64,
    mask: Option<&Tensor>,
    window: usize,
) -> Result<Tensor> {
    let length = q.dim(2)?;
    if length <= window.saturating_add(window / 4) {
        return eager_attention(q, k, v, scale, mask);
    }
    let block = window.div_ceil(2).max(1);
    let mut output = Vec::with_capacity(length.div_ceil(block));
    for query_start in (0..length).step_by(block) {
        let query_length = block.min(length - query_start);
        let query_end = query_start + query_length;
        let key_start = query_start.saturating_sub(window);
        let key_end = (query_end + window).min(length);
        let key_length = key_end - key_start;
        let q = q.narrow(2, query_start, query_length)?;
        let k = k.narrow(2, key_start, key_length)?;
        let v = v.narrow(2, key_start, key_length)?;
        let mask = match mask {
            Some(mask) if mask.rank() == 2 => Some(
                mask.narrow(0, query_start, query_length)?
                    .narrow(1, key_start, key_length)?
                    .reshape((1, 1, query_length, key_length))?,
            ),
            Some(mask) => Some(
                mask.narrow(2, query_start, query_length)?
                    .narrow(3, key_start, key_length)?,
            ),
            None => None,
        };
        output.push(eager_attention(&q, &k, &v, scale, mask.as_ref())?);
    }
    Tensor::cat(&output.iter().collect::<Vec<_>>(), 2)
}

fn attention_softmax(scores: &Tensor) -> Result<Tensor> {
    if matches!(scores.dtype(), DType::F16 | DType::BF16) {
        softmax(&scores.to_dtype(DType::F32)?, D::Minus1)?.to_dtype(scores.dtype())
    } else {
        softmax(scores, D::Minus1)
    }
}

fn flash_attention(
    _q: &Tensor,
    _k: &Tensor,
    _v: &Tensor,
    _lengths: &[usize],
    _scale: f32,
    _window: Option<usize>,
    implementation: AttentionImplementation,
) -> Result<Tensor> {
    candle_core::bail!("{} support is not compiled in", implementation.cli_name())
}

struct Mlp {
    input: Linear,
    output: Linear,
    compute_dtype: DType,
}

impl Mlp {
    fn load(vb: VarBuilder, config: &Config, compute_dtype: DType) -> Result<Self> {
        Ok(Self {
            input: linear_no_bias_dtype(
                config.hidden_size,
                config.intermediate_size * 2,
                vb.pp("Wi"),
                compute_dtype,
            )?,
            output: linear_no_bias_dtype(
                config.intermediate_size,
                config.hidden_size,
                vb.pp("Wo"),
                compute_dtype,
            )?,
            compute_dtype,
        })
    }
}

impl Module for Mlp {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let projected = xs.to_dtype(self.compute_dtype)?.apply(&self.input)?;

        if projected.device().is_metal() {
            return crate::kernels::metal::geglu(&projected)?.apply(&self.output);
        }
        let parts = projected.chunk(2, D::Minus1)?;
        (&parts[0].gelu_erf()? * &parts[1])?.apply(&self.output)
    }
}

struct Layer {
    attention: Attention,
    mlp: Mlp,
    attention_norm: Option<Norm>,
    mlp_norm: Norm,
    uses_local_attention: bool,
}

impl Layer {
    fn load(
        vb: VarBuilder,
        config: &Config,
        rotary: Arc<RotaryEmbedding>,
        uses_local_attention: bool,
        compute_dtype: DType,
        implementation: AttentionImplementation,
    ) -> Result<Self> {
        Ok(Self {
            attention: Attention::load(
                vb.pp("attn"),
                config,
                rotary,
                compute_dtype,
                implementation,
            )?,
            mlp: Mlp::load(vb.pp("mlp"), config, compute_dtype)?,
            attention_norm: Norm::load(
                config.hidden_size,
                config.layer_norm_eps,
                vb.pp("attn_norm"),
            )
            .ok(),
            mlp_norm: Norm::load(config.hidden_size, config.layer_norm_eps, vb.pp("mlp_norm"))?,
            uses_local_attention,
        })
    }

    fn forward(
        &self,
        xs: &Tensor,
        global_mask: Option<&Tensor>,
        local_mask: Option<&Tensor>,
        lengths: &[usize],
        local_window: usize,
    ) -> Result<Tensor> {
        let normalized = match &self.attention_norm {
            Some(norm) => xs.apply(norm)?,
            None => xs.clone(),
        };
        let mask = if self.uses_local_attention {
            local_mask.cloned()
        } else {
            global_mask.cloned()
        };
        let attention = self
            .attention
            .forward(
                &normalized,
                mask.as_ref(),
                lengths,
                self.uses_local_attention.then_some(local_window),
            )?
            .to_dtype(xs.dtype())?;
        let xs = (attention + xs)?;
        let mlp = xs.apply(&self.mlp_norm)?.apply(&self.mlp)?;
        let dtype = xs.dtype();
        xs + mlp.to_dtype(dtype)?
    }
}

pub struct Encoder {
    embeddings: Embedding,
    norm: Norm,
    layers: Vec<Layer>,
    final_norm: Norm,
    local_attention_size: usize,
    implementation: AttentionImplementation,
    dtype: DType,
    local_masks: Mutex<HashMap<usize, Tensor>>,
}

impl Encoder {
    pub fn load(
        vb: VarBuilder,
        config: &Config,
        compute_dtype: DType,
        implementation: AttentionImplementation,
    ) -> Result<Self> {
        let global_rotary = Arc::new(RotaryEmbedding::new(
            config,
            config.rope_parameters["full_attention"].rope_theta,
            vb.device(),
        )?);
        let local_rotary = Arc::new(RotaryEmbedding::new(
            config,
            config.rope_parameters["sliding_attention"].rope_theta,
            vb.device(),
        )?);
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for index in 0..config.num_hidden_layers {
            let local = index % config.global_attn_every_n_layers != 0;
            layers.push(Layer::load(
                vb.pp(format!("model.layers.{index}")),
                config,
                if local {
                    local_rotary.clone()
                } else {
                    global_rotary.clone()
                },
                local,
                compute_dtype,
                implementation,
            )?);
        }
        Ok(Self {
            embeddings: embedding(
                config.vocab_size,
                config.hidden_size,
                vb.pp("model.embeddings.tok_embeddings"),
            )?,
            norm: Norm::load(
                config.hidden_size,
                config.layer_norm_eps,
                vb.pp("model.embeddings.norm"),
            )?,
            layers,
            final_norm: Norm::load(
                config.hidden_size,
                config.layer_norm_eps,
                vb.pp("model.final_norm"),
            )?,
            local_attention_size: config.local_attention,
            implementation,
            dtype: compute_dtype,
            local_masks: Mutex::new(HashMap::new()),
        })
    }

    pub fn forward(
        &self,
        ids: &Tensor,
        mask: &Tensor,
        lengths: &[usize],
        has_padding: bool,
    ) -> Result<Tensor> {
        let length = ids.dim(1)?;
        let uses_eager_masks = matches!(
            self.implementation,
            AttentionImplementation::Auto | AttentionImplementation::Eager
        );
        let global_mask = (uses_eager_masks && has_padding)
            .then(|| global_attention_mask(mask, length, self.dtype))
            .transpose()?;
        let local_mask = uses_eager_masks
            .then(|| self.local_mask(length, ids.device()))
            .transpose()?;
        let local_mask = match (&local_mask, &global_mask) {
            (Some(local), Some(global)) => Some(global.broadcast_add(local)?),
            _ => local_mask,
        };
        let local_window = self.local_attention_size / 2;
        let mut xs = ids.apply(&self.embeddings)?.apply(&self.norm)?;
        for layer in &self.layers {
            xs = layer.forward(
                &xs,
                global_mask.as_ref(),
                local_mask.as_ref(),
                lengths,
                local_window,
            )?;
        }
        xs.apply(&self.final_norm)
    }

    fn local_mask(&self, length: usize, device: &Device) -> Result<Tensor> {
        let mut masks = self
            .local_masks
            .lock()
            .map_err(|_| candle_core::Error::Msg("local attention mask cache poisoned".into()))?;
        if let Some(mask) = masks.get(&length) {
            return Ok(mask.clone());
        }
        let mask = local_attention_mask(length, self.local_attention_size / 2, self.dtype, device)?;
        masks.insert(length, mask.clone());
        Ok(mask)
    }
}

enum Norm {
    #[cfg(feature = "metal")]
    FusedMetal {
        weight: Tensor,
        eps: f64,
    },
    Default(LayerNorm),
}

impl Norm {
    fn load(size: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        let weight = vb.get_with_hints(size, "weight", Init::Const(1.0))?;

        if weight.device().is_metal() {
            return Ok(Self::FusedMetal { weight, eps });
        }
        Ok(Self::Default(LayerNorm::new_no_bias(weight, eps)))
    }
}

impl Module for Norm {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        match self {
            Self::FusedMetal { weight, eps } => crate::kernels::metal::layer_norm(xs, weight, *eps),
            Self::Default(norm) => norm.forward(xs),
        }
    }
}

fn linear_no_bias_dtype(
    input: usize,
    output: usize,
    vb: VarBuilder,
    dtype: DType,
) -> Result<Linear> {
    Ok(Linear::new(
        vb.get((output, input), "weight")?.to_dtype(dtype)?,
        None,
    ))
}

fn global_attention_mask(mask: &Tensor, target_length: usize, dtype: DType) -> Result<Tensor> {
    let (batch, source_length) = mask.dims2()?;
    let expanded = mask
        .unsqueeze(1)?
        .unsqueeze(2)?
        .expand((batch, 1, target_length, source_length))?
        .to_dtype(dtype)?;
    ((1.0 - expanded)? * ATTENTION_MASK_VALUE as f64)?.to_dtype(dtype)
}

fn local_attention_mask(
    length: usize,
    max_distance: usize,
    dtype: DType,
    device: &Device,
) -> Result<Tensor> {
    let mask: Vec<_> = (0..length)
        .flat_map(|left| {
            (0..length).map(move |right| {
                if left.abs_diff(right) > max_distance {
                    ATTENTION_MASK_VALUE
                } else {
                    0.0
                }
            })
        })
        .collect();
    Tensor::from_vec(mask, (length, length), device)?.to_dtype(dtype)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::IndexOp;

    #[test]
    fn metal_windowed_attention_matches_dense_attention() -> Result<()> {
        let device = Device::new_metal(0)?;
        let (batch, heads, length, dim) = (2usize, 16usize, 800usize, 64usize);
        let window = 64;
        let tensor = |offset: f64| {
            Tensor::arange(0u32, (batch * heads * length * dim) as u32, &device)
                .and_then(|t| t.to_dtype(DType::F32))
                .and_then(|t| t.affine(0.0007, offset))
                .and_then(|t| t.sin())
                .and_then(|t| t.reshape((batch, heads, length, dim)))
        };
        let scale = (dim as f64).powf(-0.5) as f32;

        for dtype in [DType::F32, DType::F16, DType::BF16] {
            let q = tensor(0.0)?.to_dtype(dtype)?;
            let k = tensor(0.4)?.to_dtype(dtype)?;
            let v = tensor(0.9)?.to_dtype(dtype)?;

            let band = local_attention_mask(length, window, dtype, &device)?;
            let padding = Tensor::ones((batch, length), DType::F32, &device)?;
            let combined = global_attention_mask(&padding, length, dtype)?.broadcast_add(&band)?;

            for mask in [&band, &combined] {
                let dense = candle_nn::ops::sdpa(
                    &q,
                    &k,
                    &v,
                    Some(&mask.broadcast_as((batch, heads, length, length))?),
                    false,
                    scale,
                    1.0,
                )?;
                let windowed = metal_windowed_sdpa(&q, &k, &v, mask, scale, window, heads)?;

                assert_eq!(windowed.dims(), dense.dims());
                assert_eq!(windowed.dtype(), dtype);
                let dense = dense
                    .to_dtype(DType::F32)?
                    .flatten_all()?
                    .to_vec1::<f32>()?;
                let windowed = windowed
                    .to_dtype(DType::F32)?
                    .flatten_all()?
                    .to_vec1::<f32>()?;
                assert!(
                    windowed.iter().all(|value| value.is_finite()),
                    "{dtype:?}: windowed attention produced non-finite values"
                );
                let difference = dense
                    .iter()
                    .zip(windowed)
                    .map(|(left, right)| (left - right).abs())
                    .fold(0f32, f32::max);
                let tolerance = match dtype {
                    DType::F32 => 1e-5,
                    DType::F16 => 2e-3,
                    _ => 2e-2,
                };
                assert!(
                    difference <= tolerance,
                    "{dtype:?}: maximum difference was {difference}"
                );
            }
        }
        Ok(())
    }

    #[test]
    fn fused_metal_merge_heads_matches_candle() -> Result<()> {
        let device = Device::new_metal(0)?;
        let (batch, heads, length, dim) = (3usize, 16usize, 37usize, 64usize);
        let values: Vec<f32> = (0..batch * heads * length * dim)
            .map(|index| ((index * 31 % 2039) as f32 - 1019.0) / 8.0)
            .collect();
        for dtype in [DType::F32, DType::F16, DType::BF16] {
            let attention = Tensor::from_vec(values.clone(), (batch, heads, length, dim), &device)?
                .to_dtype(dtype)?;
            let expected = attention
                .transpose(1, 2)?
                .reshape((batch, length, heads * dim))?
                .to_dtype(DType::F32)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            let actual = crate::kernels::metal::merge_heads(&attention)?;

            assert_eq!(actual.dims(), [batch, length, heads * dim]);
            assert_eq!(actual.dtype(), dtype);
            assert!(actual.is_contiguous());
            assert_eq!(
                actual
                    .to_dtype(DType::F32)?
                    .flatten_all()?
                    .to_vec1::<f32>()?,
                expected,
                "{dtype:?} must be exact"
            );
        }
        Ok(())
    }

    #[test]
    fn fused_metal_rope_qkv_matches_candle() -> Result<()> {
        let device = Device::new_metal(0)?;
        let (batch, length, heads, dim) = (2usize, 37usize, 16usize, 64usize);
        let rows = length + 11;
        let table = |offset: f64| {
            Tensor::arange(0u32, (rows * dim / 2) as u32, &device)
                .and_then(|t| t.to_dtype(DType::F32))
                .and_then(|t| t.affine(0.013, offset))
                .and_then(|t| t.sin())
                .and_then(|t| t.reshape((rows, dim / 2)))
        };
        let rotary = RotaryEmbedding {
            cos: table(0.0)?,
            sin: table(0.7)?,
        };
        let values: Vec<f32> = (0..batch * length * 3 * heads * dim)
            .map(|index| ((index * 53 % 211) as f32 - 105.0) / 37.0)
            .collect();

        for dtype in [DType::F32, DType::F16, DType::BF16] {
            let qkv = Tensor::from_vec(values.clone(), (batch, length, 3, heads, dim), &device)?
                .to_dtype(dtype)?;
            let (expected_q, expected_k, expected_v) = rotary.apply_split(&qkv)?;
            let (actual_q, actual_k, actual_v) = rotary.apply_qkv_metal(&qkv)?;

            for tensor in [&actual_q, &actual_k, &actual_v] {
                assert_eq!(tensor.dims(), [batch, heads, length, dim]);
                assert!(tensor.is_contiguous());
            }
            let tolerance = match dtype {
                DType::F32 => 1e-6,
                DType::F16 => 1e-2,
                _ => 1e-1,
            };
            for (actual, expected) in [
                (actual_q, expected_q),
                (actual_k, expected_k),
                (actual_v, expected_v),
            ] {
                let actual = actual
                    .to_dtype(DType::F32)?
                    .flatten_all()?
                    .to_vec1::<f32>()?;
                let expected = expected
                    .to_dtype(DType::F32)?
                    .flatten_all()?
                    .to_vec1::<f32>()?;
                let difference = actual
                    .iter()
                    .zip(expected)
                    .map(|(left, right)| (left - right).abs())
                    .fold(0f32, f32::max);
                assert!(
                    difference <= tolerance,
                    "{dtype:?}: maximum difference was {difference}"
                );
            }
        }
        Ok(())
    }

    #[test]
    fn fused_metal_geglu_matches_candle() -> Result<()> {
        let device = Device::new_metal(0)?;
        let rows = 97;
        let inner = 2624;
        let values: Vec<f32> = (0..rows * inner * 2)
            .map(|index| ((index * 37 % 2003) as f32 - 1001.0) / 97.0)
            .collect();
        for dtype in [DType::F32, DType::F16, DType::BF16] {
            let input =
                Tensor::from_vec(values.clone(), (rows, inner * 2), &device)?.to_dtype(dtype)?;
            let parts = input.chunk(2, D::Minus1)?;
            let expected = (&parts[0].gelu_erf()? * &parts[1])?;
            let actual = crate::kernels::metal::geglu(&input)?;

            assert_eq!(actual.dims(), expected.dims());
            assert_eq!(actual.dtype(), dtype);
            let expected = expected
                .to_dtype(DType::F32)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            let actual = actual
                .to_dtype(DType::F32)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            let tolerance = match dtype {
                DType::F32 => 1e-5,
                DType::F16 => 1e-1,
                _ => 5e-1,
            };
            let difference = expected
                .iter()
                .zip(actual)
                .map(|(left, right)| (left - right).abs())
                .fold(0f32, f32::max);
            assert!(
                difference <= tolerance,
                "{dtype:?}: maximum difference was {difference}"
            );
        }
        Ok(())
    }

    #[test]
    fn fused_metal_layer_norm_is_no_worse_than_the_fallback() -> Result<()> {
        let device = Device::new_metal(0)?;
        let hidden = 1024;
        let rows = 64;
        let values: Vec<f32> = (0..rows * hidden)
            .map(|index| ((index * 37 % 997) as f32 - 498.0) / 137.0)
            .collect();
        let weight_f32 = Tensor::arange(0u32, hidden as u32, &device)?
            .to_dtype(DType::F32)?
            .affine(1.0 / hidden as f64, 0.5)?;
        let flat = |tensor: Tensor| -> Result<Vec<f32>> {
            tensor.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()
        };
        let worst = |truth: &[f32], candidate: &[f32]| {
            truth
                .iter()
                .zip(candidate)
                .map(|(left, right)| (left - right).abs())
                .fold(0f32, f32::max)
        };

        for dtype in [DType::F32, DType::F16, DType::BF16] {
            let xs =
                Tensor::from_vec(values.clone(), (1, rows, hidden), &device)?.to_dtype(dtype)?;
            let weight = weight_f32.to_dtype(dtype)?;

            let truth = flat(
                LayerNorm::new_no_bias(weight.to_dtype(DType::F32)?, 1e-5)
                    .forward(&xs.to_dtype(DType::F32)?)?,
            )?;
            let fallback = flat(LayerNorm::new_no_bias(weight.clone(), 1e-5).forward(&xs)?)?;
            let zero = Tensor::zeros(hidden, dtype, &device)?;
            let candle_fused = flat(LayerNorm::new(weight.clone(), zero, 1e-5).forward(&xs)?)?;
            let fused = crate::kernels::metal::layer_norm(&xs, &weight, 1e-5)?;
            assert_eq!(fused.dims(), xs.dims());
            assert_eq!(fused.dtype(), dtype);
            let fused = flat(fused)?;

            let ours = worst(&truth, &fused);
            let theirs = worst(&truth, &candle_fused);
            let reference = worst(&truth, &fallback);

            assert!(
                ours <= reference.max(1e-6) * 1.5,
                "{dtype:?}: kernel is {ours} from f32, fallback is {reference}"
            );
            if dtype == DType::F32 {
                assert!(
                    ours < 1e-5,
                    "f32 should differ only by reduction order: {ours}"
                );
                continue;
            }
            assert!(
                ours < theirs,
                "{dtype:?}: kernel is {ours} from f32, candle's fused path is {theirs}; being \
                 closer than that is the whole point of this kernel"
            );
        }
        Ok(())
    }

    #[test]
    fn rotary_tables_resolve_every_position() -> Result<()> {
        let config: Config = serde_json::from_str(
            r#"{
                "vocab_size": 50368,
                "hidden_size": 1024,
                "num_hidden_layers": 28,
                "num_attention_heads": 16,
                "intermediate_size": 2624,
                "max_position_embeddings": 8192,
                "layer_norm_eps": 1e-05,
                "global_attn_every_n_layers": 3,
                "local_attention": 128,
                "rope_parameters": {
                    "full_attention": { "rope_theta": 160000.0 },
                    "sliding_attention": { "rope_theta": 10000.0 }
                }
            }"#,
        )
        .unwrap();
        let rotary = RotaryEmbedding::new(&config, 160_000.0, &Device::Cpu)?;
        assert_eq!(rotary.cos.dtype(), DType::F32);
        assert_eq!(rotary.sin.dtype(), DType::F32);

        for position in [1usize, 255, 257, 2047, 2049, 4095, 4097, 8190] {
            let row = rotary
                .sin
                .narrow(0, position, 1)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            let next = rotary
                .sin
                .narrow(0, position + 1, 1)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            let difference = row
                .iter()
                .zip(next)
                .map(|(left, right)| (left - right).abs())
                .fold(0f32, f32::max);
            assert!(
                difference > 1e-7,
                "positions {position} and {} share a table row",
                position + 1
            );
        }
        Ok(())
    }

    #[test]
    fn local_attention_mask_only_exposes_nearby_tokens() -> Result<()> {
        let mask = local_attention_mask(4, 1, DType::F32, &Device::Cpu)?.to_vec2::<f32>()?;

        assert_eq!(
            mask,
            vec![
                vec![0.0, 0.0, ATTENTION_MASK_VALUE, ATTENTION_MASK_VALUE],
                vec![0.0, 0.0, 0.0, ATTENTION_MASK_VALUE],
                vec![ATTENTION_MASK_VALUE, 0.0, 0.0, 0.0],
                vec![ATTENTION_MASK_VALUE, ATTENTION_MASK_VALUE, 0.0, 0.0],
            ]
        );
        Ok(())
    }

    #[test]
    fn global_attention_mask_hides_padding_tokens() -> Result<()> {
        let mask = Tensor::new(&[[1u32, 1, 0]], &Device::Cpu)?;
        let mask = global_attention_mask(&mask, 2, DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;

        assert_eq!(
            mask,
            vec![
                0.0,
                0.0,
                ATTENTION_MASK_VALUE,
                0.0,
                0.0,
                ATTENTION_MASK_VALUE
            ]
        );
        Ok(())
    }

    #[tokio::test]
    async fn fp32_hidden_states() -> anyhow::Result<()> {
        let path = crate::hub::download(
            "convaiinnovations/laya",
            "aa8c91ca088ec597df95a0d1c76b3063cb2ae5e8",
        )
        .await?;

        let device = crate::device::load()?;
        let config = Config::load(&path.join("encoder/config.json"))?;
        let weights = path.join("model.safetensors");

        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[weights], DType::F32, &device)? };
        let vb = vb.rename_f(|name| {
            name.strip_prefix("model.")
                .map(|name| format!("encoder.{name}"))
                .unwrap_or_else(|| name.to_owned())
        });
        let encoder = Encoder::load(vb, &config, DType::F32, AttentionImplementation::Eager)?;

        let tokenizer_path = path.join("tokenizer/tokenizer.json");
        let tokenizer = crate::tokenizer::from_json(&fs::read(tokenizer_path)?)?;
        let cls_id = tokenizer
            .token_to_id("[CLS]")
            .ok_or_else(|| anyhow::anyhow!("tokenizer is missing [CLS]"))?;
        let sep_id = tokenizer
            .token_to_id("[SEP]")
            .ok_or_else(|| anyhow::anyhow!("tokenizer is missing [SEP]"))?;
        let pad_id = tokenizer
            .token_to_id("[PAD]")
            .ok_or_else(|| anyhow::anyhow!("tokenizer is missing [PAD]"))?;

        let mut ids = vec![cls_id];
        ids.extend(tokenizer.encode("What is Deep Learning?")?);
        ids.push(sep_id);
        let valid_length = ids.len();

        let mut mask = vec![1f32; ids.len()];
        ids.resize(32, pad_id);
        mask.resize(32, 0.0);

        let length = ids.len();
        let ids = Tensor::from_vec(ids, (1, length), &device)?;
        let mask = Tensor::from_vec(mask, (1, length), &device)?;
        let hidden = encoder.forward(&ids, &mask, &[valid_length], true)?;

        let values = hidden.i((0, 0, 0..128))?.to_vec1::<f32>()?;
        insta::with_settings!({prepend_module_to_snapshot => false}, {
        insta::assert_yaml_snapshot!(
            "sys1__archs__modernbert__tests__modernbert_fp32_hidden_states",
            values,
            {
                "[]" => insta::rounded_redaction(3),
            }
        );
        });

        Ok(())
    }

    #[test]
    fn cpu_windowed_attention_matches_dense_attention() -> Result<()> {
        let device = Device::Cpu;
        let values = Tensor::arange(0u32, 48, &device)?
            .to_dtype(DType::F32)?
            .reshape((1, 2, 6, 4))?;
        let q = (&values / 37.0)?;
        let k = (&values / 29.0)?;
        let v = (&values / 19.0)?;
        let local = local_attention_mask(6, 2, DType::F32, &device)?;
        let padding = Tensor::new(&[[1f32, 1.0, 1.0, 1.0, 0.0, 0.0]], &device)?;
        let global = global_attention_mask(&padding, 6, DType::F32)?;

        for mask in [&local, &global.broadcast_add(&local)?] {
            let dense = eager_attention(&q, &k, &v, 0.5, Some(mask))?;
            let windowed = cpu_windowed_attention(&q, &k, &v, 0.5, Some(mask), 2)?;
            let dense = dense.flatten_all()?.to_vec1::<f32>()?;
            let windowed = windowed.flatten_all()?.to_vec1::<f32>()?;
            let difference = dense
                .iter()
                .zip(windowed)
                .map(|(left, right)| (left - right).abs())
                .fold(0f32, f32::max);
            assert!(difference < 1e-5, "maximum difference was {difference}");
        }

        Ok(())
    }

    #[test]
    fn eager_bf16_softmax_handles_padding_masks_without_nans() {
        let device = Device::Cpu;
        let scores = Tensor::new(&[[0f32, f32::NEG_INFINITY]], &device)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();

        let output = attention_softmax(&scores)
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();

        assert!(output.iter().all(|value| value.is_finite()));
        assert_eq!(output, vec![1.0, 0.0]);
    }

    #[test]
    fn bf16_attention_masks_remain_finite() {
        let device = Device::Cpu;
        let padding = Tensor::new(&[[1f32, 0.0]], &device).unwrap();
        let global = global_attention_mask(&padding, 2, DType::BF16)
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let local = local_attention_mask(4, 1, DType::BF16, &device)
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();

        assert!(global.iter().chain(&local).all(|value| value.is_finite()));
        assert!(global.iter().any(|value| *value < -1_000.0));
        assert!(local.iter().any(|value| *value < -1_000.0));
    }
}

#[cfg(feature = "cuda")]
mod cuda;
#[cfg(feature = "cuda")]
use cuda::*;

#[cfg(feature = "metal")]
pub(crate) mod metal;

use crate::models::AttentionImplementation;

use candle_core::{D, DType, Device, Result, Tensor};
use candle_nn::{Embedding, LayerNorm, Linear, Module, VarBuilder, embedding, ops::softmax};

use serde::Deserialize;
use std::{
    collections::HashMap,
    fs,
    path::Path,
    sync::{Arc, Mutex},
};

const ATTENTION_MASK_VALUE: f32 = -10_000.0;

fn layer_norm_no_bias(size: usize, eps: f64, vb: VarBuilder) -> Result<LayerNorm> {
    Ok(LayerNorm::new_no_bias(
        vb.get_with_hints(size, "weight", candle_nn::Init::Const(1.))?,
        eps,
    ))
}

#[derive(Deserialize)]
pub struct Config {
    vocab_size: usize,
    hidden_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    intermediate_size: usize,
    max_position_embeddings: usize,
    layer_norm_eps: f64,
    global_attn_every_n_layers: usize,
    local_attention: usize,
    rope_parameters: HashMap<String, RopeConfig>,
}

#[derive(Deserialize)]
struct RopeConfig {
    rope_theta: f64,
}

impl Config {
    pub fn load(path: &Path) -> anyhow::Result<Self> {
        Ok(serde_json::from_slice(&fs::read(path)?)?)
    }

    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    pub fn intermediate_size(&self) -> usize {
        self.intermediate_size
    }

    pub fn attention_heads(&self) -> usize {
        self.num_attention_heads
    }

    pub fn max_position_embeddings(&self) -> usize {
        self.max_position_embeddings
    }
}

struct RotaryEmbedding {
    sin: Tensor,
    cos: Tensor,
}

impl RotaryEmbedding {
    fn new(dtype: DType, config: &Config, rope_theta: f64, device: &Device) -> Result<Self> {
        let dim = config.hidden_size / config.num_attention_heads;
        let inv_freq: Vec<_> = (0..dim)
            .step_by(2)
            .map(|index| 1f32 / rope_theta.powf(index as f64 / dim as f64) as f32)
            .collect();
        let inv_freq_len = inv_freq.len();
        let inv_freq = Tensor::from_vec(inv_freq, (1, inv_freq_len), device)?.to_dtype(dtype)?;
        let positions = Tensor::arange(0u32, config.max_position_embeddings as u32, device)?
            .to_dtype(dtype)?
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
        let projected = projection_forward(&xs.to_dtype(self.compute_dtype)?, &self.qkv)?;

        let qkv = projected.reshape((batch, length, 3, self.heads, self.head_size))?;
        #[cfg(feature = "cuda")]
        let fused_rope = xs.device().is_cuda()
            && self.implementation != AttentionImplementation::Eager
            && matches!(qkv.dtype(), DType::F16 | DType::BF16)
            && self.rotary.cos.dtype() == DType::F32;
        #[cfg(feature = "cuda")]
        let (q, k, v) = if fused_rope {
            self.rotary.apply_qkv(&qkv)?
        } else {
            let qkv = qkv.permute((2, 0, 3, 1, 4))?;
            let q = qkv.get(0)?;
            let k = qkv.get(1)?;
            let v = qkv.get(2)?;
            let (q, k) = self.rotary.apply(&q, &k)?;
            (q, k, v)
        };
        #[cfg(not(feature = "cuda"))]
        let (q, k, v) = {
            let qkv = qkv.permute((2, 0, 3, 1, 4))?;
            let q = qkv.get(0)?;
            let k = qkv.get(1)?;
            let v = qkv.get(2)?;
            let (q, k) = self.rotary.apply(&q, &k)?;
            (q, k, v)
        };
        let scale = (self.head_size as f64).powf(-0.5);

        #[cfg(feature = "metal")]
        let attention =
            if xs.device().is_metal() && self.implementation != AttentionImplementation::Eager {
                let mask = mask
                    .map(|mask| mask.broadcast_as((batch, self.heads, length, length)))
                    .transpose()?;
                candle_nn::ops::sdpa(&q, &k, &v, mask.as_ref(), false, scale as f32, 1.0)?
            } else {
                scaled_dot_product_attention(
                    &q,
                    &k,
                    &v,
                    scale,
                    AttentionOptions {
                        mask,
                        implementation: self.implementation,
                        lengths,
                        window,
                    },
                )?
            };
        #[cfg(not(feature = "metal"))]
        let attention = scaled_dot_product_attention(
            &q,
            &k,
            &v,
            scale,
            AttentionOptions {
                mask,
                implementation: self.implementation,
                lengths,
                window,
            },
        )?;

        let mixed = attention
            .transpose(1, 2)?
            .reshape((batch, length, hidden))?;

        let output = projection_forward(&mixed, &self.projection)?;

        Ok(output)
    }
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
    #[cfg(feature = "metal")]
    if q.device().is_metal() {
        return metal::scaled_dot_product_attention(
            q,
            k,
            v,
            scale,
            metal::AttentionOptions {
                mask: options.mask,
                implementation: options.implementation,
                lengths: options.lengths,
                window: options.window,
            },
        );
    }

    if options.implementation == AttentionImplementation::Eager && q.device().is_cuda() {
        return padding_independent_eager_attention(
            q,
            k,
            v,
            scale,
            options.mask,
            options.lengths,
            options.window,
        );
    }
    match options.implementation {
        AttentionImplementation::Eager if q.device().is_cpu() && options.window.is_some() => {
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

fn padding_independent_eager_attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    scale: f64,
    mask: Option<&Tensor>,
    lengths: &[usize],
    window: Option<usize>,
) -> Result<Tensor> {
    let (batch, heads, width, _) = q.dims4()?;
    let (kb, kh, kw, _) = k.dims4()?;
    let (vb, vh, vw, value_dim) = v.dims4()?;
    if lengths.len() != batch
        || (kb, kh, kw) != (batch, heads, width)
        || (vb, vh, vw) != (batch, heads, width)
        || lengths.iter().any(|&length| length == 0 || length > width)
    {
        candle_core::bail!(
            "Padding-independent eager attention requires valid self-attention lengths/shapes"
        )
    }
    let mask = mask
        .map(|mask| mask.broadcast_as((batch, heads, width, width)))
        .transpose()?;
    let mut results = Vec::with_capacity(batch);
    for (slot, &length) in lengths.iter().enumerate() {
        let valid = |tensor: &Tensor| {
            tensor
                .narrow(0, slot, 1)?
                .narrow(2, 0, length)?
                .contiguous()
        };
        let mask = mask
            .as_ref()
            .map(|mask| {
                mask.narrow(0, slot, 1)?
                    .narrow(2, 0, length)?
                    .narrow(3, 0, length)?
                    .broadcast_as((1, heads, length, length))
            })
            .transpose()?;
        let q = valid(q)?;
        let k = valid(k)?;
        let v = valid(v)?;
        let output = match window {
            Some(window) => cpu_windowed_attention(&q, &k, &v, scale, mask.as_ref(), window)?,
            None => eager_attention(&q, &k, &v, scale, mask.as_ref())?,
        };
        results.push(if length < width {
            let padding = Tensor::zeros(
                (1, heads, width - length, value_dim),
                output.dtype(),
                output.device(),
            )?;
            Tensor::cat(&[&output, &padding], 2)?
        } else {
            output
        });
    }
    Tensor::cat(&results, 0)
}

fn eager_attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    scale: f64,
    mask: Option<&Tensor>,
) -> Result<Tensor> {
    if q.device().is_cuda() {
        return chunked_eager_attention(q, k, v, scale, mask, 128);
    }
    eager_attention_full(q, k, v, scale, mask)
}

fn eager_attention_full(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    scale: f64,
    mask: Option<&Tensor>,
) -> Result<Tensor> {
    let dtype = q.dtype();
    let opmath = if matches!(dtype, DType::F16 | DType::BF16) {
        DType::F32
    } else {
        dtype
    };
    let q = q.to_dtype(opmath)?;
    let k = k.to_dtype(opmath)?;
    let v = v.to_dtype(opmath)?;

    let operand_scale = scale.abs().sqrt();
    let q = (&q
        * if scale < 0.0 {
            -operand_scale
        } else {
            operand_scale
        })?;
    let k = (&k * operand_scale)?;
    let scores = q.matmul(&k.transpose(D::Minus2, D::Minus1)?)?;
    let scores = match mask {
        Some(mask) => scores.broadcast_add(&mask.to_dtype(opmath)?)?,
        None => scores,
    };
    softmax(&scores, D::Minus1)?.matmul(&v)?.to_dtype(dtype)
}

fn chunked_eager_attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    scale: f64,
    mask: Option<&Tensor>,
    query_chunk: usize,
) -> Result<Tensor> {
    if query_chunk == 0 {
        candle_core::bail!("Eager attention query chunk must be positive")
    }
    let dtype = q.dtype();
    let opmath = if matches!(dtype, DType::F16 | DType::BF16) {
        DType::F32
    } else {
        dtype
    };
    let (batch, heads, queries, _) = q.dims4()?;
    let keys = k.dim(2)?;
    let operand_scale = scale.abs().sqrt();
    let q = (q.to_dtype(opmath)?
        * if scale < 0.0 {
            -operand_scale
        } else {
            operand_scale
        })?;
    let k = (k.to_dtype(opmath)? * operand_scale)?.transpose(D::Minus2, D::Minus1)?;
    let v = v.to_dtype(opmath)?;
    let mask = mask
        .map(|mask| mask.broadcast_as((batch, heads, queries, keys)))
        .transpose()?;
    let mut output = Vec::with_capacity(queries.div_ceil(query_chunk));
    for start in (0..queries).step_by(query_chunk) {
        let count = query_chunk.min(queries - start);
        let scores = q.narrow(2, start, count)?.contiguous()?.matmul(&k)?;
        let scores = match &mask {
            Some(mask) => scores.broadcast_add(&mask.narrow(2, start, count)?.to_dtype(opmath)?)?,
            None => scores,
        };
        output.push(softmax(&scores, D::Minus1)?.matmul(&v)?.to_dtype(dtype)?);
    }
    Tensor::cat(&output.iter().collect::<Vec<_>>(), 2)
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

#[cfg(not(feature = "cuda"))]
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
        let projected = projection_forward(&xs.to_dtype(self.compute_dtype)?, &self.input)?;

        #[cfg(feature = "cuda")]
        if projected.device().is_cuda() {
            let product = geglu(&projected)?;

            return projection_forward(&product, &self.output);
        }
        let parts = projected.chunk(2, D::Minus1)?;
        let product = (&crate::kernels::activation::gelu_erf(&parts[0])? * &parts[1])?;

        crate::kernels::linear::forward_compatible(&product, &self.output)
    }
}

fn projection_forward(input: &Tensor, linear: &Linear) -> Result<Tensor> {
    #[cfg(feature = "cuda")]
    if input.device().is_cuda() && input.dtype() == DType::F16 && linear.bias().is_none() {
        if linear.weight().dtype() != DType::F16
            || !input.device().same_device(linear.weight().device())
        {
            candle_core::bail!("Stable FP16 projection requires matching weight dtype/device")
        }
        return input
            .contiguous()?
            .apply_op2_no_bwd(&linear.weight().contiguous()?, &StableF16Projection);
    }
    crate::kernels::linear::forward_compatible(input, linear)
}

struct Layer {
    attention: Attention,
    mlp: Mlp,
    attention_norm: Option<LayerNorm>,
    mlp_norm: LayerNorm,
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
            attention_norm: layer_norm_no_bias(
                config.hidden_size,
                config.layer_norm_eps,
                vb.pp("attn_norm"),
            )
            .ok(),
            mlp_norm: layer_norm_no_bias(
                config.hidden_size,
                config.layer_norm_eps,
                vb.pp("mlp_norm"),
            )?,
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

        let mask = match (self.uses_local_attention, local_mask, global_mask) {
            (true, Some(local_mask), Some(global_mask)) => {
                Some(global_mask.broadcast_add(local_mask)?)
            }
            (true, Some(local_mask), None) => Some(local_mask.clone()),
            (false, _, global_mask) => global_mask.cloned(),
            (true, None, _) => None,
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
        let normalized = xs.apply(&self.mlp_norm)?;

        let mlp = self.mlp.forward(&normalized)?;

        let dtype = xs.dtype();
        xs + mlp.to_dtype(dtype)?
    }
}

pub struct Encoder {
    embeddings: Embedding,
    norm: LayerNorm,
    layers: Vec<Layer>,
    final_norm: LayerNorm,
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
            vb.dtype(),
            config,
            config.rope_parameters["full_attention"].rope_theta,
            vb.device(),
        )?);
        let local_rotary = Arc::new(RotaryEmbedding::new(
            vb.dtype(),
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
            norm: layer_norm_no_bias(
                config.hidden_size,
                config.layer_norm_eps,
                vb.pp("model.embeddings.norm"),
            )?,
            layers,
            final_norm: layer_norm_no_bias(
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
    #[cfg(feature = "metal")]
    use crate::architectures::metal::ModernBertEncoder as Encoder;

    #[test]
    fn natural_geometry_eager_attention_preserves_valid_tokens() -> Result<()> {
        let device = Device::Cpu;
        let lengths = [3usize, 5];
        let q = Tensor::arange(0f32, 40f32, &device)?.reshape((2, 1, 5, 4))?;
        let q = (&q / 31.)?;
        let k = (&q * 0.7)?;
        let v = (&q + 0.2)?;
        let keep = Tensor::new(&[[1f32, 1., 1., 0., 0.], [1., 1., 1., 1., 1.]], &device)?;
        let global = global_attention_mask(&keep, 5, DType::F32)?;
        let local = local_attention_mask(5, 2, DType::F32, &device)?;
        for mask in [&global, &global.broadcast_add(&local)?] {
            let actual =
                padding_independent_eager_attention(&q, &k, &v, 0.5, Some(mask), &lengths, None)?;
            for (slot, &length) in lengths.iter().enumerate() {
                let valid = |tensor: &Tensor| {
                    tensor
                        .narrow(0, slot, 1)?
                        .narrow(2, 0, length)?
                        .contiguous()
                };
                let valid_mask = mask
                    .broadcast_as((2, 1, 5, 5))?
                    .narrow(0, slot, 1)?
                    .narrow(2, 0, length)?
                    .narrow(3, 0, length)?
                    .contiguous()?;
                let expected = eager_attention(
                    &valid(&q)?,
                    &valid(&k)?,
                    &valid(&v)?,
                    0.5,
                    Some(&valid_mask),
                )?;
                assert_eq!(
                    valid(&actual)?.flatten_all()?.to_vec1::<f32>()?,
                    expected.flatten_all()?.to_vec1::<f32>()?
                );
            }
        }
        assert!(padding_independent_eager_attention(&q, &k, &v, 0.5, None, &[0, 5], None).is_err());
        Ok(())
    }
    use candle_core::IndexOp;

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
        // SAFETY: All file-backed mmap constructors are marked `unsafe` because of the potential
        // for undefined behavior using the map if the underlying file is modified or out of
        // process.
        //
        // More information at https://github.com/RazrFalcon/memmap2-rs/blob/a02e2a/src/lib.rs#L135-L165
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

        insta::assert_yaml_snapshot!(
            "modernbert_fp32_hidden_states",
            hidden.i((0, 0, 0..128))?.to_vec1::<f32>()?,
            {
                "[]" => insta::rounded_redaction(3),
            }
        );

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
    fn eager_bf16_attention_handles_padding_masks_without_nans() {
        let device = Device::Cpu;
        let q = Tensor::zeros((1, 1, 1, 1), DType::BF16, &device).unwrap();
        let k = Tensor::zeros((1, 1, 2, 1), DType::BF16, &device).unwrap();
        let v = Tensor::new(&[2f32, 3.], &device)
            .unwrap()
            .reshape((1, 1, 2, 1))
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();
        let mask = Tensor::new(&[0f32, f32::NEG_INFINITY], &device)
            .unwrap()
            .reshape((1, 1, 1, 2))
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();
        let output = eager_attention(&q, &k, &v, 1., Some(&mask))
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();

        assert!(output.iter().all(|value| value.is_finite()));
        assert_eq!(output, vec![2.0]);
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

    #[test]
    #[ignore = "external reference data"]
    fn eager_opmath_matches_native_sdpa_fixtures() -> Result<()> {
        let root = crate::assets::fixtures("sdpa")?;
        let device =
            crate::device::load().map_err(|error| candle_core::Error::Msg(error.to_string()))?;
        for (dtype, rtol) in [("bf16", 0.016f32), ("f16", 0.001), ("f32", 1.3e-6)] {
            #[cfg(feature = "cuda")]
            if dtype == "bf16"
                && let Device::Cuda(cuda) = &device
            {
                let (major, _) = cuda
                    .cuda_stream()
                    .context()
                    .compute_capability()
                    .map_err(|error| candle_core::Error::Msg(error.to_string()))?;
                if major < 8 {
                    continue;
                }
            }
            for mask in ["masked", "unmasked"] {
                let data = candle_core::safetensors::load(
                    root.join(format!("b2-h3-l11-d64-{mask}-{dtype}.safetensors")),
                    &device,
                )?;
                let actual = eager_attention(
                    &data["q"],
                    &data["k"],
                    &data["v"],
                    0.125,
                    Some(&data["mask"]),
                )?;
                assert_eq!(actual.dtype(), data["expected"].dtype());
                assert_eq!(actual.dims(), data["expected"].dims());
                let actual = actual
                    .to_dtype(DType::F32)?
                    .flatten_all()?
                    .to_vec1::<f32>()?;
                let expected = data["expected"]
                    .to_dtype(DType::F32)?
                    .flatten_all()?
                    .to_vec1::<f32>()?;
                for (index, (actual, expected)) in actual.iter().zip(&expected).enumerate() {
                    assert!(
                        actual.is_finite()
                            && (*actual - expected).abs() <= 1e-5 + rtol * expected.abs(),
                        "{mask}/{dtype} native math SDPA at{index}: {actual} vs {expected}"
                    );
                }
            }
        }
        Ok(())
    }
}

pub(crate) mod batching;
#[cfg(feature = "cuda")]
mod cuda;
pub(crate) mod inference;
#[cfg(feature = "cuda")]
pub(crate) use cuda::compensated_bf16_projection;
#[cfg(all(test, feature = "cuda"))]
use cuda::fixed_bf16_projection;
#[cfg(feature = "cuda")]
use cuda::*;

#[cfg(feature = "metal")]
pub(crate) mod metal;

use crate::kernels::activation::sigmoid;
use crate::models::AttentionImplementation;
use candle_core::{D, DType, Result, Tensor};
use candle_nn::{Embedding, Linear as CandleLinear, VarBuilder, embedding, ops::softmax_last_dim};
use serde::Deserialize;
use std::{fs, path::Path};

#[derive(Clone, Debug, Deserialize)]
pub struct Config {
    pub dtype: Option<String>,
    pub text_config: TextConfig,
    pub vision_config: VisionConfig,
    pub image_token_id: u32,
    pub video_token_id: u32,
}

#[derive(Clone, Debug, Deserialize)]
pub struct TextConfig {
    #[serde(skip, default = "causal_attention")]
    pub full_attention_causal: bool,
    pub dtype: Option<String>,
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub layer_types: Vec<String>,
    pub linear_num_key_heads: usize,
    pub linear_num_value_heads: usize,
    pub linear_key_head_dim: usize,
    pub linear_value_head_dim: usize,
    pub linear_conv_kernel_dim: usize,
    pub rms_norm_eps: f64,
    pub rope_parameters: RopeConfig,
}

#[derive(Clone, Debug, Deserialize)]
pub struct RopeConfig {
    pub rope_theta: f64,
    pub partial_rotary_factor: f64,
    pub mrope_section: [usize; 3],
}

#[derive(Clone, Debug, Deserialize)]
pub struct VisionConfig {
    pub dtype: Option<String>,
    pub depth: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_heads: usize,
    pub in_channels: usize,
    pub patch_size: usize,
    pub temporal_patch_size: usize,
    pub spatial_merge_size: usize,
    pub out_hidden_size: usize,
    pub num_position_embeddings: usize,
}

fn causal_attention() -> bool {
    true
}

impl Config {
    pub fn load(path: &Path) -> anyhow::Result<Self> {
        let config: Self = serde_json::from_slice(&fs::read(path)?)?;
        anyhow::ensure!(
            config.text_config.layer_types.len() == config.text_config.num_hidden_layers,
            "Qwen3.5 layer_types count does not match num_hidden_layers"
        );
        Ok(config)
    }
}

fn linear(input: usize, output: usize, vb: VarBuilder) -> Result<Linear> {
    let result = Linear::new(vb.get((output, input), "weight")?, None);
    Ok(result)
}

struct RmsNorm {
    #[cfg(feature = "cuda")]
    weight: Tensor,
    scale: Tensor,
    eps: f64,
}

impl RmsNorm {
    fn load(width: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        let weight = vb.get(width, "weight")?;
        let scale = (&weight.to_dtype(DType::F32)? + 1.)?;
        Ok(Self {
            #[cfg(feature = "cuda")]
            weight,
            scale,
            eps,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        #[cfg(feature = "cuda")]
        if x.device().is_cuda() && x.dtype() == DType::BF16 {
            return cuda_rms_norm_delta(x, &self.weight, self.eps);
        }
        let dtype = x.dtype();
        let x = x.to_dtype(DType::F32)?;
        let variance = x.sqr()?.mean_keepdim(D::Minus1)?;
        let scale = (variance + self.eps)?.sqrt()?.recip()?;
        x.broadcast_mul(&scale)?
            .broadcast_mul(&self.scale)?
            .to_dtype(dtype)
    }

    fn forward_residual(&self, x: &Tensor, mixed: &Tensor) -> Result<(Tensor, Tensor)> {
        #[cfg(feature = "cuda")]
        if x.device().is_cuda()
            && x.dtype() == DType::BF16
            && mixed.dtype() == DType::BF16
            && self.weight.dtype() == DType::BF16
            && x.dims() == mixed.dims()
        {
            return cuda_residual_rms_norm_delta(x, mixed, &self.weight, self.eps);
        }
        let residual = (x + mixed)?;
        let normalized = self.forward(&residual)?;
        Ok((residual, normalized))
    }
}

struct Mlp {
    #[cfg(all(feature = "cuda", sys1_cuda_direct_ffn))]
    direct_weight: Option<crate::kernels::cuda::sm86::fused_gate_up::Prepared>,
    #[cfg(all(feature = "cuda", sys1_cuda_fused_ffn))]
    fused_weight: std::sync::Arc<std::sync::Mutex<Option<Tensor>>>,
    gate: Linear,
    up: Linear,
    down: Linear,
}

impl Mlp {
    fn load(config: &TextConfig, vb: VarBuilder) -> Result<Self> {
        let gate = linear(
            config.hidden_size,
            config.intermediate_size,
            vb.pp("gate_proj"),
        )?;
        let up = linear(
            config.hidden_size,
            config.intermediate_size,
            vb.pp("up_proj"),
        )?;
        let down = linear(
            config.intermediate_size,
            config.hidden_size,
            vb.pp("down_proj"),
        )?;
        #[cfg(all(feature = "cuda", sys1_cuda_direct_ffn))]
        let direct_weight =
            if crate::kernels::cuda::sm86::fused_gate_up::available(gate.weight(), up.weight()) {
                Some(crate::kernels::cuda::sm86::fused_gate_up::prepare(
                    gate.weight(),
                    up.weight(),
                )?)
            } else {
                None
            };
        Ok(Self {
            #[cfg(all(feature = "cuda", sys1_cuda_direct_ffn))]
            direct_weight,
            #[cfg(all(feature = "cuda", sys1_cuda_fused_ffn))]
            fused_weight: Default::default(),
            gate,
            up,
            down,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let fused_product: Option<Tensor> = {
            #[cfg(all(feature = "cuda", sys1_cuda_fused_ffn))]
            {
                let rows = x.elem_count() / x.dim(D::Minus1)?.max(1);
                if x.device().is_cuda() && x.dtype() == DType::BF16 && (512..=32768).contains(&rows)
                {
                    if let Some(weight) = self.cached_fused()? {
                        let result = crate::kernels::cuda::fused_gate_up::forward(x, &weight)?;
                        #[cfg(test)]
                        FUSED_MLP_CALLS.with(|slot| slot.set(slot.get() + 1));

                        Some(result)
                    } else {
                        None
                    }
                } else {
                    None
                }
            }
            #[cfg(not(all(feature = "cuda", sys1_cuda_fused_ffn)))]
            {
                None
            }
        };
        #[cfg(all(feature = "cuda", sys1_cuda_direct_ffn))]
        let fused_product = match fused_product {
            Some(product) => Some(product),
            None => self.direct_product(x)?,
        };
        let product = if let Some(product) = fused_product {
            product
        } else if x.device().is_cuda() && x.dtype() == DType::BF16 {
            let gate = linear_attention_projection_forward(x, &self.gate)?;

            let up = linear_attention_projection_forward(x, &self.up)?;

            crate::kernels::activation::mul_silu(&gate, &up)?
        } else {
            let activated = {
                let gate = linear_attention_projection_forward(x, &self.gate)?;

                crate::kernels::activation::silu(&gate)?
            };
            let up = linear_attention_projection_forward(x, &self.up)?;

            (activated * up)?
        };

        #[cfg(feature = "cuda")]
        if product.device().is_cuda() && product.dtype() == DType::BF16 {
            #[cfg(feature = "cuda")]
            if let Some(output) = try_nn_projection(&product, &self.down)? {
                return Ok(output);
            }
            let output = cuda_bf16_projection(&product, &self.down, false)?;

            return Ok(output);
        }
        crate::kernels::linear::forward_compatible(&product, &self.down)
    }
}

struct Rotary {
    dimension: usize,
    divisors: Vec<f32>,
    sections: [usize; 3],
}

struct RotaryTensors {
    dimension: usize,
    cos: Tensor,
    sin: Tensor,
}

impl Rotary {
    fn new(dimension: usize, theta: f64, sections: [usize; 3]) -> Result<Self> {
        if !dimension.is_multiple_of(2) || sections.iter().sum::<usize>() != dimension / 2 {
            candle_core::bail!("Qwen3.5 mRoPE sections do not match an even rotary dimension")
        }
        Ok(Self {
            dimension,
            divisors: (0..dimension / 2)
                .map(|index| theta.powf(2. * index as f64 / dimension as f64) as f32)
                .collect(),
            sections,
        })
    }

    fn prepare(
        &self,
        positions: &[[u32; 3]],
        dtype: DType,
        device: &candle_core::Device,
    ) -> Result<RotaryTensors> {
        let frequencies = positions
            .iter()
            .flat_map(|position| {
                self.divisors
                    .iter()
                    .enumerate()
                    .map(move |(index, divisor)| {
                        let axis = if index < self.sections[1] * 3 && index % 3 == 1 {
                            1
                        } else if index < self.sections[2] * 3 && index % 3 == 2 {
                            2
                        } else {
                            0
                        };
                        position[axis] as f32 / divisor
                    })
            })
            .collect::<Vec<_>>();
        let phase = Tensor::from_vec(
            frequencies,
            (1, positions.len(), 1, self.dimension / 2),
            device,
        )?;
        let cos = phase.cos()?;
        let sin = phase.sin()?;
        Ok(RotaryTensors {
            dimension: self.dimension,
            cos: Tensor::cat(&[&cos, &cos], D::Minus1)?.to_dtype(dtype)?,
            sin: Tensor::cat(&[&sin, &sin], D::Minus1)?.to_dtype(dtype)?,
        })
    }
}

impl RotaryTensors {
    fn apply(&self, x: &Tensor) -> Result<Tensor> {
        let (_, length, _, width) = x.dims4()?;
        if length != self.cos.dim(1)? || width < self.dimension {
            candle_core::bail!("Qwen3.5 rotary tensor shape does not match attention input")
        }
        let active = x.narrow(D::Minus1, 0, self.dimension)?;
        let first = active.narrow(D::Minus1, 0, self.dimension / 2)?;
        let second = active.narrow(D::Minus1, self.dimension / 2, self.dimension / 2)?;
        let rotated = Tensor::cat(&[&second.neg()?, &first], D::Minus1)?;
        let active = (active.broadcast_mul(&self.cos)? + rotated.broadcast_mul(&self.sin)?)?;
        Tensor::cat(
            &[
                &active,
                &x.narrow(D::Minus1, self.dimension, width - self.dimension)?,
            ],
            D::Minus1,
        )
    }
}

struct FullAttention {
    causal: bool,
    #[cfg(feature = "cuda")]
    merged_qkv: Option<Linear>,
    q: Linear,
    k: Linear,
    v: Linear,
    output: Linear,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    rotary: Rotary,
    implementation: AttentionImplementation,
}

fn full_attention_kv_forward(input: &Tensor, linear: &Linear) -> Result<Tensor> {
    #[cfg(feature = "cuda")]
    if let Some(output) = try_nn_projection(input, linear)? {
        return Ok(output);
    }

    #[cfg(feature = "cuda")]
    if input.device().is_cuda() && input.dtype() == DType::BF16 {
        return cuda_bf16_projection(input, linear, false);
    }
    crate::kernels::linear::forward_compatible(input, linear)
}

fn linear_attention_projection_forward(input: &Tensor, linear: &Linear) -> Result<Tensor> {
    #[cfg(feature = "cuda")]
    if let Some(output) = try_nn_projection(input, linear)? {
        return Ok(output);
    }

    #[cfg(feature = "cuda")]
    if input.device().is_cuda() && input.dtype() == DType::BF16 {
        return cuda_bf16_projection(input, linear, false);
    }
    crate::kernels::linear::forward_compatible(input, linear)
}

impl FullAttention {
    fn project_finish(
        &self,
        x: &Tensor,
        rotary: &RotaryTensors,
    ) -> Result<(Tensor, Tensor, Tensor, Tensor)> {
        let (batch, length, _) = x.dims3()?;
        #[cfg(feature = "cuda")]
        if matches!(
            option_env!("SYS1_CUDA_COMPUTE_CAP"),
            Some("80" | "90" | "120")
        ) && x.device().is_cuda()
            && x.dtype() == DType::BF16
            && self.q_norm.weight.dtype() == DType::BF16
            && self.k_norm.weight.dtype() == DType::BF16
            && (32..=1024).contains(&self.head_dim)
            && rotary.dimension <= self.head_dim
            && rotary.dimension.is_multiple_of(2)
            && rotary.cos.dtype() == DType::BF16
            && rotary.sin.dtype() == DType::BF16
        {
            if let Some(merged) = &self.merged_qkv
                && (16_384..=65_536).contains(&x.elem_count().checked_div(x.dim(2)?).unwrap_or(0))
            {
                let projected = crate::kernels::cuda::blaslt::project_unblocked(x, merged)?;
                #[cfg(test)]
                FULL_QKV_MERGED_SELECTIONS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

                let (q, k, v, gate) = cuda_full_qkv_finish(
                    &projected,
                    None,
                    (&self.q_norm.weight, &self.k_norm.weight),
                    rotary,
                    self.heads,
                    self.kv_heads,
                    self.q_norm.eps,
                )?;

                return Ok((q, k, v, gate));
            }

            let qg = linear_attention_projection_forward(x, &self.q)?;
            let k = full_attention_kv_forward(x, &self.k)?;
            let v = full_attention_kv_forward(x, &self.v)?;

            let (q, k, v, gate) = cuda_full_qkv_finish(
                &qg,
                Some((&k, &v)),
                (&self.q_norm.weight, &self.k_norm.weight),
                rotary,
                self.heads,
                self.kv_heads,
                self.q_norm.eps,
            )?;

            return Ok((q, k, v, gate));
        }
        let qg = linear_attention_projection_forward(x, &self.q)?.reshape((
            batch,
            length,
            self.heads,
            2,
            self.head_dim,
        ))?;

        let q = full_attention_norm_rotary(&qg.narrow(3, 0, 1)?.squeeze(3)?, &self.q_norm, rotary)?;
        let gate = sigmoid(&qg.narrow(3, 1, 1)?.squeeze(3)?.reshape((
            batch,
            length,
            self.heads * self.head_dim,
        ))?)?;
        let k_projected = full_attention_kv_forward(x, &self.k)?.reshape((
            batch,
            length,
            self.kv_heads,
            self.head_dim,
        ))?;

        let k = full_attention_norm_rotary(&k_projected, &self.k_norm, rotary)?;
        let v = full_attention_kv_forward(x, &self.v)?.reshape((
            batch,
            length,
            self.kv_heads,
            self.head_dim,
        ))?;

        Ok((q, k, v, gate))
    }

    fn load(
        config: &TextConfig,
        vb: VarBuilder,
        implementation: AttentionImplementation,
    ) -> Result<Self> {
        let rotary_dim =
            (config.head_dim as f64 * config.rope_parameters.partial_rotary_factor) as usize;
        let rotary = Rotary::new(
            rotary_dim,
            config.rope_parameters.rope_theta,
            config.rope_parameters.mrope_section,
        )?;
        let q = linear(
            config.hidden_size,
            config.num_attention_heads * config.head_dim * 2,
            vb.pp("q_proj"),
        )?;
        let k = linear(
            config.hidden_size,
            config.num_key_value_heads * config.head_dim,
            vb.pp("k_proj"),
        )?;
        let v = linear(
            config.hidden_size,
            config.num_key_value_heads * config.head_dim,
            vb.pp("v_proj"),
        )?;
        #[cfg(feature = "cuda")]
        let (q, k, v, merged_qkv) = {
            let geometry = config.hidden_size == 4096
                && config.num_attention_heads == 16
                && config.num_key_value_heads == 4
                && config.head_dim == 256
                && rotary_dim == 64
                && q.weight().dtype() == DType::BF16
                && q.bias().is_none()
                && k.bias().is_none()
                && v.bias().is_none();
            let hopper = match q.weight().device() {
                candle_core::Device::Cuda(device) => geometry && is_hopper(device)?,
                _ => false,
            };
            if hopper {
                let q_columns = q.weight().dim(0)?;
                let k_columns = k.weight().dim(0)?;
                let v_columns = v.weight().dim(0)?;
                let packed = Tensor::cat(&[q.weight(), k.weight(), v.weight()], 0)?;
                let q = Linear::new(packed.narrow(0, 0, q_columns)?, None);
                let k = Linear::new(packed.narrow(0, q_columns, k_columns)?, None);
                let v = Linear::new(packed.narrow(0, q_columns + k_columns, v_columns)?, None);
                (q, k, v, Some(Linear::new(packed, None)))
            } else {
                (q, k, v, None)
            }
        };
        Ok(Self {
            causal: config.full_attention_causal,
            #[cfg(feature = "cuda")]
            merged_qkv,
            q,
            k,
            v,
            output: linear(
                config.num_attention_heads * config.head_dim,
                config.hidden_size,
                vb.pp("o_proj"),
            )?,
            q_norm: RmsNorm::load(config.head_dim, config.rms_norm_eps, vb.pp("q_norm"))?,
            k_norm: RmsNorm::load(config.head_dim, config.rms_norm_eps, vb.pp("k_norm"))?,
            heads: config.num_attention_heads,
            kv_heads: config.num_key_value_heads,
            head_dim: config.head_dim,
            rotary,
            implementation,
        })
    }

    fn forward(&self, x: &Tensor, rotary: &RotaryTensors) -> Result<Tensor> {
        let (batch, length, _) = x.dims3()?;
        let (q, k, v, gate) = self.project_finish(x, rotary)?;
        let scale = (self.head_dim as f32).powf(-0.5);
        #[cfg(feature = "metal")]
        if x.device().is_metal()
            && x.dtype() != DType::BF16
            && self.implementation != AttentionImplementation::Eager
        {
            let attention = candle_nn::ops::sdpa(
                &q.transpose(1, 2)?.contiguous()?,
                &k.transpose(1, 2)?.contiguous()?,
                &v.transpose(1, 2)?.contiguous()?,
                None,
                self.causal,
                scale,
                1.0,
            )?
            .transpose(1, 2)?;

            let gated = full_attention_gate_mul(
                &attention,
                &gate,
                batch,
                length,
                self.heads * self.head_dim,
            )?;

            return gated.apply(&self.output);
        }
        let attention = attention_forward(&q, &k, &v, scale, self.causal, self.implementation)?;

        let gated =
            full_attention_gate_mul(&attention, &gate, batch, length, self.heads * self.head_dim)?;

        linear_attention_projection_forward(&gated, &self.output)
    }
}

pub(crate) fn attention_forward(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    scale: f32,
    causal: bool,
    implementation: AttentionImplementation,
) -> Result<Tensor> {
    let implementation = if implementation == AttentionImplementation::Auto {
        crate::models::attention_for_device(q.dtype(), q.device())
            .map_err(|error| candle_core::Error::Msg(error.to_string()))?
    } else {
        implementation
    };
    match implementation {
        AttentionImplementation::Auto | AttentionImplementation::Eager => {
            eager_attention(q, k, v, scale, causal)
        }
        #[cfg(feature = "cuda")]
        AttentionImplementation::FlashAttention2
        | AttentionImplementation::FlashAttentionBlackwell => {
            crate::kernels::cuda::attention::flash_attn(
                &q.contiguous()?,
                &k.contiguous()?,
                &v.contiguous()?,
                scale,
                causal,
            )
        }
        #[allow(unreachable_patterns)]
        other => candle_core::bail!("{} support is not compiled in", other.cli_name()),
    }
}

pub(crate) fn eager_attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    scale: f32,
    causal: bool,
) -> Result<Tensor> {
    let (batch, q_length, heads, width) = q.dims4()?;
    let (k_batch, k_length, kv_heads, k_width) = k.dims4()?;
    if kv_heads == 0
        || heads == 0
        || q_length == 0
        || k_length == 0
        || k_batch != batch
        || k_width != width
        || heads % kv_heads != 0
        || k.dims() != v.dims()
        || (causal && q_length != k_length)
    {
        candle_core::bail!("Unsupported eager attention dimensions")
    }
    let expand = |tensor: &Tensor| -> Result<Tensor> {
        tensor
            .to_dtype(DType::F32)?
            .unsqueeze(3)?
            .broadcast_as((batch, k_length, kv_heads, heads / kv_heads, width))?
            .reshape((batch, k_length, heads, width))?
            .transpose(1, 2)
    };
    let query = q.to_dtype(DType::F32)?.transpose(1, 2)?;
    let key = expand(k)?.transpose(D::Minus2, D::Minus1)?;
    let value = expand(v)?;

    const QUERY_CHUNK: usize = 256;
    let mut chunks = Vec::with_capacity(q_length.div_ceil(QUERY_CHUNK));
    for start in (0..q_length).step_by(QUERY_CHUNK) {
        let count = (q_length - start).min(QUERY_CHUNK);
        let mut scores =
            (contiguous_matmul(&query.narrow(2, start, count)?, &key)? * scale as f64)?;
        if causal {
            let mask = Tensor::from_vec(
                (start..start + count)
                    .flat_map(|row| {
                        (0..k_length).map(move |col| if col > row { f32::NEG_INFINITY } else { 0. })
                    })
                    .collect::<Vec<_>>(),
                (1, 1, count, k_length),
                q.device(),
            )?;
            scores = scores.broadcast_add(&mask)?;
        }
        let probabilities = softmax_last_dim(&scores.contiguous()?)?;
        chunks.push(contiguous_matmul(&probabilities, &value)?);
    }
    Tensor::cat(&chunks.iter().collect::<Vec<_>>(), 2)?
        .transpose(1, 2)?
        .to_dtype(q.dtype())
}

struct LinearAttention {
    #[cfg(feature = "cuda")]
    merged_hopper_qkvz: Option<Linear>,
    #[cfg(all(feature = "cuda", sys1_cuda_fused_ffn))]
    qkvz_weight: std::sync::Mutex<Option<Tensor>>,
    qkv: Linear,
    z: Linear,
    b: Linear,
    a: Linear,
    #[cfg(feature = "cuda")]
    ba: Option<Linear>,
    conv: Tensor,
    a_log: Tensor,
    dt_bias: Tensor,
    norm: Tensor,
    output: Linear,
    key_heads: usize,
    value_heads: usize,
    key_dim: usize,
    value_dim: usize,
    kernel: usize,
    eps: f64,
}

impl LinearAttention {
    fn load(config: &TextConfig, vb: VarBuilder) -> Result<Self> {
        let key_width = config.linear_num_key_heads * config.linear_key_head_dim;
        let value_width = config.linear_num_value_heads * config.linear_value_head_dim;
        let b = vb.pp("in_proj_b").get(
            (config.linear_num_value_heads, config.hidden_size),
            "weight",
        )?;
        let a = vb.pp("in_proj_a").get(
            (config.linear_num_value_heads, config.hidden_size),
            "weight",
        )?;
        #[cfg(feature = "cuda")]
        let ba = if vb.device().is_cuda() && vb.dtype() == DType::BF16 {
            Some(Linear::new(Tensor::cat(&[&b, &a], 0)?, None))
        } else {
            None
        };
        let qkv = linear(
            config.hidden_size,
            key_width * 2 + value_width,
            vb.pp("in_proj_qkv"),
        )?;
        let z = linear(config.hidden_size, value_width, vb.pp("in_proj_z"))?;
        #[cfg(feature = "cuda")]
        let (qkv, z, merged_hopper_qkvz) = qkvz::prepare(qkv, z)?;
        Ok(Self {
            #[cfg(feature = "cuda")]
            merged_hopper_qkvz,
            #[cfg(all(feature = "cuda", sys1_cuda_fused_ffn))]
            qkvz_weight: std::sync::Mutex::new(None),
            qkv,
            z,
            b: Linear::new(b, None),
            a: Linear::new(a, None),
            #[cfg(feature = "cuda")]
            ba,
            conv: vb.get(
                (
                    key_width * 2 + value_width,
                    1,
                    config.linear_conv_kernel_dim,
                ),
                "conv1d.weight",
            )?,
            a_log: vb
                .get(config.linear_num_value_heads, "A_log")?
                .to_dtype(DType::F32)?
                .exp()?
                .neg()?,
            dt_bias: vb
                .get(config.linear_num_value_heads, "dt_bias")?
                .to_dtype(DType::F32)?,
            norm: vb.get(config.linear_value_head_dim, "norm.weight")?,
            output: linear(value_width, config.hidden_size, vb.pp("out_proj"))?,
            key_heads: config.linear_num_key_heads,
            value_heads: config.linear_num_value_heads,
            key_dim: config.linear_key_head_dim,
            value_dim: config.linear_value_head_dim,
            kernel: config.linear_conv_kernel_dim,
            eps: config.rms_norm_eps,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (batch, length, _) = x.dims3()?;
        let key_width = self.key_heads * self.key_dim;
        let value_width = self.value_heads * self.value_dim;
        let (projected, early_z) = self.project_qkvz(x)?;

        let (q, k, v) = {
            #[cfg(feature = "cuda")]
            if x.device().is_cuda()
                && x.dtype() == DType::BF16
                && matches!(option_env!("SYS1_CUDA_COMPUTE_CAP"), Some("89" | "120"))
                && batch.saturating_mul(length) >= 512
                && self.key_dim == 128
                && self.value_dim == 128
            {
                #[cfg(test)]
                let vstorage_enabled = VSTORAGE_MODEL_REPLAY.with(std::cell::Cell::get);
                #[cfg(not(test))]
                let vstorage_enabled = true;
                if option_env!("SYS1_CUDA_COMPUTE_CAP") == Some("120")
                    && vstorage_enabled
                    && length >= 512
                {
                    #[cfg(test)]
                    VSTORAGE_MODEL_CALLS.with(|slot| slot.set(slot.get() + 1));
                    cuda_conv_vstore(
                        &projected,
                        &self.conv,
                        self.key_heads,
                        self.value_heads,
                        None,
                    )?
                } else {
                    cuda_conv_qkv(
                        &projected,
                        &self.conv,
                        self.key_heads,
                        self.value_heads,
                        None,
                    )?
                }
            } else {
                #[cfg(feature = "cuda")]
                let mixed = if x.device().is_cuda() && matches!(x.dtype(), DType::BF16 | DType::F32)
                {
                    cuda_conv_silu(&projected, &self.conv)?
                } else {
                    causal_conv_silu(&projected, &self.conv, self.kernel)?
                };
                #[cfg(feature = "metal")]
                let mixed = if x.device().is_metal()
                    && matches!(x.dtype(), DType::BF16 | DType::F16 | DType::F32)
                {
                    crate::kernels::metal::conv_silu(&projected, &self.conv)?
                } else {
                    causal_conv_silu(&projected, &self.conv, self.kernel)?
                };
                #[cfg(not(any(feature = "cuda", feature = "metal")))]
                let mixed = causal_conv_silu(&projected, &self.conv, self.kernel)?;

                let (q, k, v) = {
                    #[cfg(feature = "cuda")]
                    if x.device().is_cuda()
                        && x.dtype() == DType::BF16
                        && self.key_dim == 128
                        && self.value_dim == 128
                    {
                        cuda_fused_qkv_prepare(&mixed, self.key_heads, self.value_heads)?
                    } else {
                        let q = mixed.narrow(D::Minus1, 0, key_width)?.reshape((
                            batch,
                            length,
                            self.key_heads,
                            self.key_dim,
                        ))?;
                        let k = mixed.narrow(D::Minus1, key_width, key_width)?.reshape((
                            batch,
                            length,
                            self.key_heads,
                            self.key_dim,
                        ))?;
                        let v = mixed
                            .narrow(D::Minus1, key_width * 2, value_width)?
                            .reshape((batch, length, self.value_heads, self.value_dim))?;
                        let repeat = self.value_heads / self.key_heads;
                        let expand = |xs: Tensor| -> Result<Tensor> {
                            xs.unsqueeze(3)?
                                .broadcast_as((
                                    batch,
                                    length,
                                    self.key_heads,
                                    repeat,
                                    self.key_dim,
                                ))?
                                .reshape((batch, length, self.value_heads, self.key_dim))
                        };
                        let grouped_qk =
                            (cfg!(feature = "cuda") && x.device().is_cuda() && self.key_dim == 128)
                                || (cfg!(feature = "metal")
                                    && x.device().is_metal()
                                    && self.key_dim == 128
                                    && self.value_dim == 128);
                        let l2 = |xs: Tensor, scale: f64| -> Result<Tensor> {
                            #[cfg(feature = "cuda")]
                            if xs.device().is_cuda() && self.key_dim == 128 {
                                return cuda_l2_128(&xs, scale as f32);
                            }
                            let denom = (xs.sqr()?.sum_keepdim(D::Minus1)? + 1e-6)?.sqrt()?;
                            let normalized = xs.broadcast_div(&denom)?;
                            if scale == 1. {
                                Ok(normalized)
                            } else {
                                normalized * scale
                            }
                        };
                        let q = if grouped_qk { q } else { expand(q)? };
                        let q = l2(q.to_dtype(DType::F32)?, (self.key_dim as f64).powf(-0.5))?;
                        let k = if grouped_qk { k } else { expand(k)? };
                        let k = l2(k.to_dtype(DType::F32)?, 1.)?;
                        let v = v.to_dtype(DType::F32)?;
                        (q, k, v)
                    }
                    #[cfg(not(feature = "cuda"))]
                    {
                        let q = mixed.narrow(D::Minus1, 0, key_width)?.reshape((
                            batch,
                            length,
                            self.key_heads,
                            self.key_dim,
                        ))?;
                        let k = mixed.narrow(D::Minus1, key_width, key_width)?.reshape((
                            batch,
                            length,
                            self.key_heads,
                            self.key_dim,
                        ))?;
                        let v = mixed
                            .narrow(D::Minus1, key_width * 2, value_width)?
                            .reshape((batch, length, self.value_heads, self.value_dim))?;
                        let repeat = self.value_heads / self.key_heads;
                        let expand = |xs: Tensor| -> Result<Tensor> {
                            xs.unsqueeze(3)?
                                .broadcast_as((
                                    batch,
                                    length,
                                    self.key_heads,
                                    repeat,
                                    self.key_dim,
                                ))?
                                .reshape((batch, length, self.value_heads, self.key_dim))
                        };
                        let grouped_qk =
                            (cfg!(feature = "cuda") && x.device().is_cuda() && self.key_dim == 128)
                                || (cfg!(feature = "metal")
                                    && x.device().is_metal()
                                    && self.key_dim == 128
                                    && self.value_dim == 128);
                        let l2 = |xs: Tensor, scale: f64| -> Result<Tensor> {
                            #[cfg(feature = "cuda")]
                            if xs.device().is_cuda() && self.key_dim == 128 {
                                return cuda_l2_128(&xs, scale as f32);
                            }
                            let denom = (xs.sqr()?.sum_keepdim(D::Minus1)? + 1e-6)?.sqrt()?;
                            let normalized = xs.broadcast_div(&denom)?;
                            if scale == 1. {
                                Ok(normalized)
                            } else {
                                normalized * scale
                            }
                        };
                        let q = if grouped_qk { q } else { expand(q)? };
                        let q = l2(q.to_dtype(DType::F32)?, (self.key_dim as f64).powf(-0.5))?;
                        let k = if grouped_qk { k } else { expand(k)? };
                        let k = l2(k.to_dtype(DType::F32)?, 1.)?;
                        let v = v.to_dtype(DType::F32)?;
                        (q, k, v)
                    }
                };
                (q, k, v)
            }
            #[cfg(not(feature = "cuda"))]
            {
                #[cfg(feature = "cuda")]
                let mixed = if x.device().is_cuda() && matches!(x.dtype(), DType::BF16 | DType::F32)
                {
                    cuda_conv_silu(&projected, &self.conv)?
                } else {
                    causal_conv_silu(&projected, &self.conv, self.kernel)?
                };
                #[cfg(feature = "metal")]
                let mixed = if x.device().is_metal()
                    && matches!(x.dtype(), DType::BF16 | DType::F16 | DType::F32)
                {
                    crate::kernels::metal::conv_silu(&projected, &self.conv)?
                } else {
                    causal_conv_silu(&projected, &self.conv, self.kernel)?
                };
                #[cfg(not(any(feature = "cuda", feature = "metal")))]
                let mixed = causal_conv_silu(&projected, &self.conv, self.kernel)?;

                let (q, k, v) = {
                    #[cfg(feature = "cuda")]
                    if x.device().is_cuda()
                        && x.dtype() == DType::BF16
                        && self.key_dim == 128
                        && self.value_dim == 128
                    {
                        cuda_fused_qkv_prepare(&mixed, self.key_heads, self.value_heads)?
                    } else {
                        let q = mixed.narrow(D::Minus1, 0, key_width)?.reshape((
                            batch,
                            length,
                            self.key_heads,
                            self.key_dim,
                        ))?;
                        let k = mixed.narrow(D::Minus1, key_width, key_width)?.reshape((
                            batch,
                            length,
                            self.key_heads,
                            self.key_dim,
                        ))?;
                        let v = mixed
                            .narrow(D::Minus1, key_width * 2, value_width)?
                            .reshape((batch, length, self.value_heads, self.value_dim))?;
                        let repeat = self.value_heads / self.key_heads;
                        let expand = |xs: Tensor| -> Result<Tensor> {
                            xs.unsqueeze(3)?
                                .broadcast_as((
                                    batch,
                                    length,
                                    self.key_heads,
                                    repeat,
                                    self.key_dim,
                                ))?
                                .reshape((batch, length, self.value_heads, self.key_dim))
                        };
                        let grouped_qk =
                            (cfg!(feature = "cuda") && x.device().is_cuda() && self.key_dim == 128)
                                || (cfg!(feature = "metal")
                                    && x.device().is_metal()
                                    && self.key_dim == 128
                                    && self.value_dim == 128);
                        let l2 = |xs: Tensor, scale: f64| -> Result<Tensor> {
                            #[cfg(feature = "cuda")]
                            if xs.device().is_cuda() && self.key_dim == 128 {
                                return cuda_l2_128(&xs, scale as f32);
                            }
                            let denom = (xs.sqr()?.sum_keepdim(D::Minus1)? + 1e-6)?.sqrt()?;
                            let normalized = xs.broadcast_div(&denom)?;
                            if scale == 1. {
                                Ok(normalized)
                            } else {
                                normalized * scale
                            }
                        };
                        let q = if grouped_qk { q } else { expand(q)? };
                        let q = l2(q.to_dtype(DType::F32)?, (self.key_dim as f64).powf(-0.5))?;
                        let k = if grouped_qk { k } else { expand(k)? };
                        let k = l2(k.to_dtype(DType::F32)?, 1.)?;
                        let v = v.to_dtype(DType::F32)?;
                        (q, k, v)
                    }
                    #[cfg(not(feature = "cuda"))]
                    {
                        let q = mixed.narrow(D::Minus1, 0, key_width)?.reshape((
                            batch,
                            length,
                            self.key_heads,
                            self.key_dim,
                        ))?;
                        let k = mixed.narrow(D::Minus1, key_width, key_width)?.reshape((
                            batch,
                            length,
                            self.key_heads,
                            self.key_dim,
                        ))?;
                        let v = mixed
                            .narrow(D::Minus1, key_width * 2, value_width)?
                            .reshape((batch, length, self.value_heads, self.value_dim))?;
                        let repeat = self.value_heads / self.key_heads;
                        let expand = |xs: Tensor| -> Result<Tensor> {
                            xs.unsqueeze(3)?
                                .broadcast_as((
                                    batch,
                                    length,
                                    self.key_heads,
                                    repeat,
                                    self.key_dim,
                                ))?
                                .reshape((batch, length, self.value_heads, self.key_dim))
                        };
                        let grouped_qk =
                            (cfg!(feature = "cuda") && x.device().is_cuda() && self.key_dim == 128)
                                || (cfg!(feature = "metal")
                                    && x.device().is_metal()
                                    && self.key_dim == 128
                                    && self.value_dim == 128);
                        let l2 = |xs: Tensor, scale: f64| -> Result<Tensor> {
                            #[cfg(feature = "cuda")]
                            if xs.device().is_cuda() && self.key_dim == 128 {
                                return cuda_l2_128(&xs, scale as f32);
                            }
                            let denom = (xs.sqr()?.sum_keepdim(D::Minus1)? + 1e-6)?.sqrt()?;
                            let normalized = xs.broadcast_div(&denom)?;
                            if scale == 1. {
                                Ok(normalized)
                            } else {
                                normalized * scale
                            }
                        };
                        let q = if grouped_qk { q } else { expand(q)? };
                        let q = l2(q.to_dtype(DType::F32)?, (self.key_dim as f64).powf(-0.5))?;
                        let k = if grouped_qk { k } else { expand(k)? };
                        let k = l2(k.to_dtype(DType::F32)?, 1.)?;
                        let v = v.to_dtype(DType::F32)?;
                        (q, k, v)
                    }
                };
                (q, k, v)
            }
        };
        let (g, beta) = self.reference_gates(x)?;

        #[cfg(feature = "cuda")]
        if x.device().is_cuda() && self.key_dim == 128 {
            let output = if x.dtype() == DType::BF16
                && matches!(option_env!("SYS1_CUDA_COMPUTE_CAP"), Some("80" | "90"))
                && self.value_dim == 128
                && length >= 128
            {
                cuda_c64bfout_uniform(&q, &k, &v, &g, &beta, 32)?
            } else if x.dtype() == DType::BF16
                && option_env!("SYS1_CUDA_COMPUTE_CAP") == Some("120")
                && self.value_dim == 128
            {
                if length >= 512 {
                    if v.dtype() == DType::BF16 {
                        cuda_vstore_gdn_uniform(&q, &k, &v, &g, &beta, 32)?
                    } else {
                        cuda_c64bfout_uniform(&q, &k, &v, &g, &beta, 32)?
                    }
                } else {
                    cuda_delta_rule_token_major_bfout(&q, &k, &v, &g, &beta, true)?
                }
            } else if x.dtype() == DType::BF16
                && option_env!("SYS1_CUDA_COMPUTE_CAP") == Some("120")
                && length >= 512
                && self.value_dim <= 128
            {
                cuda_chunk64_direct(&q, &k, &v, &g, &beta, 32)?
            } else {
                cuda_delta_rule_token_major(&q, &k, &v, &g, &beta)?
            };

            let result = self.finish_delta(x, output, batch, length, value_width, early_z)?;

            return Ok(result);
        }
        #[cfg(feature = "metal")]
        let grouped_qk = (cfg!(feature = "cuda") && x.device().is_cuda() && self.key_dim == 128)
            || (cfg!(feature = "metal")
                && x.device().is_metal()
                && self.key_dim == 128
                && self.value_dim == 128);
        let q = q.transpose(1, 2)?;
        let k = k.transpose(1, 2)?;
        let v = v.transpose(1, 2)?;
        let beta = beta.transpose(1, 2)?;
        let g = g.transpose(1, 2)?;
        #[cfg(feature = "metal")]
        let output = if grouped_qk {
            let packed = crate::kernels::metal::pack_delta_f32(&q, &k, &v, &g, &beta)?;
            crate::kernels::metal::delta_rule_packed(&packed, self.value_dim)?
        } else {
            chunked_delta_rule(&q, &k, &v, &g, &beta)?
        };
        #[cfg(not(feature = "metal"))]
        let output = chunked_delta_rule(&q, &k, &v, &g, &beta)?;

        let result = self.finish_delta(x, output, batch, length, value_width, early_z)?;

        Ok(result)
    }

    fn reference_gates(&self, x: &Tensor) -> Result<(Tensor, Tensor)> {
        #[cfg(feature = "cuda")]
        if x.device().is_cuda() && x.dtype() == DType::BF16 {
            let ba = self
                .ba
                .as_ref()
                .ok_or_else(|| candle_core::Error::Msg("missing packed BF16 gates".into()))?;
            let projected = if option_env!("SYS1_CUDA_COMPUTE_CAP") == Some("120")
                && x.dim(1)? >= 1024
                && ba.weight().dims() == [64, 4096]
            {
                cuda_gate_tensor16_rows32(x, ba, true, SimtProjection::Compensated)?
            } else {
                cuda_gate_projection_bf16(x, ba)?
            };

            return cuda_packed_ba_nonlinearity(&projected, &projected, &self.dt_bias, &self.a_log);
        }
        let project = |linear: &Linear| {
            #[cfg(feature = "cuda")]
            if x.device().is_cuda() && x.dtype() == DType::BF16 {
                return cuda_gate_projection_bf16(x, linear);
            }
            crate::kernels::linear::forward_compatible(x, linear)
        };
        let b = project(&self.b)?;

        let beta = rounded_beta(&b)?;
        drop(b);
        let a = project(&self.a)?.to_dtype(DType::F32)?;

        let gate_input = a.broadcast_add(&self.dt_bias)?;
        let softplus = crate::kernels::activation::softplus(&gate_input)?;
        Ok((softplus.broadcast_mul(&self.a_log)?, beta))
    }

    #[allow(
        clippy::too_many_arguments,
        reason = "Keep the validated decoder geometry and optional early-Z cast boundary explicit"
    )]
    fn finish_delta(
        &self,
        x: &Tensor,
        output: Tensor,
        batch: usize,
        length: usize,
        value_width: usize,
        early_z: Option<Tensor>,
    ) -> Result<Tensor> {
        let gate = match early_z {
            Some(z) => z,
            None => linear_attention_projection_forward(x, &self.z)?,
        };

        #[cfg(feature = "cuda")]
        let merged_gate = option_env!("SYS1_CUDA_COMPUTE_CAP") == Some("90")
            && x.device().is_cuda()
            && output.dtype() == DType::BF16
            && self.value_heads == 32
            && self.value_dim == 128
            && gate.dims() == [batch, length, 4096]
            && gate.stride() == [length * 12288, 12288, 1];
        #[cfg(not(feature = "cuda"))]
        let merged_gate = false;
        let gate = if merged_gate {
            gate
        } else {
            gate.reshape((batch, length, self.value_heads, self.value_dim))?
        };

        #[cfg(feature = "cuda")]
        if x.device().is_cuda() && x.dtype() == DType::BF16 && self.value_dim == 128 {
            let finalized = cuda_post_finalize_bf16(&output, &gate, &self.norm, self.eps)?
                .reshape((batch, length, value_width))?;

            return linear_attention_projection_forward(&finalized, &self.output);
        }
        #[cfg(feature = "metal")]
        if x.device().is_metal() && x.dtype() == DType::BF16 && self.value_dim == 128 {
            let output = output.to_dtype(x.dtype())?.to_dtype(DType::F32)?;
            let variance = output.sqr()?.mean_keepdim(D::Minus1)?;
            let denominator = (variance + self.eps)?.sqrt()?;
            let gate_silu = gate.to_dtype(DType::F32)?.silu()?;
            let norm = self.norm.to_dtype(DType::F32)?;
            return crate::kernels::metal::post_finalize_bf16(
                &output,
                &denominator,
                &gate_silu,
                &norm,
            )?
            .reshape((batch, length, value_width))?
            .apply(&self.output);
        }
        let finalized = gated_norm_reference(&output, &gate, &self.norm, self.eps, x.dtype())?
            .reshape((batch, length, value_width))?;

        crate::kernels::linear::forward_compatible(&finalized, &self.output)
    }
}

#[cfg(any(feature = "cuda", test))]
#[allow(dead_code)]
fn blocked_projection(input: &Tensor, linear: &Linear, block_rows: usize) -> Result<Tensor> {
    if block_rows == 0 {
        candle_core::bail!("Blocked projection requires a nonzero row block")
    }
    let (batch, length, inner) = input.dims3()?;
    let rows = batch
        .checked_mul(length)
        .ok_or_else(|| candle_core::Error::Msg("Blocked projection row count overflow".into()))?;
    if rows == 0 {
        return input.apply(linear);
    }
    let columns = linear.weight().dim(0)?;
    let matrix = input.contiguous()?.reshape((rows, inner))?;
    let mut chunks = Vec::new();
    for start in (0..rows).step_by(block_rows) {
        let count = (rows - start).min(block_rows);
        let piece = matrix.narrow(0, start, count)?;
        let padded = if count == block_rows {
            piece
        } else {
            let zeros = Tensor::zeros((block_rows - count, inner), input.dtype(), input.device())?;
            Tensor::cat(&[&piece, &zeros], 0)?
        };
        chunks.push(padded.apply(linear)?.narrow(0, 0, count)?);
    }
    let output = if chunks.len() == 1 {
        chunks.pop().unwrap()
    } else {
        Tensor::cat(&chunks.iter().collect::<Vec<_>>(), 0)?
    };
    output.reshape((batch, length, columns))
}

#[cfg(test)]
fn guarded_double_radius_coefficient(inner: usize) -> f64 {
    guarded_double_radius_coefficient_impl(inner, false)
}

#[cfg(any(feature = "cuda", test))]
fn guarded_double_radius_coefficient_impl(inner: usize, tensor_dot: bool) -> f64 {
    let count = inner.div_ceil(32);
    let unit32 = f64::from(f32::EPSILON) / 2.;
    let unit64 = f64::EPSILON / 2.;
    if count == 0 || count as f64 * unit32 >= 0.25 {
        return f64::INFINITY;
    }
    let gamma = |n: usize, unit: f64| {
        let nu = (n as f64 * unit).next_up();
        (nu / (1. - nu).next_down()).next_up()
    };
    let g32 = gamma(count, unit32);
    let amplified = (1. + g32).next_up();
    let final_error = (gamma(6, unit64) * (amplified * amplified).next_up()).next_up();
    let old_error = ((g32 * g32).next_up() + final_error).next_up();

    let candidate_operations = if tensor_dot { inner } else { count + 5 };
    (old_error + gamma(candidate_operations, unit64)).next_up()
}

#[cfg(any(feature = "cuda", test))]
fn projection_tile_rows(rows: usize, columns: usize, sm120: bool) -> usize {
    if columns < 2048 {
        16
    } else if sm120 && rows >= 128 {
        128
    } else if rows >= 64 {
        32
    } else {
        16
    }
}

fn rounded_beta(input: &Tensor) -> Result<Tensor> {
    sigmoid(&input.to_dtype(DType::F32)?)?
        .to_dtype(input.dtype())?
        .to_dtype(DType::F32)
}

fn gated_norm_reference(
    input: &Tensor,
    gate: &Tensor,
    weight: &Tensor,
    epsilon: f64,
    dtype: DType,
) -> Result<Tensor> {
    let input = input.to_dtype(dtype)?.to_dtype(DType::F32)?;
    let variance = input.sqr()?.mean_keepdim(D::Minus1)?;
    let normalized = input
        .broadcast_div(&(variance + epsilon)?.sqrt()?)?
        .to_dtype(dtype)?;

    normalized
        .broadcast_mul(&weight.to_dtype(dtype)?)?
        .to_dtype(DType::F32)?
        .broadcast_mul(&gate.to_dtype(DType::F32)?.silu()?)?
        .to_dtype(dtype)
}

fn causal_conv_silu(projected: &Tensor, conv: &Tensor, kernel: usize) -> Result<Tensor> {
    crate::kernels::activation::silu(&causal_conv(projected, conv, kernel)?)
}

fn causal_conv(projected: &Tensor, conv: &Tensor, kernel: usize) -> Result<Tensor> {
    let (batch, length, channels) = projected.dims3()?;
    let accumulation_dtype = match projected.dtype() {
        DType::F16 | DType::BF16 => DType::F32,
        dtype => dtype,
    };
    let padded = projected
        .to_dtype(accumulation_dtype)?
        .pad_with_zeros(1, kernel - 1, 0)?;
    let weights = conv
        .to_dtype(accumulation_dtype)?
        .reshape((channels, kernel))?;
    let mut mixed = Tensor::zeros(
        (batch, length, channels),
        accumulation_dtype,
        projected.device(),
    )?;
    for tap in 0..kernel {
        let source = padded.narrow(1, tap, length)?;
        let weight = weights
            .narrow(1, tap, 1)?
            .squeeze(1)?
            .reshape((1, 1, channels))?;
        mixed = (mixed + source.broadcast_mul(&weight)?)?;
    }
    mixed.to_dtype(projected.dtype())
}

fn chunked_delta_rule(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    g: &Tensor,
    beta: &Tensor,
) -> Result<Tensor> {
    #[cfg(any(feature = "cuda", feature = "metal"))]
    let key_dim = q.dim(D::Minus1)?;
    #[cfg(feature = "cuda")]
    if q.device().is_cuda() && key_dim == 128 {
        candle_core::bail!("CUDA delta-rule inputs must remain token-major")
    }
    #[cfg(feature = "metal")]
    if q.device().is_metal() && key_dim == 128 {
        return crate::kernels::metal::delta_rule(q, k, v, g, beta);
    }
    chunked_delta_rule_fallback(q, k, v, g, beta)
}

fn chunked_delta_rule_fallback(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    g: &Tensor,
    beta: &Tensor,
) -> Result<Tensor> {
    let output_dtype = q.dtype();
    let accumulation = if q.device().is_cpu() {
        DType::F64
    } else {
        DType::F32
    };
    let q = q.to_dtype(accumulation)?;
    let k = k.to_dtype(accumulation)?;
    let v = v.to_dtype(accumulation)?;
    let g = g.to_dtype(accumulation)?;
    let beta = beta.to_dtype(accumulation)?;
    let (batch, heads, length, key_dim) = q.dims4()?;
    let value_dim = v.dim(D::Minus1)?;
    let mut state = Tensor::zeros((batch, heads, key_dim, value_dim), accumulation, q.device())?;
    let mut outputs = Vec::with_capacity(length.div_ceil(32));
    let full_lower = Tensor::tril2(32, accumulation, q.device())?;
    let full_identity = Tensor::eye(32, accumulation, q.device())?;
    let full_strictly_lower = (&full_lower - &full_identity)?;
    for start in (0..length).step_by(32) {
        let count = (length - start).min(32);
        let qi = q.narrow(2, start, count)?;
        let ki = k.narrow(2, start, count)?;
        let vi = v.narrow(2, start, count)?;
        let bi = beta.narrow(2, start, count)?.unsqueeze(3)?;
        let gi = g.narrow(2, start, count)?.contiguous()?.cumsum(2)?;
        let (lower, identity, strictly_lower) = if count == 32 {
            (
                full_lower.clone(),
                full_identity.clone(),
                full_strictly_lower.clone(),
            )
        } else {
            let lower = Tensor::tril2(count, accumulation, q.device())?;
            let identity = Tensor::eye(count, accumulation, q.device())?;
            let strictly_lower = (&lower - &identity)?;
            (lower, identity, strictly_lower)
        };
        let decay = gi
            .unsqueeze(3)?
            .broadcast_sub(&gi.unsqueeze(2)?)?
            .broadcast_mul(&lower)?
            .exp()?
            .broadcast_mul(&lower)?;
        let key_beta = ki.broadcast_mul(&bi)?;
        let value_beta = vi.broadcast_mul(&bi)?;
        let gi_exp = gi.exp()?;
        let l = contiguous_matmul(&key_beta, &ki.transpose(D::Minus2, D::Minus1)?)?
            .broadcast_mul(&decay)?
            .broadcast_mul(&strictly_lower)?
            .neg()?;
        let mut inverse = l.broadcast_add(&identity)?;
        let mut power = l;
        let mut covered = 2;
        while covered < count {
            power = contiguous_matmul(&power, &power)?;
            inverse = (&inverse + contiguous_matmul(&power, &inverse)?)?;
            covered *= 2;
        }
        let value = contiguous_matmul(&inverse, &value_beta)?;
        let key_cumdecay =
            contiguous_matmul(&inverse, &key_beta.broadcast_mul(&gi_exp.unsqueeze(3)?)?)?;
        let value_new = (&value - contiguous_matmul(&key_cumdecay, &state)?)?;
        let attn =
            contiguous_matmul(&qi, &ki.transpose(D::Minus2, D::Minus1)?)?.broadcast_mul(&decay)?;
        let intermediate = contiguous_matmul(&qi.broadcast_mul(&gi_exp.unsqueeze(3)?)?, &state)?;
        outputs.push((intermediate + contiguous_matmul(&attn, &value_new)?)?);
        let last = gi.narrow(2, count - 1, 1)?;
        let updated = contiguous_matmul(
            &ki.broadcast_mul(&last.broadcast_sub(&gi)?.exp()?.unsqueeze(3)?)?
                .transpose(D::Minus2, D::Minus1)?,
            &value_new,
        )?;
        state = (state.broadcast_mul(&last.exp()?.unsqueeze(3)?)? + updated)?;
    }
    Tensor::cat(&outputs.iter().collect::<Vec<_>>(), 2)?
        .transpose(1, 2)?
        .to_dtype(output_dtype)
}

fn contiguous_matmul(left: &Tensor, right: &Tensor) -> Result<Tensor> {
    crate::kernels::linear::matmul_compatible(&left.contiguous()?, &right.contiguous()?)
}

enum Mixer {
    Full(FullAttention),
    Linear(LinearAttention),
}

struct Layer {
    input_norm: RmsNorm,
    mixer: Mixer,
    post_norm: RmsNorm,
    mlp: Mlp,
}

impl Layer {
    fn load(
        config: &TextConfig,
        index: usize,
        vb: VarBuilder,
        implementation: AttentionImplementation,
    ) -> Result<Self> {
        let mixer = match config.layer_types[index].as_str() {
            "full_attention" => Mixer::Full(FullAttention::load(
                config,
                vb.pp("self_attn"),
                implementation,
            )?),
            "linear_attention" => {
                Mixer::Linear(LinearAttention::load(config, vb.pp("linear_attn"))?)
            }
            other => candle_core::bail!("unsupported Qwen3.5 layer type {other}"),
        };
        Ok(Self {
            input_norm: RmsNorm::load(
                config.hidden_size,
                config.rms_norm_eps,
                vb.pp("input_layernorm"),
            )?,
            mixer,
            post_norm: RmsNorm::load(
                config.hidden_size,
                config.rms_norm_eps,
                vb.pp("post_attention_layernorm"),
            )?,
            mlp: Mlp::load(config, vb.pp("mlp"))?,
        })
    }

    fn forward(&self, x: &Tensor, rotary: Option<&RotaryTensors>) -> Result<Tensor> {
        let normalized = self.input_norm.forward(x)?;

        let mixed = match &self.mixer {
            Mixer::Full(layer) => layer.forward(
                &normalized,
                rotary.ok_or_else(|| {
                    candle_core::Error::Msg("Missing Qwen3.5 rotary tensors".into())
                })?,
            )?,
            Mixer::Linear(layer) => layer.forward(&normalized)?,
        };

        let (x, post_normalized) = self.post_norm.forward_residual(x, &mixed)?;

        let mlp_output = self.mlp.forward(&post_normalized)?;

        let output = (&x + mlp_output)?;

        Ok(output)
    }
}

pub struct Decoder {
    embedding: Embedding,
    layers: Vec<Layer>,
    norm: RmsNorm,
}

impl Decoder {
    pub fn load(
        config: &TextConfig,
        vb: VarBuilder,
        implementation: AttentionImplementation,
    ) -> Result<Self> {
        let embedding = embedding(config.vocab_size, config.hidden_size, vb.pp("embed_tokens"))?;
        #[cfg(feature = "cuda")]
        let embedding = maybe_cpu_input_embedding(embedding, config.hidden_size)?;
        let layers = (0..config.num_hidden_layers)
            .map(|index| {
                Layer::load(
                    config,
                    index,
                    vb.pp(format!("layers.{index}")),
                    implementation,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        let norm = RmsNorm::load(config.hidden_size, config.rms_norm_eps, vb.pp("norm"))?;
        Ok(Self {
            embedding,
            layers,
            norm,
        })
    }

    pub fn forward(&self, ids: &Tensor) -> Result<Tensor> {
        let length = ids.dim(1)?;
        let positions: Vec<_> = (0..length).map(|index| [index as u32; 3]).collect();
        self.forward_embeds(&self.embed(ids)?, &positions)
    }

    pub fn embed(&self, ids: &Tensor) -> Result<Tensor> {
        if self.embedding.embeddings().device().is_cpu() && ids.device().is_cuda() {
            let cpu_ids = ids.to_device(&candle_core::Device::Cpu)?;
            return cpu_ids.apply(&self.embedding)?.to_device(ids.device());
        }
        ids.apply(&self.embedding)
    }

    pub fn forward_embeds(&self, embeds: &Tensor, positions: &[[u32; 3]]) -> Result<Tensor> {
        if positions.len() != embeds.dim(1)? {
            candle_core::bail!("Qwen3.5 position count does not match sequence length")
        }
        let rotary = self
            .layers
            .iter()
            .find_map(|layer| match &layer.mixer {
                Mixer::Full(attention) => Some(&attention.rotary),
                Mixer::Linear(_) => None,
            })
            .map(|rotary| rotary.prepare(positions, embeds.dtype(), embeds.device()))
            .transpose()?;

        let output = self.forward_layers(embeds, rotary.as_ref())?;

        Ok(output)
    }
}

#[cfg(any(feature = "cuda", test))]
fn long_down_actual_m(rows: usize, inner: usize, columns: usize, sm120: bool) -> bool {
    sm120
        && (2048..=8192).contains(&rows)
        && columns >= 2048
        && columns.checked_mul(3) == Some(inner)
}

#[test]
fn long_down_geometry_policy_has_explicit_scope() {
    for (rows, inner, columns, sm120, expected) in [
        (2048, 12288, 4096, true, true),
        (2105, 12288, 4096, true, true),
        (8192, 12288, 4096, true, true),
        (2047, 12288, 4096, true, false),
        (8193, 12288, 4096, true, false),
        (2105, 12288, 4096, false, false),
        (2105, 4096, 12288, true, false),
        (2105, 6144, 2048, true, true),
        (2105, 192, 64, true, false),
    ] {
        assert_eq!(long_down_actual_m(rows, inner, columns, sm120), expected);
    }
}

fn full_attention_norm_rotary(
    input: &Tensor,
    norm: &RmsNorm,
    rotary: &RotaryTensors,
) -> Result<Tensor> {
    #[cfg(feature = "cuda")]
    if input.device().is_cuda()
        && input.dtype() == DType::BF16
        && norm.weight.dtype() == DType::BF16
    {
        let (batch, length, heads, width) = input.dims4()?;
        let strides = input.stride();
        let stride = strides[2];
        if batch > 0
            && length > 0
            && heads > 0
            && (32..=1024).contains(&width)
            && strides[3] == 1
            && stride >= width
            && strides[1] == heads * stride
            && strides[0] == length * heads * stride
            && rotary.dimension <= width
            && rotary.dimension.is_multiple_of(2)
            && rotary.cos.dtype() == DType::BF16
            && rotary.sin.dtype() == DType::BF16
        {
            return cuda_full_norm_rope(input, &norm.weight, rotary, norm.eps);
        }
    }
    rotary.apply(&norm.forward(input)?)
}

impl Decoder {
    fn forward_layers(&self, embeds: &Tensor, rotary: Option<&RotaryTensors>) -> Result<Tensor> {
        #[cfg(feature = "cuda")]
        if embeds.device().is_cuda() && embeds.dtype() == DType::BF16 {
            return self.forward_deferred(embeds, rotary, None);
        }

        let mut x = embeds.clone();
        for layer in &self.layers {
            x = layer.forward(&x, rotary)?;
        }
        self.norm.forward(&x)
    }
}

#[derive(Clone)]
struct Linear {
    inner: CandleLinear,
    #[cfg(feature = "cuda")]
    nn_weight: std::sync::Arc<std::sync::Mutex<Option<Tensor>>>,
}
impl Linear {
    fn new(weight: Tensor, bias: Option<Tensor>) -> Self {
        Self {
            inner: CandleLinear::new(weight, bias),
            #[cfg(feature = "cuda")]
            nn_weight: std::sync::Arc::new(std::sync::Mutex::new(None)),
        }
    }
}
impl std::ops::Deref for Linear {
    type Target = CandleLinear;
    fn deref(&self) -> &CandleLinear {
        &self.inner
    }
}
impl candle_core::Module for Linear {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        candle_core::Module::forward(&self.inner, x)
    }
}

impl LinearAttention {
    fn project_qkvz(&self, x: &Tensor) -> Result<(Tensor, Option<Tensor>)> {
        #[cfg(feature = "cuda")]
        if let Some(output) = self.project_merged_hopper_qkvz(x)? {
            return Ok(output);
        }
        #[cfg(all(feature = "cuda", sys1_cuda_fused_ffn))]
        if x.device().is_cuda()
            && x.dtype() == DType::BF16
            && option_env!("SYS1_CUDA_COMPUTE_CAP") == Some("120")
        {
            let (batch, length, _) = x.dims3()?;
            let rows = batch
                .checked_mul(length)
                .ok_or_else(|| candle_core::Error::Msg("QKV/Z rows overflow".into()))?;
            #[cfg(test)]
            let enabled = QKVZ_DUAL_ENABLED.with(std::cell::Cell::get);
            #[cfg(not(test))]
            let enabled = true;
            if let Some(tile) =
                crate::kernels::cuda::fused_qkvz::select_tile(rows).filter(|_| enabled)
            {
                let packed = self
                    .qkvz_weight
                    .lock()
                    .map_err(|_| {
                        candle_core::Error::Msg("poisoned model-owned QKV/Z weight".into())
                    })?
                    .clone();
                if let Some(packed) = packed {
                    let (qkv, z) = crate::kernels::cuda::fused_qkvz::forward(x, &packed, tile)?;
                    #[cfg(test)]
                    QKVZ_DUAL_CALLS.with(|c| c.set(c.get() + 1));
                    return Ok((qkv, Some(z)));
                }
            }
        }
        Ok((linear_attention_projection_forward(x, &self.qkv)?, None))
    }
}

fn full_attention_gate_mul(
    attention: &Tensor,
    gate: &Tensor,
    batch: usize,
    length: usize,
    width: usize,
) -> Result<Tensor> {
    let flat = attention.reshape((batch, length, width))?;
    #[cfg(feature = "cuda")]
    if option_env!("SYS1_CUDA_COMPUTE_CAP") == Some("90")
        && let Some(output) = crate::kernels::cuda::pointwise_mul::try_mul(&flat, gate)?
    {
        return Ok(output);
    }
    &flat * gate
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn guarded_double_radius_coefficient_bounds_exact_rational_values() {
        for (count, nearest_exact) in [
            (1, 4.884981811276547e-15_f64),
            (80, 2.2747687422185558e-11),
            (288, 2.9471959623355573e-10),
            (576, 1.178851243061726e-9),
            (4_194_303, 0.11111104093424891),
        ] {
            let coefficient = guarded_double_radius_coefficient(count * 32);
            assert!(
                coefficient >= nearest_exact.next_up(),
                "must exceed the exact rational bound"
            );
            assert!(
                coefficient <= nearest_exact * (1. + 64. * f64::EPSILON),
                "rounding outward must not replace the derived bound with an arbitrary large radius"
            );
            assert_eq!(
                coefficient,
                guarded_double_radius_coefficient(count * 32 - 31),
                "same ceil(K/32) bound for the tail"
            );
        }
        assert!(guarded_double_radius_coefficient(0).is_infinite());
        assert!(guarded_double_radius_coefficient(4_194_304 * 32).is_infinite());
        assert!(guarded_double_radius_coefficient(usize::MAX).is_infinite());
    }

    #[test]
    fn guarded_tensor_double_radius_bounds_exact_rational_values() {
        for (count, nearest_exact) in [
            (1, 7.771561675301967e-15_f64),
            (80, 2.3022467620780365e-11),
            (288, 2.957102482384298e-10),
            (576, 1.1808331021829885e-9),
        ] {
            let coefficient = guarded_double_radius_coefficient_impl(count * 32, true);
            assert!(coefficient >= nearest_exact.next_up());
            assert!(coefficient <= nearest_exact * (1. + 64. * f64::EPSILON));
            assert!(coefficient >= guarded_double_radius_coefficient(count * 32));
        }
        assert!(guarded_double_radius_coefficient_impl(0, true).is_infinite());
        assert!(guarded_double_radius_coefficient_impl(usize::MAX, true).is_infinite());
    }

    #[test]
    fn projection_geometry_restricts_larger_tiles_to_measured_architecture() {
        for sm120 in [false, true] {
            assert_eq!(projection_tile_rows(4096, 2047, sm120), 16);
            assert_eq!(projection_tile_rows(63, 2048, sm120), 16);
            assert_eq!(projection_tile_rows(64, 2048, sm120), 32);
            assert_eq!(projection_tile_rows(127, 2048, sm120), 32);
        }
        assert_eq!(projection_tile_rows(128, 2048, true), 128);
        assert_eq!(projection_tile_rows(129, 2049, true), 128);
        assert_eq!(projection_tile_rows(128, 2048, false), 32);
        assert_eq!(projection_tile_rows(4096, 12288, false), 32);
    }

    #[test]
    fn blocked_projection_preserves_rows_storage_and_tail_shape() -> Result<()> {
        let device = candle_core::Device::Cpu;
        let weights = Tensor::from_vec(
            (0..15).map(|i| (i % 7) as f32 * 0.125).collect::<Vec<_>>(),
            (3, 5),
            &device,
        )?;
        let linear = Linear::new(weights, None);
        for (batch, length) in [(1, 1), (1, 16), (1, 17), (2, 19)] {
            let input = Tensor::from_vec(
                (0..batch * length * 5)
                    .map(|i| (i % 8) as f32 * 0.25)
                    .collect::<Vec<_>>(),
                (batch, length, 5),
                &device,
            )?;
            let actual = blocked_projection(&input, &linear, 16)?;
            let expected = input.apply(&linear)?;
            assert_eq!(actual.dims(), expected.dims());
            assert_eq!(actual.dtype(), input.dtype());
            assert_eq!(
                actual.flatten_all()?.to_vec1::<f32>()?,
                expected.flatten_all()?.to_vec1::<f32>()?
            );
            assert!(blocked_projection(&input, &linear, 0).is_err());
        }
        Ok(())
    }

    #[test]
    fn full_attention_kv_fixed_reduction_preserves_storage_and_cast_boundary() -> Result<()> {
        let devices = vec![candle_core::Device::Cpu];
        #[cfg(feature = "cuda")]
        let devices = {
            let mut devices = devices;
            devices.push(test_cuda_device()?);
            devices
        };
        for device in devices {
            for dtype in supported_rotary_test_dtypes(&device)? {
                for (batch, length, inner, columns) in [
                    (1, 2, 33, 9),
                    (2, 9, 129, 17),
                    (1, 33, 65, 65),
                    (2, 1, 4096, 1024),
                    (1, 2, 6144, 1024),
                    (1, 2, 12288, 65),
                    (1, 2, 18432, 65),
                    (1, 33, 257, 2049),
                    (1, 65, 65, 2049),
                    (2, 33, 129, 2049),
                    (1, 129, 65, 2049),
                    (1, 129, 257, 2049),
                ] {
                    let rows = batch * length;
                    let xs: Vec<f32> = (0..rows * inner)
                        .map(|i| ((i * 17 % 255) as f32 - 127.) / 64.)
                        .collect();
                    let ws: Vec<f32> = (0..columns * inner)
                        .map(|i| ((i * 29 % 127) as f32 - 63.) / 256.)
                        .collect();
                    let weights =
                        Tensor::from_vec(ws.clone(), (columns, inner), &device)?.to_dtype(dtype)?;
                    let original = Linear::new(weights.clone(), None);
                    let layer = Linear::new(weights, None);
                    assert_eq!(layer.weight().dtype(), dtype);
                    let x = Tensor::from_vec(xs.clone(), (batch, length, inner), &device)?
                        .to_dtype(dtype)?;
                    #[cfg(target_os = "macos")]
                    if device.is_cpu() && dtype == DType::F16 {
                        assert!(full_attention_kv_forward(&x, &layer).is_err());
                        assert!(linear_attention_projection_forward(&x, &layer).is_err());
                        continue;
                    }
                    let projected = full_attention_kv_forward(&x, &layer)?;
                    let linear_qkv = linear_attention_projection_forward(&x, &layer)?;
                    assert_eq!(
                        projected
                            .to_dtype(DType::F32)?
                            .flatten_all()?
                            .to_vec1::<f32>()?,
                        linear_qkv
                            .to_dtype(DType::F32)?
                            .flatten_all()?
                            .to_vec1::<f32>()?,
                        "QKV candidate preserves projection dtype and cast boundary"
                    );
                    assert_eq!(projected.dtype(), dtype, "round back BEFORE attention/norm");
                    let actual = projected
                        .to_dtype(DType::F32)?
                        .flatten_all()?
                        .to_vec1::<f32>()?;
                    if device.is_cuda() && dtype == DType::BF16 {
                        let repeated = full_attention_kv_forward(&x, &layer)?
                            .to_dtype(DType::F32)?
                            .flatten_all()?
                            .to_vec1::<f32>()?;
                        assert_eq!(
                            actual, repeated,
                            "actual repeated fixed-reduction K/V projection"
                        );
                        #[cfg(feature = "cuda")]
                        {
                            for shaped in [x.clone(), x.reshape((rows, inner))?] {
                                let head = fixed_bf16_projection(&shaped, &layer)?;
                                assert_eq!(head.dtype(), dtype);
                                assert_eq!(head.dim(D::Minus1)?, columns);
                                assert_eq!(
                                    actual,
                                    head.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?,
                                    "rank-two/three head adapter preserves validated projection"
                                );
                            }
                        }
                    }
                    if dtype != DType::F32 && !(device.is_cuda() && dtype == DType::BF16) {
                        let retained = if device.is_cpu() && dtype == DType::BF16 {
                            x.to_dtype(DType::F32)?
                                .apply(&Linear::new(original.weight().to_dtype(DType::F32)?, None))?
                                .to_dtype(dtype)?
                        } else {
                            x.apply(&original)?
                        }
                        .to_dtype(DType::F32)?
                        .flatten_all()?
                        .to_vec1::<f32>()?;
                        assert_eq!(actual, retained, "unchanged backend/dtype path");
                        continue;
                    }
                    let expected: Vec<f32> = (0..rows)
                        .flat_map(|row| {
                            let xs = &xs;
                            let ws = &ws;
                            (0..columns).map(move |column| {
                                (0..inner)
                                    .map(|k| {
                                        f64::from(xs[row * inner + k])
                                            * f64::from(ws[column * inner + k])
                                    })
                                    .sum::<f64>() as f32
                            })
                        })
                        .collect();
                    #[cfg(feature = "cuda")]
                    if device.is_cuda() && dtype == DType::BF16 {
                        let biases: Vec<f32> = (0..columns)
                            .map(|column| ((column % 17) as f32 - 8.) / 256.)
                            .collect();
                        let biased = Linear::new(
                            layer.weight().clone(),
                            Some(
                                Tensor::from_vec(biases.clone(), columns, &device)?
                                    .to_dtype(dtype)?,
                            ),
                        );
                        let expected_biased: Vec<f32> = expected
                            .iter()
                            .enumerate()
                            .map(|(i, value)| *value + biases[i % columns])
                            .collect();
                        let expected_biased = Tensor::from_vec(
                            expected_biased,
                            actual.len(),
                            &candle_core::Device::Cpu,
                        )?
                        .to_dtype(dtype)?
                        .to_dtype(DType::F32)?
                        .to_vec1::<f32>()?;
                        for shaped in [x.clone(), x.reshape((rows, inner))?] {
                            let output = fixed_bf16_projection(&shaped, &biased)?;
                            assert_eq!(output.dim(D::Minus1)?, columns);
                            assert_eq!(
                                output
                                    .to_dtype(DType::F32)?
                                    .flatten_all()?
                                    .to_vec1::<f32>()?,
                                expected_biased,
                                "biased rank-two/three projection adds FP32 bias before sole BF16 cast"
                            );
                        }
                    }
                    let expected =
                        Tensor::from_vec(expected, actual.len(), &candle_core::Device::Cpu)?
                            .to_dtype(dtype)?
                            .to_dtype(DType::F32)?
                            .to_vec1::<f32>()?;
                    assert_eq!(
                        actual, expected,
                        "exact dyadic reference {dtype:?}/{inner}/{columns}"
                    );
                }
            }
        }
        Ok(())
    }

    #[test]
    fn mlp_diagnostic_boundaries_preserve_original_expression() -> Result<()> {
        let device = candle_core::Device::Cpu;
        for dtype in [DType::F32, DType::F16, DType::BF16] {
            let weights = |output, input| -> Result<Linear> {
                let values: Vec<f32> = (0..output * input)
                    .map(|i| ((i * 13 % 31) as f32 - 15.) / 128.)
                    .collect();
                Ok(Linear::new(
                    Tensor::from_vec(values, (output, input), &device)?.to_dtype(dtype)?,
                    None,
                ))
            };
            let layer = Mlp {
                #[cfg(all(feature = "cuda", sys1_cuda_direct_ffn))]
                direct_weight: None,
                #[cfg(all(feature = "cuda", sys1_cuda_fused_ffn))]
                fused_weight: Default::default(),
                gate: weights(17, 33)?,
                up: weights(17, 33)?,
                down: weights(9, 17)?,
            };
            let values: Vec<f32> = (0..2 * 3 * 33)
                .map(|i| ((i * 7 % 29) as f32 - 14.) / 64.)
                .collect();
            let x = Tensor::from_vec(values, (2, 3, 33), &device)?.to_dtype(dtype)?;
            #[cfg(target_os = "macos")]
            if dtype == DType::F16 {
                assert!(layer.forward(&x).is_err());
                continue;
            }
            let reference = |input: &Tensor, linear: &Linear| -> Result<Tensor> {
                if dtype == DType::BF16 {
                    input
                        .to_dtype(DType::F32)?
                        .apply(&Linear::new(linear.weight().to_dtype(DType::F32)?, None))?
                        .to_dtype(dtype)
                } else {
                    input.apply(linear)
                }
            };
            let original = reference(
                &(reference(&x, &layer.gate)?.silu()? * reference(&x, &layer.up)?)?,
                &layer.down,
            )?;
            let actual = layer.forward(&x)?;
            assert_eq!(actual.dtype(), dtype);
            let values = |x: &Tensor| x.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>();
            assert_eq!(
                values(&actual)?,
                values(&original)?,
                "diagnostic-only MLP ordering {dtype:?}"
            );
        }
        Ok(())
    }

    pub(super) fn supported_rotary_test_dtypes(
        _device: &candle_core::Device,
    ) -> Result<Vec<DType>> {
        let mut dtypes = vec![DType::F32, DType::F16];
        #[cfg(feature = "cuda")]
        if let candle_core::Device::Cuda(cuda) = _device {
            let (major, _) = cuda
                .cuda_stream()
                .context()
                .compute_capability()
                .map_err(|error| candle_core::Error::Msg(error.to_string()))?;
            if major >= 8 {
                dtypes.push(DType::BF16);
            }
            return Ok(dtypes);
        }
        dtypes.push(DType::BF16);
        Ok(dtypes)
    }

    #[test]
    fn eager_causal_gqa_matches_independent_f64_scalar_reference() -> Result<()> {
        let mut devices = vec![candle_core::Device::Cpu];
        #[cfg(feature = "cuda")]
        devices.push(test_cuda_device()?);
        let (batch, length, heads, kv_heads, width) = (2, 129, 4, 2, 4);
        let make = |count, factor| {
            (0..count)
                .map(|i| ((i * 13 % 97) as f32 - 48.) * factor)
                .collect::<Vec<_>>()
        };
        let q = make(batch * length * heads * width, 0.013);
        let k = make(batch * length * kv_heads * width, 0.017);
        let v = make(batch * length * kv_heads * width, 0.019);
        let mut expected = Vec::new();
        for b in 0..batch {
            for row in 0..length {
                for h in 0..heads {
                    let kv = h / (heads / kv_heads);
                    let scores = (0..=row)
                        .map(|col| {
                            (0..width)
                                .map(|d| {
                                    q[((b * length + row) * heads + h) * width + d] as f64
                                        * k[((b * length + col) * kv_heads + kv) * width + d] as f64
                                })
                                .sum::<f64>()
                                * 0.5
                        })
                        .collect::<Vec<_>>();
                    let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                    let weights = scores
                        .iter()
                        .map(|score| (score - max).exp())
                        .collect::<Vec<_>>();
                    let total = weights.iter().sum::<f64>();
                    for d in 0..width {
                        expected.push(
                            weights
                                .iter()
                                .enumerate()
                                .map(|(col, weight)| {
                                    weight
                                        * v[((b * length + col) * kv_heads + kv) * width + d] as f64
                                })
                                .sum::<f64>()
                                / total,
                        );
                    }
                }
            }
        }
        for device in devices.drain(..) {
            let actual = eager_attention(
                &Tensor::from_vec(q.clone(), (batch, length, heads, width), &device)?,
                &Tensor::from_vec(k.clone(), (batch, length, kv_heads, width), &device)?,
                &Tensor::from_vec(v.clone(), (batch, length, kv_heads, width), &device)?,
                0.5,
                true,
            )?
            .flatten_all()?
            .to_vec1::<f32>()?;
            for (actual, expected) in actual.into_iter().zip(&expected) {
                assert!((actual as f64 - expected).abs() <= 1e-5 + 1.3e-6 * expected.abs());
            }
        }
        Ok(())
    }

    #[test]
    fn shared_rotary_is_bitwise_equal_to_legacy_per_head_preparation() -> Result<()> {
        let mut devices = vec![candle_core::Device::Cpu];
        #[cfg(feature = "cuda")]
        devices.push(test_cuda_device()?);
        for device in devices.drain(..) {
            for dtype in supported_rotary_test_dtypes(&device)? {
                for (dimension, sections, width) in [(64, [16, 8, 8], 64), (64, [16, 8, 8], 256)] {
                    let theta = 1_000_000f64;
                    let positions = (0..17)
                        .map(|i| [i * 1031, i * 7, i * 31])
                        .collect::<Vec<_>>();
                    let prepared = Rotary::new(dimension, theta, sections)?
                        .prepare(&positions, dtype, &device)?;
                    for heads in [4, 24] {
                        let x = Tensor::from_vec(
                            (0..2 * positions.len() * heads * width)
                                .map(|i| (i % 251) as f32 * 0.013 - 1.5)
                                .collect::<Vec<_>>(),
                            (2, positions.len(), heads, width),
                            &device,
                        )?
                        .to_dtype(dtype)?;

                        let phase = positions
                            .iter()
                            .flat_map(|position| {
                                (0..dimension / 2).map(move |i| {
                                    let axis = if i < sections[1] * 3 && i % 3 == 1 {
                                        1
                                    } else if i < sections[2] * 3 && i % 3 == 2 {
                                        2
                                    } else {
                                        0
                                    };
                                    position[axis] as f32
                                        / theta.powf(2. * i as f64 / dimension as f64) as f32
                                })
                            })
                            .collect::<Vec<_>>();
                        let phase = Tensor::from_vec(
                            phase,
                            (1, positions.len(), 1, dimension / 2),
                            &device,
                        )?;
                        let cos = Tensor::cat(&[&phase.cos()?, &phase.cos()?], D::Minus1)?
                            .to_dtype(dtype)?;
                        let sin = Tensor::cat(&[&phase.sin()?, &phase.sin()?], D::Minus1)?
                            .to_dtype(dtype)?;
                        let active = x.narrow(D::Minus1, 0, dimension)?;
                        let first = active.narrow(D::Minus1, 0, dimension / 2)?;
                        let second = active.narrow(D::Minus1, dimension / 2, dimension / 2)?;
                        let rotated = Tensor::cat(&[&second.neg()?, &first], D::Minus1)?;
                        let legacy = Tensor::cat(
                            &[
                                &(active.broadcast_mul(&cos)? + rotated.broadcast_mul(&sin)?)?,
                                &x.narrow(D::Minus1, dimension, width - dimension)?,
                            ],
                            D::Minus1,
                        )?;
                        let values = |tensor: Tensor| -> Result<Vec<u32>> {
                            Ok(tensor
                                .to_dtype(DType::F32)?
                                .flatten_all()?
                                .to_vec1::<f32>()?
                                .into_iter()
                                .map(f32::to_bits)
                                .collect())
                        };
                        assert_eq!(
                            values(prepared.apply(&x)?)?,
                            values(legacy)?,
                            "{dtype:?}, heads={heads}, width={width}"
                        );
                    }
                }
            }
        }
        Ok(())
    }

    #[test]
    fn eager_attention_single_key_and_chunked_causal_are_finite() {
        let mut devices = vec![candle_core::Device::Cpu];
        #[cfg(feature = "cuda")]
        devices.push(test_cuda_device().unwrap());
        #[cfg(feature = "metal")]
        devices.push(candle_core::Device::new_metal(0).unwrap());
        for device in devices.drain(..) {
            let q = Tensor::ones((1, 1, 2, 4), DType::F32, &device).unwrap();
            let k = Tensor::ones((1, 1, 1, 4), DType::F32, &device).unwrap();
            let v = Tensor::from_vec(vec![1f32, -2., 3., -4.], (1, 1, 1, 4), &device).unwrap();
            let actual = eager_attention(&q, &k, &v, 0.5, false)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            assert_eq!(actual, vec![1., -2., 3., -4., 1., -2., 3., -4.]);

            let q = Tensor::zeros((1, 257, 2, 4), DType::F32, &device).unwrap();
            let k = Tensor::zeros((1, 257, 1, 4), DType::F32, &device).unwrap();
            let v = Tensor::from_vec(
                (0..257).flat_map(|i| [i as f32; 4]).collect::<Vec<_>>(),
                (1, 257, 1, 4),
                &device,
            )
            .unwrap();
            let actual = eager_attention(&q, &k, &v, 0.5, true)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            for (index, actual) in actual.into_iter().enumerate() {
                let expected = (index / 8) as f32 * 0.5;
                assert!(
                    (actual - expected).abs() <= 1e-5 + expected.abs() * 1.3e-6,
                    "{device:?}: index {index}, expected {expected}, got {actual}"
                );
            }
        }
    }

    #[test]
    #[ignore = "external reference data"]
    fn convolution_matches_frozen_native_torch_gpu_fixtures() -> Result<()> {
        let directory = crate::assets::fixtures("qwen35_conv")?;
        let mut fixtures = std::fs::read_dir(directory)?
            .filter_map(|entry| {
                let path = entry.ok()?.path();
                (path
                    .extension()
                    .is_some_and(|extension| extension == "safetensors"))
                .then_some(path)
            })
            .collect::<Vec<_>>();
        fixtures.sort();
        assert_eq!(
            fixtures.len(),
            8,
            "all pinned native BF16/FP32 cases are required"
        );
        for path in fixtures {
            let tensors = candle_core::safetensors::load(&path, &candle_core::Device::Cpu)?;
            let input = &tensors["input"];
            let weight = &tensors["weight"];
            let kernel = weight.dim(2)?;
            let expected_conv = &tensors["expected_conv"];
            let expected = &tensors["expected"];
            let assert_close = |actual: &Tensor, reference: &Tensor, name: &str| -> Result<()> {
                let actual = actual
                    .to_dtype(DType::F32)?
                    .flatten_all()?
                    .to_vec1::<f32>()?;
                let reference = reference
                    .to_dtype(DType::F32)?
                    .flatten_all()?
                    .to_vec1::<f32>()?;
                assert_eq!(actual.len(), reference.len());

                let rtol = if input.dtype() == DType::BF16 {
                    1.6e-2
                } else {
                    1.3e-6
                };
                for (index, (actual, reference)) in actual.into_iter().zip(reference).enumerate() {
                    assert!(actual.is_finite() && reference.is_finite());
                    assert!(
                        (actual - reference).abs() <= 1e-5 + rtol * reference.abs(),
                        "{} {name} element {index}: {actual} vs {reference}",
                        path.display()
                    );
                }
                Ok(())
            };
            let convolution = causal_conv(input, weight, kernel)?;
            if input.dtype() == DType::BF16 {
                assert_eq!(
                    convolution
                        .to_dtype(DType::F32)?
                        .flatten_all()?
                        .to_vec1::<f32>()?,
                    expected_conv
                        .to_dtype(DType::F32)?
                        .flatten_all()?
                        .to_vec1::<f32>()?
                );
            }
            assert_close(&convolution, expected_conv, "CPU convolution")?;
            assert_close(
                &causal_conv_silu(input, weight, kernel)?,
                expected,
                "CPU SiLU",
            )?;
            #[cfg(feature = "cuda")]
            {
                let device = test_cuda_device()?;
                let capability = match &device {
                    candle_core::Device::Cuda(cuda) => {
                        cuda.cuda_stream()
                            .context()
                            .compute_capability()
                            .map_err(|error| candle_core::Error::Msg(error.to_string()))?
                            .0
                    }
                    _ => unreachable!(),
                };
                if input.dtype() != DType::BF16 || capability >= 8 {
                    let result =
                        cuda_conv_silu(&input.to_device(&device)?, &weight.to_device(&device)?)?;
                    assert_close(&result, expected, "CUDA SiLU")?;
                }
            }
            #[cfg(feature = "metal")]
            {
                let device = candle_core::Device::new_metal(0)?;
                let result = crate::kernels::metal::conv_silu(
                    &input.to_device(&device)?,
                    &weight.to_device(&device)?,
                )?;

                let round = |value: f32| {
                    if input.dtype() == DType::BF16 {
                        let bits = value.to_bits();
                        f32::from_bits(bits.wrapping_add(0x7fff + ((bits >> 16) & 1)) & 0xffff0000)
                    } else {
                        value
                    }
                };
                let (batch, length, channels) = input.dims3()?;
                let xs = input
                    .to_dtype(DType::F32)?
                    .flatten_all()?
                    .to_vec1::<f32>()?;
                let ws = weight
                    .to_dtype(DType::F32)?
                    .flatten_all()?
                    .to_vec1::<f32>()?;
                let mut reference = Vec::with_capacity(xs.len());
                for b in 0..batch {
                    for token in 0..length {
                        for channel in 0..channels {
                            let mut sum = 0f32;
                            for tap in 0..kernel {
                                if token + tap + 1 >= kernel {
                                    let source = token + tap + 1 - kernel;
                                    let product = round(
                                        xs[(b * length + source) * channels + channel]
                                            * ws[channel * kernel + tap],
                                    );
                                    sum = round(sum + product);
                                }
                            }
                            reference.push(round(sum / (1. + (-sum).exp())));
                        }
                    }
                }
                let reference =
                    Tensor::from_vec(reference, input.shape().clone(), &candle_core::Device::Cpu)?
                        .to_dtype(input.dtype())?;
                assert_close(&result, &reference, "Metal retained per-tap SiLU")?;
            }
        }
        Ok(())
    }

    #[test]
    #[ignore = "external reference data"]
    fn delta_rule_matches_pinned_native_chunk_and_fp64_fixtures() -> Result<()> {
        let directory = crate::assets::fixtures("qwen35_delta")?;
        let device =
            crate::device::load().map_err(|error| candle_core::Error::Msg(error.to_string()))?;
        for preparation in ["f32", "bf16"] {
            let tensors = candle_core::safetensors::load(
                directory.join(format!("b2-l65-kh2-h4-d16-{preparation}.safetensors")),
                &device,
            )?;
            let q = &tensors["query_scaled_f32"];
            let k = &tensors["key_f32"];
            let v = &tensors["value_f32"];
            let g = &tensors["gate_f32"];
            let beta = &tensors["beta_f32"];
            #[allow(unused_mut)]
            let mut actuals = vec![(
                "tensor_chunk32",
                chunked_delta_rule_fallback(
                    &q.transpose(1, 2)?,
                    &k.transpose(1, 2)?,
                    &v.transpose(1, 2)?,
                    &g.transpose(1, 2)?,
                    &beta.transpose(1, 2)?,
                )?,
            )];
            #[cfg(feature = "cuda")]
            if device.is_cuda() {
                actuals.push((
                    "cuda_expanded",
                    cuda_delta_rule_token_major(q, k, v, g, beta)?,
                ));
                let indices = Tensor::from_vec(vec![0u32, 2], 2, &device)?;
                actuals.push((
                    "cuda_grouped",
                    cuda_delta_rule_token_major(
                        &q.index_select(&indices, 2)?,
                        &k.index_select(&indices, 2)?,
                        v,
                        g,
                        beta,
                    )?,
                ));
            }
            for (implementation, actual) in actuals {
                assert_eq!(actual.dims(), [2, 65, 4, 16]);
                let actual = actual.flatten_all()?.to_vec1::<f32>()?;
                for oracle in [
                    "native_chunk_f32",
                    "native_sequential_f32",
                    "fp64_recurrence_rounded_f32",
                ] {
                    let expected = tensors[oracle].flatten_all()?.to_vec1::<f32>()?;
                    assert_eq!(actual.len(), expected.len());
                    for (index, (actual, expected)) in actual.iter().zip(&expected).enumerate() {
                        assert!(
                            actual.is_finite()
                                && expected.is_finite()
                                && (actual - expected).abs() <= 1e-5 + 1.3e-6 * expected.abs(),
                            "{preparation} {implementation} vs {oracle}, element{index}: {actual} vs {expected}"
                        );
                    }
                }
            }
        }
        Ok(())
    }

    #[test]
    #[ignore = "external reference data"]
    fn recurrent_rounding_matches_native_fixtures() -> Result<()> {
        let root = crate::assets::fixtures("qwen35_recurrent")?;
        let device =
            crate::device::load().map_err(|error| candle_core::Error::Msg(error.to_string()))?;
        for (name, dtype, rtol) in [
            ("bf16", DType::BF16, 1.6e-2f32),
            ("f16", DType::F16, 1e-3),
            ("f32", DType::F32, 1.3e-6),
        ] {
            #[cfg(feature = "cuda")]
            if dtype == DType::BF16
                && let candle_core::Device::Cuda(cuda) = &device
                && cuda
                    .cuda_stream()
                    .context()
                    .compute_capability()
                    .map_err(|error| candle_core::Error::Msg(error.to_string()))?
                    .0
                    < 8
            {
                continue;
            }
            let data = candle_core::safetensors::load(
                root.join(format!("post-finalize-{name}.safetensors")),
                &device,
            )?;
            let actual = gated_norm_reference(
                &data["recurrent_f32"],
                &data["gate"],
                &data["weight"],
                1e-6,
                dtype,
            )?;
            let values =
                |tensor: &Tensor| tensor.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>();
            let check = |actual: &Tensor, expected: &Tensor| -> Result<()> {
                for (index, (actual, expected)) in
                    values(actual)?.iter().zip(values(expected)?).enumerate()
                {
                    assert!(actual.is_finite() && expected.is_finite());
                    assert!(
                        (actual - expected).abs() <= 1e-5 + rtol * expected.abs(),
                        "native {name} fixture index={index}: {actual} vs {expected}"
                    );
                }
                Ok(())
            };
            check(&actual, &data["expected"])?;
            check(&rounded_beta(&data["beta_input"])?, &data["expected_beta"])?;
            #[cfg(feature = "cuda")]
            if device.is_cuda() && dtype == DType::BF16 {
                check(
                    &cuda_post_finalize_bf16(
                        &data["recurrent_f32"],
                        &data["gate"],
                        &data["weight"],
                        1e-6,
                    )?,
                    &data["expected"],
                )?;
            }
        }
        Ok(())
    }

    #[test]
    fn chunked_delta_rule_matches_scalar_recurrence_across_chunk_boundary() {
        let length = 70;
        let key_dim = 4;
        let value_dim = 3;
        let make = |width: usize, factor: f32| {
            (0..length * width)
                .map(|index| ((index * 13 % 29) as f32 - 14.) * factor)
                .collect::<Vec<_>>()
        };
        let q = make(key_dim, 0.01);
        let k = make(key_dim, 0.012);
        let v = make(value_dim, 0.02);
        let beta = vec![0.45f32; length];
        let g = vec![-0.1f32; length];
        let device = candle_core::Device::Cpu;
        let output = chunked_delta_rule(
            &Tensor::from_vec(q.clone(), (1, 1, length, key_dim), &device).unwrap(),
            &Tensor::from_vec(k.clone(), (1, 1, length, key_dim), &device).unwrap(),
            &Tensor::from_vec(v.clone(), (1, 1, length, value_dim), &device).unwrap(),
            &Tensor::from_vec(g.clone(), (1, 1, length), &device).unwrap(),
            &Tensor::from_vec(beta.clone(), (1, 1, length), &device).unwrap(),
        )
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
        let mut state = vec![0f32; key_dim * value_dim];
        for step in 0..length {
            let decay = g[step].exp();
            for value in &mut state {
                *value *= decay;
            }
            for column in 0..value_dim {
                let remembered = (0..key_dim)
                    .map(|row| k[step * key_dim + row] * state[row * value_dim + column])
                    .sum::<f32>();
                let delta = (v[step * value_dim + column] - remembered) * beta[step];
                for row in 0..key_dim {
                    state[row * value_dim + column] += k[step * key_dim + row] * delta;
                }
                let expected = (0..key_dim)
                    .map(|row| q[step * key_dim + row] * state[row * value_dim + column])
                    .sum::<f32>();
                assert!((output[step * value_dim + column] - expected).abs() < 1e-4);
            }
        }
    }

    #[test]
    fn chunked_delta_rule_stays_finite_with_large_negative_gates() {
        let device = candle_core::Device::Cpu;
        let q = Tensor::ones((1, 1, 64, 4), DType::F32, &device).unwrap();
        let k = Tensor::ones((1, 1, 64, 4), DType::F32, &device).unwrap();
        let v = Tensor::ones((1, 1, 64, 3), DType::F32, &device).unwrap();
        let g = Tensor::new(-8f32, &device)
            .unwrap()
            .broadcast_as((1, 1, 64))
            .unwrap();
        let beta = Tensor::new(0.5f32, &device)
            .unwrap()
            .broadcast_as((1, 1, 64))
            .unwrap();
        let output = chunked_delta_rule(&q, &k, &v, &g, &beta)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert!(output.iter().all(|value| value.is_finite()));
    }

    #[cfg(feature = "metal")]
    #[test]
    fn chunked_delta_rule_runs_with_broadcast_gates_on_metal() {
        let device = candle_core::Device::new_metal(0).unwrap();
        let q = Tensor::ones((1, 1, 64, 4), DType::F32, &device).unwrap();
        let k = Tensor::ones((1, 1, 64, 4), DType::F32, &device).unwrap();
        let v = Tensor::ones((1, 1, 64, 3), DType::F32, &device).unwrap();
        let g = Tensor::new(-8f32, &device)
            .unwrap()
            .broadcast_as((1, 1, 64))
            .unwrap();
        let beta = Tensor::new(0.5f32, &device)
            .unwrap()
            .broadcast_as((1, 1, 64))
            .unwrap();
        let output = chunked_delta_rule(&q, &k, &v, &g, &beta)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert!(output.iter().all(|value| value.is_finite()));
    }
}

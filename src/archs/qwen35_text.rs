use crate::models::AttentionImplementation;
use candle_core::{D, DType, Result, Tensor};
use candle_nn::{
    Embedding, Linear, VarBuilder, embedding,
    ops::{sigmoid, softmax},
};
use serde::Deserialize;
use std::{fs, path::Path};

#[cfg(feature = "cuda")]
use candle_core::{
    CudaStorage, Storage,
    backend::BackendStorage,
    cuda_backend::{
        WrapErr,
        cudarc::driver::{LaunchConfig, PushKernelArg},
    },
    op::BackpropOp,
};
#[cfg(feature = "cuda")]
use half::bf16;

#[cfg(feature = "cuda")]
const CUDA_PTX: &str = include_str!(env!("SYS1_KERNEL_PTX_PATH"));

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
    Ok(Linear::new(vb.get((output, input), "weight")?, None))
}

struct RmsNorm {
    weight: Tensor,
    eps: f64,
}

impl RmsNorm {
    fn load(width: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        let weight = vb.get(width, "weight")?;
        Ok(Self { weight, eps })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let dtype = x.dtype();
        let x = x.to_dtype(DType::F32)?;
        let variance = x.sqr()?.mean_keepdim(D::Minus1)?;
        let scale = (variance + self.eps)?.sqrt()?.recip()?;
        x.broadcast_mul(&scale)?
            .broadcast_mul(&(&self.weight.to_dtype(DType::F32)? + 1.)?)?
            .to_dtype(dtype)
    }
}

struct Mlp {
    gate: Linear,
    up: Linear,
    down: Linear,
}

impl Mlp {
    fn load(config: &TextConfig, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            gate: linear(
                config.hidden_size,
                config.intermediate_size,
                vb.pp("gate_proj"),
            )?,
            up: linear(
                config.hidden_size,
                config.intermediate_size,
                vb.pp("up_proj"),
            )?,
            down: linear(
                config.intermediate_size,
                config.hidden_size,
                vb.pp("down_proj"),
            )?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        (x.apply(&self.gate)?.silu()? * x.apply(&self.up)?)?.apply(&self.down)
    }
}

struct FullAttention {
    q: Linear,
    k: Linear,
    v: Linear,
    output: Linear,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    rotary_dim: usize,
    rope_theta: f64,
    mrope_section: [usize; 3],
    implementation: AttentionImplementation,
}

impl FullAttention {
    fn load(
        config: &TextConfig,
        vb: VarBuilder,
        implementation: AttentionImplementation,
    ) -> Result<Self> {
        let rotary_dim =
            (config.head_dim as f64 * config.rope_parameters.partial_rotary_factor) as usize;
        if !rotary_dim.is_multiple_of(2) {
            candle_core::bail!("Qwen3.5 rotary dimension must be even")
        }
        if config.rope_parameters.mrope_section.iter().sum::<usize>() != rotary_dim / 2 {
            candle_core::bail!("Qwen3.5 mRoPE sections do not match the rotary dimension")
        }
        Ok(Self {
            q: linear(
                config.hidden_size,
                config.num_attention_heads * config.head_dim * 2,
                vb.pp("q_proj"),
            )?,
            k: linear(
                config.hidden_size,
                config.num_key_value_heads * config.head_dim,
                vb.pp("k_proj"),
            )?,
            v: linear(
                config.hidden_size,
                config.num_key_value_heads * config.head_dim,
                vb.pp("v_proj"),
            )?,
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
            rotary_dim,
            rope_theta: config.rope_parameters.rope_theta,
            mrope_section: config.rope_parameters.mrope_section,
            implementation,
        })
    }

    fn rotary(&self, x: &Tensor, positions: &[[u32; 3]]) -> Result<Tensor> {
        let (_, length, _, _) = x.dims4()?;
        let half = self.rotary_dim / 2;
        if positions.len() != length {
            candle_core::bail!("Qwen3.5 position count does not match sequence length")
        }
        let frequencies: Vec<f32> = positions
            .iter()
            .flat_map(|position| {
                (0..half).map(move |index| {
                    let axis = if index < self.mrope_section[1] * 3 && index % 3 == 1 {
                        1
                    } else if index < self.mrope_section[2] * 3 && index % 3 == 2 {
                        2
                    } else {
                        0
                    };
                    position[axis] as f32
                        / self
                            .rope_theta
                            .powf(2. * index as f64 / self.rotary_dim as f64)
                            as f32
                })
            })
            .collect();
        let phase = Tensor::from_vec(frequencies, (1, length, 1, half), x.device())?;
        let cos = Tensor::cat(&[&phase.cos()?, &phase.cos()?], D::Minus1)?.to_dtype(x.dtype())?;
        let sin = Tensor::cat(&[&phase.sin()?, &phase.sin()?], D::Minus1)?.to_dtype(x.dtype())?;
        let active = x.narrow(D::Minus1, 0, self.rotary_dim)?;
        let first = active.narrow(D::Minus1, 0, half)?;
        let second = active.narrow(D::Minus1, half, half)?;
        let rotated = Tensor::cat(&[&second.neg()?, &first], D::Minus1)?;
        let active = (active.broadcast_mul(&cos)? + rotated.broadcast_mul(&sin)?)?;
        Tensor::cat(
            &[
                &active,
                &x.narrow(D::Minus1, self.rotary_dim, self.head_dim - self.rotary_dim)?,
            ],
            D::Minus1,
        )
    }

    fn forward(&self, x: &Tensor, positions: &[[u32; 3]]) -> Result<Tensor> {
        let (batch, length, _) = x.dims3()?;
        let qg = x
            .apply(&self.q)?
            .reshape((batch, length, self.heads, 2, self.head_dim))?;
        let q = self.rotary(
            &self.q_norm.forward(&qg.narrow(3, 0, 1)?.squeeze(3)?)?,
            positions,
        )?;
        let gate = sigmoid(&qg.narrow(3, 1, 1)?.squeeze(3)?.reshape((
            batch,
            length,
            self.heads * self.head_dim,
        ))?)?;
        let k = self.rotary(
            &self.k_norm.forward(&x.apply(&self.k)?.reshape((
                batch,
                length,
                self.kv_heads,
                self.head_dim,
            ))?)?,
            positions,
        )?;
        let v = x
            .apply(&self.v)?
            .reshape((batch, length, self.kv_heads, self.head_dim))?;
        let scale = (self.head_dim as f32).powf(-0.5);
        #[cfg(feature = "metal")]
        // BF16 Metal SDPA produced non-finite values for Clef's 256-wide heads.
        if x.device().is_metal() && x.dtype() != DType::BF16 {
            let attention = candle_nn::ops::sdpa(
                &q.transpose(1, 2)?.contiguous()?,
                &k.transpose(1, 2)?.contiguous()?,
                &v.transpose(1, 2)?.contiguous()?,
                None,
                true,
                scale,
                1.0,
            )?
            .transpose(1, 2)?;
            let gated = (attention.reshape((batch, length, self.heads * self.head_dim))? * gate)?;
            return gated.apply(&self.output);
        }
        let attention = match self.implementation {
            AttentionImplementation::Eager => {
                let repeat = self.heads / self.kv_heads;
                let repeat_kv = |xs: &Tensor| -> Result<Tensor> {
                    xs.unsqueeze(3)?
                        .broadcast_as((batch, length, self.kv_heads, repeat, self.head_dim))?
                        .reshape((batch, length, self.heads, self.head_dim))?
                        .transpose(1, 2)
                };
                let q = q.transpose(1, 2)?;
                let k = repeat_kv(&k)?.transpose(D::Minus2, D::Minus1)?;
                let v = repeat_kv(&v)?;
                let mut chunks = Vec::with_capacity(length.div_ceil(128));
                for start in (0..length).step_by(128) {
                    let count = (length - start).min(128);
                    let mask = Tensor::from_vec(
                        (start..start + count)
                            .flat_map(|row| {
                                (0..length)
                                    .map(move |col| if col > row { f32::NEG_INFINITY } else { 0. })
                            })
                            .collect::<Vec<_>>(),
                        (1, 1, count, length),
                        x.device(),
                    )?
                    .to_dtype(x.dtype())?;
                    let scores = (contiguous_matmul(&q.narrow(2, start, count)?, &k)?
                        * scale as f64)?
                        .broadcast_add(&mask)?;
                    let probs =
                        softmax(&scores.to_dtype(DType::F32)?, D::Minus1)?.to_dtype(x.dtype())?;
                    chunks.push(contiguous_matmul(&probs, &v)?);
                }
                Tensor::cat(&chunks.iter().collect::<Vec<_>>(), 2)?.transpose(1, 2)?
            }
            #[cfg(feature = "flash-attn-2")]
            AttentionImplementation::FlashAttention2 => candle_flash_attn::flash_attn(
                &q.contiguous()?,
                &k.contiguous()?,
                &v.contiguous()?,
                scale,
                true,
            )?,
            #[cfg(feature = "flash-attn-3")]
            AttentionImplementation::FlashAttention3 => candle_flash_attn_v3::flash_attn(
                &q.contiguous()?,
                &k.contiguous()?,
                &v.contiguous()?,
                scale,
                true,
                false,
            )?,
            #[allow(unreachable_patterns)]
            other => candle_core::bail!("{} support is not compiled in", other.cli_name()),
        };
        let gated = (attention.reshape((batch, length, self.heads * self.head_dim))? * gate)?;
        gated.apply(&self.output)
    }
}

struct LinearAttention {
    qkv: Linear,
    z: Linear,
    b: Linear,
    a: Linear,
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
        Ok(Self {
            qkv: linear(
                config.hidden_size,
                key_width * 2 + value_width,
                vb.pp("in_proj_qkv"),
            )?,
            z: linear(config.hidden_size, value_width, vb.pp("in_proj_z"))?,
            b: linear(
                config.hidden_size,
                config.linear_num_value_heads,
                vb.pp("in_proj_b"),
            )?,
            a: linear(
                config.hidden_size,
                config.linear_num_value_heads,
                vb.pp("in_proj_a"),
            )?,
            conv: vb.get(
                (
                    key_width * 2 + value_width,
                    1,
                    config.linear_conv_kernel_dim,
                ),
                "conv1d.weight",
            )?,
            a_log: vb.get(config.linear_num_value_heads, "A_log")?,
            dt_bias: vb.get(config.linear_num_value_heads, "dt_bias")?,
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
        let projected = x.apply(&self.qkv)?;
        #[cfg(feature = "cuda")]
        let mixed = if x.device().is_cuda() && matches!(x.dtype(), DType::BF16 | DType::F32) {
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
                .broadcast_as((batch, length, self.key_heads, repeat, self.key_dim))?
                .reshape((batch, length, self.value_heads, self.key_dim))
        };
        #[cfg(feature = "metal")]
        let grouped_qk = x.device().is_metal() && self.key_dim == 128 && self.value_dim == 128;
        #[cfg(not(feature = "metal"))]
        let grouped_qk = false;
        let q = if grouped_qk { q } else { expand(q)? };
        let k = if grouped_qk { k } else { expand(k)? };
        let l2 = |xs: Tensor| -> Result<Tensor> {
            let denom = (xs.sqr()?.sum_keepdim(D::Minus1)? + 1e-6)?.sqrt()?;
            xs.broadcast_div(&denom)
        };
        let q = (l2(q.to_dtype(DType::F32)?)? * (self.key_dim as f64).powf(-0.5))?;
        let k = l2(k.to_dtype(DType::F32)?)?;
        let v = v.to_dtype(DType::F32)?;
        let beta = sigmoid(&x.apply(&self.b)?.to_dtype(DType::F32)?)?;
        let a = x.apply(&self.a)?.to_dtype(DType::F32)?;
        let a_log = self.a_log.to_dtype(DType::F32)?.exp()?.neg()?;
        let gate_input = a.broadcast_add(&self.dt_bias.to_dtype(DType::F32)?)?;
        let softplus = (gate_input.relu()? + (gate_input.abs()?.neg()?.exp()? + 1.)?.log()?)?;
        let g = softplus.broadcast_mul(&a_log)?;
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
        self.finish_delta(x, output, batch, length, value_width)
    }

    fn finish_delta(
        &self,
        x: &Tensor,
        output: Tensor,
        batch: usize,
        length: usize,
        value_width: usize,
    ) -> Result<Tensor> {
        let gate = x
            .apply(&self.z)?
            .reshape((batch, length, self.value_heads, self.value_dim))?;
        let variance = output.sqr()?.mean_keepdim(D::Minus1)?;
        #[cfg(feature = "metal")]
        if x.device().is_metal() && x.dtype() == DType::BF16 && self.value_dim == 128 {
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
        let output = output.broadcast_div(&(variance + self.eps)?.sqrt()?)?;
        let output = output.broadcast_mul(&self.norm.to_dtype(DType::F32)?)?;
        let gate = gate.to_dtype(DType::F32)?.silu()?;
        output
            .broadcast_mul(&gate)?
            .to_dtype(x.dtype())?
            .reshape((batch, length, value_width))?
            .apply(&self.output)
    }
}

fn causal_conv_silu(projected: &Tensor, conv: &Tensor, kernel: usize) -> Result<Tensor> {
    let (batch, length, channels) = projected.dims3()?;
    let padded = projected.pad_with_zeros(1, kernel - 1, 0)?;
    let weights = conv.reshape((channels, kernel))?;
    let mut mixed = Tensor::zeros(
        (batch, length, channels),
        projected.dtype(),
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
    mixed.silu()
}

#[cfg(feature = "cuda")]
fn cuda_conv_silu(input: &Tensor, weights: &Tensor) -> Result<Tensor> {
    let (batch, length, channels) = input.dims3()?;
    let (weight_channels, groups, kernel) = weights.dims3()?;
    if weight_channels != channels || groups != 1 || kernel == 0 || input.dtype() != weights.dtype()
    {
        candle_core::bail!("Unsupported Qwen3.5 CUDA convolution shape or dtype")
    }
    let input = input.contiguous()?;
    let weights = weights.contiguous()?;
    let (input_storage, input_layout) = input.storage_and_layout();
    let (weight_storage, weight_layout) = weights.storage_and_layout();
    let (Storage::Cuda(input_storage), Storage::Cuda(weight_storage)) =
        (&*input_storage, &*weight_storage)
    else {
        candle_core::bail!("Qwen3.5 convolution requires CUDA storage")
    };
    let (input_start, input_end) = input_layout.contiguous_offsets().ok_or_else(|| {
        candle_core::Error::Msg("Qwen3.5 convolution input is not contiguous".into())
    })?;
    let (weight_start, weight_end) = weight_layout.contiguous_offsets().ok_or_else(|| {
        candle_core::Error::Msg("Qwen3.5 convolution weights are not contiguous".into())
    })?;
    let device = input_storage.device().clone();
    let count = batch * length * channels;
    let config = LaunchConfig {
        grid_dim: (count.div_ceil(256) as u32, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    let count_arg = count as i32;
    let length_arg = length as i32;
    let channels_arg = channels as i32;
    let kernel_arg = kernel as i32;
    let output = match input.dtype() {
        DType::BF16 => {
            let source = input_storage
                .as_cuda_slice::<bf16>()?
                .slice(input_start..input_end);
            let filters = weight_storage
                .as_cuda_slice::<bf16>()?
                .slice(weight_start..weight_end);
            let output = unsafe { device.alloc::<bf16>(count)? };
            let function =
                device.get_or_load_custom_func("qwen35_conv_bf16", "qwen35", CUDA_PTX)?;
            let mut launch = function.builder();
            launch.arg(&source);
            launch.arg(&filters);
            launch.arg(&output);
            launch.arg(&count_arg);
            launch.arg(&length_arg);
            launch.arg(&channels_arg);
            launch.arg(&kernel_arg);
            unsafe { launch.launch(config) }.w()?;
            Storage::Cuda(CudaStorage::wrap_cuda_slice(output, device))
        }
        DType::F32 => {
            let source = input_storage
                .as_cuda_slice::<f32>()?
                .slice(input_start..input_end);
            let filters = weight_storage
                .as_cuda_slice::<f32>()?
                .slice(weight_start..weight_end);
            let output = unsafe { device.alloc::<f32>(count)? };
            let function = device.get_or_load_custom_func("qwen35_conv_f32", "qwen35", CUDA_PTX)?;
            let mut launch = function.builder();
            launch.arg(&source);
            launch.arg(&filters);
            launch.arg(&output);
            launch.arg(&count_arg);
            launch.arg(&length_arg);
            launch.arg(&channels_arg);
            launch.arg(&kernel_arg);
            unsafe { launch.launch(config) }.w()?;
            Storage::Cuda(CudaStorage::wrap_cuda_slice(output, device))
        }
        dtype => candle_core::bail!("Unsupported Qwen3.5 CUDA convolution dtype: {dtype:?}"),
    };
    Ok(Tensor::from_storage(
        output,
        (batch, length, channels),
        BackpropOp::none(),
        false,
    ))
}

#[cfg(feature = "cuda")]
fn cuda_delta_rule(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    g: &Tensor,
    beta: &Tensor,
) -> Result<Tensor> {
    let (batch, heads, length, key_dim) = q.dims4()?;
    let value_dim = v.dim(D::Minus1)?;
    if key_dim != 128
        || k.dims() != q.dims()
        || v.dims() != [batch, heads, length, value_dim]
        || g.dims() != [batch, heads, length]
        || beta.dims() != [batch, heads, length]
    {
        candle_core::bail!("Unsupported Qwen3.5 CUDA delta-rule shape")
    }
    let input = Tensor::cat(&[q, k, v, &g.unsqueeze(3)?, &beta.unsqueeze(3)?], 3)?.contiguous()?;
    let (storage, layout) = input.storage_and_layout();
    let Storage::Cuda(storage) = &*storage else {
        candle_core::bail!("Qwen3.5 delta rule requires CUDA storage")
    };
    let (start, end) = layout
        .contiguous_offsets()
        .ok_or_else(|| candle_core::Error::Msg("Qwen3.5 delta input is not contiguous".into()))?;
    let source = storage.as_cuda_slice::<f32>()?.slice(start..end);
    let device = storage.device().clone();
    let output = unsafe { device.alloc::<f32>(batch * heads * length * value_dim)? };
    let function = device.get_or_load_custom_func("qwen35_delta_f32", "qwen35_delta", CUDA_PTX)?;
    let config = LaunchConfig {
        grid_dim: ((batch * heads) as u32, value_dim.div_ceil(4) as u32, 1),
        block_dim: (128, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut launch = function.builder();
    launch.arg(&source);
    launch.arg(&output);
    let length = length as i32;
    let value_dim = value_dim as i32;
    launch.arg(&length);
    launch.arg(&value_dim);
    unsafe { launch.launch(config) }.w()?;
    let output = Tensor::from_storage(
        Storage::Cuda(CudaStorage::wrap_cuda_slice(output, device)),
        (batch, heads, length as usize, value_dim as usize),
        BackpropOp::none(),
        false,
    );
    output.transpose(1, 2)
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
        return cuda_delta_rule(q, k, v, g, beta);
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
    let (batch, heads, length, key_dim) = q.dims4()?;
    let value_dim = v.dim(D::Minus1)?;
    let mut state = Tensor::zeros((batch, heads, key_dim, value_dim), DType::F32, q.device())?;
    let mut outputs = Vec::with_capacity(length.div_ceil(32));
    let full_lower = Tensor::tril2(32, DType::F32, q.device())?;
    let full_identity = Tensor::eye(32, DType::F32, q.device())?;
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
            let lower = Tensor::tril2(count, DType::F32, q.device())?;
            let identity = Tensor::eye(count, DType::F32, q.device())?;
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
    Tensor::cat(&outputs.iter().collect::<Vec<_>>(), 2)?.transpose(1, 2)
}

fn contiguous_matmul(left: &Tensor, right: &Tensor) -> Result<Tensor> {
    left.contiguous()?.matmul(&right.contiguous()?)
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

    fn forward(&self, x: &Tensor, positions: &[[u32; 3]]) -> Result<Tensor> {
        let normalized = self.input_norm.forward(x)?;
        let mixed = match &self.mixer {
            Mixer::Full(layer) => layer.forward(&normalized, positions)?,
            Mixer::Linear(layer) => layer.forward(&normalized)?,
        };
        let x = (x + mixed)?;
        &x + self.mlp.forward(&self.post_norm.forward(&x)?)?
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
        self.forward_embeds(&ids.apply(&self.embedding)?, &positions)
    }

    pub fn embed(&self, ids: &Tensor) -> Result<Tensor> {
        ids.apply(&self.embedding)
    }

    pub fn forward_embeds(&self, embeds: &Tensor, positions: &[[u32; 3]]) -> Result<Tensor> {
        let mut x = embeds.clone();
        for layer in &self.layers {
            x = layer.forward(&x, positions)?;
        }
        self.norm.forward(&x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "cuda")]
    #[test]
    fn conv_cuda_matches_candle_across_batches_and_padding() {
        let (batch, length, channels, kernel) = (2, 7, 31, 4);
        let cpu = candle_core::Device::Cpu;
        let projected = Tensor::from_vec(
            (0..batch * length * channels)
                .map(|index| ((index * 17 % 41) as f32 - 20.) * 0.04)
                .collect::<Vec<_>>(),
            (batch, length, channels),
            &cpu,
        )
        .unwrap();
        let weights = Tensor::from_vec(
            (0..channels * kernel)
                .map(|index| ((index * 11 % 23) as f32 - 11.) * 0.02)
                .collect::<Vec<_>>(),
            (channels, 1, kernel),
            &cpu,
        )
        .unwrap();
        let cuda = candle_core::Device::new_cuda(0).unwrap();
        for dtype in [DType::BF16, DType::F32] {
            let projected = projected.to_dtype(dtype).unwrap();
            let weights = weights.to_dtype(dtype).unwrap();
            let expected = causal_conv_silu(&projected, &weights, kernel)
                .unwrap()
                .to_dtype(DType::F32)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            let actual = cuda_conv_silu(
                &projected.to_device(&cuda).unwrap(),
                &weights.to_device(&cuda).unwrap(),
            )
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
            let max_error = expected
                .iter()
                .zip(actual)
                .map(|(expected, actual)| (expected - actual).abs())
                .fold(0f32, f32::max);
            assert!(
                max_error < 0.001,
                "{dtype:?} convolution error: {max_error}"
            );
        }
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn delta_cuda_matches_chunked_reference() {
        let batch = 2;
        let heads = 2;
        let length = 290;
        let key_dim = 128;
        let value_dim = 16;
        let values = |width: usize, factor: f32| {
            (0..batch * heads * length * width)
                .map(|index| ((index * 13 % 29) as f32 - 14.) * factor)
                .collect::<Vec<_>>()
        };
        let cpu = candle_core::Device::Cpu;
        let q = Tensor::from_vec(
            values(key_dim, 0.001),
            (batch, heads, length, key_dim),
            &cpu,
        )
        .unwrap();
        let k = Tensor::from_vec(
            values(key_dim, 0.006),
            (batch, heads, length, key_dim),
            &cpu,
        )
        .unwrap();
        let v = Tensor::from_vec(
            values(value_dim, 0.008),
            (batch, heads, length, value_dim),
            &cpu,
        )
        .unwrap();
        let g = Tensor::from_vec(
            vec![-0.002f32; batch * heads * length],
            (batch, heads, length),
            &cpu,
        )
        .unwrap();
        let beta = Tensor::from_vec(
            vec![0.5f32; batch * heads * length],
            (batch, heads, length),
            &cpu,
        )
        .unwrap();
        let reference = chunked_delta_rule(&q, &k, &v, &g, &beta)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let cuda = candle_core::Device::new_cuda(0).unwrap();
        let actual = cuda_delta_rule(
            &q.to_device(&cuda).unwrap(),
            &k.to_device(&cuda).unwrap(),
            &v.to_device(&cuda).unwrap(),
            &g.to_device(&cuda).unwrap(),
            &beta.to_device(&cuda).unwrap(),
        )
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
        let max_error = reference
            .iter()
            .zip(actual)
            .map(|(expected, actual)| (expected - actual).abs())
            .fold(0f32, f32::max);
        assert!(max_error < 1e-4, "Max delta-rule error: {max_error}");
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

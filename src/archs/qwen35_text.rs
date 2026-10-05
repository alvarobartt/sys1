use crate::models::AttentionImplementation;
use candle_core::{D, DType, Result, Tensor};
use candle_nn::{
    Embedding, Linear, VarBuilder, embedding,
    ops::{sigmoid, softmax},
};
use serde::Deserialize;
use std::{fs, path::Path};

#[derive(Clone, Debug, Deserialize)]
pub struct Config {
    pub text_config: TextConfig,
    pub vision_config: VisionConfig,
    pub image_token_id: u32,
    pub video_token_id: u32,
}

#[derive(Clone, Debug, Deserialize)]
pub struct TextConfig {
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
        Ok(Self {
            weight: vb.get(width, "weight")?,
            eps,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let dtype = x.dtype();
        let x = x.to_dtype(DType::F32)?;
        let variance = x.sqr()?.mean_keepdim(D::Minus1)?;
        let scale = (variance + self.eps)?.sqrt()?.recip()?;
        let weight = (&self.weight.to_dtype(DType::F32)? + 1.)?;
        x.broadcast_mul(&scale)?
            .broadcast_mul(&weight)?
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
        if x.device().is_metal() {
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
        let channels = key_width * 2 + value_width;
        let padded = projected.pad_with_zeros(1, self.kernel - 1, 0)?;
        let weights = self.conv.reshape((channels, self.kernel))?;
        let mut mixed = Tensor::zeros((batch, length, channels), x.dtype(), x.device())?;
        for tap in 0..self.kernel {
            let source = padded.narrow(1, tap, length)?;
            let weight = weights
                .narrow(1, tap, 1)?
                .squeeze(1)?
                .reshape((1, 1, channels))?;
            mixed = (mixed + source.broadcast_mul(&weight)?)?;
        }
        let mixed = mixed.silu()?;
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
        let q = expand(q)?;
        let k = expand(k)?;
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
        let output = chunked_delta_rule(&q, &k, &v, &g, &beta)?;
        let variance = output.sqr()?.mean_keepdim(D::Minus1)?;
        let output = output.broadcast_div(&(variance + self.eps)?.sqrt()?)?;
        let output = output.broadcast_mul(&self.norm.to_dtype(DType::F32)?)?;
        let gate = x
            .apply(&self.z)?
            .reshape((batch, length, self.value_heads, self.value_dim))?
            .to_dtype(DType::F32)?
            .silu()?;
        output
            .broadcast_mul(&gate)?
            .to_dtype(x.dtype())?
            .reshape((batch, length, value_width))?
            .apply(&self.output)
    }
}

fn chunked_delta_rule(
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
    for start in (0..length).step_by(32) {
        let count = (length - start).min(32);
        let qi = q.narrow(2, start, count)?;
        let ki = k.narrow(2, start, count)?;
        let vi = v.narrow(2, start, count)?;
        let bi = beta.narrow(2, start, count)?.unsqueeze(3)?;
        let gi = g.narrow(2, start, count)?.contiguous()?.cumsum(2)?;
        let lower = Tensor::tril2(count, DType::F32, q.device())?;
        let identity = Tensor::eye(count, DType::F32, q.device())?;
        let strictly_lower = (&lower - &identity)?;
        let decay = gi
            .unsqueeze(3)?
            .broadcast_sub(&gi.unsqueeze(2)?)?
            .broadcast_mul(&lower)?
            .exp()?
            .broadcast_mul(&lower)?;
        let key_beta = ki.broadcast_mul(&bi)?;
        let value_beta = vi.broadcast_mul(&bi)?;
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
            contiguous_matmul(&inverse, &key_beta.broadcast_mul(&gi.exp()?.unsqueeze(3)?)?)?;
        let value_new = (&value - contiguous_matmul(&key_cumdecay, &state)?)?;
        let attn =
            contiguous_matmul(&qi, &ki.transpose(D::Minus2, D::Minus1)?)?.broadcast_mul(&decay)?;
        let intermediate = contiguous_matmul(&qi.broadcast_mul(&gi.exp()?.unsqueeze(3)?)?, &state)?;
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

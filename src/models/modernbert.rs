use super::AttentionImplementation;
use candle_core::{D, DType, Device, Result, Tensor};
use candle_nn::{
    Embedding, LayerNorm, Linear, Module, VarBuilder, embedding, layer_norm_no_bias, ops::softmax,
};
use serde::Deserialize;
use std::{
    collections::HashMap,
    fs,
    path::Path,
    sync::{Arc, Mutex},
};

const ATTENTION_MASK_VALUE: f32 = -10_000.0;

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

    pub fn max_position_embeddings(&self) -> usize {
        self.max_position_embeddings
    }

    fn global_rope_theta(&self) -> f64 {
        self.rope_parameters["full_attention"].rope_theta
    }

    fn local_rope_theta(&self) -> f64 {
        self.rope_parameters["sliding_attention"].rope_theta
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
        let qkv = xs
            .to_dtype(self.compute_dtype)?
            .apply(&self.qkv)?
            .reshape((batch, length, 3, self.heads, self.head_size))?
            .permute((2, 0, 3, 1, 4))?;
        let q = qkv.get(0)?;
        let k = qkv.get(1)?;
        let v = qkv.get(2)?;
        let (q, k) = self.rotary.apply(&q, &k)?;
        let scale = (self.head_size as f64).powf(-0.5);

        #[cfg(feature = "metal")]
        let attention = if xs.device().is_metal() {
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

        attention
            .transpose(1, 2)?
            .reshape((batch, length, hidden))?
            .apply(&self.projection)
    }
}

pub(super) struct AttentionOptions<'a> {
    pub mask: Option<&'a Tensor>,
    pub implementation: AttentionImplementation,
    pub lengths: &'a [usize],
    pub window: Option<usize>,
}

pub(super) fn scaled_dot_product_attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    scale: f64,
    options: AttentionOptions<'_>,
) -> Result<Tensor> {
    match options.implementation {
        AttentionImplementation::Eager => {
            let scores = (q * scale)?.matmul(&k.transpose(D::Minus2, D::Minus1)?)?;
            let scores = match options.mask {
                Some(mask) => scores.to_dtype(mask.dtype())?.broadcast_add(mask)?,
                None => scores,
            };
            let probabilities = attention_softmax(&scores)?;
            probabilities.to_dtype(v.dtype())?.matmul(v)
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

fn attention_softmax(scores: &Tensor) -> Result<Tensor> {
    if matches!(scores.dtype(), DType::F16 | DType::BF16) {
        softmax(&scores.to_dtype(DType::F32)?, D::Minus1)?.to_dtype(scores.dtype())
    } else {
        softmax(scores, D::Minus1)
    }
}

#[cfg(any(feature = "flash-attn-2", feature = "flash-attn-3"))]
fn flash_attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    lengths: &[usize],
    scale: f32,
    window: Option<usize>,
    implementation: AttentionImplementation,
) -> Result<Tensor> {
    let (batch, heads, max_length, head_size) = q.dims4()?;
    if lengths.len() != batch || lengths.iter().any(|&length| length > max_length) {
        candle_core::bail!(
            "invalid Flash Attention sequence lengths {:?} for shape {:?}",
            lengths,
            q.shape()
        )
    }
    let q = q.transpose(1, 2)?.contiguous()?;
    let k = k.transpose(1, 2)?.contiguous()?;
    let v = v.transpose(1, 2)?.contiguous()?;
    let attention = if lengths.iter().all(|&length| length == max_length) {
        match implementation {
            #[cfg(feature = "flash-attn-2")]
            AttentionImplementation::FlashAttention2 => {
                candle_flash_attn::flash_attn_windowed(&q, &k, &v, scale, window, window)?
            }
            #[cfg(feature = "flash-attn-3")]
            AttentionImplementation::FlashAttention3 => {
                candle_flash_attn_v3::flash_attn_windowed(&q, &k, &v, scale, window, window, false)?
            }
            _ => candle_core::bail!("{} support is not compiled in", implementation.cli_name()),
        }
    } else {
        let mut indices = Vec::with_capacity(lengths.iter().sum());
        let mut cumulative = Vec::with_capacity(batch + 1);
        cumulative.push(0u32);
        for (row, &length) in lengths.iter().enumerate() {
            indices.extend((0..length).map(|column| (row * max_length + column) as u32));
            cumulative.push(cumulative.last().copied().unwrap() + length as u32);
        }
        let indices = Tensor::from_vec(
            indices,
            cumulative.last().copied().unwrap() as usize,
            q.device(),
        )?;
        let cumulative = Tensor::from_vec(cumulative, batch + 1, q.device())?;
        let pack = |tensor: &Tensor| {
            tensor
                .reshape((batch * max_length, heads, head_size))?
                .index_select(&indices, 0)
        };
        let packed_q = pack(&q)?;
        let packed_k = pack(&k)?;
        let packed_v = pack(&v)?;
        let packed = match implementation {
            #[cfg(feature = "flash-attn-2")]
            AttentionImplementation::FlashAttention2 => {
                candle_flash_attn::flash_attn_varlen_windowed(
                    &packed_q,
                    &packed_k,
                    &packed_v,
                    &cumulative,
                    &cumulative,
                    max_length,
                    max_length,
                    scale,
                    window,
                    window,
                )?
            }
            #[cfg(feature = "flash-attn-3")]
            AttentionImplementation::FlashAttention3 => {
                candle_flash_attn_v3::flash_attn_varlen_windowed(
                    &packed_q,
                    &packed_k,
                    &packed_v,
                    &cumulative,
                    &cumulative,
                    max_length,
                    max_length,
                    scale,
                    window,
                    window,
                    false,
                )?
            }
            _ => candle_core::bail!("{} support is not compiled in", implementation.cli_name()),
        };
        Tensor::zeros(
            (batch * max_length, heads, head_size),
            packed.dtype(),
            packed.device(),
        )?
        .index_add(&indices, &packed, 0)?
        .reshape((batch, max_length, heads, head_size))?
    };
    attention.transpose(1, 2)
}

#[cfg(not(any(feature = "flash-attn-2", feature = "flash-attn-3")))]
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
        let parts = xs
            .to_dtype(self.compute_dtype)?
            .apply(&self.input)?
            .chunk(2, D::Minus1)?;
        (&parts[0].gelu_erf()? * &parts[1])?.apply(&self.output)
    }
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
        local_mask: &Tensor,
        lengths: &[usize],
        local_window: usize,
    ) -> Result<Tensor> {
        let normalized = match &self.attention_norm {
            Some(norm) => xs.apply(norm)?,
            None => xs.clone(),
        };
        let mask = if self.uses_local_attention {
            Some(match global_mask {
                Some(global_mask) => global_mask.broadcast_add(local_mask)?,
                None => local_mask.clone(),
            })
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
    norm: LayerNorm,
    layers: Vec<Layer>,
    final_norm: LayerNorm,
    local_attention_size: usize,
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
            config.global_rope_theta(),
            vb.device(),
        )?);
        let local_rotary = Arc::new(RotaryEmbedding::new(
            vb.dtype(),
            config,
            config.local_rope_theta(),
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
            dtype: vb.dtype(),
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
        let global_mask = has_padding
            .then(|| global_attention_mask(mask, length, self.dtype))
            .transpose()?;
        let local_mask = self.local_mask(length, ids.device())?;
        let local_window = self.local_attention_size / 2;
        let mut xs = ids.apply(&self.embeddings)?.apply(&self.norm)?;
        for layer in &self.layers {
            xs = layer.forward(
                &xs,
                global_mask.as_ref(),
                &local_mask,
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

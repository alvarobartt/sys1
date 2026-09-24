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
}

impl Attention {
    fn load(
        vb: VarBuilder,
        config: &Config,
        rotary: Arc<RotaryEmbedding>,
        compute_dtype: DType,
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
        })
    }

    fn forward(&self, xs: &Tensor, mask: Option<&Tensor>) -> Result<Tensor> {
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
            unfused_attention(&q, &k, &v, mask, scale)?
        };
        #[cfg(not(feature = "metal"))]
        let attention = unfused_attention(&q, &k, &v, mask, scale)?;

        attention
            .transpose(1, 2)?
            .reshape((batch, length, hidden))?
            .apply(&self.projection)
    }
}

fn unfused_attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    mask: Option<&Tensor>,
    scale: f64,
) -> Result<Tensor> {
    let scores = (q * scale)?.matmul(&k.transpose(D::Minus2, D::Minus1)?)?;
    let scores = match mask {
        Some(mask) => scores.to_dtype(mask.dtype())?.broadcast_add(mask)?,
        None => scores,
    };
    let probabilities = if scores.dtype() == DType::F16 {
        softmax(&scores.to_dtype(DType::F32)?, D::Minus1)?.to_dtype(DType::F16)?
    } else {
        softmax(&scores, D::Minus1)?
    };
    probabilities.to_dtype(v.dtype())?.matmul(v)
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
    ) -> Result<Self> {
        Ok(Self {
            attention: Attention::load(vb.pp("attn"), config, rotary, compute_dtype)?,
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
            .forward(&normalized, mask.as_ref())?
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
    pub fn load(vb: VarBuilder, config: &Config, compute_dtype: DType) -> Result<Self> {
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

    pub fn forward(&self, ids: &Tensor, mask: &Tensor, has_padding: bool) -> Result<Tensor> {
        let length = ids.dim(1)?;
        let global_mask = has_padding
            .then(|| global_attention_mask(mask, length, self.dtype))
            .transpose()?;
        let local_mask = self.local_mask(length, ids.device())?;
        let mut xs = ids.apply(&self.embeddings)?.apply(&self.norm)?;
        for layer in &self.layers {
            xs = layer.forward(&xs, global_mask.as_ref(), &local_mask)?;
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
    ((1.0 - expanded)? * f32::MIN as f64)?.to_dtype(dtype)
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
                    f32::NEG_INFINITY
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
                vec![0.0, 0.0, f32::NEG_INFINITY, f32::NEG_INFINITY],
                vec![0.0, 0.0, 0.0, f32::NEG_INFINITY],
                vec![f32::NEG_INFINITY, 0.0, 0.0, 0.0],
                vec![f32::NEG_INFINITY, f32::NEG_INFINITY, 0.0, 0.0],
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

        assert_eq!(mask, vec![0.0, 0.0, f32::MIN, 0.0, 0.0, f32::MIN]);
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
        let encoder = Encoder::load(vb, &config, DType::F32)?;

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

        let mut mask = vec![1f32; ids.len()];
        ids.resize(32, pad_id);
        mask.resize(32, 0.0);

        let length = ids.len();
        let ids = Tensor::from_vec(ids, (1, length), &device)?;
        let mask = Tensor::from_vec(mask, (1, length), &device)?;
        let hidden = encoder.forward(&ids, &mask, true)?;

        insta::assert_yaml_snapshot!(
            "modernbert_fp32_hidden_states",
            hidden.i((0, 0, 0..128))?.to_vec1::<f32>()?,
            {
                "[]" => insta::rounded_redaction(3),
            }
        );

        Ok(())
    }
}

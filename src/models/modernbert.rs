use candle_core::Tensor;
use candle_nn::VarBuilder;
use candle_transformers::models::modernbert::{Config as CandleConfig, ModernBert};
use serde::Deserialize;
use std::{collections::HashMap, fs, path::Path};

#[derive(Deserialize)]
pub struct Config {
    vocab_size: usize,
    hidden_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    intermediate_size: usize,
    max_position_embeddings: usize,
    layer_norm_eps: f64,
    pad_token_id: u32,
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

    fn candle(&self) -> CandleConfig {
        CandleConfig {
            vocab_size: self.vocab_size,
            hidden_size: self.hidden_size,
            num_hidden_layers: self.num_hidden_layers,
            num_attention_heads: self.num_attention_heads,
            intermediate_size: self.intermediate_size,
            max_position_embeddings: self.max_position_embeddings,
            layer_norm_eps: self.layer_norm_eps,
            pad_token_id: self.pad_token_id,
            global_attn_every_n_layers: self.global_attn_every_n_layers,
            global_rope_theta: self.rope_parameters["full_attention"].rope_theta,
            local_attention: self.local_attention,
            local_rope_theta: self.rope_parameters["sliding_attention"].rope_theta,
            classifier_config: None,
        }
    }
}

pub struct Encoder {
    inner: ModernBert,
}

impl Encoder {
    pub fn load(vb: VarBuilder, config: &Config) -> candle_core::Result<Self> {
        Ok(Self {
            inner: ModernBert::load(vb, &config.candle())?,
        })
    }

    pub fn forward(&self, ids: &Tensor, attention_mask: &Tensor) -> candle_core::Result<Tensor> {
        self.inner.forward(ids, attention_mask)
    }
}

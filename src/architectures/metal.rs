use super::{modernbert, qwen35_text};
use crate::models::AttentionImplementation;
use candle_core::{Result, Tensor};
use candle_nn::VarBuilder;

pub(crate) enum QwenDecoder {
    Eager(qwen35_text::Decoder),
    Metal(qwen35_text::metal::Decoder),
}

impl QwenDecoder {
    pub(crate) fn load(
        config: &qwen35_text::TextConfig,
        vb: VarBuilder,
        attention: AttentionImplementation,
    ) -> Result<Self> {
        if vb.device().is_metal() {
            return qwen35_text::metal::Decoder::load(config, vb, attention).map(Self::Metal);
        }
        qwen35_text::Decoder::load(config, vb, attention).map(Self::Eager)
    }
    pub(crate) fn forward(&self, ids: &Tensor) -> Result<Tensor> {
        match self {
            Self::Eager(model) => model.forward(ids),
            Self::Metal(model) => model.forward(ids),
        }
    }
    pub(crate) fn embed(&self, ids: &Tensor) -> Result<Tensor> {
        match self {
            Self::Eager(model) => model.embed(ids),
            Self::Metal(model) => model.embed(ids),
        }
    }
    pub(crate) fn forward_embeds(&self, embeds: &Tensor, positions: &[[u32; 3]]) -> Result<Tensor> {
        match self {
            Self::Eager(model) => model.forward_embeds(embeds, positions),
            Self::Metal(model) => model.forward_embeds(embeds, positions),
        }
    }
}

pub(crate) enum ModernBertEncoder {
    Eager(modernbert::Encoder),
    Metal(modernbert::metal::Encoder),
}

impl ModernBertEncoder {
    pub(crate) fn load(
        vb: VarBuilder,
        config: &modernbert::Config,
        compute_dtype: candle_core::DType,
        attention: AttentionImplementation,
    ) -> Result<Self> {
        if vb.device().is_metal() {
            return modernbert::metal::Encoder::load(vb, config, compute_dtype, attention)
                .map(Self::Metal);
        }
        modernbert::Encoder::load(vb, config, compute_dtype, attention).map(Self::Eager)
    }
    pub(crate) fn forward(
        &self,
        ids: &Tensor,
        mask: &Tensor,
        lengths: &[usize],
        has_padding: bool,
    ) -> Result<Tensor> {
        match self {
            Self::Eager(model) => model.forward(ids, mask, lengths, has_padding),
            Self::Metal(model) => model.forward(ids, mask, lengths, has_padding),
        }
    }
}

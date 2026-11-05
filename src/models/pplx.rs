use super::{AttentionImplementation, DecisionModel};
use crate::{
    architectures::{heads::readout::Readout, qwen35_text::inference::Backbone},
    inference::qwen35::{Adapter, Qwen35},
    protocols::{Encoded, inline::InlineProtocol},
    schema::{ApiError, DecisionRequest, DecisionResponse},
};
use candle_core::{DType, Tensor};
use candle_nn::VarBuilder;
use std::path::Path;

struct InlineAdapter {
    protocol: InlineProtocol,
    head: Readout,
}
impl Adapter for InlineAdapter {
    fn splits_requests(&self) -> bool {
        true
    }
    fn plan(
        &self,
        backbone: &Backbone,
        request: &DecisionRequest,
    ) -> Result<Vec<Encoded>, ApiError> {
        self.protocol.plan(backbone, request)
    }
    fn encode(&self, backbone: &Backbone, request: &DecisionRequest) -> Result<Encoded, ApiError> {
        self.protocol.encode(backbone, request)
    }
    fn score(
        &self,
        encoded: &Encoded,
        _ids: &Tensor,
        hidden: &Tensor,
    ) -> candle_core::Result<Vec<Vec<f32>>> {
        self.head
            .forward(&[encoded], hidden)?
            .pop()
            .ok_or_else(|| candle_core::Error::Msg("missing readout result".into()))
    }
    #[cfg(feature = "cuda")]
    fn score_packed(
        &self,
        encoded: &[&Encoded],
        hidden: &Tensor,
    ) -> candle_core::Result<Vec<Vec<Vec<f32>>>> {
        self.head.forward(encoded, hidden)
    }
    fn response(
        &self,
        encoded: &Encoded,
        logits: Vec<Vec<f32>>,
    ) -> Result<DecisionResponse, ApiError> {
        self.protocol.response(encoded, logits)
    }
}

pub struct Pplx(Qwen35<InlineAdapter>);
impl Pplx {
    pub fn load(
        path: &Path,
        dtype: DType,
        max_model_len: Option<usize>,
        attention: AttentionImplementation,
    ) -> anyhow::Result<Self> {
        let config: crate::protocols::inline::Config =
            serde_json::from_slice(&std::fs::read(path.join("decision_config.json"))?)?;
        let (backbone, readout) = Backbone::load(
            path,
            dtype,
            max_model_len,
            attention,
            false,
            config.max_pixels,
            |config, vb, _attention, tokenizer| {
                let protocol = InlineProtocol::load(path, tokenizer)?;
                let weights = unsafe {
                    VarBuilder::from_mmaped_safetensors(
                        &[path.join("readout.safetensors")],
                        dtype,
                        vb.device(),
                    )?
                };
                let head = Readout::load(
                    config.text_config.hidden_size,
                    protocol.config.codes.len(),
                    weights,
                )?;
                Ok(InlineAdapter { protocol, head })
            },
        )?;
        Ok(Self(Qwen35::new(backbone, readout)))
    }
}
impl DecisionModel for Pplx {
    fn supports_images(&self) -> bool {
        true
    }
    fn predict_batch(
        &self,
        requests: Vec<DecisionRequest>,
    ) -> Vec<Result<DecisionResponse, ApiError>> {
        self.0.predict_batch(requests)
    }
    fn predict_batch_incremental(
        &self,
        requests: Vec<DecisionRequest>,
        complete: &mut dyn FnMut(usize, Result<DecisionResponse, ApiError>),
    ) {
        self.0.predict_batch_incremental(requests, complete);
    }
}

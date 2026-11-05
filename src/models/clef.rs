use super::{AttentionImplementation, DecisionModel};
use crate::{
    architectures::{heads::joint, qwen35_text::inference::Backbone},
    inference::qwen35::{Adapter, Qwen35},
    protocols::{Encoded, joint::JointProtocol},
    schema::{ApiError, DecisionRequest, DecisionResponse},
};
use candle_core::{DType, Tensor};
use candle_nn::VarBuilder;
use std::path::Path;

struct JointAdapter {
    protocol: JointProtocol,
    head: joint::JointHead,
    output_embedding: Tensor,
}
impl Adapter for JointAdapter {
    fn encode(&self, backbone: &Backbone, request: &DecisionRequest) -> Result<Encoded, ApiError> {
        self.protocol.encode(backbone, request)
    }
    fn score(
        &self,
        encoded: &Encoded,
        ids: &Tensor,
        hidden: &Tensor,
    ) -> candle_core::Result<Vec<Vec<f32>>> {
        self.head
            .forward(hidden, ids, &encoded.questions, &self.output_embedding)
    }
    fn response(
        &self,
        encoded: &Encoded,
        logits: Vec<Vec<f32>>,
    ) -> Result<DecisionResponse, ApiError> {
        self.protocol.response_logits(encoded, logits)
    }
    fn supports_videos(&self) -> bool {
        true
    }
    fn needs_confirmation(&self, response: &DecisionResponse) -> bool {
        JointProtocol::response_needs_singleton_confirmation(response)
    }
    fn confirm_batches(&self) -> bool {
        self.output_embedding
            .dim(1)
            .map_or(true, |width| width != 4096)
    }
    #[cfg(feature = "cuda")]
    fn score_packed(
        &self,
        encoded: &[&Encoded],
        hidden: &Tensor,
    ) -> candle_core::Result<Vec<Vec<Vec<f32>>>> {
        self.head
            .forward_batch(encoded, hidden, &self.output_embedding)
    }
}

pub struct Clef(Qwen35<JointAdapter>);
impl Clef {
    pub fn load(
        path: &Path,
        dtype: DType,
        max_model_len: Option<usize>,
        attention: AttentionImplementation,
    ) -> anyhow::Result<Self> {
        let (backbone, adapter) = Backbone::load(
            path,
            dtype,
            max_model_len,
            attention,
            true,
            16_777_216,
            |config, vb, attention, _tokenizer| {
                let head_config = joint::Config::load(&path.join("joint_head_config.json"))?;
                anyhow::ensure!(
                    head_config.hidden_size == config.text_config.hidden_size,
                    "Clef head hidden size differs from the decoder"
                );
                let output_embedding = vb.get(
                    (
                        config.text_config.vocab_size,
                        config.text_config.hidden_size,
                    ),
                    "lm_head.weight",
                )?;
                let head_vb = unsafe {
                    VarBuilder::from_mmaped_safetensors(
                        &[path.join("joint_head.safetensors")],
                        dtype,
                        vb.device(),
                    )?
                };
                let head = joint::JointHead::load(&head_config, head_vb, attention)?;
                Ok(JointAdapter {
                    protocol: JointProtocol::default(),
                    head,
                    output_embedding,
                })
            },
        )?;
        Ok(Self(Qwen35::new(backbone, adapter)))
    }
}
impl DecisionModel for Clef {
    fn supports_images(&self) -> bool {
        true
    }
    fn supports_videos(&self) -> bool {
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

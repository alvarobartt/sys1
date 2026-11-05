use super::{AttentionImplementation, DecisionModel};
#[cfg(feature = "metal")]
use crate::architectures::metal::ModernBertEncoder;
#[cfg(not(feature = "metal"))]
use crate::architectures::modernbert::Encoder as ModernBertEncoder;
use crate::{
    architectures::{
        heads::encoder::{HeadLayer, layer_norm, linear_dtype},
        modernbert::Config as ModernBertConfig,
    },
    device,
    inference::encoder::{TokenClassifier, execution_dtypes, validate_attention},
    protocols::ordinal::{Config, OrdinalProtocol},
    schema::{ApiError, DecisionRequest, DecisionResponse},
    tokenizer,
};
use candle_core::DType;
use candle_nn::{VarBuilder, embedding};
use std::{fs, path::Path, sync::atomic::AtomicUsize};

pub struct Laya(TokenClassifier);
impl Laya {
    pub fn load(
        path: &Path,
        dtype: Option<DType>,
        max_model_len: Option<usize>,
        attention: AttentionImplementation,
    ) -> anyhow::Result<Self> {
        let device = device::load()?;
        let dtype = crate::models::dtype::resolve(path, crate::models::Architecture::Laya, dtype)?;
        let (model_dtype, compute_dtype) = execution_dtypes(dtype, device.is_cuda());
        Self::build_variant(path, model_dtype, compute_dtype, max_model_len, attention)
    }

    fn build_variant(
        path: &Path,
        model_dtype: DType,
        compute_dtype: DType,
        max_model_len: Option<usize>,
        attention: AttentionImplementation,
    ) -> anyhow::Result<Self> {
        let device = device::load()?;
        let attention = if attention == AttentionImplementation::Auto {
            crate::models::attention_for_device(compute_dtype, &device)?
        } else {
            attention
        };
        validate_attention(attention, &device, compute_dtype)?;
        let mut config: Config =
            serde_json::from_slice(&fs::read(path.join("rl_agent_config.json"))?)?;
        let encoder_config = ModernBertConfig::load(&path.join("encoder/config.json"))?;
        if let Some(max_model_len) = max_model_len {
            anyhow::ensure!(
                max_model_len <= encoder_config.max_position_embeddings(),
                "--max-model-len {max_model_len} exceeds the encoder limit of {}",
                encoder_config.max_position_embeddings()
            );
            config.max_len = max_model_len;
        }
        let weights = path.join("model.safetensors");
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[weights], model_dtype, &device)? };
        let encoder_vb = vb.clone().rename_f(|name| {
            name.strip_prefix("model.")
                .map(|name| format!("encoder.{name}"))
                .unwrap_or_else(|| name.to_owned())
        });
        let encoder =
            ModernBertEncoder::load(encoder_vb, &encoder_config, compute_dtype, attention)?;
        let head = (0..2)
            .map(|index| {
                HeadLayer::load(
                    vb.pp(format!("head.layers.{index}")),
                    encoder_config.hidden_size(),
                    compute_dtype,
                    attention,
                )
            })
            .collect::<candle_core::Result<Vec<_>>>()?;
        let tokenizer_path = path.join("tokenizer/tokenizer.v1.json");
        let legacy_tokenizer_path = path.join("tokenizer/tokenizer.json");
        let tokenizer_path = if tokenizer_path.exists() {
            &tokenizer_path
        } else {
            &legacy_tokenizer_path
        };
        let tokenizer_json = fs::read(tokenizer_path)?;
        let tokenizer = tokenizer::from_json(&tokenizer_json)?;
        let token = |values: &[&str]| {
            values
                .iter()
                .find_map(|value| tokenizer.token_to_id(value))
                .ok_or_else(|| anyhow::anyhow!("tokenizer is missing one of {values:?}"))
        };
        let pad_id = token(&["[PAD]", "<pad>"])?;
        let cls_id = token(&["[CLS]", "<bos>"])?;
        let sep_id = token(&["[SEP]", "<eos>"])?;
        let mask_id = token(&["[MASK]", "<mask>"])?;
        let model = TokenClassifier {
            activation_budget: AtomicUsize::new(usize::MAX),
            activation_bytes_per_token: encoder_config
                .hidden_size()
                .saturating_mul(16)
                .saturating_add(encoder_config.intermediate_size().saturating_mul(3))
                .saturating_mul(4),
            attention_bytes_per_cell: if matches!(
                attention,
                AttentionImplementation::Auto | AttentionImplementation::Eager
            ) {
                encoder_config
                    .attention_heads()
                    .max(head.iter().map(|layer| layer.heads).max().unwrap_or(1))
                    .saturating_mul(12)
            } else {
                0
            },
            protocol: OrdinalProtocol {
                tokenizer,
                config,
                pad_id,
                cls_id,
                sep_id,
                mask_id,
            },
            encoder,
            head,
            type_embedding: embedding(3, encoder_config.hidden_size(), vb.pp("type_emb"))?,
            scorer_norm: layer_norm(encoder_config.hidden_size(), 1e-5, vb.pp("scorer.0"))?,
            scorer_in: linear_dtype(
                encoder_config.hidden_size(),
                encoder_config.hidden_size(),
                vb.pp("scorer.1"),
                compute_dtype,
            )?,
            scorer_out: linear_dtype(
                encoder_config.hidden_size(),
                1,
                vb.pp("scorer.3"),
                compute_dtype,
            )?,
            device,
            compute_dtype,
        };
        model.refresh_activation_budget();
        Ok(Self(model))
    }
}
impl DecisionModel for Laya {
    fn predict_batch(
        &self,
        requests: Vec<DecisionRequest>,
    ) -> Vec<Result<DecisionResponse, ApiError>> {
        self.0.predict_batch(requests)
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocols::ordinal::{bucket, probability};
    use serde_json::json;
    #[tokio::test]
    async fn fp32_logits_and_probabilities() -> anyhow::Result<()> {
        let models = [
            (
                "laya",
                "convaiinnovations/laya",
                "aa8c91ca088ec597df95a0d1c76b3063cb2ae5e8",
            ),
            (
                "laya_typed_decisions",
                "convaiinnovations/laya-typed-decisions",
                "1a793eb568e6718f15941d08f85432581df534e3",
            ),
            (
                "laya_multilingual",
                "convaiinnovations/laya-multilingual",
                "e4e9ddf21a7b1903b7acffd8814ad4307bf63a67",
            ),
        ];

        for (snapshot, model_id, revision) in models {
            let path = crate::hub::download(model_id, revision).await?;
            let model = Laya::load(
                &path,
                Some(DType::F32),
                None,
                AttentionImplementation::Eager,
            )?;
            let request: DecisionRequest = serde_json::from_value(json!({
                "state": {
                    "message": "I was charged twice for invoice 4411. Please refund me today.",
                    "account_tier": "enterprise"
                },
                "questions": {
                    "route": {
                        "type": "choice",
                        "instructions": "Where should this ticket go?",
                        "criteria": {
                            "billing": "payments, refunds, invoices",
                            "bug": "the product is broken",
                            "account": "login or access"
                        }
                    },
                    "urgency": {
                        "type": "score",
                        "instructions": "How urgent is this message?",
                        "criteria": [
                            "routine, no rush",
                            "today",
                            "urgent",
                            "critical, about to churn"
                        ]
                    },
                    "escalate": {
                        "type": "noul",
                        "instructions": "Escalate to a human immediately?"
                    }
                }
            }))?;

            let prepared = model
                .0
                .protocol
                .prepare(request)
                .expect("prepare the test request");
            let logits = model.0.forward(std::slice::from_ref(&prepared))?;
            let probabilities: Vec<_> = prepared
                .items
                .iter()
                .zip(&logits)
                .map(|(item, logits)| {
                    let bucket = bucket(item.question.kind, logits.len());
                    let temperature = model
                        .0
                        .protocol
                        .config
                        .temperature_by_options
                        .get(&bucket)
                        .copied()
                        .unwrap_or(model.0.protocol.config.temperature[item.question.kind])
                        .clamp(0.5, 5.0);
                    probability(logits, temperature)
                })
                .collect();

            insta::assert_yaml_snapshot!(format!("{snapshot}_fp32_logits"), logits, {
                "[][]" => insta::rounded_redaction(3),
            });
            insta::assert_yaml_snapshot!(
                format!("{snapshot}_fp32_probabilities"),
                probabilities,
                {
                    "[][]" => insta::rounded_redaction(4),
                }
            );
        }

        Ok(())
    }
}

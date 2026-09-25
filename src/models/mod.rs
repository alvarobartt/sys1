mod laya;
mod modernbert;

use crate::schema::{ApiError, DecisionRequest, DecisionResponse};

use anyhow::{Context, bail};
use candle_core::DType;
use clap::ValueEnum;
use serde::Deserialize;
use std::{fs, path::Path};

pub use laya::Laya;

pub const LAYA_MODEL_ID: &str = "convaiinnovations/laya";
pub const LAYA_TYPED_DECISIONS_MODEL_ID: &str = "convaiinnovations/laya-typed-decisions";
pub const LAYA_MULTILINGUAL_MODEL_ID: &str = "convaiinnovations/laya-multilingual";

pub const LAYA_MODEL_IDS: &[&str] = &[
    LAYA_MODEL_ID,
    LAYA_TYPED_DECISIONS_MODEL_ID,
    LAYA_MULTILINGUAL_MODEL_ID,
];

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, ValueEnum)]
pub enum AttentionImplementation {
    #[default]
    Eager,
    #[value(name = "flash-attn-2")]
    FlashAttention2,
    #[value(name = "flash-attn-3")]
    FlashAttention3,
}

impl AttentionImplementation {
    pub fn cli_name(self) -> &'static str {
        match self {
            Self::Eager => "eager",
            Self::FlashAttention2 => "flash-attn-2",
            Self::FlashAttention3 => "flash-attn-3",
        }
    }

    pub fn validate(self, dtype: DType) -> anyhow::Result<()> {
        let name = self.cli_name();
        let enabled = match self {
            Self::Eager => return Ok(()),
            Self::FlashAttention2 => cfg!(feature = "flash-attn-2"),
            Self::FlashAttention3 => cfg!(feature = "flash-attn-3"),
        };
        anyhow::ensure!(
            enabled,
            "--attention {name} requires a binary built with --features {name}"
        );
        anyhow::ensure!(
            matches!(dtype, DType::F16 | DType::BF16),
            "--attention {name} requires --dtype f16 or --dtype bf16"
        );
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Architecture {
    Laya,
}

impl Architecture {
    pub fn from_model_id(model_id: &str) -> anyhow::Result<Self> {
        if LAYA_MODEL_IDS.contains(&model_id) {
            Ok(Self::Laya)
        } else {
            bail!(
                "unsupported model id {model_id:?}; supported models: {}",
                LAYA_MODEL_IDS.join(", ")
            )
        }
    }

    pub fn from_path(path: &Path) -> anyhow::Result<Self> {
        let root_config = path.join("config.json");
        let encoder_config = path.join("encoder/config.json");
        let config_path = if root_config.is_file() {
            root_config
        } else {
            encoder_config
        };
        let config: ModelConfig = serde_json::from_slice(
            &fs::read(&config_path)
                .with_context(|| format!("failed to read {}", config_path.display()))?,
        )
        .with_context(|| format!("failed to parse {}", config_path.display()))?;
        let laya_config_path = path.join("rl_agent_config.json");
        if config.model_type == "modernbert" && laya_config_path.is_file() {
            let laya_config: LayaIdentity = serde_json::from_slice(
                &fs::read(&laya_config_path)
                    .with_context(|| format!("failed to read {}", laya_config_path.display()))?,
            )
            .with_context(|| format!("failed to parse {}", laya_config_path.display()))?;
            if matches!(
                laya_config.model_name.as_str(),
                "rl-agent" | "laya-typed-decisions"
            ) {
                return Ok(Self::Laya);
            }
        }
        bail!(
            "unsupported model architecture {:?} in {}",
            config.model_type,
            config_path.display()
        )
    }
}

#[derive(Deserialize)]
struct ModelConfig {
    model_type: String,
}

#[derive(Deserialize)]
struct LayaIdentity {
    model_name: String,
}

pub enum Model {
    Laya(Laya),
}

pub trait DecisionModel: Send + Sync {
    fn predict_batch(
        &self,
        requests: Vec<DecisionRequest>,
    ) -> Vec<Result<DecisionResponse, ApiError>>;
}

impl DecisionModel for Model {
    fn predict_batch(
        &self,
        requests: Vec<DecisionRequest>,
    ) -> Vec<Result<DecisionResponse, ApiError>> {
        match self {
            Self::Laya(model) => model.predict_batch(requests),
        }
    }
}

pub fn load(
    path: &Path,
    architecture: Architecture,
    dtype: DType,
    max_model_len: Option<usize>,
    attention: AttentionImplementation,
) -> anyhow::Result<Model> {
    match architecture {
        Architecture::Laya => Laya::load(path, dtype, max_model_len, attention).map(Model::Laya),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn routes_the_supported_hub_models() {
        for model_id in LAYA_MODEL_IDS {
            assert_eq!(
                Architecture::from_model_id(model_id).unwrap(),
                Architecture::Laya
            );
        }
        assert!(Architecture::from_model_id("owner/other").is_err());
    }
}

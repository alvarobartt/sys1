mod laya;
mod modernbert;

use crate::schema::{ApiError, DecisionRequest, DecisionResponse};

use anyhow::{Context, bail};
use candle_core::DType;
use serde::Deserialize;
use std::{fs, path::Path};

pub use laya::Laya;

pub const LAYA_MODEL_ID: &str = "convaiinnovations/laya";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Architecture {
    Laya,
}

impl Architecture {
    pub fn from_model_id(model_id: &str) -> anyhow::Result<Self> {
        match model_id {
            LAYA_MODEL_ID => Ok(Self::Laya),
            _ => bail!("unsupported model id {model_id:?}; supported model: {LAYA_MODEL_ID}"),
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
            if laya_config.model_name == "rl-agent" {
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

pub fn load(path: &Path, architecture: Architecture, dtype: DType) -> anyhow::Result<Model> {
    match architecture {
        Architecture::Laya => Laya::load(path, dtype).map(Model::Laya),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn routes_the_supported_hub_model() {
        assert_eq!(
            Architecture::from_model_id(LAYA_MODEL_ID).unwrap(),
            Architecture::Laya
        );
        assert!(Architecture::from_model_id("owner/other").is_err());
    }
}

mod clef;
mod dtype;
mod laya;

use crate::schema::{ApiError, DecisionRequest, DecisionResponse};

use anyhow::{Context, bail};
use candle_core::{DType, Device};
use clap::ValueEnum;
use serde::Deserialize;
use std::{fs, path::Path};

pub use clef::Clef;
pub use laya::Laya;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, ValueEnum)]
pub enum AttentionImplementation {
    #[default]
    Auto,
    #[value(name = "eager")]
    Eager,
    #[value(name = "flash-attn-2")]
    FlashAttention2,
    #[value(name = "flash-attn-blackwell")]
    FlashAttentionBlackwell,
}

impl AttentionImplementation {
    pub fn cli_name(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Eager => "eager",
            Self::FlashAttention2 => "flash-attn-2",
            Self::FlashAttentionBlackwell => "flash-attn-blackwell",
        }
    }

    pub fn validate(self, dtype: DType) -> anyhow::Result<()> {
        let name = self.cli_name();
        let enabled = match self {
            Self::Auto => return Ok(()),
            Self::Eager => return Ok(()),
            Self::FlashAttention2 | Self::FlashAttentionBlackwell => cfg!(feature = "cuda"),
        };
        anyhow::ensure!(
            enabled,
            "{name} attention requires a binary built with --features cuda"
        );
        anyhow::ensure!(
            matches!(dtype, DType::F16 | DType::BF16),
            "{name} attention requires --dtype f16 or --dtype bf16"
        );
        Ok(())
    }
}

pub(crate) fn automatic_attention(
    dtype: DType,
    capability: Option<(i32, i32)>,
) -> AttentionImplementation {
    if !matches!(dtype, DType::F16 | DType::BF16) {
        return AttentionImplementation::Eager;
    }
    #[cfg(feature = "cuda")]
    if capability.is_some_and(|(major, _)| (8..=9).contains(&major)) {
        return AttentionImplementation::FlashAttention2;
    }
    #[cfg(feature = "cuda")]
    if capability == Some((12, 0)) {
        return AttentionImplementation::FlashAttentionBlackwell;
    }
    let _ = capability;
    AttentionImplementation::Eager
}

pub(crate) fn attention_for_device(
    dtype: DType,
    device: &Device,
) -> anyhow::Result<AttentionImplementation> {
    if device.is_metal() {
        return Ok(AttentionImplementation::Auto);
    }
    let capability = None;
    #[cfg(feature = "cuda")]
    let capability = match device {
        Device::Cuda(cuda) if matches!(dtype, DType::F16 | DType::BF16) => Some(
            cuda.cuda_stream()
                .context()
                .compute_capability()
                .context("Failed to query CUDA compute capability")?,
        ),
        _ => capability,
    };
    let _ = device;
    Ok(automatic_attention(dtype, capability))
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Architecture {
    Laya,
    Qwen35,
}

impl Architecture {
    fn from_model_type(model_type: &str) -> Option<Self> {
        match model_type {
            "laya" => Some(Self::Laya),
            "qwen3_5" => Some(Self::Qwen35),
            _ => None,
        }
    }

    pub fn from_path(path: &Path) -> anyhow::Result<Self> {
        let config_path = path.join("config.json");
        if config_path.is_file() {
            let config = ModelConfig::load(&config_path)?;
            if let Some(architecture) = Self::from_model_type(&config.model_type) {
                return Ok(architecture);
            }
            if config.model_type != "modernbert" {
                bail!(
                    "unsupported model type {:?} in {}",
                    config.model_type,
                    config_path.display()
                );
            }
        }

        Self::from_legacy_laya(path)
    }

    fn from_legacy_laya(path: &Path) -> anyhow::Result<Self> {
        let config_path = path.join("encoder/config.json");
        let config = ModelConfig::load(&config_path)?;
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

/// Resolve the checkpoint's declared dtype, falling back to Safetensors metadata.
pub fn model_default_dtype(path: &Path, architecture: Architecture) -> anyhow::Result<DType> {
    dtype::resolve(path, architecture, None)
}

#[derive(Deserialize)]
struct ModelConfig {
    model_type: String,
}

impl ModelConfig {
    fn load(path: &Path) -> anyhow::Result<Self> {
        serde_json::from_slice(
            &fs::read(path).with_context(|| format!("failed to read {}", path.display()))?,
        )
        .with_context(|| format!("failed to parse {}", path.display()))
    }
}

#[derive(Deserialize)]
struct LayaIdentity {
    model_name: String,
}

pub enum Model {
    Laya(Box<Laya>),
    Clef(Box<Clef>),
}

pub trait DecisionModel: Send + Sync {
    fn supports_images(&self) -> bool {
        false
    }

    fn supports_videos(&self) -> bool {
        false
    }

    fn predict_batch(
        &self,
        requests: Vec<DecisionRequest>,
    ) -> Vec<Result<DecisionResponse, ApiError>>;

    fn predict_batch_incremental(
        &self,
        requests: Vec<DecisionRequest>,
        complete: &mut dyn FnMut(usize, Result<DecisionResponse, ApiError>),
    ) {
        for (slot, response) in self.predict_batch(requests).into_iter().enumerate() {
            complete(slot, response);
        }
    }
}

impl DecisionModel for Model {
    fn supports_images(&self) -> bool {
        matches!(self, Self::Clef(_))
    }

    fn supports_videos(&self) -> bool {
        matches!(self, Self::Clef(_))
    }

    fn predict_batch(
        &self,
        requests: Vec<DecisionRequest>,
    ) -> Vec<Result<DecisionResponse, ApiError>> {
        match self {
            Self::Laya(model) => model.predict_batch(requests),
            Self::Clef(model) => model.predict_batch(requests),
        }
    }
    fn predict_batch_incremental(
        &self,
        requests: Vec<DecisionRequest>,
        complete: &mut dyn FnMut(usize, Result<DecisionResponse, ApiError>),
    ) {
        match self {
            Self::Laya(model) => model.predict_batch_incremental(requests, complete),
            Self::Clef(model) => model.predict_batch_incremental(requests, complete),
        }
    }
}

pub fn load(
    path: &Path,
    architecture: Architecture,
    dtype: Option<DType>,
    max_model_len: Option<usize>,
    attention: AttentionImplementation,
) -> anyhow::Result<Model> {
    match architecture {
        Architecture::Laya => Laya::load(path, dtype, max_model_len, attention)
            .map(Box::new)
            .map(Model::Laya),
        Architecture::Qwen35 => Clef::load(
            path,
            match dtype {
                Some(dtype) => dtype,
                None => model_default_dtype(path, architecture)?,
            },
            max_model_len,
            attention,
        )
        .map(Box::new)
        .map(Model::Clef),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn automatic_attention_handles_cpu_and_unsupported_cuda_devices() {
        for dtype in [DType::F32, DType::F16, DType::BF16] {
            assert_eq!(
                attention_for_device(dtype, &Device::Cpu).unwrap(),
                AttentionImplementation::Eager
            );
            assert_eq!(
                automatic_attention(dtype, Some((7, 5))),
                AttentionImplementation::Eager
            );
            assert_eq!(
                automatic_attention(dtype, Some((10, 0))),
                AttentionImplementation::Eager
            );
        }
        for capability in [(8, 0), (8, 6), (8, 9), (9, 0), (12, 0)] {
            assert_eq!(
                automatic_attention(DType::F32, Some(capability)),
                AttentionImplementation::Eager
            );
        }
        let supported = if cfg!(feature = "cuda") {
            AttentionImplementation::FlashAttention2
        } else {
            AttentionImplementation::Eager
        };
        for capability in [(8, 0), (8, 6), (8, 9), (9, 0)] {
            assert_eq!(automatic_attention(DType::F16, Some(capability)), supported);
        }
        let supported = if cfg!(feature = "cuda") {
            AttentionImplementation::FlashAttentionBlackwell
        } else {
            AttentionImplementation::Eager
        };
        assert_eq!(automatic_attention(DType::BF16, Some((12, 0))), supported);
    }

    #[test]
    fn identifies_qwen35_from_config_independent_of_repository_name() {
        let path = std::env::temp_dir().join(format!("sys1-arch-test-{}", std::process::id()));
        std::fs::create_dir_all(&path).unwrap();
        std::fs::write(path.join("config.json"), br#"{"model_type":"qwen3_5"}"#).unwrap();
        assert_eq!(
            Architecture::from_path(&path).unwrap(),
            Architecture::Qwen35
        );
        std::fs::remove_file(path.join("config.json")).unwrap();
        std::fs::remove_dir(path).unwrap();
    }

    #[test]
    fn identifies_laya_with_root_hub_config() {
        let path = std::env::temp_dir().join(format!("sys1-laya-arch-test-{}", std::process::id()));
        std::fs::create_dir_all(path.join("encoder")).unwrap();
        std::fs::write(path.join("config.json"), br#"{"model_type":"laya"}"#).unwrap();
        std::fs::write(
            path.join("encoder/config.json"),
            br#"{"model_type":"modernbert"}"#,
        )
        .unwrap();
        std::fs::write(
            path.join("rl_agent_config.json"),
            br#"{"model_name":"rl-agent"}"#,
        )
        .unwrap();
        assert_eq!(Architecture::from_path(&path).unwrap(), Architecture::Laya);
        std::fs::write(path.join("config.json"), br#"{"model_type":"modernbert"}"#).unwrap();
        assert_eq!(Architecture::from_path(&path).unwrap(), Architecture::Laya);
        std::fs::remove_file(path.join("config.json")).unwrap();
        assert_eq!(Architecture::from_path(&path).unwrap(), Architecture::Laya);
        std::fs::remove_dir_all(path).unwrap();
    }
}

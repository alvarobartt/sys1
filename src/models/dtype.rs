use super::Architecture;
use anyhow::{Context, bail, ensure};
use candle_core::{DType, safetensors::MmapedSafetensors};
use std::{fs, path::Path};

pub(super) fn parse(value: &str) -> anyhow::Result<DType> {
    match value {
        "float32" | "f32" | "fp32" => Ok(DType::F32),
        "float16" | "f16" | "fp16" => Ok(DType::F16),
        "bfloat16" | "bf16" => Ok(DType::BF16),
        _ => bail!("Unsupported checkpoint dtype {value:?}"),
    }
}

fn declared(path: &Path, architecture: Architecture) -> anyhow::Result<Option<DType>> {
    let mut configs = vec![path.join("config.json")];
    if architecture == Architecture::Laya {
        configs.push(path.join("encoder/config.json"));
    }
    for config in configs {
        if !config.is_file() {
            continue;
        }
        let json: serde_json::Value = serde_json::from_slice(&fs::read(&config)?)
            .with_context(|| format!("parsing {}", config.display()))?;
        if let Some(value) = json.get("dtype").filter(|value| !value.is_null()) {
            return parse(value.as_str().context("config dtype must be a string")?)
                .with_context(|| format!("reading dtype in {}", config.display()))
                .map(Some);
        }
    }
    Ok(None)
}

// Inspect headers without loading tensor values into CPU or GPU memory.
fn stored(path: &Path) -> anyhow::Result<Vec<(DType, usize)>> {
    let mut counts = Vec::<(DType, usize)>::new();
    for entry in fs::read_dir(path)? {
        let entry = entry?;
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if !(name == "model.safetensors"
            || name == "joint_head.safetensors"
            || (name.starts_with("model-") && name.ends_with(".safetensors")))
        {
            continue;
        }
        // SAFETY: Checkpoint files must remain immutable while mapped, as with the
        // model loaders' existing mmap-backed weight loading.
        let tensors = unsafe { MmapedSafetensors::new(entry.path())? };
        for (_, tensor) in tensors.tensors() {
            let dtype = DType::try_from(tensor.dtype())?;
            if !matches!(dtype, DType::BF16 | DType::F16 | DType::F32) {
                continue;
            }
            let elements = tensor.shape().iter().product::<usize>();
            if let Some((_, count)) = counts.iter_mut().find(|(kind, _)| *kind == dtype) {
                *count += elements;
            } else {
                counts.push((dtype, elements));
            }
        }
    }
    Ok(counts)
}

/// Explicit override, then config dtype, then the predominant floating-point
/// weight dtype. Backend selection must never silently narrow BF16 to FP16.
pub(super) fn resolve(
    path: &Path,
    architecture: Architecture,
    requested: Option<DType>,
) -> anyhow::Result<DType> {
    let declared = declared(path, architecture)?;
    let stored = stored(path)?;
    let dtype = match requested.or(declared) {
        Some(dtype) => dtype,
        None => {
            let max = stored.iter().map(|(_, count)| *count).max().context(
                "No config dtype or floating-point Safetensors weights; set --dtype explicitly",
            )?;
            let candidates: Vec<_> = stored.iter().filter(|(_, count)| *count == max).collect();
            ensure!(
                candidates.len() == 1,
                "Ambiguous Safetensors dtype; set --dtype explicitly"
            );
            candidates[0].0
        }
    };
    ensure!(
        dtype != DType::F16
            || (declared != Some(DType::BF16)
                && !stored.iter().any(|(kind, _)| *kind == DType::BF16)),
        "Unsafe checkpoint conversion from bf16 to f16: FP16 has a narrower exponent range and may produce non-finite hidden states; use --dtype bf16 or --dtype f32"
    );
    Ok(dtype)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, Tensor};

    fn weights(path: &Path, name: &str, dtype: DType, elements: usize) {
        Tensor::zeros(elements, dtype, &Device::Cpu)
            .unwrap()
            .save_safetensors("weight", path.join(name))
            .unwrap();
    }

    #[test]
    fn config_default_and_explicit_override_preserve_bf16() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("config.json"), br#"{"dtype":"bfloat16"}"#).unwrap();
        weights(dir.path(), "model.safetensors", DType::BF16, 4);
        assert_eq!(
            resolve(dir.path(), Architecture::Qwen35, None).unwrap(),
            DType::BF16
        );
        assert_eq!(
            resolve(dir.path(), Architecture::Qwen35, Some(DType::F32)).unwrap(),
            DType::F32
        );
        let error = resolve(dir.path(), Architecture::Qwen35, Some(DType::F16)).unwrap_err();
        assert!(error.to_string().contains("bf16 to f16"));
    }

    #[test]
    fn safetensors_fallback_handles_shards_and_small_fp32_auxiliary_weights() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(
            dir.path().join("config.json"),
            br#"{"model_type":"qwen3_5"}"#,
        )
        .unwrap();
        weights(
            dir.path(),
            "model-00001-of-00002.safetensors",
            DType::BF16,
            4,
        );
        weights(
            dir.path(),
            "model-00002-of-00002.safetensors",
            DType::BF16,
            4,
        );
        weights(dir.path(), "joint_head.safetensors", DType::F32, 1);
        assert_eq!(
            resolve(dir.path(), Architecture::Qwen35, None).unwrap(),
            DType::BF16
        );
        assert!(resolve(dir.path(), Architecture::Qwen35, Some(DType::F16)).is_err());
    }

    #[test]
    fn bf16_weights_cannot_be_narrowed_even_with_an_incorrect_fp16_config() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("config.json"), br#"{"dtype":"float16"}"#).unwrap();
        weights(dir.path(), "model.safetensors", DType::BF16, 4);
        assert!(resolve(dir.path(), Architecture::Laya, None).is_err());
        assert_eq!(
            resolve(dir.path(), Architecture::Laya, Some(DType::F32)).unwrap(),
            DType::F32
        );
    }

    #[test]
    fn laya_uses_encoder_config_without_implicit_backend_autocast() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("config.json"), br#"{"model_type":"laya"}"#).unwrap();
        fs::create_dir(dir.path().join("encoder")).unwrap();
        fs::write(
            dir.path().join("encoder/config.json"),
            br#"{"dtype":"float32"}"#,
        )
        .unwrap();
        weights(dir.path(), "model.safetensors", DType::F16, 4);
        assert_eq!(
            resolve(dir.path(), Architecture::Laya, None).unwrap(),
            DType::F32
        );
        assert_eq!(
            resolve(dir.path(), Architecture::Laya, Some(DType::F16)).unwrap(),
            DType::F16
        );
        fs::write(dir.path().join("config.json"), br#"{"dtype":"bf16"}"#).unwrap();
        assert_eq!(
            resolve(dir.path(), Architecture::Laya, None).unwrap(),
            DType::BF16
        );
        assert!(resolve(dir.path(), Architecture::Laya, Some(DType::F16)).is_err());
    }

    #[test]
    fn absent_or_ambiguous_metadata_requires_an_explicit_dtype() {
        let dir = tempfile::tempdir().unwrap();
        assert!(resolve(dir.path(), Architecture::Qwen35, None).is_err());
        weights(
            dir.path(),
            "model-00001-of-00002.safetensors",
            DType::F16,
            4,
        );
        weights(
            dir.path(),
            "model-00002-of-00002.safetensors",
            DType::F32,
            4,
        );
        assert!(resolve(dir.path(), Architecture::Qwen35, None).is_err());
        assert_eq!(
            resolve(dir.path(), Architecture::Qwen35, Some(DType::F32)).unwrap(),
            DType::F32
        );
        fs::write(dir.path().join("config.json"), br#"{"dtype":"unknown"}"#).unwrap();
        assert!(resolve(dir.path(), Architecture::Qwen35, None).is_err());
    }
}

use candle_core::{DType, Result, Tensor};
use std::{
    cell::Cell,
    collections::HashMap,
    path::PathBuf,
    sync::{
        OnceLock,
        atomic::{AtomicUsize, Ordering},
    },
};

static ROOT: OnceLock<Option<PathBuf>> = OnceLock::new();
static NEXT: AtomicUsize = AtomicUsize::new(0);
thread_local! { static REQUEST: Cell<usize> = const { Cell::new(0) }; }

fn root() -> Option<&'static PathBuf> {
    ROOT.get_or_init(|| std::env::var_os("SYS1_NUMERICS_DIR").map(PathBuf::from))
        .as_ref()
}

fn directory() -> Option<PathBuf> {
    root().map(|root| root.join(format!("request-{:04}", REQUEST.get())))
}

pub(crate) fn enabled() -> bool {
    root().is_some()
}

#[cfg(feature = "cuda")]
pub(crate) fn full_attention_reference_enabled(index: usize) -> bool {
    index == 3
        && enabled()
        && std::env::var_os("SYS1_NUMERICS_FULL_ATTENTION_INPUTS").is_some()
        && std::env::var_os("SYS1_NUMERICS_INPUTS_ONLY").is_none()
}

pub(crate) fn layer_capture(index: usize, stage: &str, tensor: &Tensor) -> Result<()> {
    if !enabled() || index >= 4 {
        return Ok(());
    }
    let tensor = if tensor.rank() > 3 {
        tensor.flatten_from(2)?
    } else {
        tensor.clone()
    };
    capture(&format!("layer-{index:02}-{stage}"), &tensor)
}

pub(crate) fn begin() -> Result<()> {
    if root().is_some() {
        REQUEST.set(NEXT.fetch_add(1, Ordering::Relaxed));
        std::fs::create_dir_all(directory().unwrap())?;
    }
    Ok(())
}

pub(crate) fn capture(name: &str, tensor: &Tensor) -> Result<()> {
    if std::env::var_os("SYS1_NUMERICS_INPUTS_ONLY").is_some() {
        return Ok(());
    }
    let Some(directory) = directory() else {
        return Ok(());
    };
    let mut tensors = HashMap::new();

    let full_linear_inputs = std::env::var_os("SYS1_NUMERICS_FULL_LINEAR_INPUTS").is_some()
        && is_first_layer_linear_input(name);
    let full_attention_inputs = std::env::var_os("SYS1_NUMERICS_FULL_ATTENTION_INPUTS").is_some()
        && (is_first_full_attention_boundary(name) || is_first_head_attention_boundary(name));
    let tensor = if tensor.rank() == 3
        && tensor.dim(1)? > 64
        && !full_linear_inputs
        && !full_attention_inputs
    {
        let length = tensor.dim(1)?;
        let indices = Tensor::from_vec(
            (0..64)
                .map(|i| (i * (length - 1) / 63) as u32)
                .collect::<Vec<_>>(),
            64,
            tensor.device(),
        )?;
        tensors.insert("positions", indices.clone());
        sample_positions(tensor, &indices)?
    } else {
        tensor.clone()
    };
    tensors.insert("values", tensor.to_dtype(DType::F32)?.contiguous()?);
    candle_core::safetensors::save(&tensors, directory.join(format!("{name}.safetensors")))
}

fn sample_positions(tensor: &Tensor, indices: &Tensor) -> Result<Tensor> {
    tensor.contiguous()?.index_select(indices, 1)
}

fn is_first_layer_linear_input(name: &str) -> bool {
    matches!(
        name,
        "layer-00-normalized"
            | "modernbert-normalized"
            | "modernbert-qkv-projected"
            | "modernbert-attention-q"
            | "modernbert-attention-k"
            | "modernbert-attention-v"
            | "modernbert-attention-mixed"
            | "modernbert-attention-output"
            | "modernbert-post-normalized"
            | "modernbert-mlp-projected"
            | "modernbert-mlp-product"
            | "modernbert-mlp-output"
            | "layer-00-linear-projected"
            | "layer-00-linear-z-projected"
            | "layer-00-linear-conv"
            | "layer-00-linear-delta"
            | "layer-00-linear-finalized"
            | "layer-00-mixed"
            | "layer-00-linear-a-projection"
            | "layer-00-linear-b-projection"
            | "layer-00-linear-q"
            | "layer-00-linear-k"
            | "layer-00-linear-v"
            | "layer-00-linear-g"
            | "layer-00-linear-beta"
            | "layer-00-post-normalized"
            | "layer-00-mlp-gate-projected"
            | "layer-00-mlp-gate-activated"
            | "layer-00-mlp-up-projected"
            | "layer-00-mlp-product"
            | "layer-00-mlp-output"
    )
}

fn is_first_full_attention_boundary(name: &str) -> bool {
    name == "layer-03-normalized"
        || name == "layer-03-mixed"
        || matches!(
            name,
            "layer-03-full-qg-projected"
                | "layer-03-full-k-projected"
                | "layer-03-full-q"
                | "layer-03-full-k"
                | "layer-03-full-v"
                | "layer-03-full-gate"
                | "layer-03-full-attention"
                | "layer-03-full-reference-f32"
                | "layer-03-full-reference-ordinary"
                | "layer-03-full-gated"
        )
}

fn is_first_head_attention_boundary(name: &str) -> bool {
    matches!(name, "decoder-output" | "head-normalized")
        || matches!(
            name,
            "head-evidence-0-query-input"
                | "head-evidence-0"
                | "head-evidence-0-memory-input"
                | "head-evidence-0-q-projected"
                | "head-evidence-0-k-projected"
                | "head-evidence-0-v-projected"
                | "head-evidence-0-attended"
                | "head-evidence-0-output-projected"
                | "head-evidence-0-feedforward-input"
                | "head-evidence-0-feedforward-projected"
                | "head-evidence-0-feedforward-activated"
                | "head-evidence-0-feedforward-output"
        )
}

pub(crate) fn json(name: &str, value: &impl serde::Serialize) -> Result<()> {
    if let Some(directory) = directory() {
        let data = serde_json::to_vec(value).map_err(candle_core::Error::wrap)?;
        std::fs::write(directory.join(format!("{name}.json")), data)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    #[test]
    fn complete_head_capture_is_bounded_to_first_routing_attention() {
        for stage in [
            "query-input",
            "memory-input",
            "q-projected",
            "k-projected",
            "v-projected",
            "attended",
            "output-projected",
            "feedforward-input",
            "feedforward-projected",
            "feedforward-activated",
            "feedforward-output",
        ] {
            assert!(super::is_first_head_attention_boundary(&format!(
                "head-evidence-0-{stage}"
            )));
            assert!(!super::is_first_head_attention_boundary(&format!(
                "head-evidence-1-{stage}"
            )));
        }
        assert!(super::is_first_head_attention_boundary("decoder-output"));
        assert!(super::is_first_head_attention_boundary("head-normalized"));
        assert!(!super::is_first_head_attention_boundary("head-fields-0"));
    }
    #[test]
    fn complete_attention_capture_is_limited_to_first_full_boundary() {
        for stage in [
            "normalized",
            "mixed",
            "full-qg-projected",
            "full-k-projected",
            "full-q",
            "full-k",
            "full-v",
            "full-gate",
            "full-attention",
            "full-reference-f32",
            "full-reference-ordinary",
            "full-gated",
        ] {
            assert!(super::is_first_full_attention_boundary(&format!(
                "layer-03-{stage}"
            )));
            assert!(!super::is_first_full_attention_boundary(&format!(
                "layer-07-{stage}"
            )));
        }
        for stage in [
            "layer-03-output",
            "layer-03-mlp-output",
            "layer-03-linear-q",
            "decoder-output",
        ] {
            assert!(!super::is_first_full_attention_boundary(stage));
        }
    }

    #[test]
    fn sampled_capture_preserves_strided_sequence_values() -> candle_core::Result<()> {
        use candle_core::{DType, Device, IndexOp, Tensor};
        #[allow(unused_mut)]
        let mut devices = vec![Device::Cpu];
        #[cfg(feature = "cuda")]
        devices.push(Device::new_cuda(0)?);
        for device in devices {
            let mut dtypes = vec![DType::F32, DType::F16];
            #[cfg(feature = "cuda")]
            let supports_bf16 = match &device {
                Device::Cuda(cuda) => {
                    use candle_core::cuda_backend::WrapErr;
                    cuda.cuda_stream().context().compute_capability().w()?.0 >= 8
                }
                _ => true,
            };
            #[cfg(not(feature = "cuda"))]
            let supports_bf16 = true;

            if supports_bf16 {
                dtypes.push(DType::BF16);
            }
            for dtype in dtypes {
                let values = Tensor::from_vec(
                    (0..520).map(|i| (i % 128) as f32).collect::<Vec<_>>(),
                    (1, 130, 4),
                    &device,
                )?
                .to_dtype(dtype)?;
                let strided = values.narrow(2, 1, 2)?;
                assert!(!strided.is_contiguous());
                let indices = Tensor::from_vec(vec![0u32, 64, 129], 3, &device)?;
                let sampled = super::sample_positions(&strided, &indices)?;
                assert_eq!(sampled.dims(), &[1, 3, 2]);
                assert_eq!(
                    sampled.to_dtype(DType::F32)?.to_vec3::<f32>()?,
                    vec![vec![vec![1., 2.], vec![1., 2.], vec![5., 6.]]]
                );
                assert_eq!(
                    strided
                        .i((0, 129))?
                        .to_dtype(DType::F32)?
                        .to_vec1::<f32>()?,
                    vec![5., 6.]
                );
            }
        }
        Ok(())
    }

    #[test]
    fn complete_linear_capture_is_limited_to_first_layer_recurrence_inputs() {
        assert!(super::is_first_layer_linear_input("layer-00-normalized"));
        assert!(!super::is_first_layer_linear_input("layer-01-normalized"));
        for stage in [
            "q",
            "k",
            "v",
            "g",
            "beta",
            "a-projection",
            "b-projection",
            "projected",
            "z-projected",
            "conv",
            "delta",
            "finalized",
        ] {
            assert!(super::is_first_layer_linear_input(&format!(
                "layer-00-linear-{stage}"
            )));
            assert!(!super::is_first_layer_linear_input(&format!(
                "layer-01-linear-{stage}"
            )));
        }
        for stage in [
            "post-normalized",
            "mixed",
            "mlp-gate-projected",
            "mlp-gate-activated",
            "mlp-up-projected",
            "mlp-product",
            "mlp-output",
        ] {
            assert!(super::is_first_layer_linear_input(&format!(
                "layer-00-{stage}"
            )));
            assert!(!super::is_first_layer_linear_input(&format!(
                "layer-01-{stage}"
            )));
        }
        for stage in ["embedding", "decoder-output"] {
            assert!(!super::is_first_layer_linear_input(stage));
        }
    }

    #[test]
    fn complete_modernbert_capture_has_an_explicit_first_layer_allowlist() {
        for stage in [
            "normalized",
            "qkv-projected",
            "attention-q",
            "attention-k",
            "attention-v",
            "attention-mixed",
            "attention-output",
            "post-normalized",
            "mlp-projected",
            "mlp-product",
            "mlp-output",
        ] {
            assert!(super::is_first_layer_linear_input(&format!(
                "modernbert-{stage}"
            )));
        }
        assert!(!super::is_first_layer_linear_input("modernbert-any-stage"));
    }
}

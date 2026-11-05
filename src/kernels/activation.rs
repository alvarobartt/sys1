#[cfg(feature = "cuda")]
use super::cuda::activation::*;

use candle_core::{DType, Result, Tensor};

pub(crate) fn gelu_erf(input: &Tensor) -> Result<Tensor> {
    #[cfg(feature = "metal")]
    if input.device().is_metal() {
        return input.gelu_erf();
    }
    match input.dtype() {
        DType::F16 | DType::BF16 => input
            .to_dtype(DType::F32)?
            .gelu_erf()?
            .to_dtype(input.dtype()),
        _ => input.gelu_erf(),
    }
}

pub(crate) fn silu(input: &Tensor) -> Result<Tensor> {
    if input.device().is_cuda() && input.dtype() == DType::BF16 {
        #[cfg(feature = "cuda")]
        {
            cuda_bf16_activation(input, None, 0)
        }
        #[cfg(not(feature = "cuda"))]
        {
            input.to_dtype(DType::F32)?.silu()?.to_dtype(input.dtype())
        }
    } else {
        input.silu()
    }
}

pub(crate) fn sigmoid(input: &Tensor) -> Result<Tensor> {
    if input.device().is_cuda() && input.dtype() == DType::BF16 {
        #[cfg(feature = "cuda")]
        {
            cuda_bf16_activation(input, None, 1)
        }
        #[cfg(not(feature = "cuda"))]
        {
            candle_nn::ops::sigmoid(&input.to_dtype(DType::F32)?)?.to_dtype(input.dtype())
        }
    } else {
        candle_nn::ops::sigmoid(input)
    }
}

pub(crate) fn mul_silu(input: &Tensor, up: &Tensor) -> Result<Tensor> {
    #[cfg(feature = "cuda")]
    if input.device().is_cuda() && input.dtype() == DType::BF16 {
        return cuda_bf16_activation(input, Some(up), 2);
    }
    silu(input)? * up
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn cuda_bf16_silu_sigmoid_use_single_output_rounding() -> Result<()> {
        let device = crate::device::load().map_err(|e| candle_core::Error::Msg(e.to_string()))?;
        if !device.is_cuda() {
            return Ok(());
        }
        #[cfg(feature = "cuda")]
        if let candle_core::Device::Cuda(cuda) = &device
            && cuda
                .cuda_stream()
                .context()
                .compute_capability()
                .map_err(|e| candle_core::Error::Msg(e.to_string()))?
                .0
                < 8
        {
            println!("ACTIVATION_OPMATH_SKIPPED BF16 requires SM80");
            return Ok(());
        }
        let input = Tensor::from_vec(
            (0..2049).map(|i| i as f32 / 128. - 8.).collect::<Vec<_>>(),
            2049,
            &device,
        )?
        .to_dtype(DType::BF16)?;
        let f32_input = input.to_dtype(DType::F32)?;
        let observed = [silu(&input)?, sigmoid(&input)?];
        let expected = [
            f32_input.silu()?.to_dtype(DType::BF16)?,
            candle_nn::ops::sigmoid(&f32_input)?.to_dtype(DType::BF16)?,
        ];
        let xs = f32_input.to_vec1::<f32>()?;
        for (kind, (a, b)) in observed.iter().zip(&expected).enumerate() {
            let av = a.to_dtype(DType::F32)?.to_vec1::<f32>()?;
            let bv = b.to_dtype(DType::F32)?.to_vec1::<f32>()?;
            assert_eq!(
                av.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                bv.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
            );
            for (x, y) in xs.iter().zip(av) {
                let sig = 1. / (1. + (-(*x as f64)).exp());
                let reference = if kind == 0 { *x as f64 * sig } else { sig };
                assert!(
                    y.is_finite() && (y as f64 - reference).abs() <= 1e-5 + 0.016 * reference.abs()
                );
            }
        }
        Ok(())
    }

    #[test]
    #[ignore = "external reference data"]
    fn promoted_gelu_matches_native_torch_fixtures() -> Result<()> {
        let root = crate::assets::fixtures("activation")?;
        let device =
            crate::device::load().map_err(|error| candle_core::Error::Msg(error.to_string()))?;
        for (name, rtol) in [("bf16", 0.016f32), ("f16", 0.001), ("f32", 1.3e-6)] {
            #[cfg(feature = "cuda")]
            if name == "bf16"
                && let candle_core::Device::Cuda(cuda) = &device
            {
                let (major, _) = cuda
                    .cuda_stream()
                    .context()
                    .compute_capability()
                    .map_err(|error| candle_core::Error::Msg(error.to_string()))?;
                if major < 8 {
                    continue;
                }
            }
            let data = candle_core::safetensors::load(
                root.join(format!("gelu-erf-{name}.safetensors")),
                &device,
            )?;
            let actual = gelu_erf(&data["input"])?
                .to_dtype(DType::F32)?
                .to_vec1::<f32>()?;
            let expected = data["expected"].to_dtype(DType::F32)?.to_vec1::<f32>()?;
            assert_eq!(actual.len(), 2049);
            for (index, (actual, expected)) in actual.iter().zip(&expected).enumerate() {
                assert!(
                    actual.is_finite()
                        && (*actual - expected).abs() <= 1e-5 + rtol * expected.abs(),
                    "{name} GELU at{index}: {actual} vs {expected}"
                );
            }
        }
        Ok(())
    }
}

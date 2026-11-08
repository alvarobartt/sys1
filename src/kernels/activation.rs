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

pub(crate) fn softplus(input: &Tensor) -> Result<Tensor> {
    if input.device().is_cpu() {
        let values = input
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let values = values
            .into_iter()
            .map(|x| x.max(0.) + (-x.abs()).exp().ln_1p())
            .collect::<Vec<_>>();
        return Tensor::from_vec(values, input.shape(), input.device())?.to_dtype(input.dtype());
    }
    input.relu()? + (input.abs()?.neg()?.exp()? + 1.)?.log()?
}

#[cfg(test)]
mod tests {
    #[test]
    fn cpu_softplus_preserves_small_negative_gate_values() -> Result<()> {
        let device = candle_core::Device::Cpu;
        let values = [-50f32, -20., -0.0001, 0., 20., 80.];
        let input = Tensor::from_slice(&values, (2, 3), &device)?;
        let output = softplus(&input)?;
        assert_eq!(output.dims(), &[2, 3]);
        for (input, actual) in values.iter().zip(output.flatten_all()?.to_vec1::<f32>()?) {
            let x = f64::from(*input);
            let expected = x.max(0.) + (-x.abs()).exp().ln_1p();
            assert!(actual > 0. && actual.is_finite());
            assert!(
                (f64::from(actual) - expected).abs() <= 4. * f64::from(f32::EPSILON) * expected
            );
        }
        Ok(())
    }

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
}

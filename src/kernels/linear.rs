#[cfg(feature = "cuda")]
use super::cuda::linear::*;

use candle_core::{DType, Result, Tensor};
use candle_nn::Linear;

pub(crate) fn forward_compatible(input: &Tensor, linear: &Linear) -> Result<Tensor> {
    if !input.device().is_cpu() || input.dtype() != DType::BF16 {
        return input.apply(linear);
    }
    if linear.weight().dtype() != DType::BF16 {
        candle_core::bail!("CPU BF16 linear requires matching input and weight dtypes")
    }
    let projection = Linear::new(linear.weight().to_dtype(DType::F32)?, None);
    let product = input
        .to_dtype(DType::F32)?
        .apply(&projection)?
        .to_dtype(DType::BF16)?;
    match linear.bias() {
        Some(bias) => product.broadcast_add(bias),
        None => Ok(product),
    }
}

pub(crate) fn matmul_compatible(left: &Tensor, right: &Tensor) -> Result<Tensor> {
    if left.device().is_cpu()
        && right.device().is_cpu()
        && left.dtype() == DType::BF16
        && right.dtype() == DType::BF16
    {
        return left
            .to_dtype(DType::F32)?
            .matmul(&right.to_dtype(DType::F32)?)?
            .to_dtype(DType::BF16);
    }
    left.matmul(right)
}

pub(crate) fn forward(input: &Tensor, linear: &Linear) -> Result<Tensor> {
    #[cfg(feature = "metal")]
    if input.device().is_metal() {
        return input.apply(linear);
    }
    let Some(bias) = linear.bias() else {
        return forward_compatible(input, linear);
    };
    if !matches!(input.dtype(), DType::BF16 | DType::F16) {
        return input.apply(linear);
    }
    let width = input.dim(candle_core::D::Minus1)?;
    if width == 0 || input.elem_count() == 0 {
        return input.apply(linear);
    }
    if linear.weight().dtype() != input.dtype() || bias.dtype() != input.dtype() {
        candle_core::bail!("Single-rounding linear requires matching input, weight and bias dtypes")
    }
    let mut shape = input.dims().to_vec();
    let rows = input.elem_count() / width;
    let matrix = input.contiguous()?.reshape((rows, width))?;
    #[cfg(feature = "cuda")]
    let product = if input.device().is_cuda() {
        matmul_fp32(&matrix, linear.weight())?
    } else {
        matrix
            .to_dtype(DType::F32)?
            .matmul(&linear.weight().to_dtype(DType::F32)?.t()?)?
    };
    #[cfg(not(feature = "cuda"))]
    let product = matrix
        .to_dtype(DType::F32)?
        .matmul(&linear.weight().to_dtype(DType::F32)?.t()?)?;
    *shape.last_mut().unwrap() = linear.weight().dim(0)?;
    product
        .broadcast_add(&bias.to_dtype(DType::F32)?)?
        .to_dtype(input.dtype())?
        .reshape(shape)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn cpu_bf16_compatible_linear_preserves_cast_before_bias_and_all_row_ranks() -> Result<()> {
        let device = candle_core::Device::Cpu;
        let values = |x: &Tensor| x.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>();
        for shape in [vec![3, 5], vec![2, 3, 5], vec![2, 2, 3, 5]] {
            let count: usize = shape.iter().product();
            let input = Tensor::from_vec(
                (0..count)
                    .map(|i| (i as f32 - 7.) / 16.)
                    .collect::<Vec<_>>(),
                shape,
                &device,
            )?
            .to_dtype(DType::BF16)?;
            let weight = Tensor::from_vec(
                (0..20).map(|i| (i as f32 - 9.) / 32.).collect::<Vec<_>>(),
                (4, 5),
                &device,
            )?
            .to_dtype(DType::BF16)?;
            let projected = input
                .to_dtype(DType::F32)?
                .apply(&Linear::new(weight.to_dtype(DType::F32)?, None))?
                .to_dtype(DType::BF16)?;
            let unbiased = Linear::new(weight.clone(), None);
            assert_eq!(
                values(&forward_compatible(&input, &unbiased)?)?,
                values(&projected)?
            );
            assert_eq!(values(&forward(&input, &unbiased)?)?, values(&projected)?);
            let bias = Tensor::from_vec(vec![1_f32 / 256., -1_f32 / 256., 0., 1.], 4, &device)?
                .to_dtype(DType::BF16)?;
            let expected = projected.broadcast_add(&bias)?;
            let biased = Linear::new(weight, Some(bias));
            assert_eq!(
                values(&forward_compatible(&input, &biased)?)?,
                values(&expected)?
            );
        }

        let input =
            Tensor::from_vec(vec![1., 1_f32 / 256.], (1, 2), &device)?.to_dtype(DType::BF16)?;
        let weight = Tensor::from_vec(vec![1., 1.], (1, 2), &device)?.to_dtype(DType::BF16)?;
        let bias = Tensor::from_vec(vec![1_f32 / 256.], 1, &device)?.to_dtype(DType::BF16)?;
        let layer = Linear::new(weight, Some(bias));
        assert_eq!(values(&forward_compatible(&input, &layer)?)?, vec![1.]);
        assert_eq!(values(&forward(&input, &layer)?)?, vec![1.0078125]);
        Ok(())
    }

    #[test]
    fn cpu_bf16_compatible_matmul_preserves_strided_batched_shapes() -> Result<()> {
        let device = candle_core::Device::Cpu;
        let input = Tensor::from_vec(
            (0..24).map(|i| (i as f32 - 11.) / 16.).collect::<Vec<_>>(),
            (2, 3, 4),
            &device,
        )?
        .to_dtype(DType::BF16)?;
        let weight = Tensor::from_vec(
            (0..40).map(|i| (i as f32 - 19.) / 32.).collect::<Vec<_>>(),
            (2, 5, 4),
            &device,
        )?
        .to_dtype(DType::BF16)?
        .transpose(1, 2)?;
        let expected = input
            .to_dtype(DType::F32)?
            .matmul(&weight.to_dtype(DType::F32)?)?
            .to_dtype(DType::BF16)?;
        let actual = matmul_compatible(&input, &weight)?;
        assert_eq!(actual.dims(), [2, 3, 5]);
        assert_eq!(actual.dtype(), DType::BF16);
        assert_eq!(
            actual
                .to_dtype(DType::F32)?
                .flatten_all()?
                .to_vec1::<f32>()?,
            expected
                .to_dtype(DType::F32)?
                .flatten_all()?
                .to_vec1::<f32>()?
        );
        Ok(())
    }

    #[test]
    #[ignore = "external reference data"]
    fn bias_rounding_matches_native_torch_projection_fixtures() -> Result<()> {
        let root = crate::assets::fixtures("linear")?;

        #[cfg(not(feature = "cuda"))]
        let device = candle_core::Device::Cpu;
        #[cfg(feature = "cuda")]
        let device =
            crate::device::load().map_err(|error| candle_core::Error::Msg(error.to_string()))?;
        for name in ["bf16", "f16", "f32"] {
            #[cfg(feature = "cuda")]
            if name == "bf16"
                && device.is_cuda()
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
            let path = root.join(format!("b2-q3-k7-w128-{name}.safetensors"));
            let data = candle_core::safetensors::load(&path, &device)?;
            let weight = &data["weight"];
            let bias = &data["bias"];
            let q = Linear::new(weight.narrow(0, 0, 128)?, Some(bias.narrow(0, 0, 128)?));
            let kv = Linear::new(weight.narrow(0, 128, 256)?, Some(bias.narrow(0, 128, 256)?));
            let query = forward(&data["query"], &q)?;
            let memory = forward(&data["memory"], &kv)?;
            for (key, actual) in [
                ("q", query),
                ("k", memory.narrow(2, 0, 128)?),
                ("v", memory.narrow(2, 128, 128)?),
            ] {
                let values =
                    |tensor: &Tensor| tensor.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>();
                assert_eq!(
                    values(&actual)?,
                    values(&data[&format!("expected_{key}")])?,
                    "{name} {key}"
                );
            }
        }
        Ok(())
    }

    #[cfg(feature = "metal")]
    #[test]
    fn metal_preserves_main_matmul_cast_before_bias() -> Result<()> {
        let device = candle_core::Device::new_metal(0)?;
        for (dtype, half_ulp) in [(DType::BF16, 1f32 / 256.), (DType::F16, 1f32 / 2048.)] {
            let input = Tensor::from_vec(vec![1f32, half_ulp], (1, 2), &device)?.to_dtype(dtype)?;
            let weight = Tensor::from_vec(vec![1f32, 1.], (1, 2), &device)?.to_dtype(dtype)?;
            let bias = Tensor::from_vec(vec![half_ulp], 1, &device)?.to_dtype(dtype)?;
            let actual = forward(&input, &Linear::new(weight, Some(bias)))?;

            assert_eq!(actual.dtype(), dtype);
            assert_eq!(
                actual
                    .to_dtype(DType::F32)?
                    .flatten_all()?
                    .to_vec1::<f32>()?,
                vec![1.]
            );
        }
        Ok(())
    }
}

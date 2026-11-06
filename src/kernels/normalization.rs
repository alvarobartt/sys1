use super::cpu::f32_slice;
use candle_core::{D, DType, Result, Tensor, cpu::kernels::VecOps};
use candle_nn::LayerNorm;
use rayon::prelude::*;

pub(crate) fn layer_norm(input: &Tensor, layer: &LayerNorm) -> Result<Tensor> {
    if cpu_f32(input) && layer.bias().is_none() && layer.remove_mean() {
        return normalize_f32(input, layer.weight(), layer.eps());
    }
    input.apply(layer)
}

fn cpu_f32(input: &Tensor) -> bool {
    input.device().is_cpu()
        && input.dtype() == DType::F32
        && input.is_contiguous()
        && input.elem_count() != 0
}

fn normalize_f32(input: &Tensor, weight: &Tensor, epsilon: f64) -> Result<Tensor> {
    let width = input.dim(D::Minus1)?;
    if width == 0 || weight.dims() != [width] || !epsilon.is_finite() || epsilon < 0. {
        candle_core::bail!("Invalid CPU normalization width, scale or epsilon")
    }
    let weight = weight.contiguous()?;
    let (input_storage, input_layout) = input.storage_and_layout();
    let (weight_storage, weight_layout) = weight.storage_and_layout();
    let source = f32_slice(&input_storage, input_layout)?;
    let weight = f32_slice(&weight_storage, weight_layout)?;
    let mut output = vec![0f32; source.len()];
    let reciprocal_width = (1. / width as f64) as f32;
    output
        .par_chunks_mut(width)
        .zip(source.par_chunks(width))
        .for_each_init(
            || vec![0f32; width],
            |squared, (output, input)| {
                let mean = sum(input) * reciprocal_width + 0.;
                for ((source, output), squared) in input.iter().zip(&mut *output).zip(&mut *squared)
                {
                    *output = *source - mean;
                    *squared = *output * *output;
                }
                let variance = sum(squared) * reciprocal_width + 0.;
                let denominator = (variance + epsilon as f32).sqrt();
                for (output, weight) in output.iter_mut().zip(weight) {
                    *output = (*output / denominator) * weight;
                }
            },
        );
    Tensor::from_vec(output, input.shape(), input.device())
}

fn sum(values: &[f32]) -> f32 {
    let mut result = 0.;
    // The reducer reads values.len() valid elements and writes one local result.
    unsafe { f32::vec_reduce_sum(values.as_ptr(), &mut result, values.len()) };
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    fn bits(input: &Tensor) -> Result<Vec<u32>> {
        Ok(input
            .flatten_all()?
            .to_vec1::<f32>()?
            .iter()
            .map(|v| v.to_bits())
            .collect())
    }

    #[test]
    fn cpu_normalization_preserves_tensor_reductions_and_rounding() -> Result<()> {
        let device = Device::Cpu;
        for width in [16, 768, 1024, 4096] {
            for offset in [0f32, 10_000.] {
                let input = Tensor::from_vec(
                    (0..4 * width)
                        .map(|i| offset + ((i * 17 % 257) as f32 - 128.) / 31.)
                        .collect::<Vec<_>>(),
                    (4, width),
                    &device,
                )?
                .narrow(0, 1, 3)?;
                let weight = Tensor::from_vec(
                    (0..width)
                        .map(|i| (i % 17) as f32 / 16. + 0.5)
                        .collect::<Vec<_>>(),
                    width,
                    &device,
                )?;
                let layer = LayerNorm::new_no_bias(weight, 1e-5);
                assert_eq!(
                    bits(&layer_norm(&input, &layer)?)?,
                    bits(&input.apply(&layer)?)?
                );
            }
        }
        Ok(())
    }

    #[test]
    fn cpu_normalization_retains_biased_and_strided_fallbacks() -> Result<()> {
        let device = Device::Cpu;
        let input = Tensor::arange(0f32, 48f32, &device)?
            .reshape((3, 4, 4))?
            .transpose(0, 1)?;
        let weight = Tensor::ones(4, DType::F32, &device)?;
        let bias = Tensor::ones(4, DType::F32, &device)?;
        for layer in [
            LayerNorm::new(weight.clone(), bias, 1e-5),
            LayerNorm::new_no_bias(weight, 1e-5),
        ] {
            assert_eq!(
                bits(&layer_norm(&input, &layer)?)?,
                bits(&input.apply(&layer)?)?
            );
        }
        Ok(())
    }
}

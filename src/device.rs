use candle_core::Device;

pub fn validate_available() -> anyhow::Result<()> {
    #[cfg(feature = "cuda")]
    anyhow::ensure!(
        candle_core::utils::cuda_is_available(),
        "CUDA backend selected, but no CUDA device is available"
    );
    #[cfg(feature = "metal")]
    anyhow::ensure!(
        candle_core::utils::metal_is_available(),
        "Metal backend selected, but no Metal device is available"
    );
    Ok(())
}

#[cfg(all(feature = "cpu", not(any(feature = "cuda", feature = "metal"))))]
pub fn load() -> anyhow::Result<Device> {
    Ok(Device::Cpu)
}

#[cfg(all(feature = "cuda", not(any(feature = "cpu", feature = "metal"))))]
pub fn load() -> anyhow::Result<Device> {
    use anyhow::Context;
    use candle_core::cuda_backend::{self, cudarc::cublas::sys};

    let device = Device::new_cuda(0)?;

    cuda_backend::set_gemm_reduced_precision_bf16(false);
    cuda_backend::set_gemm_reduced_precision_f16(false);
    cuda_backend::set_gemm_reduced_precision_f32(false);
    if let Device::Cuda(cuda) = &device {
        let blas = cuda.cublas_handle();

        unsafe {
            sys::cublasSetMathMode(
                *blas.handle(),
                sys::cublasMath_t::CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION,
            )
            .result()
        }
        .context("Failed to require FP32 cuBLAS partial reductions")?;
    }
    Ok(device)
}

#[cfg(all(test, feature = "cuda", not(any(feature = "cpu", feature = "metal"))))]
mod tests {
    #[test]
    fn cuda_integer_reductions_match_exact_values_and_indices() -> anyhow::Result<()> {
        use candle_core::Tensor;
        let device = super::load()?;
        for values in [vec![3i64, 11, 7], vec![-11i64, -3, -7]] {
            let tensor = Tensor::from_vec(values.clone(), 3, &device)?;
            assert_eq!(
                tensor.min(0)?.to_scalar::<i64>()?,
                *values.iter().min().unwrap()
            );
            assert_eq!(
                tensor.max(0)?.to_scalar::<i64>()?,
                *values.iter().max().unwrap()
            );
            assert_eq!(tensor.argmin(0)?.to_scalar::<u32>()?, 0);
            assert_eq!(tensor.argmax(0)?.to_scalar::<u32>()?, 1);
        }
        for values in [
            vec![3u32, 11, 7],
            vec![u32::MAX - 11, u32::MAX - 3, u32::MAX - 7],
        ] {
            let tensor = Tensor::from_vec(values.clone(), 3, &device)?;
            assert_eq!(
                tensor.min(0)?.to_scalar::<u32>()?,
                *values.iter().min().unwrap()
            );
            assert_eq!(
                tensor.max(0)?.to_scalar::<u32>()?,
                *values.iter().max().unwrap()
            );
            assert_eq!(tensor.argmin(0)?.to_scalar::<u32>()?, 0);
            assert_eq!(tensor.argmax(0)?.to_scalar::<u32>()?, 1);
        }
        Ok(())
    }

    #[test]
    fn cuda_handle_disallows_low_precision_partial_reductions() -> anyhow::Result<()> {
        use candle_core::cuda_backend::{self, cudarc::cublas::sys};
        let candle_core::Device::Cuda(cuda) = super::load()? else {
            anyhow::bail!("CUDA build did not load a CUDA device")
        };
        let blas = cuda.cublas_handle();
        let mut mode = sys::cublasMath_t::CUBLAS_DEFAULT_MATH;
        unsafe { sys::cublasGetMathMode(*blas.handle(), &mut mode).result()? };
        assert_ne!(
            mode as u32
                & sys::cublasMath_t::CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION as u32,
            0
        );
        assert!(!cuda_backend::gemm_reduced_precision_bf16());
        assert!(!cuda_backend::gemm_reduced_precision_f16());
        assert!(!cuda_backend::gemm_reduced_precision_f32());
        Ok(())
    }
}

#[cfg(all(feature = "metal", not(any(feature = "cpu", feature = "cuda"))))]
pub fn load() -> anyhow::Result<Device> {
    Ok(Device::new_metal(0)?)
}

#[cfg(not(any(
    all(feature = "cpu", not(any(feature = "cuda", feature = "metal"))),
    all(feature = "cuda", not(any(feature = "cpu", feature = "metal"))),
    all(feature = "metal", not(any(feature = "cpu", feature = "cuda")))
)))]
pub fn load() -> anyhow::Result<Device> {
    unreachable!()
}

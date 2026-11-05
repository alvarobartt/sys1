use candle_core::{DType, Result, Tensor};

pub(crate) fn matmul_fp32(input: &Tensor, weight: &Tensor) -> Result<Tensor> {
    use candle_core::cuda_backend::{
        WrapErr,
        cudarc::{
            cublas::{result, sys},
            driver::{DevicePtr, DevicePtrMut},
        },
    };
    use candle_core::{CudaStorage, Storage, backend::BackendStorage, op::BackpropOp};
    let (rows, inner) = input.dims2()?;
    let (columns, weight_inner) = weight.dims2()?;
    if inner != weight_inner
        || input.dtype() != weight.dtype()
        || !input.device().same_device(weight.device())
    {
        candle_core::bail!(
            "Mixed-output linear requires matching input/weight device, width and dtype"
        )
    }
    let dims = |value| {
        i32::try_from(value)
            .map_err(|_| candle_core::Error::Msg("Linear dimension exceeds cuBLAS limits".into()))
    };
    let (m, n, k) = (dims(columns)?, dims(rows)?, dims(inner)?);
    let input = input.contiguous()?;
    let weight = weight.contiguous()?;
    let (xs, xl) = input.storage_and_layout();
    let (ws, wl) = weight.storage_and_layout();
    let (Storage::Cuda(xs), Storage::Cuda(ws)) = (&*xs, &*ws) else {
        candle_core::bail!("Mixed-output linear requires CUDA storage")
    };
    let (xstart, xend) = xl
        .contiguous_offsets()
        .ok_or_else(|| candle_core::Error::Msg("Linear input is not contiguous".into()))?;
    let (wstart, wend) = wl
        .contiguous_offsets()
        .ok_or_else(|| candle_core::Error::Msg("Linear weight is not contiguous".into()))?;
    let device = xs.device().clone();
    let stream = device.cuda_stream();
    let blas = device.cublas_handle();
    let count = rows
        .checked_mul(columns)
        .ok_or_else(|| candle_core::Error::Msg("Linear output size overflow".into()))?;
    let mut output = unsafe { device.alloc::<f32>(count)? };
    let alpha = 1f32;
    let beta = 0f32;
    macro_rules! gemm {
        ($ty:ty, $dtype:expr) => {{
            let x = xs.as_cuda_slice::<$ty>()?.slice(xstart..xend);
            let w = ws.as_cuda_slice::<$ty>()?.slice(wstart..wend);
            let (xptr, _xguard) = x.device_ptr(&stream);
            let (wptr, _wguard) = w.device_ptr(&stream);
            let (outptr, _outguard) = output.device_ptr_mut(&stream);

            unsafe {
                result::gemm_strided_batched_ex(
                    *blas.handle(),
                    sys::cublasOperation_t::CUBLAS_OP_T,
                    sys::cublasOperation_t::CUBLAS_OP_N,
                    m,
                    n,
                    k,
                    (&alpha as *const f32).cast(),
                    wptr as *const _,
                    $dtype,
                    k,
                    0,
                    xptr as *const _,
                    $dtype,
                    k,
                    0,
                    (&beta as *const f32).cast(),
                    outptr as *mut _,
                    sys::cudaDataType_t::CUDA_R_32F,
                    m,
                    0,
                    1,
                    sys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
                    sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT_TENSOR_OP,
                )
            }
            .w()?;
        }};
    }
    match input.dtype() {
        DType::F16 => gemm!(half::f16, sys::cudaDataType_t::CUDA_R_16F),
        DType::BF16 => gemm!(half::bf16, sys::cudaDataType_t::CUDA_R_16BF),
        dtype => candle_core::bail!("Unsupported mixed-output linear dtype {dtype:?}"),
    }
    Ok(Tensor::from_storage(
        Storage::Cuda(CudaStorage::wrap_cuda_slice(output, device)),
        (rows, columns),
        BackpropOp::none(),
        false,
    ))
}

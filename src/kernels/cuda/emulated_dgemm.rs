use candle_core::{
    CudaStorage, DType, Result, Storage, Tensor,
    backend::BackendStorage,
    cuda_backend::{
        WrapErr,
        cudarc::{
            cublas::{CudaBlas, Gemm, GemmConfig, sys},
            driver::{CudaSlice, CudaStream, DevicePtrMut},
        },
    },
    op::BackpropOp,
};
use std::sync::{Arc, Mutex};

mod ffi {
    use super::sys;
    #[link(name = "cublas")]
    unsafe extern "C" {
        pub(super) fn cublasSetFixedPointEmulationMantissaControl(
            handle: sys::cublasHandle_t,
            control: sys::cudaEmulationMantissaControl_t,
        ) -> sys::cublasStatus_t;
        pub(super) fn cublasSetFixedPointEmulationMantissaBitOffset(
            handle: sys::cublasHandle_t,
            offset: i32,
        ) -> sys::cublasStatus_t;
        pub(super) fn cublasSetFixedPointEmulationMaxMantissaBitCount(
            handle: sys::cublasHandle_t,
            bits: i32,
        ) -> sys::cublasStatus_t;
    }
}

pub(crate) const WORKSPACE_BYTES: usize = 384 * 1024 * 1024;
const MAX_MANTISSA_BITS: usize = 79;
struct State {
    stream: Arc<CudaStream>,
    blas: CudaBlas,
    workspace: CudaSlice<u8>,
}
static STATE: Mutex<Option<State>> = Mutex::new(None);

fn error(message: &str) -> candle_core::Error {
    candle_core::Error::Msg(message.into())
}

fn workspace_bound(m: usize, n: usize, k: usize) -> Option<usize> {
    let pm = m.checked_add(1023)?.checked_div(1024)?.checked_mul(1024)?;
    let pn = n.checked_add(1023)?.checked_div(1024)?.checked_mul(1024)?;
    let pk = k.checked_add(127)?.checked_div(128)?.checked_mul(128)?;
    let slices = (MAX_MANTISSA_BITS + 1).div_ceil(8);
    let gemm = pm
        .checked_add(pn)?
        .checked_mul(pk)?
        .checked_mul(slices)?
        .checked_add(pm.checked_add(pn)?.checked_mul(4)?)?;
    let nk = k.checked_add(63)?.checked_div(64)?;
    let dynamic = m
        .checked_add(n)?
        .checked_mul(nk)?
        .checked_add(m.checked_mul(n)?)?
        .checked_mul(4)?;
    gemm.max(dynamic)
        .checked_mul(5)?
        .checked_add(3)?
        .checked_div(4)?
        .checked_add(128 * 1024 * 1024)
}

pub(crate) fn resident_workspace_bytes() -> usize {
    STATE
        .lock()
        .ok()
        .map_or(0, |s| if s.is_some() { WORKSPACE_BYTES } else { 0 })
}

pub(crate) fn dot(input: &Tensor, weight: &Tensor) -> Result<Tensor> {
    let result = dot_impl(input, weight);
    if result.is_err()
        && let Ok(mut state) = STATE.lock()
    {
        state.take();
    }
    result
}

fn dot_impl(input: &Tensor, weight: &Tensor) -> Result<Tensor> {
    let (rows, inner) = input.dims2()?;
    let (columns, wk) = weight.dims2()?;
    if option_env!("SYS1_CUDA_COMPUTE_CAP") != Some("86")
        || !(8192..=16384).contains(&rows)
        || inner != 1024
        || columns != 2048
        || wk != inner
        || input.dtype() != DType::F64
        || weight.dtype() != DType::F64
        || !input.device().same_device(weight.device())
        || workspace_bound(rows, columns, inner).is_none_or(|n| n > WORKSPACE_BYTES)
    {
        candle_core::bail!("Unsupported SM86 dynamic FP64 dot geometry or workspace")
    }
    let (xs, xl) = input.storage_and_layout();
    let (ws, wl) = weight.storage_and_layout();
    let (Storage::Cuda(xs), Storage::Cuda(ws)) = (&*xs, &*ws) else {
        candle_core::bail!("SM86 dynamic FP64 dot requires CUDA")
    };
    let (a, z) = xl
        .contiguous_offsets()
        .ok_or_else(|| error("emulated input layout"))?;
    let x = xs.as_cuda_slice::<f64>()?.slice(a..z);
    let (a, z) = wl
        .contiguous_offsets()
        .ok_or_else(|| error("emulated weight layout"))?;
    let w = ws.as_cuda_slice::<f64>()?.slice(a..z);
    let device = xs.device().clone();
    let stream = device.cuda_stream();
    let mut output = unsafe { device.alloc::<f64>(rows * columns)? };
    let mut slot = STATE
        .lock()
        .map_err(|_| error("emulated FP64 state poisoned"))?;
    if slot
        .as_ref()
        .is_none_or(|s| !Arc::ptr_eq(&s.stream, &stream))
    {
        slot.take();
        let blas = CudaBlas::new(stream.clone()).w()?;
        let handle = *blas.handle();
        let mut version = 0;
        unsafe {
            sys::cublasGetVersion_v2(handle, &mut version)
                .result()
                .w()?;
        }
        if version < 130100 {
            candle_core::bail!("SM86 FP64 emulation requires production CUDA13.0.2 cuBLAS")
        }
        unsafe {
            sys::cublasSetMathMode(
                handle,
                sys::cublasMath_t::CUBLAS_FP64_EMULATED_FIXEDPOINT_MATH,
            )
            .result()
            .w()?;
            sys::cublasSetEmulationStrategy(
                handle,
                sys::cublasEmulationStrategy_t::CUBLAS_EMULATION_STRATEGY_EAGER,
            )
            .result()
            .w()?;
            ffi::cublasSetFixedPointEmulationMantissaControl(
                handle,
                sys::cudaEmulationMantissaControl::CUDA_EMULATION_MANTISSA_CONTROL_DYNAMIC,
            )
            .result()
            .w()?;
            ffi::cublasSetFixedPointEmulationMantissaBitOffset(handle, 0)
                .result()
                .w()?;
            ffi::cublasSetFixedPointEmulationMaxMantissaBitCount(handle, MAX_MANTISSA_BITS as i32)
                .result()
                .w()?;
        }
        let workspace = unsafe { device.alloc::<u8>(WORKSPACE_BYTES)? };
        *slot = Some(State {
            stream: stream.clone(),
            blas,
            workspace,
        });
    }
    let state = slot
        .as_mut()
        .ok_or_else(|| error("missing emulated FP64 state"))?;
    let (workspace, _workspace_guard) = state.workspace.device_ptr_mut(&stream);
    unsafe {
        sys::cublasSetWorkspace_v2(*state.blas.handle(), workspace as *mut _, WORKSPACE_BYTES)
            .result()
            .w()?;
        state
            .blas
            .gemm(
                GemmConfig {
                    transa: sys::cublasOperation_t::CUBLAS_OP_T,
                    transb: sys::cublasOperation_t::CUBLAS_OP_N,
                    m: columns as i32,
                    n: rows as i32,
                    k: inner as i32,
                    alpha: 1.0,
                    beta: 0.0,
                    lda: inner as i32,
                    ldb: inner as i32,
                    ldc: columns as i32,
                },
                &w,
                &x,
                &mut output,
            )
            .w()?;
    }
    #[cfg(test)]
    DOT_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    drop(_workspace_guard);
    drop(slot);
    Ok(Tensor::from_storage(
        Storage::Cuda(CudaStorage::wrap_cuda_slice(output, device)),
        (rows, columns),
        BackpropOp::none(),
        false,
    ))
}

#[cfg(test)]
static DOT_CALLS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

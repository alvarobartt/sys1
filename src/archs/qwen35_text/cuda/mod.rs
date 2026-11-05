use super::*;

pub(crate) use crate::kernels::cuda::qwen35::gated_delta;
use candle_core::{
    CudaStorage, Storage,
    backend::BackendStorage,
    cuda_backend::{
        WrapErr,
        cudarc::driver::{LaunchConfig, PushKernelArg},
    },
    op::BackpropOp,
};
use half::bf16;

mod normalization;
pub(crate) use normalization::*;

mod projection;
pub(crate) use projection::*;

mod convolution;
pub(crate) use convolution::*;

mod recurrence;
pub(crate) use recurrence::*;

mod packed;
pub(crate) use packed::*;

mod attention;
mod decoder;
mod linear_attention;
pub(crate) use attention::*;

pub(crate) mod prepared_convolution;

pub(crate) mod profiling;

pub(crate) const CUDA_PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/cuda.ptx"));

pub(crate) mod qkvz;

#[cfg(test)]
thread_local! {
    pub(crate) static VSTORAGE_MODEL_REPLAY:std::cell::Cell<bool>=const{std::cell::Cell::new(true)};
    pub(crate) static VSTORAGE_MODEL_CALLS:std::cell::Cell<usize>=const{std::cell::Cell::new(0)};
}

#[cfg(test)]
thread_local! {
    pub(crate) static QWEN_NN_REPLAY:std::cell::Cell<bool>=const{std::cell::Cell::new(true)};
    pub(crate) static QWEN_NN_CALLS:std::cell::Cell<usize>=const{std::cell::Cell::new(0)};
    pub(crate) static QWEN_NN_WEIGHT_BYTES:std::cell::Cell<usize>=const{std::cell::Cell::new(0)};
}

#[cfg(all(test, feature = "cuda", sys1_cuda_fused_ffn))]
thread_local! {
    pub(crate) static FUSED_MLP_CALLS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

#[cfg(all(test, feature = "cuda", sys1_cuda_fused_ffn))]
thread_local! {
    pub(crate) static QKVZ_DUAL_ENABLED: std::cell::Cell<bool> = const { std::cell::Cell::new(true) };
    pub(crate) static QKVZ_DUAL_CALLS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

#[cfg(all(test, feature = "cuda", sys1_cuda_direct_ffn))]
thread_local! {static DIRECT_MLP_CALLS:std::cell::Cell<usize>=const{std::cell::Cell::new(0)};}

pub(crate) fn is_hopper(device: &candle_core::CudaDevice) -> Result<bool> {
    Ok(device.cuda_stream().context().compute_capability().w()? == (9, 0))
}

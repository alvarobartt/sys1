use candle_core::backend::BackendStorage;
use candle_core::cuda_backend::{
    WrapErr,
    cudarc::driver::{LaunchConfig, PushKernelArg},
};
use candle_core::{CudaStorage, DType, Device, Result, Storage, Tensor, op::BackpropOp};
use half::bf16;

pub(crate) fn try_mul(left: &Tensor, right: &Tensor) -> Result<Option<Tensor>> {
    if left.dtype() != DType::BF16
        || right.dtype() != DType::BF16
        || left.dims() != right.dims()
        || !left.device().same_device(right.device())
        || left.elem_count() < 512
        || left.elem_count() > u32::MAX as usize
    {
        return Ok(None);
    }
    let Device::Cuda(_) = left.device() else {
        return Ok(None);
    };
    let (ls, ll) = left.storage_and_layout();
    let (rs, rl) = right.storage_and_layout();
    let (Some((a, z)), Some((b, y))) = (ll.contiguous_offsets(), rl.contiguous_offsets()) else {
        return Ok(None);
    };
    if a % 8 != 0 || b % 8 != 0 {
        return Ok(None);
    }
    let (Storage::Cuda(ls), Storage::Cuda(rs)) = (&*ls, &*rs) else {
        return Ok(None);
    };
    let l = ls.as_cuda_slice::<bf16>()?.slice(a..z);
    let r = rs.as_cuda_slice::<bf16>()?.slice(b..y);
    let device = ls.device().clone();
    let count = left.elem_count() as u64;
    let mut output = unsafe { device.alloc::<bf16>(count as usize)? };
    let function = device.get_or_load_custom_func(
        "mul_bf16_vec8",
        "sys1_pointwise_mul",
        include_str!(concat!(env!("OUT_DIR"), "/cuda.ptx")),
    )?;
    let mut launch = function.builder();
    launch.arg(&l).arg(&r).arg(&mut output).arg(&count);
    unsafe {
        launch.launch(LaunchConfig {
            grid_dim: (
                ((count as usize).div_ceil(8).div_ceil(256).min(4096)) as u32,
                1,
                1,
            ),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        })
    }
    .w()?;
    Ok(Some(Tensor::from_storage(
        Storage::Cuda(CudaStorage::wrap_cuda_slice(output, device)),
        left.shape().clone(),
        BackpropOp::none(),
        false,
    )))
}

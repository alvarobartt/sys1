use candle_core::{
    CudaStorage, DType, Result, Storage, Tensor,
    backend::BackendStorage,
    cuda_backend::{
        WrapErr,
        cudarc::driver::{LaunchConfig, PushKernelArg},
    },
    op::BackpropOp,
};
use half::bf16;

const PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/cuda.ptx"));

pub(crate) fn project(
    input: &Tensor,
    linear: &candle_nn::Linear,
    radius_coefficient: f64,
) -> Result<Tensor> {
    let (batch, length, inner) = input.dims3()?;
    let (columns, weight_inner) = linear.weight().dims2()?;
    let bias = linear.bias().ok_or_else(|| {
        candle_core::Error::Msg("Guarded biased projection requires a bias".into())
    })?;
    let rows = batch
        .checked_mul(length)
        .ok_or_else(|| candle_core::Error::Msg("Guarded biased projection row overflow".into()))?;
    let count = rows.checked_mul(columns).ok_or_else(|| {
        candle_core::Error::Msg("Guarded biased projection output overflow".into())
    })?;
    if inner != 1024
        || weight_inner != inner
        || columns != 2048
        || rows < 128
        || input.dtype() != DType::BF16
        || linear.weight().dtype() != DType::BF16
        || bias.dtype() != DType::BF16
        || bias.dims() != [columns]
        || !input.device().same_device(linear.weight().device())
        || !input.device().same_device(bias.device())
        || [input.elem_count(), linear.weight().elem_count(), count]
            .iter()
            .any(|&n| n > i32::MAX as usize)
        || !radius_coefficient.is_finite()
        || radius_coefficient <= 0.
    {
        candle_core::bail!("Unsupported guarded biased projection shape, dtype or bound")
    }
    let input = input.contiguous()?;
    let weight = linear.weight().contiguous()?;
    let bias = bias.contiguous()?;
    let (xs, xl) = input.storage_and_layout();
    let (ws, wl) = weight.storage_and_layout();
    let (bs, bl) = bias.storage_and_layout();
    let (Storage::Cuda(xs), Storage::Cuda(ws), Storage::Cuda(bs)) = (&*xs, &*ws, &*bs) else {
        candle_core::bail!("Guarded biased projection requires CUDA storage")
    };
    let offsets = |layout: &candle_core::Layout| {
        layout.contiguous_offsets().ok_or_else(|| {
            candle_core::Error::Msg("Guarded biased projection requires contiguous storage".into())
        })
    };
    let (x0, x1) = offsets(xl)?;
    let (w0, w1) = offsets(wl)?;
    let (b0, b1) = offsets(bl)?;
    let x = xs.as_cuda_slice::<bf16>()?.slice(x0..x1);
    let w = ws.as_cuda_slice::<bf16>()?.slice(w0..w1);
    let b = bs.as_cuda_slice::<bf16>()?.slice(b0..b1);
    let device = xs.device().clone();
    let mut output = unsafe { device.alloc::<bf16>(count)? };
    let function =
        device.get_or_load_custom_func("biased_projection_guard_bf16", "sys1_qwen35", PTX)?;
    let mut launch = function.builder();
    let (rows_arg, inner_arg, columns_arg) = (rows as i32, inner as i32, columns as i32);
    launch
        .arg(&x)
        .arg(&w)
        .arg(&b)
        .arg(&mut output)
        .arg(&rows_arg)
        .arg(&inner_arg)
        .arg(&columns_arg)
        .arg(&radius_coefficient);
    unsafe {
        launch.launch(LaunchConfig {
            grid_dim: (rows.div_ceil(16) as u32, columns.div_ceil(32) as u32, 1),
            block_dim: (128, 1, 1),
            shared_mem_bytes: 0,
        })
    }
    .w()?;
    Ok(Tensor::from_storage(
        Storage::Cuda(CudaStorage::wrap_cuda_slice(output, device)),
        (batch, length, columns),
        BackpropOp::none(),
        false,
    ))
}

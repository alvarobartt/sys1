use candle_core::{
    CudaStorage, D, Result, Storage, Tensor,
    backend::BackendStorage,
    cuda_backend::{
        WrapErr,
        cudarc::driver::{LaunchConfig, PushKernelArg},
    },
    op::BackpropOp,
};

const KERNEL: &str = include_str!(concat!(env!("OUT_DIR"), "/qwen35.ptx"));

pub(crate) fn forward(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    g: &Tensor,
    beta: &Tensor,
) -> Result<Tensor> {
    let (batch, heads, length, key_dim) = q.dims4()?;
    let value_dim = v.dim(D::Minus1)?;
    if key_dim != 128
        || k.dims() != q.dims()
        || v.dims() != [batch, heads, length, value_dim]
        || g.dims() != [batch, heads, length]
        || beta.dims() != [batch, heads, length]
    {
        candle_core::bail!("Unsupported Qwen3.5 CUDA delta-rule shape")
    }
    let input = Tensor::cat(&[q, k, v, &g.unsqueeze(3)?, &beta.unsqueeze(3)?], 3)?.contiguous()?;
    let (storage, layout) = input.storage_and_layout();
    let Storage::Cuda(storage) = &*storage else {
        candle_core::bail!("Qwen3.5 delta rule requires CUDA storage")
    };
    let (start, end) = layout
        .contiguous_offsets()
        .ok_or_else(|| candle_core::Error::Msg("Qwen3.5 delta input is not contiguous".into()))?;
    let source = storage.as_cuda_slice::<f32>()?.slice(start..end);
    let device = storage.device().clone();
    let output = unsafe { device.alloc::<f32>(batch * heads * length * value_dim)? };
    let function = device.get_or_load_custom_func("qwen35_delta_f32", "qwen35_delta", KERNEL)?;
    let config = LaunchConfig {
        grid_dim: ((batch * heads) as u32, value_dim.div_ceil(4) as u32, 1),
        block_dim: (128, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut launch = function.builder();
    launch.arg(&source);
    launch.arg(&output);
    let length = length as i32;
    let value_dim = value_dim as i32;
    launch.arg(&length);
    launch.arg(&value_dim);
    unsafe { launch.launch(config) }.w()?;
    let output = Tensor::from_storage(
        Storage::Cuda(CudaStorage::wrap_cuda_slice(output, device)),
        (batch, heads, length as usize, value_dim as usize),
        BackpropOp::none(),
        false,
    );
    output.transpose(1, 2)
}

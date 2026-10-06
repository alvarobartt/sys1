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

const KERNEL: &str = include_str!(concat!(env!("OUT_DIR"), "/qwen35.ptx"));

pub(crate) fn forward(input: &Tensor, weights: &Tensor) -> Result<Tensor> {
    let (batch, length, channels) = input.dims3()?;
    let (weight_channels, groups, kernel) = weights.dims3()?;
    if weight_channels != channels || groups != 1 || kernel == 0 || input.dtype() != weights.dtype()
    {
        candle_core::bail!("Unsupported Qwen3.5 CUDA convolution shape or dtype")
    }
    let input = input.contiguous()?;
    let weights = weights.contiguous()?;
    let (input_storage, input_layout) = input.storage_and_layout();
    let (weight_storage, weight_layout) = weights.storage_and_layout();
    let (Storage::Cuda(input_storage), Storage::Cuda(weight_storage)) =
        (&*input_storage, &*weight_storage)
    else {
        candle_core::bail!("Qwen3.5 convolution requires CUDA storage")
    };
    let (input_start, input_end) = input_layout.contiguous_offsets().ok_or_else(|| {
        candle_core::Error::Msg("Qwen3.5 convolution input is not contiguous".into())
    })?;
    let (weight_start, weight_end) = weight_layout.contiguous_offsets().ok_or_else(|| {
        candle_core::Error::Msg("Qwen3.5 convolution weights are not contiguous".into())
    })?;
    let device = input_storage.device().clone();
    let count = batch * length * channels;
    let config = LaunchConfig {
        grid_dim: (count.div_ceil(256) as u32, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    let count_arg = count as i32;
    let length_arg = length as i32;
    let channels_arg = channels as i32;
    let kernel_arg = kernel as i32;
    let output = match input.dtype() {
        DType::BF16 => {
            let source = input_storage
                .as_cuda_slice::<bf16>()?
                .slice(input_start..input_end);
            let filters = weight_storage
                .as_cuda_slice::<bf16>()?
                .slice(weight_start..weight_end);
            let output = unsafe { device.alloc::<bf16>(count)? };
            let function = device.get_or_load_custom_func("qwen35_conv_bf16", "qwen35", KERNEL)?;
            let mut launch = function.builder();
            launch.arg(&source);
            launch.arg(&filters);
            launch.arg(&output);
            launch.arg(&count_arg);
            launch.arg(&length_arg);
            launch.arg(&channels_arg);
            launch.arg(&kernel_arg);
            unsafe { launch.launch(config) }.w()?;
            Storage::Cuda(CudaStorage::wrap_cuda_slice(output, device))
        }
        DType::F32 => {
            let source = input_storage
                .as_cuda_slice::<f32>()?
                .slice(input_start..input_end);
            let filters = weight_storage
                .as_cuda_slice::<f32>()?
                .slice(weight_start..weight_end);
            let output = unsafe { device.alloc::<f32>(count)? };
            let function = device.get_or_load_custom_func("qwen35_conv_f32", "qwen35", KERNEL)?;
            let mut launch = function.builder();
            launch.arg(&source);
            launch.arg(&filters);
            launch.arg(&output);
            launch.arg(&count_arg);
            launch.arg(&length_arg);
            launch.arg(&channels_arg);
            launch.arg(&kernel_arg);
            unsafe { launch.launch(config) }.w()?;
            Storage::Cuda(CudaStorage::wrap_cuda_slice(output, device))
        }
        dtype => candle_core::bail!("Unsupported Qwen3.5 CUDA convolution dtype: {dtype:?}"),
    };
    Ok(Tensor::from_storage(
        output,
        (batch, length, channels),
        BackpropOp::none(),
        false,
    ))
}

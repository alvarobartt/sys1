use candle_core::cuda_backend::{
    WrapErr,
    cudarc::driver::{LaunchConfig, PushKernelArg},
};
use candle_core::{
    CudaStorage, DType, Result, Storage, Tensor, backend::BackendStorage, op::BackpropOp,
};
use half::bf16;
const PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/fused_gate_up.ptx"));
const SHARED: u32 = 73728;
fn error<E: std::fmt::Display>(e: E) -> candle_core::Error {
    candle_core::Error::Msg(format!("fused BF16 MLP: {e}"))
}

pub(crate) fn available(gate: &Tensor, up: &Tensor) -> bool {
    gate.dims() == [12288, 4096]
        && up.dims() == gate.dims()
        && gate.dtype() == DType::BF16
        && up.dtype() == DType::BF16
        && gate.device().is_cuda()
        && gate.device().same_device(up.device())
}

pub(crate) fn prepare(gate: &Tensor, up: &Tensor) -> Result<Tensor> {
    if !available(gate, up) {
        candle_core::bail!("unsupported fused weight geometry");
    }
    let (n, k) = gate.dims2()?;
    let packed = Tensor::cat(
        &[
            &gate.reshape((n / 32, 32, k))?,
            &up.reshape((n / 32, 32, k))?,
        ],
        1,
    )?
    .reshape((2 * n, k))?
    .t()?
    .contiguous()?;
    let candle_core::Device::Cuda(cuda) = gate.device() else {
        unreachable!()
    };
    cuda.cuda_stream()
        .context()
        .bind_to_thread()
        .map_err(error)?;
    {
        let name = "fused_ffn_bf16";
        let function = cuda.get_or_load_custom_func(name, "sys1_fused_ffn", PTX)?;
        function.set_attribute(candle_core::cuda_backend::cudarc::driver::sys::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,SHARED as i32).w()?;
    }
    Ok(packed)
}

pub(crate) fn forward(input: &Tensor, weight: &Tensor) -> Result<Tensor> {
    let (batch, length, k) = input.dims3()?;
    let rows = batch
        .checked_mul(length)
        .ok_or_else(|| error("rows overflow"))?;
    let (wk, twon) = weight.dims2()?;
    let n = twon / 2;
    if k != 4096
        || twon != 24576
        || wk != k
        || !(512..=32768).contains(&rows)
        || input.dtype() != DType::BF16
        || weight.dtype() != DType::BF16
        || !input.device().same_device(weight.device())
    {
        candle_core::bail!("unsupported fused FFN input");
    }
    let input = input.contiguous()?;
    let (xs, xl) = input.storage_and_layout();
    let (ws, wl) = weight.storage_and_layout();
    let (Storage::Cuda(xs), Storage::Cuda(ws)) = (&*xs, &*ws) else {
        candle_core::bail!("CUDA storage required");
    };
    let (xstart, xend) = xl
        .contiguous_offsets()
        .ok_or_else(|| error("noncontiguous input"))?;
    let (wstart, wend) = wl
        .contiguous_offsets()
        .ok_or_else(|| error("noncontiguous packed weights"))?;
    let xstore = xs.as_cuda_slice::<bf16>()?;
    let wstore = ws.as_cuda_slice::<bf16>()?;
    if xend > xstore.len() || wend > wstore.len() || xstart % 8 != 0 || wstart % 8 != 0 {
        candle_core::bail!("fused operand bounds/alignment differ");
    }
    let x = xstore.slice(xstart..xend);
    let w = wstore.slice(wstart..wend);
    let device = xs.device().clone();
    let stream = device.cuda_stream();
    stream.context().bind_to_thread().map_err(error)?;
    let count = rows
        .checked_mul(n)
        .ok_or_else(|| error("output overflow"))?;
    let mut output = unsafe { device.alloc::<bf16>(count)? };

    let name = "fused_ffn_bf16";
    let function = device.get_or_load_custom_func(name, "sys1_fused_ffn", PTX)?;
    let swizzle = if rows >= 16384 { 32u32 } else { 8u32 };
    let ri = i32::try_from(rows).map_err(error)?;
    let ni = i32::try_from(n).map_err(error)?;
    let ki = i32::try_from(k).map_err(error)?;
    super::activation::with_sigmoid_lookup(&device, |table| {
        let mut launch = function.builder();
        launch.arg(&w);
        launch.arg(&x);
        launch.arg(table);
        launch.arg(&mut output);

        launch.arg(&ri);
        launch.arg(&ni);
        launch.arg(&ki);
        unsafe {
            launch.launch(LaunchConfig {
                grid_dim: (
                    (rows as u32).div_ceil(128) * swizzle,
                    (n as u32 / 128).div_ceil(swizzle),
                    1,
                ),
                block_dim: (256, 1, 1),
                shared_mem_bytes: SHARED,
            })
        }
        .w()?;
        Ok(())
    })?;
    Ok(Tensor::from_storage(
        Storage::Cuda(CudaStorage::wrap_cuda_slice(output, device)),
        (batch, length, n),
        BackpropOp::none(),
        false,
    ))
}

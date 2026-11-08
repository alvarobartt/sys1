use candle_core::cuda_backend::{
    WrapErr,
    cudarc::driver::{LaunchConfig, PushKernelArg},
};
use candle_core::{
    CudaStorage, DType, Result, Storage, Tensor, backend::BackendStorage, op::BackpropOp,
};
use half::bf16;

const PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/fused_gate_up.ptx"));
const MODULE: &str = "sys1_fused_ffn";

fn error<E: std::fmt::Display>(e: E) -> candle_core::Error {
    candle_core::Error::Msg(format!("direct QKV/Z projection: {e}"))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Tile {
    Narrow,
    Wide,
}

impl Tile {
    fn dimensions(self) -> (u32, u32, u32, &'static str) {
        match self {
            Self::Narrow => (128, 128, 49152, "qkvz_narrow_bf16"),
            Self::Wide => (256, 256, 73728, "qkvz_wide_bf16"),
        }
    }
}

pub(crate) fn select_tile(rows: usize) -> Option<Tile> {
    match rows {
        513..=544 => Some(Tile::Narrow),
        641..=896 => Some(Tile::Wide),
        897..=1152 => Some(Tile::Narrow),
        1153..=1408 => Some(Tile::Wide),
        1409..=1664 => Some(Tile::Narrow),
        1665..=1920 => Some(Tile::Wide),
        2177..=2432 => Some(Tile::Wide),
        2433..=2688 => Some(Tile::Narrow),
        2689..=2944 => Some(Tile::Wide),
        2945..=3200 => Some(Tile::Narrow),
        3201..=3456 => Some(Tile::Wide),
        3713..=3968 => Some(Tile::Wide),
        3969..=4224 => Some(Tile::Narrow),
        4225..=4480 => Some(Tile::Wide),
        4481..=4736 => Some(Tile::Narrow),
        4737..=4992 => Some(Tile::Wide),
        5249..=5504 => Some(Tile::Wide),
        5505..=5760 => Some(Tile::Narrow),
        5761..=6016 => Some(Tile::Wide),
        6017..=6144 => Some(Tile::Narrow),
        6145..=6400 => Some(Tile::Wide),
        6657..=6912 => Some(Tile::Wide),
        6913..=7168 => Some(Tile::Narrow),
        7169..=7424 => Some(Tile::Wide),
        7425..=7680 => Some(Tile::Narrow),
        7681..=7936 => Some(Tile::Wide),
        16384..=32768 => Some(Tile::Wide),
        _ => None,
    }
}

pub(crate) fn available(qkv: &Tensor, z: &Tensor) -> bool {
    qkv.dims() == [8192, 4096]
        && z.dims() == [4096, 4096]
        && qkv.dtype() == DType::BF16
        && z.dtype() == DType::BF16
        && qkv.device().is_cuda()
        && qkv.device().same_device(z.device())
}

pub(crate) fn prepare(qkv: &Tensor, z: &Tensor) -> Result<Tensor> {
    if !available(qkv, z) {
        candle_core::bail!("unsupported direct QKV/Z weights");
    }
    let qkv = qkv.contiguous()?;
    let z = z.contiguous()?;
    let (qs, ql) = qkv.storage_and_layout();
    let (zs, zl) = z.storage_and_layout();
    let (Storage::Cuda(qs), Storage::Cuda(zs)) = (&*qs, &*zs) else {
        candle_core::bail!("CUDA QKV/Z storage required");
    };
    let (qb, qe) = ql.contiguous_offsets().ok_or_else(|| error("QKV layout"))?;
    let (zb, ze) = zl.contiguous_offsets().ok_or_else(|| error("Z layout"))?;
    let qstore = qs.as_cuda_slice::<bf16>()?;
    let zstore = zs.as_cuda_slice::<bf16>()?;
    if qe > qstore.len() || ze > zstore.len() || qe - qb != 8192 * 4096 || ze - zb != 4096 * 4096 {
        candle_core::bail!("QKV/Z source extent differs");
    }
    let q = qstore.slice(qb..qe);
    let z = zstore.slice(zb..ze);
    let device = qs.device().clone();
    let stream = device.cuda_stream();
    stream.context().bind_to_thread().map_err(error)?;
    let count = 4096usize * 12288;

    let mut packed = unsafe { device.alloc::<bf16>(count)? };
    let function = device.get_or_load_custom_func("qkvz_pack_bf16", MODULE, PTX)?;
    let mut launch = function.builder();
    launch.arg(&q);
    launch.arg(&z);
    launch.arg(&mut packed);
    unsafe {
        launch.launch(LaunchConfig {
            grid_dim: ((count as u32).div_ceil(256), 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        })
    }
    .w()?;
    for tile in [Tile::Narrow, Tile::Wide] {
        let (_, _, shared, name) = tile.dimensions();
        let function = device.get_or_load_custom_func(name, MODULE, PTX)?;
        function.set_attribute(candle_core::cuda_backend::cudarc::driver::sys::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, shared as i32).w()?;
    }
    Ok(Tensor::from_storage(
        Storage::Cuda(CudaStorage::wrap_cuda_slice(packed, device)),
        (4096, 12288),
        BackpropOp::none(),
        false,
    ))
}

pub(crate) fn forward(input: &Tensor, packed: &Tensor, tile: Tile) -> Result<(Tensor, Tensor)> {
    let (batch, length, k) = input.dims3()?;
    let rows = batch
        .checked_mul(length)
        .ok_or_else(|| error("row overflow"))?;
    if k != 4096
        || !(512..=32768).contains(&rows)
        || input.dtype() != DType::BF16
        || packed.dtype() != DType::BF16
        || packed.dims() != [4096, 12288]
        || !input.device().same_device(packed.device())
    {
        candle_core::bail!("unsupported direct QKV/Z input");
    }
    let input = input.contiguous()?;
    let (xs, xl) = input.storage_and_layout();
    let (ws, wl) = packed.storage_and_layout();
    let (Storage::Cuda(xs), Storage::Cuda(ws)) = (&*xs, &*ws) else {
        candle_core::bail!("CUDA QKV/Z storage required");
    };
    let (xb, xe) = xl
        .contiguous_offsets()
        .ok_or_else(|| error("input layout"))?;
    let (wb, we) = wl
        .contiguous_offsets()
        .ok_or_else(|| error("packed layout"))?;
    let xstore = xs.as_cuda_slice::<bf16>()?;
    let wstore = ws.as_cuda_slice::<bf16>()?;
    if xe > xstore.len()
        || we > wstore.len()
        || xe - xb != rows * k
        || we - wb != k * 12288
        || xb % 8 != 0
        || wb % 8 != 0
    {
        candle_core::bail!("QKV/Z operand bounds/alignment differ");
    }
    let x = xstore.slice(xb..xe);
    let w = wstore.slice(wb..we);
    let device = xs.device().clone();
    device
        .cuda_stream()
        .context()
        .bind_to_thread()
        .map_err(error)?;
    let mut qkv = unsafe { device.alloc::<bf16>(rows * 8192)? };
    let mut z = unsafe { device.alloc::<bf16>(rows * 4096)? };
    let (width, threads, shared, name) = tile.dimensions();
    let function = device.get_or_load_custom_func(name, MODULE, PTX)?;
    let m = i32::try_from(rows).map_err(error)?;
    let swizzle = if rows >= 16384 { 32u32 } else { 8u32 };
    let mut launch = function.builder();
    launch.arg(&x);
    launch.arg(&w);
    launch.arg(&mut qkv);
    launch.arg(&mut z);
    launch.arg(&m);
    unsafe {
        launch.launch(LaunchConfig {
            grid_dim: (
                (rows as u32).div_ceil(128) * swizzle,
                (12288u32 / width).div_ceil(swizzle),
                1,
            ),
            block_dim: (threads, 1, 1),
            shared_mem_bytes: shared,
        })
    }
    .w()?;
    let wrap = |buffer, n| {
        Tensor::from_storage(
            Storage::Cuda(CudaStorage::wrap_cuda_slice(buffer, device.clone())),
            (batch, length, n),
            BackpropOp::none(),
            false,
        )
    };
    Ok((wrap(qkv, 8192), wrap(z, 4096)))
}

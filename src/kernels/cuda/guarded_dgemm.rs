use candle_core::{
    CudaStorage, DType, Result, Storage, Tensor,
    backend::BackendStorage,
    cuda_backend::{
        WrapErr,
        cudarc::driver::{CudaStream, LaunchConfig, PushKernelArg},
    },
    op::BackpropOp,
};
use half::bf16;
use std::sync::{Arc, Mutex};

const PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/cuda.ptx"));

const CACHE_ENTRIES: usize = 96;
#[cfg(sys1_cuda_fp64_emulation)]
const CACHE_BYTES: usize = 128 * 1024 * 1024;
#[cfg(not(sys1_cuda_fp64_emulation))]
const CACHE_BYTES: usize = 1152 * 1024 * 1024;
struct Entry {
    source: Tensor,
    wide: Tensor,
    absolute: Tensor,
    safe: Tensor,
    bytes: usize,
}
struct Cache {
    stream: Arc<CudaStream>,
    entries: Vec<Entry>,
}
static CACHE: Mutex<Option<Cache>> = Mutex::new(None);

fn error(message: &str) -> candle_core::Error {
    candle_core::Error::Msg(message.into())
}

fn prepared(weight: &Tensor) -> Result<(Tensor, Tensor, Tensor)> {
    let candle_core::Device::Cuda(cuda) = weight.device() else {
        candle_core::bail!("DGEMM needs CUDA")
    };
    let stream = cuda.cuda_stream();
    let mut slot = CACHE.lock().map_err(|_| error("DGEMM cache poisoned"))?;
    if slot
        .as_ref()
        .is_none_or(|c| !Arc::ptr_eq(&c.stream, &stream))
    {
        *slot = Some(Cache {
            stream,
            entries: Vec::new(),
        });
    }
    let cache = slot.as_mut().ok_or_else(|| error("missing DGEMM cache"))?;
    if let Some(index) = cache
        .entries
        .iter()
        .position(|e| e.source.id() == weight.id())
    {
        let entry = cache.entries.remove(index);
        let result = (
            entry.wide.clone(),
            entry.absolute.clone(),
            entry.safe.clone(),
        );
        cache.entries.push(entry);
        return Ok(result);
    }
    let (columns, inner) = weight.dims2()?;
    let bytes = columns
        .checked_mul(inner)
        .and_then(|n| n.checked_mul(8))
        .and_then(|n| {
            columns
                .checked_mul(12)
                .and_then(|stats| n.checked_add(stats))
        })
        .ok_or_else(|| error("DGEMM prepared weight size overflow"))?;
    if bytes > CACHE_BYTES {
        candle_core::bail!("DGEMM prepared weight exceeds cache bound")
    }

    while cache.entries.len() >= CACHE_ENTRIES
        || cache.entries.iter().map(|e| e.bytes).sum::<usize>() > CACHE_BYTES - bytes
    {
        cache.entries.remove(0);
    }
    let contiguous = weight.contiguous()?;
    let (storage, layout) = contiguous.storage_and_layout();
    let Storage::Cuda(storage) = &*storage else {
        candle_core::bail!("DGEMM weight needs CUDA")
    };
    let (first, last) = layout
        .contiguous_offsets()
        .ok_or_else(|| error("DGEMM weight layout"))?;
    let w = storage.as_cuda_slice::<bf16>()?.slice(first..last);
    let device = storage.device().clone();
    let mut wide = unsafe { device.alloc::<f64>(columns * inner)? };
    let mut absolute = unsafe { device.alloc::<f64>(columns)? };
    let mut safe = unsafe { device.alloc::<u32>(columns)? };
    let function =
        device.get_or_load_custom_func("qwen35_guarded_dgemm_weight", "sys1_qwen35_dgemm", PTX)?;
    let mut launch = function.builder();
    let (ki, ni) = (inner as i32, columns as i32);
    launch
        .arg(&w)
        .arg(&mut wide)
        .arg(&mut absolute)
        .arg(&mut safe)
        .arg(&ki)
        .arg(&ni);
    unsafe { launch.launch(LaunchConfig::for_num_elems(columns as u32)) }.w()?;
    let wide = Tensor::from_storage(
        Storage::Cuda(CudaStorage::wrap_cuda_slice(wide, device.clone())),
        (columns, inner),
        BackpropOp::none(),
        false,
    );
    let absolute = Tensor::from_storage(
        Storage::Cuda(CudaStorage::wrap_cuda_slice(absolute, device.clone())),
        columns,
        BackpropOp::none(),
        false,
    );
    let safe = Tensor::from_storage(
        Storage::Cuda(CudaStorage::wrap_cuda_slice(safe, device)),
        columns,
        BackpropOp::none(),
        false,
    );
    cache.entries.push(Entry {
        source: weight.clone(),
        wide: wide.clone(),
        absolute: absolute.clone(),
        safe: safe.clone(),
        bytes,
    });
    Ok((wide, absolute, safe))
}

pub(crate) fn project(
    input: &Tensor,
    linear: &candle_nn::Linear,
    coefficient: f64,
) -> Result<Tensor> {
    let result = project_impl(input, linear, coefficient);
    if result.is_err()
        && let Ok(mut slot) = CACHE.lock()
    {
        slot.take();
    }
    result
}

fn project_impl(input: &Tensor, linear: &candle_nn::Linear, coefficient: f64) -> Result<Tensor> {
    let (batch, length, inner) = input.dims3()?;
    let (columns, weight_inner) = linear.weight().dims2()?;
    let has_bias = i32::from(linear.bias().is_some());

    let bias = linear.bias().unwrap_or(linear.weight());
    let rows = batch
        .checked_mul(length)
        .ok_or_else(|| error("DGEMM rows overflow"))?;
    let count = rows
        .checked_mul(columns)
        .ok_or_else(|| error("DGEMM output overflow"))?;
    let supported = (has_bias != 0 && inner == 1024 && columns == 2048 && rows >= 128)
        || (has_bias == 0 && inner == 4096 && columns == 64 && rows >= 1536);
    let supported = supported
        || has_bias != 0
            && (64..=65536).contains(&rows)
            && ((inner == 1024 && matches!(columns, 1024 | 2048 | 3072 | 4096))
                || (inner == 4096 && columns == 1024));
    if !supported
        || weight_inner != inner
        || input.dtype() != DType::BF16
        || linear.weight().dtype() != DType::BF16
        || bias.dtype() != DType::BF16
        || (has_bias != 0 && bias.dims() != [columns])
        || !input.device().same_device(linear.weight().device())
        || !input.device().same_device(bias.device())
        || !coefficient.is_finite()
        || coefficient <= 0.
        || [input.elem_count(), linear.weight().elem_count(), count]
            .iter()
            .any(|&n| n > i32::MAX as usize)
    {
        candle_core::bail!("Unsupported guarded DGEMM input or bound")
    }
    let (wide_weight, weight_absolute, weight_safe) = prepared(linear.weight())?;
    let input = input.contiguous()?;
    let weight = linear.weight().contiguous()?;
    let bias = bias.contiguous()?;
    let (xs, xl) = input.storage_and_layout();
    let (ws, wl) = weight.storage_and_layout();
    let (bs, bl) = bias.storage_and_layout();
    let (Storage::Cuda(xs), Storage::Cuda(ws), Storage::Cuda(bs)) = (&*xs, &*ws, &*bs) else {
        candle_core::bail!("Guarded DGEMM requires CUDA")
    };
    let offsets = |layout: &candle_core::Layout| {
        layout
            .contiguous_offsets()
            .ok_or_else(|| error("DGEMM layout"))
    };
    let (a, z) = offsets(xl)?;
    let x = xs.as_cuda_slice::<bf16>()?.slice(a..z);
    let (a, z) = offsets(wl)?;
    let w = ws.as_cuda_slice::<bf16>()?.slice(a..z);
    let (a, z) = offsets(bl)?;
    let b = bs.as_cuda_slice::<bf16>()?.slice(a..z);
    let device = xs.device().clone();
    let (ri, ki, ni) = (rows as i32, inner as i32, columns as i32);
    let mut wide_input = unsafe { device.alloc::<f64>(rows * inner)? };
    let mut maximum = unsafe { device.alloc::<f64>(rows)? };
    let mut input_safe = unsafe { device.alloc::<u32>(rows)? };
    let function =
        device.get_or_load_custom_func("qwen35_guarded_dgemm_input", "sys1_qwen35_dgemm", PTX)?;
    let mut launch = function.builder();
    launch
        .arg(&x)
        .arg(&mut wide_input)
        .arg(&mut maximum)
        .arg(&mut input_safe)
        .arg(&ri)
        .arg(&ki);
    unsafe {
        launch.launch(LaunchConfig {
            grid_dim: (rows as u32, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        })
    }
    .w()?;
    let wide_input = Tensor::from_storage(
        Storage::Cuda(CudaStorage::wrap_cuda_slice(wide_input, device.clone())),
        (rows, inner),
        BackpropOp::none(),
        false,
    );

    #[cfg(sys1_cuda_fp64_emulation)]
    let dot = if option_env!("SYS1_CUDA_COMPUTE_CAP") == Some("86")
        && inner == 1024
        && columns == 2048
        && (8192..=16384).contains(&rows)
    {
        crate::kernels::cuda::emulated_dgemm::dot(&wide_input, &wide_weight)?
    } else {
        wide_input.matmul(&wide_weight.t()?)?
    };
    #[cfg(not(sys1_cuda_fp64_emulation))]
    let dot = wide_input.matmul(&wide_weight.t()?)?;
    let (ds, dl) = dot.storage_and_layout();
    let (as_, al) = weight_absolute.storage_and_layout();
    let (ss, sl) = weight_safe.storage_and_layout();
    let (Storage::Cuda(ds), Storage::Cuda(as_), Storage::Cuda(ss)) = (&*ds, &*as_, &*ss) else {
        candle_core::bail!("DGEMM prepared storage differs")
    };
    let (a, z) = offsets(dl)?;
    let d = ds.as_cuda_slice::<f64>()?.slice(a..z);
    let (a, z) = offsets(al)?;
    let absolute = as_.as_cuda_slice::<f64>()?.slice(a..z);
    let (a, z) = offsets(sl)?;
    let safe_w = ss.as_cuda_slice::<u32>()?.slice(a..z);
    let mut output = unsafe { device.alloc::<bf16>(count)? };
    let function =
        device.get_or_load_custom_func("qwen35_guarded_dgemm_finish", "sys1_qwen35_dgemm", PTX)?;
    let mut launch = function.builder();
    let null_counter = 0u64;
    launch
        .arg(&x)
        .arg(&w)
        .arg(&b)
        .arg(&d)
        .arg(&maximum)
        .arg(&absolute)
        .arg(&input_safe)
        .arg(&safe_w)
        .arg(&mut output)
        .arg(&ri)
        .arg(&ki)
        .arg(&ni)
        .arg(&has_bias)
        .arg(&coefficient)
        .arg(&null_counter);
    unsafe {
        launch.launch(LaunchConfig {
            grid_dim: (count.div_ceil(256) as u32, 1, 1),
            block_dim: (256, 1, 1),
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

pub(crate) fn head_cache163_memory() -> (usize, usize, usize) {
    let sm86_emulation =
        cfg!(sys1_cuda_fp64_emulation) && option_env!("SYS1_CUDA_COMPUTE_CAP") == Some("86");
    if !sm86_emulation && !matches!(option_env!("SYS1_CUDA_COMPUTE_CAP"), Some("80" | "90")) {
        return (0, 0, 0);
    }
    let resident = CACHE
        .lock()
        .ok()
        .and_then(|s| {
            s.as_ref().map(|c| {
                (
                    c.entries.len(),
                    c.entries.iter().map(|e| e.bytes).sum::<usize>(),
                )
            })
        })
        .unwrap_or((0, 0));
    #[cfg(sys1_cuda_fp64_emulation)]
    let (workspace_resident, workspace_capacity) = (
        crate::kernels::cuda::emulated_dgemm::resident_workspace_bytes(),
        crate::kernels::cuda::emulated_dgemm::WORKSPACE_BYTES,
    );
    #[cfg(not(sys1_cuda_fp64_emulation))]
    let (workspace_resident, workspace_capacity) = (0usize, 0usize);
    (
        resident.0,
        resident.1.saturating_add(workspace_resident),
        CACHE_BYTES
            .saturating_sub(resident.1)
            .saturating_add(workspace_capacity.saturating_sub(workspace_resident)),
    )
}
#[cfg(test)]
static HEAD_ROUTE163_CALLS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

#[cfg(test)]
pub(crate) fn record_head_route163() {
    HEAD_ROUTE163_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
}

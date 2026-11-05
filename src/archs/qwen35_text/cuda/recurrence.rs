use super::*;

pub(crate) fn cuda_delta_rule_token_major(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    g: &Tensor,
    beta: &Tensor,
) -> Result<Tensor> {
    if matches!(
        option_env!("SYS1_CUDA_COMPUTE_CAP"),
        Some("90" | "80" | "86" | "89")
    ) && q.dim(1)? >= 512
    {
        return cuda_delta_rule_precomputed_h200_impl(q, k, v, g, beta, true);
    }
    cuda_delta_rule_token_major_impl(q, k, v, g, beta, true)
}

pub(crate) fn cuda_delta_rule_token_major_impl(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    g: &Tensor,
    beta: &Tensor,
    subwarp16: bool,
) -> Result<Tensor> {
    let (batch, length, key_heads, key_dim) = q.dims4()?;
    let (value_batch, value_length, heads, value_dim) = v.dims4()?;
    if key_dim != 128
        || key_heads == 0
        || heads == 0
        || heads % key_heads != 0
        || value_dim == 0
        || k.dims() != q.dims()
        || value_batch != batch
        || value_length != length
        || g.dims() != [batch, length, heads]
        || beta.dims() != [batch, length, heads]
    {
        candle_core::bail!("Unsupported Qwen3.5 CUDA delta-rule shape")
    }
    let q = q.contiguous()?;
    let k = k.contiguous()?;
    let v = v.contiguous()?;
    let g = g.contiguous()?;
    let beta = beta.contiguous()?;
    let (q_storage, q_layout) = q.storage_and_layout();
    let (k_storage, k_layout) = k.storage_and_layout();
    let (v_storage, v_layout) = v.storage_and_layout();
    let (g_storage, g_layout) = g.storage_and_layout();
    let (beta_storage, beta_layout) = beta.storage_and_layout();
    let (
        Storage::Cuda(q_storage),
        Storage::Cuda(k_storage),
        Storage::Cuda(v_storage),
        Storage::Cuda(g_storage),
        Storage::Cuda(beta_storage),
    ) = (
        &*q_storage,
        &*k_storage,
        &*v_storage,
        &*g_storage,
        &*beta_storage,
    )
    else {
        candle_core::bail!("Qwen3.5 delta rule requires CUDA storage")
    };
    let offsets = |layout: &candle_core::Layout| {
        layout
            .contiguous_offsets()
            .ok_or_else(|| candle_core::Error::Msg("Qwen3.5 delta input is not contiguous".into()))
    };
    let (q_start, q_end) = offsets(q_layout)?;
    let (k_start, k_end) = offsets(k_layout)?;
    let (v_start, v_end) = offsets(v_layout)?;
    let (g_start, g_end) = offsets(g_layout)?;
    let (beta_start, beta_end) = offsets(beta_layout)?;
    let query = q_storage.as_cuda_slice::<f32>()?.slice(q_start..q_end);
    let key = k_storage.as_cuda_slice::<f32>()?.slice(k_start..k_end);
    let value = v_storage.as_cuda_slice::<f32>()?.slice(v_start..v_end);
    let gate = g_storage.as_cuda_slice::<f32>()?.slice(g_start..g_end);
    let beta = beta_storage
        .as_cuda_slice::<f32>()?
        .slice(beta_start..beta_end);
    let device = q_storage.device().clone();
    let output = unsafe { device.alloc::<f32>(batch * heads * length * value_dim)? };
    let name = if subwarp16 {
        "qwen35_delta_subwarp16_f32"
    } else {
        "qwen35_delta_f32"
    };
    let function = device.get_or_load_custom_func(name, "sys1_qwen35", CUDA_PTX)?;
    let config = LaunchConfig {
        grid_dim: (
            (batch * heads) as u32,
            value_dim.div_ceil(if subwarp16 { 8 } else { 4 }) as u32,
            1,
        ),
        block_dim: (128, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut launch = function.builder();
    launch.arg(&query);
    launch.arg(&key);
    launch.arg(&value);
    launch.arg(&gate);
    launch.arg(&beta);
    launch.arg(&output);
    let length = length as i32;
    let heads_arg = heads as i32;
    let key_heads_arg = key_heads as i32;
    let value_dim = value_dim as i32;
    launch.arg(&length);
    launch.arg(&heads_arg);
    launch.arg(&key_heads_arg);
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

pub(crate) fn cuda_delta_rule_precomputed_h200_impl(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    g: &Tensor,
    beta: &Tensor,
    subwarp16: bool,
) -> Result<Tensor> {
    let (batch, length, key_heads, key_dim) = q.dims4()?;
    let (value_batch, value_length, heads, value_dim) = v.dims4()?;
    if key_dim != 128
        || key_heads == 0
        || heads == 0
        || heads % key_heads != 0
        || value_dim == 0
        || k.dims() != q.dims()
        || value_batch != batch
        || value_length != length
        || g.dims() != [batch, length, heads]
        || beta.dims() != [batch, length, heads]
    {
        candle_core::bail!("Unsupported Qwen3.5 CUDA delta-rule shape")
    }
    let q = q.contiguous()?;
    let k = k.contiguous()?;
    let v = v.contiguous()?;
    let g = g.contiguous()?;
    let beta = beta.contiguous()?;
    let (q_storage, q_layout) = q.storage_and_layout();
    let (k_storage, k_layout) = k.storage_and_layout();
    let (v_storage, v_layout) = v.storage_and_layout();
    let (g_storage, g_layout) = g.storage_and_layout();
    let (beta_storage, beta_layout) = beta.storage_and_layout();
    let (
        Storage::Cuda(q_storage),
        Storage::Cuda(k_storage),
        Storage::Cuda(v_storage),
        Storage::Cuda(g_storage),
        Storage::Cuda(beta_storage),
    ) = (
        &*q_storage,
        &*k_storage,
        &*v_storage,
        &*g_storage,
        &*beta_storage,
    )
    else {
        candle_core::bail!("Qwen3.5 delta rule requires CUDA storage")
    };
    let offsets = |layout: &candle_core::Layout| {
        layout
            .contiguous_offsets()
            .ok_or_else(|| candle_core::Error::Msg("Qwen3.5 delta input is not contiguous".into()))
    };
    let (q_start, q_end) = offsets(q_layout)?;
    let (k_start, k_end) = offsets(k_layout)?;
    let (v_start, v_end) = offsets(v_layout)?;
    let (g_start, g_end) = offsets(g_layout)?;
    let (beta_start, beta_end) = offsets(beta_layout)?;
    let query = q_storage.as_cuda_slice::<f32>()?.slice(q_start..q_end);
    let key = k_storage.as_cuda_slice::<f32>()?.slice(k_start..k_end);
    let value = v_storage.as_cuda_slice::<f32>()?.slice(v_start..v_end);
    let gate = g_storage.as_cuda_slice::<f32>()?.slice(g_start..g_end);
    let beta = beta_storage
        .as_cuda_slice::<f32>()?
        .slice(beta_start..beta_end);
    let device = q_storage.device().clone();
    let decay = unsafe { device.alloc::<f32>(batch * length * heads)? };
    let exp = device.get_or_load_custom_func("qwen35_preexp_gate_f32", "sys1_qwen35", CUDA_PTX)?;
    let count = (batch * length * heads) as u32;
    let mut exp_launch = exp.builder();
    exp_launch.arg(&gate);
    exp_launch.arg(&decay);
    exp_launch.arg(&count);
    unsafe { exp_launch.launch(LaunchConfig::for_num_elems(count)) }.w()?;
    let output = unsafe { device.alloc::<f32>(batch * heads * length * value_dim)? };
    let ampere_warp32 = option_env!("SYS1_CUDA_COMPUTE_CAP") == Some("80");

    let hopper_uniform_prefetch8 = option_env!("SYS1_CUDA_COMPUTE_CAP") == Some("90")
        && batch == 1
        && length >= 512
        && value_dim == 128
        && subwarp16;

    let ampere_uniform_prefetch8 =
        matches!(option_env!("SYS1_CUDA_COMPUTE_CAP"), Some("80" | "86"))
            && batch == 1
            && length >= 512
            && heads == 32
            && key_heads == 16
            && value_dim == 128
            && subwarp16;
    let uniform_prefetch8 = hopper_uniform_prefetch8 || ampere_uniform_prefetch8;
    let operand_reuse =
        option_env!("SYS1_CUDA_COMPUTE_CAP") == Some("89") && subwarp16 && value_dim == 128;
    let name = if operand_reuse {
        "qwen35_uniform_delta_reuse_f32"
    } else if uniform_prefetch8 {
        "qwen35_uniform_delta_prefetch8_preexp_f32"
    } else if ampere_warp32 {
        "qwen35_delta_warp32_preexp_f32"
    } else {
        "qwen35_delta_subwarp16_preexp_f32"
    };
    let function = device.get_or_load_custom_func(name, "sys1_qwen35", CUDA_PTX)?;
    let config = LaunchConfig {
        grid_dim: (
            (batch * heads) as u32,
            value_dim.div_ceil(if operand_reuse {
                16
            } else if uniform_prefetch8 {
                8
            } else if ampere_warp32 {
                4
            } else if subwarp16 {
                8
            } else {
                4
            }) as u32,
            1,
        ),
        block_dim: (128, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut launch = function.builder();
    launch.arg(&query);
    launch.arg(&key);
    launch.arg(&value);
    launch.arg(&decay);
    launch.arg(&beta);
    launch.arg(&output);
    let length = length as i32;
    let heads_arg = heads as i32;
    let key_heads_arg = key_heads as i32;
    let value_dim = value_dim as i32;
    launch.arg(&length);
    launch.arg(&heads_arg);
    launch.arg(&key_heads_arg);
    launch.arg(&value_dim);
    unsafe { launch.launch(config) }.w()?;
    let output = Tensor::from_storage(
        Storage::Cuda(CudaStorage::wrap_cuda_slice(output, device)),
        if operand_reuse {
            (batch, length as usize, heads, value_dim as usize)
        } else {
            (batch, heads, length as usize, value_dim as usize)
        },
        BackpropOp::none(),
        false,
    );
    if operand_reuse {
        Ok(output)
    } else {
        output.transpose(1, 2)
    }
}

pub(crate) fn cuda_delta_rule_packed_sequences(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    g: &Tensor,
    beta: &Tensor,
    sequences: &PackedSequences,
    subwarp16: bool,
) -> Result<Tensor> {
    let (batch, length, key_heads, key_dim) = q.dims4()?;
    if !subwarp16 || batch != 1 || length != sequences.total {
        candle_core::bail!("packed delta unsupported layout")
    }
    let (value_batch, value_length, heads, value_dim) = v.dims4()?;
    if key_dim != 128
        || key_heads == 0
        || heads == 0
        || heads % key_heads != 0
        || value_dim == 0
        || k.dims() != q.dims()
        || value_batch != batch
        || value_length != length
        || g.dims() != [batch, length, heads]
        || beta.dims() != [batch, length, heads]
    {
        candle_core::bail!("Unsupported Qwen3.5 CUDA delta-rule shape")
    }
    let q = q.contiguous()?;
    let k = k.contiguous()?;
    let v = v.contiguous()?;
    let g = g.contiguous()?;
    let beta = beta.contiguous()?;
    let (q_storage, q_layout) = q.storage_and_layout();
    let (k_storage, k_layout) = k.storage_and_layout();
    let (v_storage, v_layout) = v.storage_and_layout();
    let (g_storage, g_layout) = g.storage_and_layout();
    let (beta_storage, beta_layout) = beta.storage_and_layout();
    let (
        Storage::Cuda(q_storage),
        Storage::Cuda(k_storage),
        Storage::Cuda(v_storage),
        Storage::Cuda(g_storage),
        Storage::Cuda(beta_storage),
    ) = (
        &*q_storage,
        &*k_storage,
        &*v_storage,
        &*g_storage,
        &*beta_storage,
    )
    else {
        candle_core::bail!("Qwen3.5 delta rule requires CUDA storage")
    };
    let offsets = |layout: &candle_core::Layout| {
        layout
            .contiguous_offsets()
            .ok_or_else(|| candle_core::Error::Msg("Qwen3.5 delta input is not contiguous".into()))
    };
    let (q_start, q_end) = offsets(q_layout)?;
    let (k_start, k_end) = offsets(k_layout)?;
    let (v_start, v_end) = offsets(v_layout)?;
    let (g_start, g_end) = offsets(g_layout)?;
    let (beta_start, beta_end) = offsets(beta_layout)?;
    let query = q_storage.as_cuda_slice::<f32>()?.slice(q_start..q_end);
    let key = k_storage.as_cuda_slice::<f32>()?.slice(k_start..k_end);
    let value = v_storage.as_cuda_slice::<f32>()?.slice(v_start..v_end);
    let gate = g_storage.as_cuda_slice::<f32>()?.slice(g_start..g_end);
    let beta = beta_storage
        .as_cuda_slice::<f32>()?
        .slice(beta_start..beta_end);
    let (metadata_storage, metadata_layout) = sequences.cumulative.storage_and_layout();
    let Storage::Cuda(metadata_storage) = &*metadata_storage else {
        candle_core::bail!("packed metadata must be CUDA")
    };
    let (start, end) = offsets(metadata_layout)?;
    let cumulative = metadata_storage.as_cuda_slice::<u32>()?.slice(start..end);
    let device = q_storage.device().clone();
    let ampere_prefetch8 = matches!(option_env!("SYS1_CUDA_COMPUTE_CAP"), Some("80" | "86"))
        && length >= 512
        && heads == 32
        && key_heads == 16
        && value_dim == 128
        && q_start.is_multiple_of(4)
        && k_start.is_multiple_of(4)
        && v_start.is_multiple_of(4);

    let ampere_warp32 = (matches!(option_env!("SYS1_CUDA_COMPUTE_CAP"), Some("80" | "90"))
        && length >= 512)
        || ampere_prefetch8;
    let hopper_columns2 = option_env!("SYS1_CUDA_COMPUTE_CAP") == Some("90") && length >= 1024;
    let hopper_prefetch8 = hopper_columns2 && value_dim == 128;

    let hopper_prefetch4 =
        hopper_prefetch8 && heads >= 32 && sequences.lengths.len().saturating_mul(heads) >= 64;
    let precomputed_decay = if ampere_warp32 {
        let values = unsafe { device.alloc::<f32>(length * heads)? };
        let function =
            device.get_or_load_custom_func("qwen35_preexp_gate_f32", "sys1_qwen35", CUDA_PTX)?;
        let count = (length * heads) as u32;
        let mut launch = function.builder();
        launch.arg(&gate);
        launch.arg(&values);
        launch.arg(&count);
        unsafe { launch.launch(LaunchConfig::for_num_elems(count)) }.w()?;
        Some(values)
    } else {
        None
    };
    let output = unsafe { device.alloc::<f32>(batch * heads * length * value_dim)? };
    let operand_reuse = option_env!("SYS1_CUDA_COMPUTE_CAP") == Some("89") && value_dim == 128;
    let name = if operand_reuse {
        "qwen35_packed_delta_reuse_f32"
    } else if ampere_prefetch8 {
        "qwen35_packed_delta_prefetch8_preexp_f32"
    } else if hopper_prefetch4 {
        "qwen35_packed_delta_prefetch8_columns4_preexp_f32"
    } else if hopper_prefetch8 {
        "qwen35_packed_delta_prefetch8_preexp_f32"
    } else if hopper_columns2 {
        "qwen35_packed_delta_warp32_cols2_preexp_f32"
    } else if ampere_warp32 {
        "qwen35_packed_delta_warp32_preexp_f32"
    } else {
        "qwen35_packed_delta_subwarp16_f32"
    };
    let function = device.get_or_load_custom_func(name, "sys1_qwen35", CUDA_PTX)?;
    let config = LaunchConfig {
        grid_dim: (
            (sequences.lengths.len() * heads) as u32,
            value_dim.div_ceil(if operand_reuse {
                16
            } else if ampere_prefetch8 {
                8
            } else if hopper_prefetch4 {
                16
            } else if hopper_columns2 {
                8
            } else if ampere_warp32 {
                4
            } else if subwarp16 {
                8
            } else {
                4
            }) as u32,
            1,
        ),
        block_dim: (128, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut launch = function.builder();
    launch.arg(&query);
    launch.arg(&key);
    launch.arg(&value);
    if let Some(decay) = &precomputed_decay {
        launch.arg(decay);
    } else {
        launch.arg(&gate);
    }
    launch.arg(&beta);
    launch.arg(&output);
    launch.arg(&cumulative);
    let length = length as i32;
    let heads_arg = heads as i32;
    let key_heads_arg = key_heads as i32;
    let value_dim = value_dim as i32;
    launch.arg(&length);
    launch.arg(&heads_arg);
    launch.arg(&key_heads_arg);
    launch.arg(&value_dim);
    unsafe { launch.launch(config) }.w()?;
    let output = Tensor::from_storage(
        Storage::Cuda(CudaStorage::wrap_cuda_slice(output, device)),
        (batch, length as usize, heads, value_dim as usize),
        BackpropOp::none(),
        false,
    );
    Ok(output)
}

pub(crate) struct ChunkScratch {
    pub(crate) stream: std::sync::Arc<candle_core::cuda_backend::cudarc::driver::CudaStream>,
    pub(crate) w: Option<candle_core::cuda_backend::cudarc::driver::CudaSlice<f32>>,
    pub(crate) u: Option<candle_core::cuda_backend::cudarc::driver::CudaSlice<f32>>,
    pub(crate) c: Option<candle_core::cuda_backend::cudarc::driver::CudaSlice<f32>>,
    pub(crate) decays: Option<candle_core::cuda_backend::cudarc::driver::CudaSlice<f32>>,
    pub(crate) snapshots: Option<candle_core::cuda_backend::cudarc::driver::CudaSlice<f32>>,
}

pub(crate) fn grow_chunk_buffer<'a>(
    buffer: &'a mut Option<candle_core::cuda_backend::cudarc::driver::CudaSlice<f32>>,
    device: &candle_core::CudaDevice,
    count: usize,
) -> Result<&'a mut candle_core::cuda_backend::cudarc::driver::CudaSlice<f32>> {
    if buffer.as_ref().is_none_or(|x| x.len() < count) {
        *buffer = Some(unsafe { device.alloc::<f32>(count)? });
    }
    buffer
        .as_mut()
        .ok_or_else(|| candle_core::Error::Msg("missing chunk buffer".into()))
}

pub(crate) static CHUNK64_SCRATCH: std::sync::Mutex<Option<ChunkScratch>> =
    std::sync::Mutex::new(None);

pub(crate) fn cuda_chunk64_direct(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    g: &Tensor,
    beta: &Tensor,
    tile_columns: usize,
    profiling: bool,
) -> Result<Tensor> {
    if [q, k, v, g, beta].iter().any(|x| x.dtype() != DType::F32) {
        candle_core::bail!("chunked recurrence requires FP32 inputs");
    }
    let state_rows = 128usize;
    let carry_shared = 58400u32;
    if tile_columns != 32 {
        candle_core::bail!("C64 state requires N32");
    }
    let (batch, length, key_heads, key_dim) = q.dims4()?;
    let (value_batch, value_length, heads, value_dim) = v.dims4()?;
    if key_dim != 128
        || key_heads == 0
        || heads == 0
        || heads % key_heads != 0
        || value_dim == 0
        || k.dims() != q.dims()
        || value_batch != batch
        || value_length != length
        || g.dims() != [batch, length, heads]
        || beta.dims() != [batch, length, heads]
    {
        candle_core::bail!("Unsupported Qwen3.5 CUDA delta-rule shape")
    }
    let q = q.contiguous()?;
    let k = k.contiguous()?;
    let v = v.contiguous()?;
    let g = g.contiguous()?;
    let beta = beta.contiguous()?;
    let (q_storage, q_layout) = q.storage_and_layout();
    let (k_storage, k_layout) = k.storage_and_layout();
    let (v_storage, v_layout) = v.storage_and_layout();
    let (g_storage, g_layout) = g.storage_and_layout();
    let (beta_storage, beta_layout) = beta.storage_and_layout();
    let (
        Storage::Cuda(q_storage),
        Storage::Cuda(k_storage),
        Storage::Cuda(v_storage),
        Storage::Cuda(g_storage),
        Storage::Cuda(beta_storage),
    ) = (
        &*q_storage,
        &*k_storage,
        &*v_storage,
        &*g_storage,
        &*beta_storage,
    )
    else {
        candle_core::bail!("Qwen3.5 delta rule requires CUDA storage")
    };
    let offsets = |layout: &candle_core::Layout| {
        layout
            .contiguous_offsets()
            .ok_or_else(|| candle_core::Error::Msg("Qwen3.5 delta input is not contiguous".into()))
    };
    let (q_start, q_end) = offsets(q_layout)?;
    let (k_start, k_end) = offsets(k_layout)?;
    let (v_start, v_end) = offsets(v_layout)?;
    let (g_start, g_end) = offsets(g_layout)?;
    let (beta_start, beta_end) = offsets(beta_layout)?;
    let query = q_storage.as_cuda_slice::<f32>()?.slice(q_start..q_end);
    let key = k_storage.as_cuda_slice::<f32>()?.slice(k_start..k_end);
    let value = v_storage.as_cuda_slice::<f32>()?.slice(v_start..v_end);
    let gate = g_storage.as_cuda_slice::<f32>()?.slice(g_start..g_end);
    let beta = beta_storage
        .as_cuda_slice::<f32>()?
        .slice(beta_start..beta_end);
    let device = q_storage.device().clone();

    if tile_columns != 32
        || batch == 0
        || length == 0
        || value_dim > 128
        || [
            q.elem_count(),
            k.elem_count(),
            v.elem_count(),
            g.elem_count(),
        ]
        .iter()
        .any(|n| *n > i32::MAX as usize)
    {
        candle_core::bail!("chunk probe requires nonempty F32 K128,V<=128,small indices");
    }
    let instances = batch
        .checked_mul(heads)
        .and_then(|x| x.checked_mul(length.div_ceil(64)))
        .ok_or_else(|| candle_core::Error::Msg("chunk count overflow".into()))?;
    let w_count = instances
        .checked_mul(64 * 128)
        .ok_or_else(|| candle_core::Error::Msg("chunk W overflow".into()))?;
    let u_count = instances
        .checked_mul(64 * value_dim)
        .ok_or_else(|| candle_core::Error::Msg("chunk U overflow".into()))?;
    let c_count = instances
        .checked_mul(64 * 64)
        .ok_or_else(|| candle_core::Error::Msg("chunk C overflow".into()))?;
    let d_count = instances
        .checked_mul(128)
        .ok_or_else(|| candle_core::Error::Msg("decay count overflow".into()))?;
    if [w_count, u_count, c_count, d_count]
        .iter()
        .any(|n| *n > i32::MAX as usize)
    {
        candle_core::bail!("chunk scratch index overflow");
    }
    let result = (|| -> Result<Tensor> {
        let mut slot = CHUNK64_SCRATCH
            .lock()
            .map_err(|_| candle_core::Error::Msg("poisoned chunk scratch".into()))?;
        let stream = device.cuda_stream();
        if slot
            .as_ref()
            .is_none_or(|old| !std::sync::Arc::ptr_eq(&old.stream, &stream))
        {
            *slot = Some(ChunkScratch {
                stream,
                w: None,
                u: None,
                c: None,
                decays: None,
                snapshots: None,
            });
        }
        let workspace = slot
            .as_mut()
            .ok_or_else(|| candle_core::Error::Msg("missing chunk scratch".into()))?;
        let ChunkScratch {
            w,
            u,
            c,
            decays,
            snapshots,
            ..
        } = workspace;
        if profiling {
            device.cuda_stream().synchronize().w()?;
        }
        let mut stage_started = Instant::now();
        let w = grow_chunk_buffer(w, &device, w_count)?;
        let u = grow_chunk_buffer(u, &device, u_count)?;
        let c = grow_chunk_buffer(c, &device, c_count)?;
        let decays = grow_chunk_buffer(decays, &device, d_count)?;
        let output_count = batch
            .checked_mul(length)
            .and_then(|n| n.checked_mul(heads))
            .and_then(|n| n.checked_mul(value_dim))
            .ok_or_else(|| candle_core::Error::Msg("register-carry C64 output overflow".into()))?;
        if output_count > i32::MAX as usize {
            candle_core::bail!("padded output index overflow");
        }
        let snapshot_count = 1usize;
        if snapshot_count > i32::MAX as usize {
            candle_core::bail!("snapshot index overflow");
        }
        let snapshots = grow_chunk_buffer(snapshots, &device, snapshot_count)?;
        let mut output = unsafe { device.alloc::<f32>(output_count)? };
        if profiling {
            println!(
                "C64_FLAGS_WORKSPACE retained_bytes={} output_bytes={} single_entry_per_process=true owned_stream_identity=true output_never_reused=true amortized_warm_operator_not_serving=true",
                (w.len() + u.len() + c.len() + decays.len() + snapshots.len()) * 4,
                output_count * 4
            );
        }
        if profiling {
            device.cuda_stream().synchronize().w()?;
            println!(
                "C64_FLAGS_STAGE part=scratch batch={batch} length={length} columns={tile_columns} ms={} diagnostic_not_serving=true",
                stage_started.elapsed().as_secs_f64() * 1000.
            );
            stage_started = Instant::now();
        }
        let length_arg = length as i32;
        let heads_arg = heads as i32;
        let key_heads_arg = key_heads as i32;
        let value_dim_arg = value_dim as i32;
        {
            let function = device.get_or_load_custom_func(
                "qwen35_c64mma_prepare_f32",
                "sys1_chunk_delta",
                CUDA_PTX,
            )?;
            function.set_attribute(candle_core::cuda_backend::cudarc::driver::sys::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,86528).w()?;
            let mut launch = function.builder();
            launch.arg(&query);
            launch.arg(&key);
            launch.arg(&value);
            launch.arg(&gate);
            launch.arg(&beta);
            launch.arg(&mut *w);
            launch.arg(&mut *u);
            launch.arg(&mut *c);
            launch.arg(&mut *decays);
            launch.arg(&length_arg);
            launch.arg(&heads_arg);
            launch.arg(&key_heads_arg);
            launch.arg(&value_dim_arg);
            unsafe {
                launch.launch(LaunchConfig {
                    grid_dim: (instances as u32, 1, 1),
                    block_dim: (1024, 1, 1),
                    shared_mem_bytes: 86528,
                })
            }
            .w()?;
        }
        if profiling {
            device.cuda_stream().synchronize().w()?;
            println!(
                "C64_FLAGS_STAGE part=prepare batch={batch} length={length} columns={tile_columns} ms={} diagnostic_not_serving=true",
                stage_started.elapsed().as_secs_f64() * 1000.
            );
            stage_started = Instant::now();
        }
        {
            let name = "qwen35_c64flag_uniform32_f32";
            let function = device.get_or_load_custom_func(name, "sys1_chunk_delta", CUDA_PTX)?;
            if profiling {
                println!(
                    "C64_FLAGS_RESOURCE kernel={name} registers={} local_bytes={} shared_bytes={} diagnostic_not_serving=true",
                    function.num_regs().w()?,
                    function.local_size_bytes().w()?,
                    function.shared_size_bytes().w()?
                );
            }
            function.set_attribute(candle_core::cuda_backend::cudarc::driver::sys::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,carry_shared as i32).w()?;
            let mut launch = function.builder();
            launch.arg(&query);
            launch.arg(&key);
            launch.arg(&gate);
            launch.arg(&*w);
            launch.arg(&*u);
            launch.arg(&*c);
            launch.arg(&*decays);
            launch.arg(&mut output);
            launch.arg(&mut *snapshots);
            launch.arg(&length_arg);
            launch.arg(&heads_arg);
            launch.arg(&key_heads_arg);
            launch.arg(&value_dim_arg);
            unsafe {
                launch.launch(LaunchConfig {
                    grid_dim: (
                        (batch * heads) as u32,
                        value_dim.div_ceil(tile_columns) as u32,
                        1,
                    ),
                    block_dim: (((state_rows / 16) * (tile_columns / 8) * 32) as u32, 1, 1),
                    shared_mem_bytes: carry_shared,
                })
            }
            .w()?;
        }
        if profiling {
            device.cuda_stream().synchronize().w()?;
            println!(
                "C64_FLAGS_STAGE part=state batch={batch} length={length} columns={tile_columns} ms={} diagnostic_not_serving=true",
                stage_started.elapsed().as_secs_f64() * 1000.
            );
        }

        let output = Tensor::from_storage(
            Storage::Cuda(CudaStorage::wrap_cuda_slice(output, device.clone())),
            (batch, length, heads, value_dim),
            BackpropOp::none(),
            false,
        );
        Ok(output)
    })();
    if result.is_err()
        && let Ok(mut slot) = CHUNK64_SCRATCH.lock()
    {
        slot.take();
    }
    result
}

pub(crate) fn cuda_c64_varlen(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    g: &Tensor,
    beta: &Tensor,
    sequences: &PackedSequences,
    profiling: bool,
) -> Result<Tensor> {
    if [q, k, v, g, beta].iter().any(|x| x.dtype() != DType::F32) {
        candle_core::bail!("chunked recurrence requires FP32 inputs");
    }
    let state_rows = 128usize;
    let tile_columns = 32usize;
    let carry_shared = 58400u32;
    if q.dim(0)? != 1 || q.dim(1)? != sequences.total {
        candle_core::bail!("segmented C64 shape mismatch")
    }
    if sequences.c64_chunk_count == 0 {
        return cuda_delta_rule_packed_sequences(q, k, v, g, beta, sequences, true);
    }
    if [
        k,
        v,
        g,
        beta,
        &sequences.cumulative,
        &sequences.c64_chunks,
        &sequences.c64_long_ids,
        &sequences.c64_short_ids,
    ]
    .iter()
    .any(|t| !q.device().same_device(t.device()))
    {
        candle_core::bail!("segmented C64 device mismatch")
    }
    let (batch, length, key_heads, key_dim) = q.dims4()?;
    let (value_batch, value_length, heads, value_dim) = v.dims4()?;
    if key_dim != 128
        || key_heads == 0
        || heads == 0
        || heads % key_heads != 0
        || value_dim == 0
        || k.dims() != q.dims()
        || value_batch != batch
        || value_length != length
        || g.dims() != [batch, length, heads]
        || beta.dims() != [batch, length, heads]
    {
        candle_core::bail!("Unsupported Qwen3.5 CUDA delta-rule shape")
    }
    let q = q.contiguous()?;
    let k = k.contiguous()?;
    let v = v.contiguous()?;
    let g = g.contiguous()?;
    let beta = beta.contiguous()?;
    let (q_storage, q_layout) = q.storage_and_layout();
    let (k_storage, k_layout) = k.storage_and_layout();
    let (v_storage, v_layout) = v.storage_and_layout();
    let (g_storage, g_layout) = g.storage_and_layout();
    let (beta_storage, beta_layout) = beta.storage_and_layout();
    let (
        Storage::Cuda(q_storage),
        Storage::Cuda(k_storage),
        Storage::Cuda(v_storage),
        Storage::Cuda(g_storage),
        Storage::Cuda(beta_storage),
    ) = (
        &*q_storage,
        &*k_storage,
        &*v_storage,
        &*g_storage,
        &*beta_storage,
    )
    else {
        candle_core::bail!("Qwen3.5 delta rule requires CUDA storage")
    };
    let offsets = |layout: &candle_core::Layout| {
        layout
            .contiguous_offsets()
            .ok_or_else(|| candle_core::Error::Msg("Qwen3.5 delta input is not contiguous".into()))
    };
    let (q_start, q_end) = offsets(q_layout)?;
    let (k_start, k_end) = offsets(k_layout)?;
    let (v_start, v_end) = offsets(v_layout)?;
    let (g_start, g_end) = offsets(g_layout)?;
    let (beta_start, beta_end) = offsets(beta_layout)?;
    let query = q_storage.as_cuda_slice::<f32>()?.slice(q_start..q_end);
    let key = k_storage.as_cuda_slice::<f32>()?.slice(k_start..k_end);
    let value = v_storage.as_cuda_slice::<f32>()?.slice(v_start..v_end);
    let gate = g_storage.as_cuda_slice::<f32>()?.slice(g_start..g_end);
    let beta = beta_storage
        .as_cuda_slice::<f32>()?
        .slice(beta_start..beta_end);
    let (cumulative_storage, cumulative_layout) = sequences.cumulative.storage_and_layout();
    let Storage::Cuda(cumulative_storage) = &*cumulative_storage else {
        candle_core::bail!("CUDA metadata required")
    };
    let (start, end) = offsets(cumulative_layout)?;
    let cumulative = cumulative_storage.as_cuda_slice::<u32>()?.slice(start..end);
    let (chunk_offsets_storage, chunk_offsets_layout) = sequences.c64_chunks.storage_and_layout();
    let Storage::Cuda(chunk_offsets_storage) = &*chunk_offsets_storage else {
        candle_core::bail!("CUDA metadata required")
    };
    let (start, end) = offsets(chunk_offsets_layout)?;
    let chunk_offsets = chunk_offsets_storage
        .as_cuda_slice::<u32>()?
        .slice(start..end);
    let (long_ids_storage, long_ids_layout) = sequences.c64_long_ids.storage_and_layout();
    let Storage::Cuda(long_ids_storage) = &*long_ids_storage else {
        candle_core::bail!("CUDA metadata required")
    };
    let (start, end) = offsets(long_ids_layout)?;
    let long_ids = long_ids_storage.as_cuda_slice::<u32>()?.slice(start..end);
    let (short_ids_storage, short_ids_layout) = sequences.c64_short_ids.storage_and_layout();
    let Storage::Cuda(short_ids_storage) = &*short_ids_storage else {
        candle_core::bail!("CUDA metadata required")
    };
    let (start, end) = offsets(short_ids_layout)?;
    let short_ids = short_ids_storage.as_cuda_slice::<u32>()?.slice(start..end);
    let device = q_storage.device().clone();

    if tile_columns != 32
        || batch == 0
        || length == 0
        || value_dim > 128
        || [
            q.elem_count(),
            k.elem_count(),
            v.elem_count(),
            g.elem_count(),
        ]
        .iter()
        .any(|n| *n > i32::MAX as usize)
    {
        candle_core::bail!("chunk probe requires nonempty F32 K128,V<=128,small indices");
    }
    let instances = heads
        .checked_mul(sequences.c64_chunk_count)
        .ok_or_else(|| candle_core::Error::Msg("chunk count overflow".into()))?;
    let w_count = instances
        .checked_mul(64 * 128)
        .ok_or_else(|| candle_core::Error::Msg("chunk W overflow".into()))?;
    let u_count = instances
        .checked_mul(64 * value_dim)
        .ok_or_else(|| candle_core::Error::Msg("chunk U overflow".into()))?;
    let c_count = instances
        .checked_mul(64 * 64)
        .ok_or_else(|| candle_core::Error::Msg("chunk C overflow".into()))?;
    let d_count = instances
        .checked_mul(128)
        .ok_or_else(|| candle_core::Error::Msg("decay count overflow".into()))?;
    if [w_count, u_count, c_count, d_count]
        .iter()
        .any(|n| *n > i32::MAX as usize)
    {
        candle_core::bail!("chunk scratch index overflow");
    }
    let result = (|| -> Result<Tensor> {
        let mut slot = CHUNK64_SCRATCH
            .lock()
            .map_err(|_| candle_core::Error::Msg("poisoned chunk scratch".into()))?;
        let stream = device.cuda_stream();
        if slot
            .as_ref()
            .is_none_or(|old| !std::sync::Arc::ptr_eq(&old.stream, &stream))
        {
            *slot = Some(ChunkScratch {
                stream,
                w: None,
                u: None,
                c: None,
                decays: None,
                snapshots: None,
            });
        }
        let workspace = slot
            .as_mut()
            .ok_or_else(|| candle_core::Error::Msg("missing chunk scratch".into()))?;
        let ChunkScratch {
            w,
            u,
            c,
            decays,
            snapshots,
            ..
        } = workspace;
        if profiling {
            device.cuda_stream().synchronize().w()?;
        }
        let mut stage_started = Instant::now();
        let w = grow_chunk_buffer(w, &device, w_count)?;
        let u = grow_chunk_buffer(u, &device, u_count)?;
        let c = grow_chunk_buffer(c, &device, c_count)?;
        let decays = grow_chunk_buffer(decays, &device, d_count)?;
        let output_count = length
            .checked_mul(heads)
            .and_then(|n| n.checked_mul(value_dim))
            .ok_or_else(|| candle_core::Error::Msg("segmented C64 output overflow".into()))?;
        if output_count > i32::MAX as usize {
            candle_core::bail!("padded output index overflow");
        }
        let snapshot_count = 1usize;
        if snapshot_count > i32::MAX as usize {
            candle_core::bail!("snapshot index overflow");
        }
        let snapshots = grow_chunk_buffer(snapshots, &device, snapshot_count)?;
        let mut output = unsafe { device.alloc::<f32>(output_count)? };
        if profiling {
            println!(
                "C64_CARRY_WORKSPACE retained_bytes={} output_bytes={} single_entry_per_process=true owned_stream_identity=true output_never_reused=true amortized_warm_operator_not_serving=true",
                (w.len() + u.len() + c.len() + decays.len() + snapshots.len()) * 4,
                output_count * 4
            );
        }
        if profiling {
            device.cuda_stream().synchronize().w()?;
            println!(
                "C64_CARRY_STAGE part=scratch batch={batch} length={length} columns={tile_columns} ms={} diagnostic_not_serving=true",
                stage_started.elapsed().as_secs_f64() * 1000.
            );
            stage_started = Instant::now();
        }
        let sequence_count = sequences.lengths.len() as i32;
        let length_arg = length as i32;
        let heads_arg = heads as i32;
        let key_heads_arg = key_heads as i32;
        let value_dim_arg = value_dim as i32;
        {
            let function = device.get_or_load_custom_func(
                "qwen35_c64v_prepare_f32",
                "sys1_chunk_delta",
                CUDA_PTX,
            )?;
            function.set_attribute(candle_core::cuda_backend::cudarc::driver::sys::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,86528).w()?;
            let mut launch = function.builder();
            launch.arg(&query);
            launch.arg(&key);
            launch.arg(&value);
            launch.arg(&gate);
            launch.arg(&beta);
            launch.arg(&mut *w);
            launch.arg(&mut *u);
            launch.arg(&mut *c);
            launch.arg(&mut *decays);
            launch.arg(&cumulative);
            launch.arg(&chunk_offsets);
            launch.arg(&sequence_count);
            launch.arg(&length_arg);
            launch.arg(&heads_arg);
            launch.arg(&key_heads_arg);
            launch.arg(&value_dim_arg);
            unsafe {
                launch.launch(LaunchConfig {
                    grid_dim: (instances as u32, 1, 1),
                    block_dim: (1024, 1, 1),
                    shared_mem_bytes: 86528,
                })
            }
            .w()?;
        }
        if profiling {
            device.cuda_stream().synchronize().w()?;
            println!(
                "C64_CARRY_STAGE part=prepare batch={batch} length={length} columns={tile_columns} ms={} diagnostic_not_serving=true",
                stage_started.elapsed().as_secs_f64() * 1000.
            );
            stage_started = Instant::now();
        }
        {
            if sequences.c64_short_count > 0 {
                let function = device.get_or_load_custom_func(
                    "qwen35_c64v_short_f32",
                    "sys1_c64v_short",
                    CUDA_PTX,
                )?;
                let mut launch = function.builder();
                launch
                    .arg(&query)
                    .arg(&key)
                    .arg(&value)
                    .arg(&gate)
                    .arg(&beta)
                    .arg(&mut output)
                    .arg(&cumulative)
                    .arg(&short_ids)
                    .arg(&length_arg)
                    .arg(&heads_arg)
                    .arg(&key_heads_arg)
                    .arg(&value_dim_arg);
                unsafe {
                    launch.launch(LaunchConfig {
                        grid_dim: (
                            (sequences.c64_short_count * heads) as u32,
                            value_dim.div_ceil(8) as u32,
                            1,
                        ),
                        block_dim: (128, 1, 1),
                        shared_mem_bytes: 0,
                    })
                }
                .w()?;
            }
            let name = "qwen35_c64flag_varlen32_f32";
            let function = device.get_or_load_custom_func(name, "sys1_chunk_delta", CUDA_PTX)?;
            if profiling {
                println!(
                    "C64_CARRY_RESOURCE kernel={name} registers={} local_bytes={} shared_bytes={} diagnostic_not_serving=true",
                    function.num_regs().w()?,
                    function.local_size_bytes().w()?,
                    function.shared_size_bytes().w()?
                );
            }
            function.set_attribute(candle_core::cuda_backend::cudarc::driver::sys::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,carry_shared as i32).w()?;
            let mut launch = function.builder();
            launch.arg(&query);
            launch.arg(&key);
            launch.arg(&gate);
            launch.arg(&*w);
            launch.arg(&*u);
            launch.arg(&*c);
            launch.arg(&*decays);
            launch.arg(&mut output);
            launch.arg(&mut *snapshots);
            launch.arg(&cumulative);
            launch.arg(&chunk_offsets);
            launch.arg(&long_ids);
            launch.arg(&length_arg);
            launch.arg(&heads_arg);
            launch.arg(&key_heads_arg);
            launch.arg(&value_dim_arg);
            unsafe {
                launch.launch(LaunchConfig {
                    grid_dim: (
                        (sequences.c64_long_count * heads) as u32,
                        value_dim.div_ceil(tile_columns) as u32,
                        1,
                    ),
                    block_dim: (((state_rows / 16) * (tile_columns / 8) * 32) as u32, 1, 1),
                    shared_mem_bytes: carry_shared,
                })
            }
            .w()?;
        }
        if profiling {
            device.cuda_stream().synchronize().w()?;
            println!(
                "C64_CARRY_STAGE part=state batch={batch} length={length} columns={tile_columns} ms={} diagnostic_not_serving=true",
                stage_started.elapsed().as_secs_f64() * 1000.
            );
        }

        let output = Tensor::from_storage(
            Storage::Cuda(CudaStorage::wrap_cuda_slice(output, device.clone())),
            (1, length, heads, value_dim),
            BackpropOp::none(),
            false,
        );
        Ok(output)
    })();
    if result.is_err()
        && let Ok(mut slot) = CHUNK64_SCRATCH.lock()
    {
        slot.take();
    }
    result
}

pub(crate) fn cuda_delta_rule_token_major_bfout(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    g: &Tensor,
    beta: &Tensor,
    subwarp16: bool,
) -> Result<Tensor> {
    if !subwarp16 {
        candle_core::bail!("BF16-output scalar requires selected subwarp16");
    }
    let (batch, length, key_heads, key_dim) = q.dims4()?;
    let (value_batch, value_length, heads, value_dim) = v.dims4()?;
    if key_dim != 128
        || key_heads == 0
        || heads == 0
        || heads % key_heads != 0
        || value_dim == 0
        || k.dims() != q.dims()
        || value_batch != batch
        || value_length != length
        || g.dims() != [batch, length, heads]
        || beta.dims() != [batch, length, heads]
    {
        candle_core::bail!("Unsupported Qwen3.5 CUDA delta-rule shape")
    }
    let q = q.contiguous()?;
    let k = k.contiguous()?;
    let v = v.contiguous()?;
    let g = g.contiguous()?;
    let beta = beta.contiguous()?;
    let (q_storage, q_layout) = q.storage_and_layout();
    let (k_storage, k_layout) = k.storage_and_layout();
    let (v_storage, v_layout) = v.storage_and_layout();
    let (g_storage, g_layout) = g.storage_and_layout();
    let (beta_storage, beta_layout) = beta.storage_and_layout();
    let (
        Storage::Cuda(q_storage),
        Storage::Cuda(k_storage),
        Storage::Cuda(v_storage),
        Storage::Cuda(g_storage),
        Storage::Cuda(beta_storage),
    ) = (
        &*q_storage,
        &*k_storage,
        &*v_storage,
        &*g_storage,
        &*beta_storage,
    )
    else {
        candle_core::bail!("Qwen3.5 delta rule requires CUDA storage")
    };
    let offsets = |layout: &candle_core::Layout| {
        layout
            .contiguous_offsets()
            .ok_or_else(|| candle_core::Error::Msg("Qwen3.5 delta input is not contiguous".into()))
    };
    let (q_start, q_end) = offsets(q_layout)?;
    let (k_start, k_end) = offsets(k_layout)?;
    let (v_start, v_end) = offsets(v_layout)?;
    let (g_start, g_end) = offsets(g_layout)?;
    let (beta_start, beta_end) = offsets(beta_layout)?;
    let query = q_storage.as_cuda_slice::<f32>()?.slice(q_start..q_end);
    let key = k_storage.as_cuda_slice::<f32>()?.slice(k_start..k_end);
    let value = v_storage.as_cuda_slice::<f32>()?.slice(v_start..v_end);
    let gate = g_storage.as_cuda_slice::<f32>()?.slice(g_start..g_end);
    let beta = beta_storage
        .as_cuda_slice::<f32>()?
        .slice(beta_start..beta_end);
    let device = q_storage.device().clone();
    let output = unsafe { device.alloc::<bf16>(batch * heads * length * value_dim)? };

    let reuse = cuda_short_value_reuse_selected(
        &device,
        2,
        heads,
        value_dim,
        (batch as u128) * (length as u128),
        length,
    )?;
    let short_columns = if reuse { 32 } else { 8 };
    let name = if reuse {
        "qwen35_shortvalues_2_bf16"
    } else {
        "qwen35_uniform_delta_bfout"
    };
    let function = device.get_or_load_custom_func(name, "sys1_qwen35", CUDA_PTX)?;
    let config = LaunchConfig {
        grid_dim: (
            (batch * heads) as u32,
            value_dim.div_ceil(short_columns) as u32,
            1,
        ),
        block_dim: (128, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut launch = function.builder();
    launch.arg(&query);
    launch.arg(&key);
    launch.arg(&value);
    launch.arg(&gate);
    launch.arg(&beta);
    launch.arg(&output);
    let length = length as i32;
    let heads_arg = heads as i32;
    let key_heads_arg = key_heads as i32;
    let value_dim = value_dim as i32;
    launch.arg(&length);
    launch.arg(&heads_arg);
    launch.arg(&key_heads_arg);
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

pub(crate) fn cuda_delta_rule_packed_bfout(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    g: &Tensor,
    beta: &Tensor,
    sequences: &PackedSequences,
    subwarp16: bool,
) -> Result<Tensor> {
    let (batch, length, key_heads, key_dim) = q.dims4()?;
    if !subwarp16 || batch != 1 || length != sequences.total {
        candle_core::bail!("packed delta unsupported layout")
    }
    let (value_batch, value_length, heads, value_dim) = v.dims4()?;
    if key_dim != 128
        || key_heads == 0
        || heads == 0
        || heads % key_heads != 0
        || value_dim == 0
        || k.dims() != q.dims()
        || value_batch != batch
        || value_length != length
        || g.dims() != [batch, length, heads]
        || beta.dims() != [batch, length, heads]
    {
        candle_core::bail!("Unsupported Qwen3.5 CUDA delta-rule shape")
    }
    let q = q.contiguous()?;
    let k = k.contiguous()?;
    let v = v.contiguous()?;
    let g = g.contiguous()?;
    let beta = beta.contiguous()?;
    let (q_storage, q_layout) = q.storage_and_layout();
    let (k_storage, k_layout) = k.storage_and_layout();
    let (v_storage, v_layout) = v.storage_and_layout();
    let (g_storage, g_layout) = g.storage_and_layout();
    let (beta_storage, beta_layout) = beta.storage_and_layout();
    let (
        Storage::Cuda(q_storage),
        Storage::Cuda(k_storage),
        Storage::Cuda(v_storage),
        Storage::Cuda(g_storage),
        Storage::Cuda(beta_storage),
    ) = (
        &*q_storage,
        &*k_storage,
        &*v_storage,
        &*g_storage,
        &*beta_storage,
    )
    else {
        candle_core::bail!("Qwen3.5 delta rule requires CUDA storage")
    };
    let offsets = |layout: &candle_core::Layout| {
        layout
            .contiguous_offsets()
            .ok_or_else(|| candle_core::Error::Msg("Qwen3.5 delta input is not contiguous".into()))
    };
    let (q_start, q_end) = offsets(q_layout)?;
    let (k_start, k_end) = offsets(k_layout)?;
    let (v_start, v_end) = offsets(v_layout)?;
    let (g_start, g_end) = offsets(g_layout)?;
    let (beta_start, beta_end) = offsets(beta_layout)?;
    let query = q_storage.as_cuda_slice::<f32>()?.slice(q_start..q_end);
    let key = k_storage.as_cuda_slice::<f32>()?.slice(k_start..k_end);
    let value = v_storage.as_cuda_slice::<f32>()?.slice(v_start..v_end);
    let gate = g_storage.as_cuda_slice::<f32>()?.slice(g_start..g_end);
    let beta = beta_storage
        .as_cuda_slice::<f32>()?
        .slice(beta_start..beta_end);
    let (metadata_storage, metadata_layout) = sequences.cumulative.storage_and_layout();
    let Storage::Cuda(metadata_storage) = &*metadata_storage else {
        candle_core::bail!("packed metadata must be CUDA")
    };
    let (start, end) = offsets(metadata_layout)?;
    let cumulative = metadata_storage.as_cuda_slice::<u32>()?.slice(start..end);
    let device = q_storage.device().clone();
    let output = unsafe { device.alloc::<bf16>(batch * heads * length * value_dim)? };

    let longest = *sequences.lengths.iter().max().unwrap_or(&0);
    let staged = option_env!("SYS1_CUDA_COMPUTE_CAP") == Some("90")
        && value_dim == 128
        && heads >= 32
        && (65..512).contains(&longest)
        && q_start % 4 == 0
        && k_start % 4 == 0
        && v_start % 4 == 0;
    let precomputed_decay = if staged {
        let values = unsafe { device.alloc::<f32>(length * heads)? };
        let function =
            device.get_or_load_custom_func("qwen35_preexp_gate_f32", "sys1_qwen35", CUDA_PTX)?;
        let count = (length * heads) as u32;
        let mut launch = function.builder();
        launch.arg(&gate);
        launch.arg(&values);
        launch.arg(&count);
        unsafe { launch.launch(LaunchConfig::for_num_elems(count)) }.w()?;
        Some(values)
    } else {
        None
    };
    let reuse = !staged
        && cuda_short_value_reuse_selected(
            &device,
            1,
            heads,
            value_dim,
            sequences.total as u128,
            longest,
        )?;
    let short_columns = if staged {
        16
    } else if reuse {
        32
    } else {
        8
    };
    let name = if staged {
        "qwen35_packed_delta_prefetch8_columns4_preexp_bf16"
    } else if reuse {
        "qwen35_shortvalues_1_bf16"
    } else {
        "qwen35_packed_delta_bfout"
    };
    let function = device.get_or_load_custom_func(name, "sys1_qwen35", CUDA_PTX)?;
    let config = LaunchConfig {
        grid_dim: (
            (sequences.lengths.len() * heads) as u32,
            value_dim.div_ceil(short_columns) as u32,
            1,
        ),
        block_dim: (128, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut launch = function.builder();
    launch.arg(&query);
    launch.arg(&key);
    launch.arg(&value);
    if let Some(decay) = &precomputed_decay {
        launch.arg(decay);
    } else {
        launch.arg(&gate);
    }
    launch.arg(&beta);
    launch.arg(&output);
    launch.arg(&cumulative);
    let length = length as i32;
    let heads_arg = heads as i32;
    let key_heads_arg = key_heads as i32;
    let value_dim = value_dim as i32;
    launch.arg(&length);
    launch.arg(&heads_arg);
    launch.arg(&key_heads_arg);
    launch.arg(&value_dim);
    unsafe { launch.launch(config) }.w()?;
    let output = Tensor::from_storage(
        Storage::Cuda(CudaStorage::wrap_cuda_slice(output, device)),
        (batch, length as usize, heads, value_dim as usize),
        BackpropOp::none(),
        false,
    );
    Ok(output)
}

pub(crate) fn cuda_c64bfout_uniform(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    g: &Tensor,
    beta: &Tensor,
    tile_columns: usize,
    profiling: bool,
) -> Result<Tensor> {
    if q.dims().len() == 4
        && v.dims().len() == 4
        && q.dim(2)? == 16
        && q.dim(3)? == 128
        && v.dim(2)? == 32
        && v.dim(3)? == 128
        && q.dim(0)? > 0
        && q.dim(1)? > 0
    {
        let (batch, length, _, _) = q.dims4()?;
        let total = batch
            .checked_mul(length)
            .ok_or_else(|| candle_core::Error::Msg("uniform AOT token overflow".into()))?;
        let packed_q = q.reshape((1, total, 16, 128))?;
        let packed_k = k.reshape((1, total, 16, 128))?;
        let packed_v = v.reshape((1, total, 32, 128))?;
        let packed_g = g.reshape((1, total, 32))?;
        let packed_beta = beta.reshape((1, total, 32))?;

        if option_env!("SYS1_CUDA_COMPUTE_CAP") == Some("80")
            && length >= 128
            && tile_columns == 32
            && [q, k, v, g, beta].iter().all(|x| x.dtype() == DType::F32)
        {
            let sequences = PackedSequences::new(&vec![length; batch], q.device())?;
            return cuda_c64bfout_varlen(
                &packed_q,
                &packed_k,
                &packed_v,
                &packed_g,
                &packed_beta,
                &sequences,
                profiling,
            )?
            .reshape((batch, length, 32, 128));
        }
        if let Some(output) = gated_delta::try_forward(
            &packed_q,
            &packed_k,
            &packed_v,
            &packed_g,
            &packed_beta,
            &vec![length; batch],
        )? {
            return output.reshape((batch, length, 32, 128));
        }
    }

    if [q, k, v, g, beta].iter().any(|x| x.dtype() != DType::F32) {
        candle_core::bail!("chunked recurrence requires FP32 inputs");
    }
    if tile_columns != 32 {
        candle_core::bail!("C64 state requires N32 caller contract");
    }
    let low_precision_state = option_env!("SYS1_CUDA_COMPUTE_CAP") == Some("90");
    let (tile_columns, state_rows, carry_shared) = if low_precision_state {
        (64usize, 64usize, 98336u32)
    } else {
        (32usize, 128usize, 58400u32)
    };
    let (batch, length, key_heads, key_dim) = q.dims4()?;
    let (value_batch, value_length, heads, value_dim) = v.dims4()?;
    if key_dim != 128
        || key_heads == 0
        || heads == 0
        || heads % key_heads != 0
        || value_dim == 0
        || k.dims() != q.dims()
        || value_batch != batch
        || value_length != length
        || g.dims() != [batch, length, heads]
        || beta.dims() != [batch, length, heads]
    {
        candle_core::bail!("Unsupported Qwen3.5 CUDA delta-rule shape")
    }
    let q = q.contiguous()?;
    let k = k.contiguous()?;
    let v = v.contiguous()?;
    let g = g.contiguous()?;
    let beta = beta.contiguous()?;
    let (q_storage, q_layout) = q.storage_and_layout();
    let (k_storage, k_layout) = k.storage_and_layout();
    let (v_storage, v_layout) = v.storage_and_layout();
    let (g_storage, g_layout) = g.storage_and_layout();
    let (beta_storage, beta_layout) = beta.storage_and_layout();
    let (
        Storage::Cuda(q_storage),
        Storage::Cuda(k_storage),
        Storage::Cuda(v_storage),
        Storage::Cuda(g_storage),
        Storage::Cuda(beta_storage),
    ) = (
        &*q_storage,
        &*k_storage,
        &*v_storage,
        &*g_storage,
        &*beta_storage,
    )
    else {
        candle_core::bail!("Qwen3.5 delta rule requires CUDA storage")
    };
    let offsets = |layout: &candle_core::Layout| {
        layout
            .contiguous_offsets()
            .ok_or_else(|| candle_core::Error::Msg("Qwen3.5 delta input is not contiguous".into()))
    };
    let (q_start, q_end) = offsets(q_layout)?;
    let (k_start, k_end) = offsets(k_layout)?;
    let (v_start, v_end) = offsets(v_layout)?;
    let (g_start, g_end) = offsets(g_layout)?;
    let (beta_start, beta_end) = offsets(beta_layout)?;
    let query = q_storage.as_cuda_slice::<f32>()?.slice(q_start..q_end);
    let key = k_storage.as_cuda_slice::<f32>()?.slice(k_start..k_end);
    let value = v_storage.as_cuda_slice::<f32>()?.slice(v_start..v_end);
    let gate = g_storage.as_cuda_slice::<f32>()?.slice(g_start..g_end);
    let beta = beta_storage
        .as_cuda_slice::<f32>()?
        .slice(beta_start..beta_end);
    let device = q_storage.device().clone();

    if ![32usize, 64].contains(&tile_columns)
        || batch == 0
        || length == 0
        || value_dim > 128
        || [
            q.elem_count(),
            k.elem_count(),
            v.elem_count(),
            g.elem_count(),
        ]
        .iter()
        .any(|n| *n > i32::MAX as usize)
    {
        candle_core::bail!("chunk probe requires nonempty F32 K128,V<=128,small indices");
    }
    let instances = batch
        .checked_mul(heads)
        .and_then(|x| x.checked_mul(length.div_ceil(64)))
        .ok_or_else(|| candle_core::Error::Msg("chunk count overflow".into()))?;
    let w_count = instances
        .checked_mul(64 * 128)
        .ok_or_else(|| candle_core::Error::Msg("chunk W overflow".into()))?;
    let u_count = instances
        .checked_mul(64 * value_dim)
        .ok_or_else(|| candle_core::Error::Msg("chunk U overflow".into()))?;
    let c_count = instances
        .checked_mul(64 * 64)
        .ok_or_else(|| candle_core::Error::Msg("chunk C overflow".into()))?;
    let d_count = instances
        .checked_mul(128)
        .ok_or_else(|| candle_core::Error::Msg("decay count overflow".into()))?;
    if [w_count, u_count, c_count, d_count]
        .iter()
        .any(|n| *n > i32::MAX as usize)
    {
        candle_core::bail!("chunk scratch index overflow");
    }
    let result = (|| -> Result<Tensor> {
        let mut slot = CHUNK64_SCRATCH
            .lock()
            .map_err(|_| candle_core::Error::Msg("poisoned chunk scratch".into()))?;
        let stream = device.cuda_stream();
        if slot
            .as_ref()
            .is_none_or(|old| !std::sync::Arc::ptr_eq(&old.stream, &stream))
        {
            *slot = Some(ChunkScratch {
                stream,
                w: None,
                u: None,
                c: None,
                decays: None,
                snapshots: None,
            });
        }
        let workspace = slot
            .as_mut()
            .ok_or_else(|| candle_core::Error::Msg("missing chunk scratch".into()))?;
        let ChunkScratch {
            w,
            u,
            c,
            decays,
            snapshots,
            ..
        } = workspace;
        if profiling {
            device.cuda_stream().synchronize().w()?;
        }
        let mut stage_started = Instant::now();
        let w = grow_chunk_buffer(w, &device, w_count)?;
        let u = grow_chunk_buffer(u, &device, u_count)?;
        let c = grow_chunk_buffer(c, &device, c_count)?;
        let decays = grow_chunk_buffer(decays, &device, d_count)?;
        let output_count = batch
            .checked_mul(length)
            .and_then(|n| n.checked_mul(heads))
            .and_then(|n| n.checked_mul(value_dim))
            .ok_or_else(|| candle_core::Error::Msg("register-carry C64 output overflow".into()))?;
        if output_count > i32::MAX as usize {
            candle_core::bail!("padded output index overflow");
        }
        let guarded = !low_precision_state
            && matches!(option_env!("SYS1_CUDA_COMPUTE_CAP"), Some("90" | "120"));
        let snapshot_count = if guarded { instances } else { 1usize };
        if snapshot_count > i32::MAX as usize {
            candle_core::bail!("snapshot index overflow");
        }
        let snapshots = grow_chunk_buffer(snapshots, &device, snapshot_count)?;
        let mut output = unsafe { device.alloc::<bf16>(output_count)? };
        if profiling {
            println!(
                "C64_FLAGS_WORKSPACE retained_bytes={} output_bytes={} single_entry_per_process=true owned_stream_identity=true output_never_reused=true amortized_warm_operator_not_serving=true",
                (w.len() + u.len() + c.len() + decays.len() + snapshots.len()) * 4,
                output_count * 2
            );
        }
        if profiling {
            device.cuda_stream().synchronize().w()?;
            println!(
                "C64_FLAGS_STAGE part=scratch batch={batch} length={length} columns={tile_columns} ms={} diagnostic_not_serving=true",
                stage_started.elapsed().as_secs_f64() * 1000.
            );
            stage_started = Instant::now();
        }
        let length_arg = length as i32;
        let heads_arg = heads as i32;
        let key_heads_arg = key_heads as i32;
        let value_dim_arg = value_dim as i32;
        {
            let function = device.get_or_load_custom_func(
                if guarded {
                    "qwen35_c64zguard_uniform_prepare_f32"
                } else {
                    if matches!(option_env!("SYS1_CUDA_COMPUTE_CAP"), Some("90" | "120")) {
                        if option_env!("SYS1_CUDA_COMPUTE_CAP") == Some("90") {
                            "qwen35_c64prepvload2_uniform_prepare_f32"
                        } else {
                            "qwen35_c64sub8vector_uniform_prepare_f32"
                        }
                    } else {
                        "qwen35_c64slice_uniform_prepare_f32"
                    }
                },
                "sys1_chunk_delta",
                CUDA_PTX,
            )?;
            function.set_attribute(candle_core::cuda_backend::cudarc::driver::sys::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,49664).w()?;
            let mut launch = function.builder();
            launch.arg(&query);
            launch.arg(&key);
            launch.arg(&value);
            launch.arg(&gate);
            launch.arg(&beta);
            launch.arg(&mut *w);
            launch.arg(&mut *u);
            launch.arg(&mut *c);
            launch.arg(&mut *decays);
            if guarded {
                launch.arg(&mut *snapshots);
            }
            launch.arg(&length_arg);
            launch.arg(&heads_arg);
            launch.arg(&key_heads_arg);
            launch.arg(&value_dim_arg);
            unsafe {
                launch.launch(LaunchConfig {
                    grid_dim: (instances as u32, 1, 1),
                    block_dim: (512, 1, 1),
                    shared_mem_bytes: 49664,
                })
            }
            .w()?;
        }
        if profiling {
            device.cuda_stream().synchronize().w()?;
            println!(
                "C64_FLAGS_STAGE part=prepare batch={batch} length={length} columns={tile_columns} ms={} diagnostic_not_serving=true",
                stage_started.elapsed().as_secs_f64() * 1000.
            );
            stage_started = Instant::now();
        }
        {
            let name = if low_precision_state {
                "qwen35_c64fp16phase3vec2_n64_uniform_f32"
            } else if matches!(option_env!("SYS1_CUDA_COMPUTE_CAP"), Some("90" | "120")) {
                "qwen35_c64zguard_n32_uniform32_f32"
            } else {
                "qwen35_c64bfout_uniform32_f32"
            };
            let function = device.get_or_load_custom_func(name, "sys1_chunk_delta", CUDA_PTX)?;
            if profiling {
                println!(
                    "C64_FLAGS_RESOURCE kernel={name} registers={} local_bytes={} shared_bytes={} diagnostic_not_serving=true",
                    function.num_regs().w()?,
                    function.local_size_bytes().w()?,
                    function.shared_size_bytes().w()?
                );
            }
            function.set_attribute(candle_core::cuda_backend::cudarc::driver::sys::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,carry_shared as i32).w()?;
            let mut launch = function.builder();
            launch.arg(&query);
            launch.arg(&key);
            launch.arg(&gate);
            launch.arg(&*w);
            launch.arg(&*u);
            launch.arg(&*c);
            launch.arg(&*decays);
            launch.arg(&mut output);
            launch.arg(&mut *snapshots);
            launch.arg(&length_arg);
            launch.arg(&heads_arg);
            launch.arg(&key_heads_arg);
            launch.arg(&value_dim_arg);
            unsafe {
                launch.launch(LaunchConfig {
                    grid_dim: (
                        (batch * heads) as u32,
                        value_dim.div_ceil(tile_columns) as u32,
                        1,
                    ),
                    block_dim: (((state_rows / 16) * (tile_columns / 8) * 32) as u32, 1, 1),
                    shared_mem_bytes: carry_shared,
                })
            }
            .w()?;
        }
        if profiling {
            device.cuda_stream().synchronize().w()?;
            println!(
                "C64_FLAGS_STAGE part=state batch={batch} length={length} columns={tile_columns} ms={} diagnostic_not_serving=true",
                stage_started.elapsed().as_secs_f64() * 1000.
            );
        }

        let output = Tensor::from_storage(
            Storage::Cuda(CudaStorage::wrap_cuda_slice(output, device.clone())),
            (batch, length, heads, value_dim),
            BackpropOp::none(),
            false,
        );
        Ok(output)
    })();
    if result.is_err()
        && let Ok(mut slot) = CHUNK64_SCRATCH.lock()
    {
        slot.take();
    }
    result
}

pub(crate) fn cuda_c64bfout_varlen(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    g: &Tensor,
    beta: &Tensor,
    sequences: &PackedSequences,
    profiling: bool,
) -> Result<Tensor> {
    cuda_c64bfout_varlen_prepared(q, k, v, (g, beta), sequences, profiling, None)
}

pub(crate) fn cuda_c64bfout_varlen_prepared(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    gates: (&Tensor, &Tensor),
    sequences: &PackedSequences,
    profiling: bool,
    prepared: Option<gated_delta::PreparedQv>,
) -> Result<Tensor> {
    let (g, beta) = gates;
    let output = if let Some(ticket) = prepared {
        gated_delta::try_forward_prepared(q, k, v, g, beta, &sequences.lengths, ticket)?
    } else {
        gated_delta::try_forward(q, k, v, g, beta, &sequences.lengths)?
    };
    if let Some(output) = output {
        return Ok(output);
    }

    if [q, k, v, g, beta].iter().any(|x| x.dtype() != DType::F32) {
        candle_core::bail!("chunked recurrence requires FP32 inputs");
    }

    if q.dim(0)? != 1 || q.dim(1)? != sequences.total {
        candle_core::bail!("segmented C64 shape mismatch")
    }
    if sequences.c64_chunk_count == 0 {
        return cuda_delta_rule_packed_bfout(q, k, v, g, beta, sequences, true);
    }
    if [
        k,
        v,
        g,
        beta,
        &sequences.cumulative,
        &sequences.c64_chunks,
        &sequences.c64_long_ids,
        &sequences.c64_short_ids,
    ]
    .iter()
    .any(|t| !q.device().same_device(t.device()))
    {
        candle_core::bail!("segmented C64 device mismatch")
    }
    let (batch, length, key_heads, key_dim) = q.dims4()?;
    let (value_batch, value_length, heads, value_dim) = v.dims4()?;
    if key_dim != 128
        || key_heads == 0
        || heads == 0
        || heads % key_heads != 0
        || value_dim == 0
        || k.dims() != q.dims()
        || value_batch != batch
        || value_length != length
        || g.dims() != [batch, length, heads]
        || beta.dims() != [batch, length, heads]
    {
        candle_core::bail!("Unsupported Qwen3.5 CUDA delta-rule shape")
    }
    let q = q.contiguous()?;
    let k = k.contiguous()?;
    let v = v.contiguous()?;
    let g = g.contiguous()?;
    let beta = beta.contiguous()?;
    let (q_storage, q_layout) = q.storage_and_layout();
    let (k_storage, k_layout) = k.storage_and_layout();
    let (v_storage, v_layout) = v.storage_and_layout();
    let (g_storage, g_layout) = g.storage_and_layout();
    let (beta_storage, beta_layout) = beta.storage_and_layout();
    let (
        Storage::Cuda(q_storage),
        Storage::Cuda(k_storage),
        Storage::Cuda(v_storage),
        Storage::Cuda(g_storage),
        Storage::Cuda(beta_storage),
    ) = (
        &*q_storage,
        &*k_storage,
        &*v_storage,
        &*g_storage,
        &*beta_storage,
    )
    else {
        candle_core::bail!("Qwen3.5 delta rule requires CUDA storage")
    };
    let offsets = |layout: &candle_core::Layout| {
        layout
            .contiguous_offsets()
            .ok_or_else(|| candle_core::Error::Msg("Qwen3.5 delta input is not contiguous".into()))
    };
    let (q_start, q_end) = offsets(q_layout)?;
    let (k_start, k_end) = offsets(k_layout)?;
    let (v_start, v_end) = offsets(v_layout)?;
    let (g_start, g_end) = offsets(g_layout)?;
    let (beta_start, beta_end) = offsets(beta_layout)?;
    let query = q_storage.as_cuda_slice::<f32>()?.slice(q_start..q_end);
    let key = k_storage.as_cuda_slice::<f32>()?.slice(k_start..k_end);
    let value = v_storage.as_cuda_slice::<f32>()?.slice(v_start..v_end);
    let gate = g_storage.as_cuda_slice::<f32>()?.slice(g_start..g_end);
    let beta = beta_storage
        .as_cuda_slice::<f32>()?
        .slice(beta_start..beta_end);
    let (cumulative_storage, cumulative_layout) = sequences.cumulative.storage_and_layout();
    let Storage::Cuda(cumulative_storage) = &*cumulative_storage else {
        candle_core::bail!("CUDA metadata required")
    };
    let (start, end) = offsets(cumulative_layout)?;
    let cumulative = cumulative_storage.as_cuda_slice::<u32>()?.slice(start..end);
    let (chunk_offsets_storage, chunk_offsets_layout) = sequences.c64_chunks.storage_and_layout();
    let Storage::Cuda(chunk_offsets_storage) = &*chunk_offsets_storage else {
        candle_core::bail!("CUDA metadata required")
    };
    let (start, end) = offsets(chunk_offsets_layout)?;
    let chunk_offsets = chunk_offsets_storage
        .as_cuda_slice::<u32>()?
        .slice(start..end);
    let (long_ids_storage, long_ids_layout) = sequences.c64_long_ids.storage_and_layout();
    let Storage::Cuda(long_ids_storage) = &*long_ids_storage else {
        candle_core::bail!("CUDA metadata required")
    };
    let (start, end) = offsets(long_ids_layout)?;
    let long_ids = long_ids_storage.as_cuda_slice::<u32>()?.slice(start..end);
    let (short_ids_storage, short_ids_layout) = sequences.c64_short_ids.storage_and_layout();
    let Storage::Cuda(short_ids_storage) = &*short_ids_storage else {
        candle_core::bail!("CUDA metadata required")
    };
    let (start, end) = offsets(short_ids_layout)?;
    let short_ids = short_ids_storage.as_cuda_slice::<u32>()?.slice(start..end);
    let device = q_storage.device().clone();
    let sm_count = cuda_c64_sm_count(&device)?;
    let wide_blocks = sequences
        .c64_long_count
        .checked_mul(heads)
        .and_then(|n| n.checked_mul(value_dim.div_ceil(64)))
        .ok_or_else(|| candle_core::Error::Msg("wide recurrent grid overflow".into()))?;
    let low_precision_state = option_env!("SYS1_CUDA_COMPUTE_CAP") == Some("90");

    let sm80_prepack =
        option_env!("SYS1_CUDA_COMPUTE_CAP") == Some("80") && sequences.c64_long_count > 0;

    let use_prepack = sm80_prepack
        || (low_precision_state
            && sequences
                .lengths
                .iter()
                .filter(|&&n| n >= 128)
                .copied()
                .sum::<usize>()
                >= 4096);

    let sm80_full01 = option_env!("SYS1_CUDA_COMPUTE_CAP") == Some("80");
    let use_wide = sm80_full01
        || low_precision_state
        || cuda_c64_scheduled_geometry(sequences, heads, value_dim, sm_count)?;
    let _ = wide_blocks;
    let (tile_columns, state_rows, carry_shared) = if use_prepack {
        (32usize, 64usize, 49184u32)
    } else if use_wide {
        (
            64usize,
            64usize,
            if matches!(
                option_env!("SYS1_CUDA_COMPUTE_CAP"),
                Some("80" | "90" | "120")
            ) {
                98336u32
            } else {
                82976u32
            },
        )
    } else {
        (32usize, 128usize, 58400u32)
    };
    if use_wide {
        static NOTED: std::sync::OnceLock<()> = std::sync::OnceLock::new();
        NOTED.get_or_init(|| {
            tracing::info!(
                value_tile = tile_columns,
                threads = (state_rows / 16) * (tile_columns / 8) * 32,
                shared_bytes = carry_shared,
                sm_count,
                "Using automatic recurrent carry geometry"
            )
        });
    }

    if ![32usize, 64].contains(&tile_columns)
        || batch == 0
        || length == 0
        || value_dim > 128
        || [
            q.elem_count(),
            k.elem_count(),
            v.elem_count(),
            g.elem_count(),
        ]
        .iter()
        .any(|n| *n > i32::MAX as usize)
    {
        candle_core::bail!("chunk probe requires nonempty F32 K128,V<=128,small indices");
    }
    let instances = heads
        .checked_mul(sequences.c64_chunk_count)
        .ok_or_else(|| candle_core::Error::Msg("chunk count overflow".into()))?;
    let w_count = instances
        .checked_mul(64 * 128)
        .ok_or_else(|| candle_core::Error::Msg("chunk W overflow".into()))?;
    let u_count = instances
        .checked_mul(64 * value_dim)
        .ok_or_else(|| candle_core::Error::Msg("chunk U overflow".into()))?;
    let c_count = instances
        .checked_mul(64 * 64)
        .ok_or_else(|| candle_core::Error::Msg("chunk C overflow".into()))?;
    let d_count = instances
        .checked_mul(128)
        .ok_or_else(|| candle_core::Error::Msg("decay count overflow".into()))?;
    if [w_count, u_count, c_count, d_count]
        .iter()
        .any(|n| *n > i32::MAX as usize)
    {
        candle_core::bail!("chunk scratch index overflow");
    }
    let result = (|| -> Result<Tensor> {
        let mut slot = CHUNK64_SCRATCH
            .lock()
            .map_err(|_| candle_core::Error::Msg("poisoned chunk scratch".into()))?;
        let stream = device.cuda_stream();
        if slot
            .as_ref()
            .is_none_or(|old| !std::sync::Arc::ptr_eq(&old.stream, &stream))
        {
            *slot = Some(ChunkScratch {
                stream,
                w: None,
                u: None,
                c: None,
                decays: None,
                snapshots: None,
            });
        }
        let workspace = slot
            .as_mut()
            .ok_or_else(|| candle_core::Error::Msg("missing chunk scratch".into()))?;
        let ChunkScratch {
            w,
            u,
            c,
            decays,
            snapshots,
            ..
        } = workspace;
        if profiling {
            device.cuda_stream().synchronize().w()?;
        }
        let mut stage_started = Instant::now();
        let w = grow_chunk_buffer(w, &device, w_count)?;
        let u = grow_chunk_buffer(u, &device, u_count)?;
        let c = grow_chunk_buffer(c, &device, c_count)?;
        let decays = grow_chunk_buffer(decays, &device, d_count)?;
        let output_count = length
            .checked_mul(heads)
            .and_then(|n| n.checked_mul(value_dim))
            .ok_or_else(|| candle_core::Error::Msg("segmented C64 output overflow".into()))?;
        if output_count > i32::MAX as usize {
            candle_core::bail!("padded output index overflow");
        }
        let guarded =
            matches!(option_env!("SYS1_CUDA_COMPUTE_CAP"), Some("90" | "120")) && !use_wide;
        let snapshot_count = if guarded { instances } else { 1usize };
        if snapshot_count > i32::MAX as usize {
            candle_core::bail!("snapshot index overflow");
        }
        let snapshots = grow_chunk_buffer(snapshots, &device, snapshot_count)?;
        let mut output = unsafe { device.alloc::<bf16>(output_count)? };
        let packed_count = instances
            .checked_mul(28672)
            .ok_or_else(|| candle_core::Error::Msg("prepacked C64 allocation overflow".into()))?;
        let mut prepacked = if use_prepack {
            Some(unsafe { device.alloc::<u16>(packed_count)? })
        } else {
            None
        };
        let mut pack_flags = if use_prepack {
            Some(unsafe { device.alloc::<u32>(instances)? })
        } else {
            None
        };

        if profiling {
            println!(
                "C64_CARRY_WORKSPACE retained_bytes={} output_bytes={} single_entry_per_process=true owned_stream_identity=true output_never_reused=true amortized_warm_operator_not_serving=true",
                (w.len() + u.len() + c.len() + decays.len() + snapshots.len()) * 4,
                output_count * 2
            );
        }
        if profiling {
            device.cuda_stream().synchronize().w()?;
            println!(
                "C64_CARRY_STAGE part=scratch batch={batch} length={length} columns={tile_columns} ms={} diagnostic_not_serving=true",
                stage_started.elapsed().as_secs_f64() * 1000.
            );
            stage_started = Instant::now();
        }
        let sequence_count = sequences.lengths.len() as i32;
        let length_arg = length as i32;
        let heads_arg = heads as i32;
        let key_heads_arg = key_heads as i32;
        let value_dim_arg = value_dim as i32;
        {
            let function = device.get_or_load_custom_func(
                if use_prepack {
                    "qwen35_c64storepack15_varlen_prepare_f32"
                } else if guarded {
                    "qwen35_c64zguard_varlen_prepare_f32"
                } else {
                    if matches!(
                        option_env!("SYS1_CUDA_COMPUTE_CAP"),
                        Some("80" | "90" | "120")
                    ) {
                        if option_env!("SYS1_CUDA_COMPUTE_CAP") == Some("90") {
                            "qwen35_c64prepvload2_varlen_prepare_f32"
                        } else {
                            "qwen35_c64sub8vector_varlen_prepare_f32"
                        }
                    } else {
                        "qwen35_c64slice_varlen_prepare_f32"
                    }
                },
                "sys1_chunk_delta",
                CUDA_PTX,
            )?;
            function.set_attribute(candle_core::cuda_backend::cudarc::driver::sys::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,49664).w()?;
            let mut launch = function.builder();
            launch.arg(&query);
            launch.arg(&key);
            launch.arg(&value);
            launch.arg(&gate);
            launch.arg(&beta);
            launch.arg(&mut *w);
            launch.arg(&mut *u);
            launch.arg(&mut *c);
            launch.arg(&mut *decays);
            if guarded {
                launch.arg(&mut *snapshots);
            }
            launch.arg(&cumulative);
            launch.arg(&chunk_offsets);
            launch.arg(&sequence_count);
            launch.arg(&length_arg);
            launch.arg(&heads_arg);
            launch.arg(&key_heads_arg);
            launch.arg(&value_dim_arg);
            if use_prepack {
                launch.arg(prepacked.as_mut().ok_or_else(|| {
                    candle_core::Error::Msg("missing fused packed C64 buffer".into())
                })?);
                launch.arg(
                    pack_flags
                        .as_mut()
                        .ok_or_else(|| candle_core::Error::Msg("missing fused C64 flags".into()))?,
                );
            }
            unsafe {
                launch.launch(LaunchConfig {
                    grid_dim: (instances as u32, 1, 1),
                    block_dim: (512, 1, 1),
                    shared_mem_bytes: 49664,
                })
            }
            .w()?;
        }
        if profiling {
            device.cuda_stream().synchronize().w()?;
            println!(
                "C64_CARRY_STAGE part=prepare batch={batch} length={length} columns={tile_columns} ms={} diagnostic_not_serving=true",
                stage_started.elapsed().as_secs_f64() * 1000.
            );
            stage_started = Instant::now();
        }
        {
            if sequences.c64_short_count > 0 {
                let short_tokens = sequences
                    .lengths
                    .iter()
                    .filter(|&&n| {
                        n < (if option_env!("SYS1_CUDA_COMPUTE_CAP") == Some("90") {
                            128
                        } else {
                            512
                        })
                    })
                    .map(|&n| n as u128)
                    .sum();
                let longest_short = sequences
                    .lengths
                    .iter()
                    .filter(|&&n| {
                        n < (if option_env!("SYS1_CUDA_COMPUTE_CAP") == Some("90") {
                            128
                        } else {
                            512
                        })
                    })
                    .copied()
                    .max()
                    .unwrap_or(0);
                let reuse = cuda_short_value_reuse_selected(
                    &device,
                    0,
                    heads,
                    value_dim,
                    short_tokens,
                    longest_short,
                )?;
                let short_columns = if reuse { 32 } else { 8 };
                let name = if reuse {
                    "qwen35_shortvalues_0_bf16"
                } else {
                    "qwen35_c64bfout_short_bf16"
                };
                let function = device.get_or_load_custom_func(name, "sys1_c64v_short", CUDA_PTX)?;
                let mut launch = function.builder();
                launch
                    .arg(&query)
                    .arg(&key)
                    .arg(&value)
                    .arg(&gate)
                    .arg(&beta)
                    .arg(&mut output)
                    .arg(&cumulative)
                    .arg(&short_ids)
                    .arg(&length_arg)
                    .arg(&heads_arg)
                    .arg(&key_heads_arg)
                    .arg(&value_dim_arg);
                unsafe {
                    launch.launch(LaunchConfig {
                        grid_dim: (
                            (sequences.c64_short_count * heads) as u32,
                            value_dim.div_ceil(short_columns) as u32,
                            1,
                        ),
                        block_dim: (128, 1, 1),
                        shared_mem_bytes: 0,
                    })
                }
                .w()?;
            }
            let name = if use_prepack {
                "qwen35_c64packed32v40_n32_varlen_f32"
            } else if low_precision_state {
                "qwen35_c64fp16phase3vec2_n64_varlen_f32"
            } else if use_wide {
                if matches!(
                    option_env!("SYS1_CUDA_COMPUTE_CAP"),
                    Some("80" | "90" | "120")
                ) {
                    "qwen35_c64full01_n64_varlen_f32"
                } else {
                    "qwen35_c64wide_varlen_f32"
                }
            } else if matches!(option_env!("SYS1_CUDA_COMPUTE_CAP"), Some("90" | "120")) {
                "qwen35_c64zguard_n32_varlen32_f32"
            } else {
                "qwen35_c64bfout_varlen32_f32"
            };
            let function = device.get_or_load_custom_func(name, "sys1_chunk_delta", CUDA_PTX)?;
            if profiling {
                println!(
                    "C64_CARRY_RESOURCE kernel={name} registers={} local_bytes={} shared_bytes={} diagnostic_not_serving=true",
                    function.num_regs().w()?,
                    function.local_size_bytes().w()?,
                    function.shared_size_bytes().w()?
                );
            }
            function.set_attribute(candle_core::cuda_backend::cudarc::driver::sys::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,carry_shared as i32).w()?;
            let mut launch = function.builder();
            launch.arg(&query);
            launch.arg(&key);
            launch.arg(&gate);
            launch.arg(&*w);
            launch.arg(&*u);
            launch.arg(&*c);
            launch.arg(&*decays);
            launch.arg(&mut output);
            launch.arg(&mut *snapshots);
            launch.arg(&cumulative);
            launch.arg(&chunk_offsets);
            launch.arg(&long_ids);
            launch.arg(&length_arg);
            launch.arg(&heads_arg);
            launch.arg(&key_heads_arg);
            launch.arg(&value_dim_arg);
            if use_prepack {
                launch.arg(
                    prepacked.as_ref().ok_or_else(|| {
                        candle_core::Error::Msg("missing packed C64 input".into())
                    })?,
                );
                launch.arg(
                    pack_flags
                        .as_ref()
                        .ok_or_else(|| candle_core::Error::Msg("missing C64 flag input".into()))?,
                );
            }

            let value_tiles = value_dim.div_ceil(tile_columns);
            let sequence_heads = sequences.c64_long_count.checked_mul(heads).ok_or_else(|| {
                candle_core::Error::Msg("carry sequence-head grid overflow".into())
            })?;
            let (carry_x, carry_y) = if use_prepack {
                (
                    sequence_heads.checked_mul(value_tiles).ok_or_else(|| {
                        candle_core::Error::Msg("value-fast carry grid overflow".into())
                    })?,
                    1usize,
                )
            } else {
                (sequence_heads, value_tiles)
            };
            let carry_x = u32::try_from(carry_x)
                .map_err(|_| candle_core::Error::Msg("carry x grid exceeds u32".into()))?;
            let carry_y = u32::try_from(carry_y)
                .map_err(|_| candle_core::Error::Msg("carry y grid exceeds u32".into()))?;
            unsafe {
                launch.launch(LaunchConfig {
                    grid_dim: (carry_x, carry_y, 1),
                    block_dim: (((state_rows / 16) * (tile_columns / 8) * 32) as u32, 1, 1),
                    shared_mem_bytes: carry_shared,
                })
            }
            .w()?;
        }
        if profiling {
            device.cuda_stream().synchronize().w()?;
            println!(
                "C64_CARRY_STAGE part=state batch={batch} length={length} columns={tile_columns} ms={} diagnostic_not_serving=true",
                stage_started.elapsed().as_secs_f64() * 1000.
            );
        }

        let output = Tensor::from_storage(
            Storage::Cuda(CudaStorage::wrap_cuda_slice(output, device.clone())),
            (1, length, heads, value_dim),
            BackpropOp::none(),
            false,
        );
        Ok(output)
    })();
    if result.is_err()
        && let Ok(mut slot) = CHUNK64_SCRATCH.lock()
    {
        slot.take();
    }
    result
}

pub(crate) fn cuda_c64_sm_count(device: &candle_core::CudaDevice) -> Result<usize> {
    type Context = std::sync::Arc<candle_core::cuda_backend::cudarc::driver::CudaContext>;
    static CACHE: std::sync::Mutex<Option<(Context, usize)>> = std::sync::Mutex::new(None);
    let context = device.cuda_stream().context().clone();
    let mut slot = CACHE
        .lock()
        .map_err(|_| candle_core::Error::Msg("poisoned CUDA SM count cache".into()))?;
    if let Some((old, count)) = slot.as_ref()
        && std::sync::Arc::ptr_eq(old, &context)
    {
        return Ok(*count);
    }
    let count=context.attribute(candle_core::cuda_backend::cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT).w()?;
    let count = usize::try_from(count)
        .map_err(|_| candle_core::Error::Msg("invalid CUDA SM count".into()))?;
    if count == 0 {
        candle_core::bail!("zero CUDA SM count");
    }
    *slot = Some((context, count));
    Ok(count)
}

pub(crate) fn cuda_c64_schedule_makespan(
    lengths: &[usize],
    heads: usize,
    tiles: usize,
    sms: usize,
) -> Result<u64> {
    if sms == 0 {
        candle_core::bail!("zero SM count");
    }
    let mut chunks = lengths
        .iter()
        .filter(|&&n| n >= 512)
        .map(|&n| n.div_ceil(64) as u64)
        .collect::<Vec<_>>();
    chunks.sort_unstable_by(|a, b| b.cmp(a));
    let mut slots = std::collections::BinaryHeap::from(vec![std::cmp::Reverse(0u64); sms]);

    for _ in 0..tiles {
        for &chunks in &chunks {
            for _ in 0..heads {
                let mut earliest = slots
                    .peek_mut()
                    .ok_or_else(|| candle_core::Error::Msg("empty CTA schedule".into()))?;
                earliest.0 = earliest
                    .0
                    .checked_add(chunks)
                    .ok_or_else(|| candle_core::Error::Msg("CTA schedule overflow".into()))?;
            }
        }
    }
    Ok(slots.into_iter().map(|x| x.0).max().unwrap_or(0))
}

pub(crate) fn cuda_c64_scheduled_geometry(
    sequences: &PackedSequences,
    heads: usize,
    value_dim: usize,
    sms: usize,
) -> Result<bool> {
    if value_dim != 128 || sequences.c64_long_count == 0 {
        return Ok(false);
    }
    let key = (heads, value_dim, sms);
    let mut cache = sequences
        .c64_geometry_cache
        .lock()
        .map_err(|_| candle_core::Error::Msg("poisoned recurrent geometry cache".into()))?;
    if let Some((old, wide)) = *cache
        && old == key
    {
        return Ok(wide);
    }
    let narrow = cuda_c64_schedule_makespan(&sequences.lengths, heads, 4, sms)?;
    let wide = cuda_c64_schedule_makespan(&sequences.lengths, heads, 2, sms)?;

    let use_wide = u128::from(narrow) * 3 >= u128::from(wide) * 4;
    *cache = Some((key, use_wide));
    Ok(use_wide)
}

pub(crate) fn cuda_vstore_delta_packed(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    g: &Tensor,
    beta: &Tensor,
    sequences: &PackedSequences,
    subwarp16: bool,
) -> Result<Tensor> {
    if option_env!("SYS1_CUDA_COMPUTE_CAP") != Some("120") {
        candle_core::bail!("vstore full GDN probe requires SM120");
    }

    if [q, k, g, beta].iter().any(|x| x.dtype() != DType::F32) || v.dtype() != DType::BF16 {
        candle_core::bail!("vstore packed input types");
    }
    let (batch, length, key_heads, key_dim) = q.dims4()?;
    if !subwarp16 || batch != 1 || length != sequences.total {
        candle_core::bail!("packed delta unsupported layout")
    }
    let (value_batch, value_length, heads, value_dim) = v.dims4()?;
    if key_dim != 128
        || key_heads == 0
        || heads == 0
        || heads % key_heads != 0
        || value_dim == 0
        || k.dims() != q.dims()
        || value_batch != batch
        || value_length != length
        || g.dims() != [batch, length, heads]
        || beta.dims() != [batch, length, heads]
    {
        candle_core::bail!("Unsupported Qwen3.5 CUDA delta-rule shape")
    }
    let q = q.contiguous()?;
    let k = k.contiguous()?;
    let v = v.contiguous()?;
    let g = g.contiguous()?;
    let beta = beta.contiguous()?;
    let (q_storage, q_layout) = q.storage_and_layout();
    let (k_storage, k_layout) = k.storage_and_layout();
    let (v_storage, v_layout) = v.storage_and_layout();
    let (g_storage, g_layout) = g.storage_and_layout();
    let (beta_storage, beta_layout) = beta.storage_and_layout();
    let (
        Storage::Cuda(q_storage),
        Storage::Cuda(k_storage),
        Storage::Cuda(v_storage),
        Storage::Cuda(g_storage),
        Storage::Cuda(beta_storage),
    ) = (
        &*q_storage,
        &*k_storage,
        &*v_storage,
        &*g_storage,
        &*beta_storage,
    )
    else {
        candle_core::bail!("Qwen3.5 delta rule requires CUDA storage")
    };
    let offsets = |layout: &candle_core::Layout| {
        layout
            .contiguous_offsets()
            .ok_or_else(|| candle_core::Error::Msg("Qwen3.5 delta input is not contiguous".into()))
    };
    let (q_start, q_end) = offsets(q_layout)?;
    let (k_start, k_end) = offsets(k_layout)?;
    let (v_start, v_end) = offsets(v_layout)?;
    let (g_start, g_end) = offsets(g_layout)?;
    let (beta_start, beta_end) = offsets(beta_layout)?;
    let query = q_storage.as_cuda_slice::<f32>()?.slice(q_start..q_end);
    let key = k_storage.as_cuda_slice::<f32>()?.slice(k_start..k_end);
    let value = v_storage.as_cuda_slice::<bf16>()?.slice(v_start..v_end);
    let gate = g_storage.as_cuda_slice::<f32>()?.slice(g_start..g_end);
    let beta = beta_storage
        .as_cuda_slice::<f32>()?
        .slice(beta_start..beta_end);
    let (metadata_storage, metadata_layout) = sequences.cumulative.storage_and_layout();
    let Storage::Cuda(metadata_storage) = &*metadata_storage else {
        candle_core::bail!("packed metadata must be CUDA")
    };
    let (start, end) = offsets(metadata_layout)?;
    let cumulative = metadata_storage.as_cuda_slice::<u32>()?.slice(start..end);
    let device = q_storage.device().clone();
    let output = unsafe { device.alloc::<bf16>(batch * heads * length * value_dim)? };

    let reuse = cuda_vstore_short_selected(
        &device,
        1,
        heads,
        value_dim,
        sequences.total as u128,
        *sequences.lengths.iter().max().unwrap_or(&0),
    )?;
    let short_columns = if reuse { 32 } else { 8 };
    let name = if reuse {
        "qwen35_vstoreshort_1_bf16"
    } else {
        "qwen35_vstore_packed_delta_bfout"
    };
    let function = device.get_or_load_custom_func(name, "sys1_qwen35", CUDA_PTX)?;
    let config = LaunchConfig {
        grid_dim: (
            (sequences.lengths.len() * heads) as u32,
            value_dim.div_ceil(short_columns) as u32,
            1,
        ),
        block_dim: (128, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut launch = function.builder();
    launch.arg(&query);
    launch.arg(&key);
    launch.arg(&value);
    launch.arg(&gate);
    launch.arg(&beta);
    launch.arg(&output);
    launch.arg(&cumulative);
    let length = length as i32;
    let heads_arg = heads as i32;
    let key_heads_arg = key_heads as i32;
    let value_dim = value_dim as i32;
    launch.arg(&length);
    launch.arg(&heads_arg);
    launch.arg(&key_heads_arg);
    launch.arg(&value_dim);
    unsafe { launch.launch(config) }.w()?;
    let output = Tensor::from_storage(
        Storage::Cuda(CudaStorage::wrap_cuda_slice(output, device)),
        (batch, length as usize, heads, value_dim as usize),
        BackpropOp::none(),
        false,
    );
    Ok(output)
}

pub(crate) fn cuda_vstore_gdn_varlen(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    g: &Tensor,
    beta: &Tensor,
    sequences: &PackedSequences,
    profiling: bool,
) -> Result<Tensor> {
    if option_env!("SYS1_CUDA_COMPUTE_CAP") != Some("120") {
        candle_core::bail!("vstore full GDN probe requires SM120");
    }

    if [q, k, g, beta].iter().any(|x| x.dtype() != DType::F32) || v.dtype() != DType::BF16 {
        candle_core::bail!("chunked recurrence requires FP32 inputs");
    }

    if q.dim(0)? != 1 || q.dim(1)? != sequences.total {
        candle_core::bail!("segmented C64 shape mismatch")
    }
    if sequences.c64_chunk_count == 0 {
        return cuda_vstore_delta_packed(q, k, v, g, beta, sequences, true);
    }
    if [
        k,
        v,
        g,
        beta,
        &sequences.cumulative,
        &sequences.c64_chunks,
        &sequences.c64_long_ids,
        &sequences.c64_short_ids,
    ]
    .iter()
    .any(|t| !q.device().same_device(t.device()))
    {
        candle_core::bail!("segmented C64 device mismatch")
    }
    let (batch, length, key_heads, key_dim) = q.dims4()?;
    let (value_batch, value_length, heads, value_dim) = v.dims4()?;
    if key_dim != 128
        || key_heads == 0
        || heads == 0
        || heads % key_heads != 0
        || value_dim == 0
        || k.dims() != q.dims()
        || value_batch != batch
        || value_length != length
        || g.dims() != [batch, length, heads]
        || beta.dims() != [batch, length, heads]
    {
        candle_core::bail!("Unsupported Qwen3.5 CUDA delta-rule shape")
    }
    let q = q.contiguous()?;
    let k = k.contiguous()?;
    let v = v.contiguous()?;
    let g = g.contiguous()?;
    let beta = beta.contiguous()?;
    let (q_storage, q_layout) = q.storage_and_layout();
    let (k_storage, k_layout) = k.storage_and_layout();
    let (v_storage, v_layout) = v.storage_and_layout();
    let (g_storage, g_layout) = g.storage_and_layout();
    let (beta_storage, beta_layout) = beta.storage_and_layout();
    let (
        Storage::Cuda(q_storage),
        Storage::Cuda(k_storage),
        Storage::Cuda(v_storage),
        Storage::Cuda(g_storage),
        Storage::Cuda(beta_storage),
    ) = (
        &*q_storage,
        &*k_storage,
        &*v_storage,
        &*g_storage,
        &*beta_storage,
    )
    else {
        candle_core::bail!("Qwen3.5 delta rule requires CUDA storage")
    };
    let offsets = |layout: &candle_core::Layout| {
        layout
            .contiguous_offsets()
            .ok_or_else(|| candle_core::Error::Msg("Qwen3.5 delta input is not contiguous".into()))
    };
    let (q_start, q_end) = offsets(q_layout)?;
    let (k_start, k_end) = offsets(k_layout)?;
    let (v_start, v_end) = offsets(v_layout)?;
    let (g_start, g_end) = offsets(g_layout)?;
    let (beta_start, beta_end) = offsets(beta_layout)?;
    let query = q_storage.as_cuda_slice::<f32>()?.slice(q_start..q_end);
    let key = k_storage.as_cuda_slice::<f32>()?.slice(k_start..k_end);
    let value = v_storage.as_cuda_slice::<bf16>()?.slice(v_start..v_end);
    let gate = g_storage.as_cuda_slice::<f32>()?.slice(g_start..g_end);
    let beta = beta_storage
        .as_cuda_slice::<f32>()?
        .slice(beta_start..beta_end);
    let (cumulative_storage, cumulative_layout) = sequences.cumulative.storage_and_layout();
    let Storage::Cuda(cumulative_storage) = &*cumulative_storage else {
        candle_core::bail!("CUDA metadata required")
    };
    let (start, end) = offsets(cumulative_layout)?;
    let cumulative = cumulative_storage.as_cuda_slice::<u32>()?.slice(start..end);
    let (chunk_offsets_storage, chunk_offsets_layout) = sequences.c64_chunks.storage_and_layout();
    let Storage::Cuda(chunk_offsets_storage) = &*chunk_offsets_storage else {
        candle_core::bail!("CUDA metadata required")
    };
    let (start, end) = offsets(chunk_offsets_layout)?;
    let chunk_offsets = chunk_offsets_storage
        .as_cuda_slice::<u32>()?
        .slice(start..end);
    let (long_ids_storage, long_ids_layout) = sequences.c64_long_ids.storage_and_layout();
    let Storage::Cuda(long_ids_storage) = &*long_ids_storage else {
        candle_core::bail!("CUDA metadata required")
    };
    let (start, end) = offsets(long_ids_layout)?;
    let long_ids = long_ids_storage.as_cuda_slice::<u32>()?.slice(start..end);
    let (short_ids_storage, short_ids_layout) = sequences.c64_short_ids.storage_and_layout();
    let Storage::Cuda(short_ids_storage) = &*short_ids_storage else {
        candle_core::bail!("CUDA metadata required")
    };
    let (start, end) = offsets(short_ids_layout)?;
    let short_ids = short_ids_storage.as_cuda_slice::<u32>()?.slice(start..end);
    let device = q_storage.device().clone();
    let sm_count = cuda_c64_sm_count(&device)?;
    let wide_blocks = sequences
        .c64_long_count
        .checked_mul(heads)
        .and_then(|n| n.checked_mul(value_dim.div_ceil(64)))
        .ok_or_else(|| candle_core::Error::Msg("wide recurrent grid overflow".into()))?;
    let use_wide = cuda_c64_scheduled_geometry(sequences, heads, value_dim, sm_count)?;
    let _ = wide_blocks;
    let (tile_columns, state_rows, carry_shared) = if use_wide {
        (
            64usize,
            64usize,
            if option_env!("SYS1_CUDA_COMPUTE_CAP") == Some("120") {
                98336u32
            } else {
                82976u32
            },
        )
    } else {
        (32usize, 128usize, 58400u32)
    };
    if use_wide {
        static NOTED: std::sync::OnceLock<()> = std::sync::OnceLock::new();
        NOTED.get_or_init(|| {
            tracing::info!(
                value_tile = 64,
                threads = 1024,
                shared_bytes = carry_shared,
                sm_count,
                "Using wider recurrent value tiles selected by cached CTA schedule cost"
            )
        });
    }

    if ![32usize, 64].contains(&tile_columns)
        || batch == 0
        || length == 0
        || value_dim > 128
        || [
            q.elem_count(),
            k.elem_count(),
            v.elem_count(),
            g.elem_count(),
        ]
        .iter()
        .any(|n| *n > i32::MAX as usize)
    {
        candle_core::bail!("chunk probe requires nonempty F32 K128,V<=128,small indices");
    }
    let instances = heads
        .checked_mul(sequences.c64_chunk_count)
        .ok_or_else(|| candle_core::Error::Msg("chunk count overflow".into()))?;
    let w_count = instances
        .checked_mul(64 * 128)
        .ok_or_else(|| candle_core::Error::Msg("chunk W overflow".into()))?;
    let u_count = instances
        .checked_mul(64 * value_dim)
        .ok_or_else(|| candle_core::Error::Msg("chunk U overflow".into()))?;
    let c_count = instances
        .checked_mul(64 * 64)
        .ok_or_else(|| candle_core::Error::Msg("chunk C overflow".into()))?;
    let d_count = instances
        .checked_mul(128)
        .ok_or_else(|| candle_core::Error::Msg("decay count overflow".into()))?;
    if [w_count, u_count, c_count, d_count]
        .iter()
        .any(|n| *n > i32::MAX as usize)
    {
        candle_core::bail!("chunk scratch index overflow");
    }
    let result = (|| -> Result<Tensor> {
        let mut slot = CHUNK64_SCRATCH
            .lock()
            .map_err(|_| candle_core::Error::Msg("poisoned chunk scratch".into()))?;
        let stream = device.cuda_stream();
        if slot
            .as_ref()
            .is_none_or(|old| !std::sync::Arc::ptr_eq(&old.stream, &stream))
        {
            *slot = Some(ChunkScratch {
                stream,
                w: None,
                u: None,
                c: None,
                decays: None,
                snapshots: None,
            });
        }
        let workspace = slot
            .as_mut()
            .ok_or_else(|| candle_core::Error::Msg("missing chunk scratch".into()))?;
        let ChunkScratch {
            w,
            u,
            c,
            decays,
            snapshots,
            ..
        } = workspace;
        if profiling {
            device.cuda_stream().synchronize().w()?;
        }
        let mut stage_started = Instant::now();
        let w = grow_chunk_buffer(w, &device, w_count)?;
        let u = grow_chunk_buffer(u, &device, u_count)?;
        let c = grow_chunk_buffer(c, &device, c_count)?;
        let decays = grow_chunk_buffer(decays, &device, d_count)?;
        let output_count = length
            .checked_mul(heads)
            .and_then(|n| n.checked_mul(value_dim))
            .ok_or_else(|| candle_core::Error::Msg("segmented C64 output overflow".into()))?;
        if output_count > i32::MAX as usize {
            candle_core::bail!("padded output index overflow");
        }
        let guarded = option_env!("SYS1_CUDA_COMPUTE_CAP") == Some("120") && !use_wide;
        let snapshot_count = if guarded { instances } else { 1usize };
        if snapshot_count > i32::MAX as usize {
            candle_core::bail!("snapshot index overflow");
        }
        let snapshots = grow_chunk_buffer(snapshots, &device, snapshot_count)?;
        let mut output = unsafe { device.alloc::<bf16>(output_count)? };
        if profiling {
            println!(
                "C64_CARRY_WORKSPACE retained_bytes={} output_bytes={} single_entry_per_process=true owned_stream_identity=true output_never_reused=true amortized_warm_operator_not_serving=true",
                (w.len() + u.len() + c.len() + decays.len() + snapshots.len()) * 4,
                output_count * 2
            );
        }
        if profiling {
            device.cuda_stream().synchronize().w()?;
            println!(
                "C64_CARRY_STAGE part=scratch batch={batch} length={length} columns={tile_columns} ms={} diagnostic_not_serving=true",
                stage_started.elapsed().as_secs_f64() * 1000.
            );
            stage_started = Instant::now();
        }
        let sequence_count = sequences.lengths.len() as i32;
        let length_arg = length as i32;
        let heads_arg = heads as i32;
        let key_heads_arg = key_heads as i32;
        let value_dim_arg = value_dim as i32;
        {
            let function = device.get_or_load_custom_func(
                if guarded {
                    "qwen35_c64vstoreguard_varlen_prepare_f32"
                } else {
                    if option_env!("SYS1_CUDA_COMPUTE_CAP") == Some("120") {
                        "qwen35_c64vstore_varlen_prepare_f32"
                    } else {
                        "qwen35_c64slice_varlen_prepare_f32"
                    }
                },
                "sys1_chunk_delta",
                CUDA_PTX,
            )?;
            function.set_attribute(candle_core::cuda_backend::cudarc::driver::sys::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,49664).w()?;
            let mut launch = function.builder();
            launch.arg(&query);
            launch.arg(&key);
            launch.arg(&value);
            launch.arg(&gate);
            launch.arg(&beta);
            launch.arg(&mut *w);
            launch.arg(&mut *u);
            launch.arg(&mut *c);
            launch.arg(&mut *decays);
            if guarded {
                launch.arg(&mut *snapshots);
            }
            launch.arg(&cumulative);
            launch.arg(&chunk_offsets);
            launch.arg(&sequence_count);
            launch.arg(&length_arg);
            launch.arg(&heads_arg);
            launch.arg(&key_heads_arg);
            launch.arg(&value_dim_arg);
            unsafe {
                launch.launch(LaunchConfig {
                    grid_dim: (instances as u32, 1, 1),
                    block_dim: (512, 1, 1),
                    shared_mem_bytes: 49664,
                })
            }
            .w()?;
        }
        if profiling {
            device.cuda_stream().synchronize().w()?;
            println!(
                "C64_CARRY_STAGE part=prepare batch={batch} length={length} columns={tile_columns} ms={} diagnostic_not_serving=true",
                stage_started.elapsed().as_secs_f64() * 1000.
            );
            stage_started = Instant::now();
        }
        {
            if sequences.c64_short_count > 0 {
                let short_tokens = sequences
                    .lengths
                    .iter()
                    .filter(|&&n| n < 512)
                    .map(|&n| n as u128)
                    .sum();
                let longest_short = sequences
                    .lengths
                    .iter()
                    .filter(|&&n| n < 512)
                    .copied()
                    .max()
                    .unwrap_or(0);
                let reuse = cuda_vstore_short_selected(
                    &device,
                    0,
                    heads,
                    value_dim,
                    short_tokens,
                    longest_short,
                )?;
                let short_columns = if reuse { 32 } else { 8 };
                let name = if reuse {
                    "qwen35_vstoreshort_0_bf16"
                } else {
                    "qwen35_vstore_short_bf16"
                };
                let function = device.get_or_load_custom_func(name, "sys1_c64v_short", CUDA_PTX)?;
                let mut launch = function.builder();
                launch
                    .arg(&query)
                    .arg(&key)
                    .arg(&value)
                    .arg(&gate)
                    .arg(&beta)
                    .arg(&mut output)
                    .arg(&cumulative)
                    .arg(&short_ids)
                    .arg(&length_arg)
                    .arg(&heads_arg)
                    .arg(&key_heads_arg)
                    .arg(&value_dim_arg);
                unsafe {
                    launch.launch(LaunchConfig {
                        grid_dim: (
                            (sequences.c64_short_count * heads) as u32,
                            value_dim.div_ceil(short_columns) as u32,
                            1,
                        ),
                        block_dim: (128, 1, 1),
                        shared_mem_bytes: 0,
                    })
                }
                .w()?;
            }
            let name = if use_wide {
                if option_env!("SYS1_CUDA_COMPUTE_CAP") == Some("120") {
                    "qwen35_c64full01_n64_varlen_f32"
                } else {
                    "qwen35_c64wide_varlen_f32"
                }
            } else if option_env!("SYS1_CUDA_COMPUTE_CAP") == Some("120") {
                "qwen35_c64zguard_n32_varlen32_f32"
            } else {
                "qwen35_c64bfout_varlen32_f32"
            };
            let function = device.get_or_load_custom_func(name, "sys1_chunk_delta", CUDA_PTX)?;
            if profiling {
                println!(
                    "C64_CARRY_RESOURCE kernel={name} registers={} local_bytes={} shared_bytes={} diagnostic_not_serving=true",
                    function.num_regs().w()?,
                    function.local_size_bytes().w()?,
                    function.shared_size_bytes().w()?
                );
            }
            function.set_attribute(candle_core::cuda_backend::cudarc::driver::sys::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,carry_shared as i32).w()?;
            let mut launch = function.builder();
            launch.arg(&query);
            launch.arg(&key);
            launch.arg(&gate);
            launch.arg(&*w);
            launch.arg(&*u);
            launch.arg(&*c);
            launch.arg(&*decays);
            launch.arg(&mut output);
            launch.arg(&mut *snapshots);
            launch.arg(&cumulative);
            launch.arg(&chunk_offsets);
            launch.arg(&long_ids);
            launch.arg(&length_arg);
            launch.arg(&heads_arg);
            launch.arg(&key_heads_arg);
            launch.arg(&value_dim_arg);
            unsafe {
                launch.launch(LaunchConfig {
                    grid_dim: (
                        (sequences.c64_long_count * heads) as u32,
                        value_dim.div_ceil(tile_columns) as u32,
                        1,
                    ),
                    block_dim: (((state_rows / 16) * (tile_columns / 8) * 32) as u32, 1, 1),
                    shared_mem_bytes: carry_shared,
                })
            }
            .w()?;
        }
        if profiling {
            device.cuda_stream().synchronize().w()?;
            println!(
                "C64_CARRY_STAGE part=state batch={batch} length={length} columns={tile_columns} ms={} diagnostic_not_serving=true",
                stage_started.elapsed().as_secs_f64() * 1000.
            );
        }

        let output = Tensor::from_storage(
            Storage::Cuda(CudaStorage::wrap_cuda_slice(output, device.clone())),
            (1, length, heads, value_dim),
            BackpropOp::none(),
            false,
        );
        Ok(output)
    })();
    if result.is_err()
        && let Ok(mut slot) = CHUNK64_SCRATCH.lock()
    {
        slot.take();
    }
    result
}

pub(crate) fn cuda_vstore_gdn_uniform(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    g: &Tensor,
    beta: &Tensor,
    tile_columns: usize,
    profiling: bool,
) -> Result<Tensor> {
    if option_env!("SYS1_CUDA_COMPUTE_CAP") != Some("120") {
        candle_core::bail!("vstore full GDN probe requires SM120");
    }

    if [q, k, g, beta].iter().any(|x| x.dtype() != DType::F32) || v.dtype() != DType::BF16 {
        candle_core::bail!("chunked recurrence requires FP32 inputs");
    }
    let state_rows = 128usize;
    let carry_shared = 58400u32;
    if tile_columns != 32 {
        candle_core::bail!("C64 state requires N32");
    }
    let (batch, length, key_heads, key_dim) = q.dims4()?;
    let (value_batch, value_length, heads, value_dim) = v.dims4()?;
    if key_dim != 128
        || key_heads == 0
        || heads == 0
        || heads % key_heads != 0
        || value_dim == 0
        || k.dims() != q.dims()
        || value_batch != batch
        || value_length != length
        || g.dims() != [batch, length, heads]
        || beta.dims() != [batch, length, heads]
    {
        candle_core::bail!("Unsupported Qwen3.5 CUDA delta-rule shape")
    }
    let q = q.contiguous()?;
    let k = k.contiguous()?;
    let v = v.contiguous()?;
    let g = g.contiguous()?;
    let beta = beta.contiguous()?;
    let (q_storage, q_layout) = q.storage_and_layout();
    let (k_storage, k_layout) = k.storage_and_layout();
    let (v_storage, v_layout) = v.storage_and_layout();
    let (g_storage, g_layout) = g.storage_and_layout();
    let (beta_storage, beta_layout) = beta.storage_and_layout();
    let (
        Storage::Cuda(q_storage),
        Storage::Cuda(k_storage),
        Storage::Cuda(v_storage),
        Storage::Cuda(g_storage),
        Storage::Cuda(beta_storage),
    ) = (
        &*q_storage,
        &*k_storage,
        &*v_storage,
        &*g_storage,
        &*beta_storage,
    )
    else {
        candle_core::bail!("Qwen3.5 delta rule requires CUDA storage")
    };
    let offsets = |layout: &candle_core::Layout| {
        layout
            .contiguous_offsets()
            .ok_or_else(|| candle_core::Error::Msg("Qwen3.5 delta input is not contiguous".into()))
    };
    let (q_start, q_end) = offsets(q_layout)?;
    let (k_start, k_end) = offsets(k_layout)?;
    let (v_start, v_end) = offsets(v_layout)?;
    let (g_start, g_end) = offsets(g_layout)?;
    let (beta_start, beta_end) = offsets(beta_layout)?;
    let query = q_storage.as_cuda_slice::<f32>()?.slice(q_start..q_end);
    let key = k_storage.as_cuda_slice::<f32>()?.slice(k_start..k_end);
    let value = v_storage.as_cuda_slice::<bf16>()?.slice(v_start..v_end);
    let gate = g_storage.as_cuda_slice::<f32>()?.slice(g_start..g_end);
    let beta = beta_storage
        .as_cuda_slice::<f32>()?
        .slice(beta_start..beta_end);
    let device = q_storage.device().clone();

    if tile_columns != 32
        || batch == 0
        || length == 0
        || value_dim > 128
        || [
            q.elem_count(),
            k.elem_count(),
            v.elem_count(),
            g.elem_count(),
        ]
        .iter()
        .any(|n| *n > i32::MAX as usize)
    {
        candle_core::bail!("chunk probe requires nonempty F32 K128,V<=128,small indices");
    }
    let instances = batch
        .checked_mul(heads)
        .and_then(|x| x.checked_mul(length.div_ceil(64)))
        .ok_or_else(|| candle_core::Error::Msg("chunk count overflow".into()))?;
    let w_count = instances
        .checked_mul(64 * 128)
        .ok_or_else(|| candle_core::Error::Msg("chunk W overflow".into()))?;
    let u_count = instances
        .checked_mul(64 * value_dim)
        .ok_or_else(|| candle_core::Error::Msg("chunk U overflow".into()))?;
    let c_count = instances
        .checked_mul(64 * 64)
        .ok_or_else(|| candle_core::Error::Msg("chunk C overflow".into()))?;
    let d_count = instances
        .checked_mul(128)
        .ok_or_else(|| candle_core::Error::Msg("decay count overflow".into()))?;
    if [w_count, u_count, c_count, d_count]
        .iter()
        .any(|n| *n > i32::MAX as usize)
    {
        candle_core::bail!("chunk scratch index overflow");
    }
    let result = (|| -> Result<Tensor> {
        let mut slot = CHUNK64_SCRATCH
            .lock()
            .map_err(|_| candle_core::Error::Msg("poisoned chunk scratch".into()))?;
        let stream = device.cuda_stream();
        if slot
            .as_ref()
            .is_none_or(|old| !std::sync::Arc::ptr_eq(&old.stream, &stream))
        {
            *slot = Some(ChunkScratch {
                stream,
                w: None,
                u: None,
                c: None,
                decays: None,
                snapshots: None,
            });
        }
        let workspace = slot
            .as_mut()
            .ok_or_else(|| candle_core::Error::Msg("missing chunk scratch".into()))?;
        let ChunkScratch {
            w,
            u,
            c,
            decays,
            snapshots,
            ..
        } = workspace;
        if profiling {
            device.cuda_stream().synchronize().w()?;
        }
        let mut stage_started = Instant::now();
        let w = grow_chunk_buffer(w, &device, w_count)?;
        let u = grow_chunk_buffer(u, &device, u_count)?;
        let c = grow_chunk_buffer(c, &device, c_count)?;
        let decays = grow_chunk_buffer(decays, &device, d_count)?;
        let output_count = batch
            .checked_mul(length)
            .and_then(|n| n.checked_mul(heads))
            .and_then(|n| n.checked_mul(value_dim))
            .ok_or_else(|| candle_core::Error::Msg("register-carry C64 output overflow".into()))?;
        if output_count > i32::MAX as usize {
            candle_core::bail!("padded output index overflow");
        }
        let guarded = option_env!("SYS1_CUDA_COMPUTE_CAP") == Some("120");
        let snapshot_count = if guarded { instances } else { 1usize };
        if snapshot_count > i32::MAX as usize {
            candle_core::bail!("snapshot index overflow");
        }
        let snapshots = grow_chunk_buffer(snapshots, &device, snapshot_count)?;
        let mut output = unsafe { device.alloc::<bf16>(output_count)? };
        if profiling {
            println!(
                "C64_FLAGS_WORKSPACE retained_bytes={} output_bytes={} single_entry_per_process=true owned_stream_identity=true output_never_reused=true amortized_warm_operator_not_serving=true",
                (w.len() + u.len() + c.len() + decays.len() + snapshots.len()) * 4,
                output_count * 2
            );
        }
        if profiling {
            device.cuda_stream().synchronize().w()?;
            println!(
                "C64_FLAGS_STAGE part=scratch batch={batch} length={length} columns={tile_columns} ms={} diagnostic_not_serving=true",
                stage_started.elapsed().as_secs_f64() * 1000.
            );
            stage_started = Instant::now();
        }
        let length_arg = length as i32;
        let heads_arg = heads as i32;
        let key_heads_arg = key_heads as i32;
        let value_dim_arg = value_dim as i32;
        {
            let function = device.get_or_load_custom_func(
                if guarded {
                    "qwen35_c64vstoreguard_uniform_prepare_f32"
                } else {
                    if option_env!("SYS1_CUDA_COMPUTE_CAP") == Some("120") {
                        "qwen35_c64vstore_uniform_prepare_f32"
                    } else {
                        "qwen35_c64slice_uniform_prepare_f32"
                    }
                },
                "sys1_chunk_delta",
                CUDA_PTX,
            )?;
            function.set_attribute(candle_core::cuda_backend::cudarc::driver::sys::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,49664).w()?;
            let mut launch = function.builder();
            launch.arg(&query);
            launch.arg(&key);
            launch.arg(&value);
            launch.arg(&gate);
            launch.arg(&beta);
            launch.arg(&mut *w);
            launch.arg(&mut *u);
            launch.arg(&mut *c);
            launch.arg(&mut *decays);
            if guarded {
                launch.arg(&mut *snapshots);
            }
            launch.arg(&length_arg);
            launch.arg(&heads_arg);
            launch.arg(&key_heads_arg);
            launch.arg(&value_dim_arg);
            unsafe {
                launch.launch(LaunchConfig {
                    grid_dim: (instances as u32, 1, 1),
                    block_dim: (512, 1, 1),
                    shared_mem_bytes: 49664,
                })
            }
            .w()?;
        }
        if profiling {
            device.cuda_stream().synchronize().w()?;
            println!(
                "C64_FLAGS_STAGE part=prepare batch={batch} length={length} columns={tile_columns} ms={} diagnostic_not_serving=true",
                stage_started.elapsed().as_secs_f64() * 1000.
            );
            stage_started = Instant::now();
        }
        {
            let name = if option_env!("SYS1_CUDA_COMPUTE_CAP") == Some("120") {
                "qwen35_c64zguard_n32_uniform32_f32"
            } else {
                "qwen35_c64bfout_uniform32_f32"
            };
            let function = device.get_or_load_custom_func(name, "sys1_chunk_delta", CUDA_PTX)?;
            if profiling {
                println!(
                    "C64_FLAGS_RESOURCE kernel={name} registers={} local_bytes={} shared_bytes={} diagnostic_not_serving=true",
                    function.num_regs().w()?,
                    function.local_size_bytes().w()?,
                    function.shared_size_bytes().w()?
                );
            }
            function.set_attribute(candle_core::cuda_backend::cudarc::driver::sys::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,carry_shared as i32).w()?;
            let mut launch = function.builder();
            launch.arg(&query);
            launch.arg(&key);
            launch.arg(&gate);
            launch.arg(&*w);
            launch.arg(&*u);
            launch.arg(&*c);
            launch.arg(&*decays);
            launch.arg(&mut output);
            launch.arg(&mut *snapshots);
            launch.arg(&length_arg);
            launch.arg(&heads_arg);
            launch.arg(&key_heads_arg);
            launch.arg(&value_dim_arg);
            unsafe {
                launch.launch(LaunchConfig {
                    grid_dim: (
                        (batch * heads) as u32,
                        value_dim.div_ceil(tile_columns) as u32,
                        1,
                    ),
                    block_dim: (((state_rows / 16) * (tile_columns / 8) * 32) as u32, 1, 1),
                    shared_mem_bytes: carry_shared,
                })
            }
            .w()?;
        }
        if profiling {
            device.cuda_stream().synchronize().w()?;
            println!(
                "C64_FLAGS_STAGE part=state batch={batch} length={length} columns={tile_columns} ms={} diagnostic_not_serving=true",
                stage_started.elapsed().as_secs_f64() * 1000.
            );
        }

        let output = Tensor::from_storage(
            Storage::Cuda(CudaStorage::wrap_cuda_slice(output, device.clone())),
            (batch, length, heads, value_dim),
            BackpropOp::none(),
            false,
        );
        Ok(output)
    })();
    if result.is_err()
        && let Ok(mut slot) = CHUNK64_SCRATCH.lock()
    {
        slot.take();
    }
    result
}

use super::*;

pub(crate) fn cuda_l2_128(input: &Tensor, scale: f32) -> Result<Tensor> {
    if input.dtype() != DType::F32 || input.dim(D::Minus1)? != 128 {
        candle_core::bail!("Qwen L2 requires CUDA FP32 width128");
    }
    let input = input.contiguous()?;
    let (storage, layout) = input.storage_and_layout();
    let Storage::Cuda(storage) = &*storage else {
        candle_core::bail!("Qwen L2 requires CUDA storage");
    };
    let (start, end) = layout
        .contiguous_offsets()
        .ok_or_else(|| candle_core::Error::Msg("Qwen L2 noncontiguous input".into()))?;
    let source = storage.as_cuda_slice::<f32>()?.slice(start..end);
    let device = storage.device().clone();
    let count = input.elem_count();
    let rows = u32::try_from(count / 128)
        .map_err(|_| candle_core::Error::Msg("Qwen L2 row count overflow".into()))?;
    if rows == 0 {
        candle_core::bail!("Qwen L2 empty input");
    }
    let output = unsafe { device.alloc::<f32>(count)? };
    let function = device.get_or_load_custom_func("qwen35_l2_128_f32", "sys1_qwen35", CUDA_PTX)?;
    let mut launch = function.builder();
    launch.arg(&source);
    launch.arg(&output);
    launch.arg(&rows);
    launch.arg(&scale);
    unsafe {
        launch.launch(LaunchConfig {
            grid_dim: (rows.div_ceil(4), 1, 1),
            block_dim: (128, 1, 1),
            shared_mem_bytes: 0,
        })
    }
    .w()?;
    Ok(Tensor::from_storage(
        Storage::Cuda(CudaStorage::wrap_cuda_slice(output, device)),
        input.shape().clone(),
        BackpropOp::none(),
        false,
    ))
}

pub(crate) fn cuda_rms_norm_delta(
    input: &Tensor,
    delta_weight: &Tensor,
    epsilon: f64,
) -> Result<Tensor> {
    let columns = input.dim(D::Minus1)?;
    cuda_rms_norm_delta_with_block_size(
        input,
        delta_weight,
        epsilon,
        columns.next_power_of_two().clamp(32, 1024) as u32,
    )
}

pub(crate) fn cuda_rms_norm_delta_with_block_size(
    input: &Tensor,
    delta_weight: &Tensor,
    epsilon: f64,
    block_size: u32,
) -> Result<Tensor> {
    let columns = input.dim(D::Minus1)?;
    let vector = columns == 4096 && block_size == 1024;
    if columns == 0
        || delta_weight.dims() != [columns]
        || delta_weight.dtype() != DType::BF16
        || !(32..=1024).contains(&block_size)
        || !block_size.is_power_of_two()
    {
        candle_core::bail!("Unsupported Qwen3.5 CUDA RMSNorm shape or dtype")
    }
    let input = input.contiguous()?;
    let delta_weight = delta_weight.contiguous()?;
    let (input_storage, input_layout) = input.storage_and_layout();
    let (weight_storage, weight_layout) = delta_weight.storage_and_layout();
    let (Storage::Cuda(input_storage), Storage::Cuda(weight_storage)) =
        (&*input_storage, &*weight_storage)
    else {
        candle_core::bail!("Qwen3.5 RMSNorm requires CUDA storage")
    };
    let (input_start, input_end) = input_layout
        .contiguous_offsets()
        .ok_or_else(|| candle_core::Error::Msg("Qwen3.5 RMSNorm input is not contiguous".into()))?;
    let (weight_start, weight_end) = weight_layout.contiguous_offsets().ok_or_else(|| {
        candle_core::Error::Msg("Qwen3.5 RMSNorm weight is not contiguous".into())
    })?;
    let source = input_storage
        .as_cuda_slice::<bf16>()?
        .slice(input_start..input_end);
    let weight = weight_storage
        .as_cuda_slice::<bf16>()?
        .slice(weight_start..weight_end);
    let device = input_storage.device().clone();
    let count = input.elem_count();
    let output = unsafe { device.alloc::<bf16>(count)? };
    let function = device.get_or_load_custom_func(
        if vector {
            "qwen35_hidden_rms_vec4_bf16"
        } else {
            "qwen35_rms_norm_delta_bf16"
        },
        "sys1_qwen35",
        CUDA_PTX,
    )?;
    let config = LaunchConfig {
        grid_dim: ((count / columns) as u32, 1, 1),
        block_dim: (if vector { 256 } else { block_size }, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut launch = function.builder();
    launch.arg(&source);
    launch.arg(&weight);
    launch.arg(&output);
    let columns = columns as i32;
    let epsilon = epsilon as f32;
    launch.arg(&columns);
    launch.arg(&epsilon);
    unsafe { launch.launch(config) }.w()?;
    Ok(Tensor::from_storage(
        Storage::Cuda(CudaStorage::wrap_cuda_slice(output, device)),
        input.shape().clone(),
        BackpropOp::none(),
        false,
    ))
}

pub(crate) fn cuda_residual_rms_norm_delta(
    input: &Tensor,
    mixed: &Tensor,
    delta_weight: &Tensor,
    epsilon: f64,
) -> Result<(Tensor, Tensor)> {
    let columns = input.dim(D::Minus1)?;
    let vector = columns == 4096;
    if columns == 0
        || input.dims() != mixed.dims()
        || input.dtype() != DType::BF16
        || mixed.dtype() != DType::BF16
        || delta_weight.dims() != [columns]
        || delta_weight.dtype() != DType::BF16
    {
        candle_core::bail!("Unsupported Qwen3.5 CUDA fused residual/RMSNorm shape or dtype")
    }
    let input = input.contiguous()?;
    let mixed = mixed.contiguous()?;
    let delta_weight = delta_weight.contiguous()?;
    let (input_storage, input_layout) = input.storage_and_layout();
    let (mixed_storage, mixed_layout) = mixed.storage_and_layout();
    let (weight_storage, weight_layout) = delta_weight.storage_and_layout();
    let (Storage::Cuda(input_storage), Storage::Cuda(mixed_storage), Storage::Cuda(weight_storage)) =
        (&*input_storage, &*mixed_storage, &*weight_storage)
    else {
        candle_core::bail!("Qwen3.5 fused residual/RMSNorm requires CUDA storage")
    };
    let slice = |layout: &candle_core::Layout| {
        layout.contiguous_offsets().ok_or_else(|| {
            candle_core::Error::Msg("Qwen3.5 fused residual/RMSNorm input is not contiguous".into())
        })
    };
    let (input_start, input_end) = slice(input_layout)?;
    let (mixed_start, mixed_end) = slice(mixed_layout)?;
    let (weight_start, weight_end) = slice(weight_layout)?;
    let source = input_storage
        .as_cuda_slice::<bf16>()?
        .slice(input_start..input_end);
    let mixed_source = mixed_storage
        .as_cuda_slice::<bf16>()?
        .slice(mixed_start..mixed_end);
    let weight = weight_storage
        .as_cuda_slice::<bf16>()?
        .slice(weight_start..weight_end);
    let device = input_storage.device().clone();
    let count = input.elem_count();
    let rows = u32::try_from(count / columns).map_err(|_| {
        candle_core::Error::Msg("Qwen3.5 fused residual/RMSNorm row count overflow".into())
    })?;
    let residual = unsafe { device.alloc::<bf16>(count)? };
    let normalized = unsafe { device.alloc::<bf16>(count)? };
    let function = device.get_or_load_custom_func(
        if vector {
            "qwen35_hidden_residual_rms_vec4_bf16"
        } else {
            "qwen35_residual_rms_norm_delta_bf16"
        },
        "sys1_qwen35",
        CUDA_PTX,
    )?;
    let mut launch = function.builder();
    launch.arg(&source);
    launch.arg(&mixed_source);
    launch.arg(&weight);
    launch.arg(&residual);
    launch.arg(&normalized);
    let columns = i32::try_from(columns).map_err(|_| {
        candle_core::Error::Msg("Qwen3.5 fused residual/RMSNorm width overflow".into())
    })?;
    let epsilon = epsilon as f32;
    launch.arg(&columns);
    launch.arg(&epsilon);
    unsafe {
        launch.launch(LaunchConfig {
            grid_dim: (rows, 1, 1),
            block_dim: (
                if vector {
                    256
                } else {
                    (columns as u32).next_power_of_two().clamp(32, 1024)
                },
                1,
                1,
            ),
            shared_mem_bytes: 0,
        })
    }
    .w()?;
    let wrap = |data| {
        Tensor::from_storage(
            Storage::Cuda(CudaStorage::wrap_cuda_slice(data, device.clone())),
            input.shape().clone(),
            BackpropOp::none(),
            false,
        )
    };
    Ok((wrap(residual), wrap(normalized)))
}

#[cfg(test)]
pub(crate) static FULL_QKV_MERGED_SELECTIONS: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

pub(crate) fn cuda_post_finalize_original(
    input: &Tensor,
    gate: &Tensor,
    norm: &Tensor,
    epsilon: f64,
) -> Result<Tensor> {
    let columns = input.dim(D::Minus1)?;
    if input.dtype() != DType::F32
        || gate.dtype() != DType::BF16
        || norm.dtype() != DType::BF16
        || input.dims() != gate.dims()
        || norm.dims() != [columns]
    {
        candle_core::bail!("Unsupported Qwen3.5 CUDA post-finalize shape or dtype")
    }
    let input = input.contiguous()?;
    let gate = gate.contiguous()?;
    let norm = norm.contiguous()?;
    let (input_storage, input_layout) = input.storage_and_layout();
    let (gate_storage, gate_layout) = gate.storage_and_layout();
    let (norm_storage, norm_layout) = norm.storage_and_layout();
    let (Storage::Cuda(input_storage), Storage::Cuda(gate_storage), Storage::Cuda(norm_storage)) =
        (&*input_storage, &*gate_storage, &*norm_storage)
    else {
        candle_core::bail!("Qwen3.5 post-finalize requires CUDA storage")
    };
    let (input_start, input_end) = input_layout.contiguous_offsets().ok_or_else(|| {
        candle_core::Error::Msg("Qwen3.5 post-finalize input is not contiguous".into())
    })?;
    let (gate_start, gate_end) = gate_layout.contiguous_offsets().ok_or_else(|| {
        candle_core::Error::Msg("Qwen3.5 post-finalize gate is not contiguous".into())
    })?;
    let (norm_start, norm_end) = norm_layout.contiguous_offsets().ok_or_else(|| {
        candle_core::Error::Msg("Qwen3.5 post-finalize norm is not contiguous".into())
    })?;
    let source = input_storage
        .as_cuda_slice::<f32>()?
        .slice(input_start..input_end);
    let gate = gate_storage
        .as_cuda_slice::<bf16>()?
        .slice(gate_start..gate_end);
    let norm = norm_storage
        .as_cuda_slice::<bf16>()?
        .slice(norm_start..norm_end);
    let device = input_storage.device().clone();
    let count = input.elem_count();
    let output = unsafe { device.alloc::<bf16>(count)? };
    let function =
        device.get_or_load_custom_func("qwen35_post_finalize_bf16", "sys1_qwen35", CUDA_PTX)?;
    let config = LaunchConfig {
        grid_dim: ((count / columns) as u32, 1, 1),
        block_dim: (128, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut launch = function.builder();
    launch.arg(&source);
    launch.arg(&gate);
    launch.arg(&norm);
    launch.arg(&output);
    let columns = columns as i32;
    let epsilon = epsilon as f32;
    launch.arg(&columns);
    launch.arg(&epsilon);
    unsafe { launch.launch(config) }.w()?;
    Ok(Tensor::from_storage(
        Storage::Cuda(CudaStorage::wrap_cuda_slice(output, device)),
        input.shape().clone(),
        BackpropOp::none(),
        false,
    ))
}

pub(crate) fn maybe_cpu_input_embedding(
    embedding: Embedding,
    hidden_size: usize,
) -> Result<Embedding> {
    if option_env!("SYS1_CUDA_COMPUTE_CAP") != Some("86")
        || embedding.embeddings().dtype() != DType::BF16
    {
        return Ok(embedding);
    }
    let candle_core::Device::Cuda(cuda) = embedding.embeddings().device() else {
        return Ok(embedding);
    };
    let (_, total) = cuda
        .cuda_stream()
        .context()
        .mem_get_info()
        .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
    let host_limit = std::fs::read_to_string("/sys/fs/cgroup/memory.max")
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok());
    if total <= 32usize * 1024 * 1024 * 1024
        && host_limit.is_some_and(|bytes| bytes >= 8usize * 1024 * 1024 * 1024)
    {
        let bytes = embedding.embeddings().elem_count() * 2;
        let weights = embedding
            .embeddings()
            .to_device(&candle_core::Device::Cpu)?;
        tracing::info!(
            CPU_embedding_bytes = bytes,
            GPU_total_bytes = total,
            "Qwen input embedding prepared on CPU for SM86 activation capacity"
        );
        return Ok(Embedding::new(weights, hidden_size));
    }
    Ok(embedding)
}

pub(crate) fn cuda_fused_qkv_prepare(
    mixed: &Tensor,
    key_heads: usize,
    value_heads: usize,
) -> Result<(Tensor, Tensor, Tensor)> {
    let (batch, length, channels) = mixed.dims3()?;
    if batch == 0
        || length == 0
        || key_heads == 0
        || value_heads == 0
        || !value_heads.is_multiple_of(key_heads)
        || mixed.dtype() != DType::BF16
        || mixed.elem_count() > i32::MAX as usize
        || key_heads
            .checked_mul(2)
            .and_then(|x| x.checked_add(value_heads))
            .and_then(|x| x.checked_mul(128))
            != Some(channels)
    {
        candle_core::bail!("unsupported packed QKV preparation");
    }
    let mixed = mixed.contiguous()?;
    let (storage, layout) = mixed.storage_and_layout();
    let Storage::Cuda(storage) = &*storage else {
        candle_core::bail!("packed QKV requires CUDA");
    };
    let (start, end) = layout
        .contiguous_offsets()
        .ok_or_else(|| candle_core::Error::Msg("QKV layout".into()))?;
    let source = storage.as_cuda_slice::<bf16>()?.slice(start..end);
    let device = storage.device().clone();
    let q_count = batch * length * key_heads * 128;
    let v_count = batch * length * value_heads * 128;
    let mut q = unsafe { device.alloc::<f32>(q_count)? };
    let mut k = unsafe { device.alloc::<f32>(q_count)? };
    let mut v = unsafe { device.alloc::<f32>(v_count)? };
    let rows = (q_count / 128) as u32;
    let function = device.get_or_load_custom_func(
        "qwen35_fused_qkv_prepare_bf16",
        "sys1_fused_qkv",
        CUDA_PTX,
    )?;
    let mut launch = function.builder();
    launch.arg(&source);
    launch.arg(&mut q);
    launch.arg(&mut k);
    launch.arg(&mut v);
    let kh = key_heads as i32;
    let vh = value_heads as i32;
    let scale = 128f64.powf(-0.5) as f32;
    launch.arg(&rows);
    launch.arg(&kh);
    launch.arg(&vh);
    launch.arg(&scale);
    unsafe {
        launch.launch(LaunchConfig {
            grid_dim: (rows.div_ceil(4), 1, 1),
            block_dim: (128, 1, 1),
            shared_mem_bytes: 0,
        })
    }
    .w()?;
    let wrap = |data, heads| {
        Tensor::from_storage(
            Storage::Cuda(CudaStorage::wrap_cuda_slice(data, device.clone())),
            (batch, length, heads, 128),
            BackpropOp::none(),
            false,
        )
    };
    Ok((wrap(q, key_heads), wrap(k, key_heads), wrap(v, value_heads)))
}

pub(crate) fn cuda_post_finalize_bf16(
    input: &Tensor,
    gate: &Tensor,
    norm: &Tensor,
    epsilon: f64,
) -> Result<Tensor> {
    if input.dtype() == DType::BF16 {
        cuda_post_finalize_bf16_warp(input, gate, norm, epsilon)
    } else if input.rank() == 4 && input.stride()[3] == 1 && input.elem_count() > 0 {
        cuda_post_finalize_strided(input, gate, norm, epsilon)
    } else {
        cuda_post_finalize_original(input, gate, norm, epsilon)
    }
}

pub(crate) fn cuda_post_finalize_strided(
    input: &Tensor,
    gate: &Tensor,
    norm: &Tensor,
    epsilon: f64,
) -> Result<Tensor> {
    let columns = input.dim(D::Minus1)?;
    if input.dtype() != DType::F32
        || gate.dtype() != DType::BF16
        || norm.dtype() != DType::BF16
        || input.dims() != gate.dims()
        || norm.dims() != [columns]
    {
        candle_core::bail!("Unsupported Qwen3.5 CUDA post-finalize shape or dtype")
    }
    let (batch, length, heads, width) = input.dims4()?;
    if width == 0 || batch == 0 || length == 0 || heads == 0 || input.stride()[3] != 1 {
        candle_core::bail!("unsupported strided post-finalize layout");
    }
    let columns_arg = i32::try_from(columns)
        .map_err(|_| candle_core::Error::Msg("post-finalize width overflow".into()))?;
    let length_arg =
        u64::try_from(length).map_err(|_| candle_core::Error::Msg("length overflow".into()))?;
    let heads_arg =
        u64::try_from(heads).map_err(|_| candle_core::Error::Msg("heads overflow".into()))?;
    let strides: Vec<u64> = input.stride()[..3]
        .iter()
        .map(|&s| u64::try_from(s).map_err(|_| candle_core::Error::Msg("stride overflow".into())))
        .collect::<Result<_>>()?;
    let gate = gate.contiguous()?;
    let norm = norm.contiguous()?;
    let (input_storage, input_layout) = input.storage_and_layout();
    let (gate_storage, gate_layout) = gate.storage_and_layout();
    let (norm_storage, norm_layout) = norm.storage_and_layout();
    let (Storage::Cuda(input_storage), Storage::Cuda(gate_storage), Storage::Cuda(norm_storage)) =
        (&*input_storage, &*gate_storage, &*norm_storage)
    else {
        candle_core::bail!("Qwen3.5 post-finalize requires CUDA storage")
    };
    let input_start = input_layout.start_offset();
    let span =
        input
            .dims()
            .iter()
            .zip(input.stride())
            .try_fold(1usize, |span, (&dim, &stride)| {
                dim.checked_sub(1)
                    .and_then(|n| n.checked_mul(stride))
                    .and_then(|n| span.checked_add(n))
                    .ok_or_else(|| candle_core::Error::Msg("post-finalize span overflow".into()))
            })?;
    let input_end = input_start
        .checked_add(span)
        .ok_or_else(|| candle_core::Error::Msg("post-finalize offset overflow".into()))?;
    if input_end > input_storage.as_cuda_slice::<f32>()?.len() {
        candle_core::bail!("post-finalize input span exceeds allocation");
    }
    if !std::sync::Arc::ptr_eq(
        &input_storage.device().cuda_stream(),
        &gate_storage.device().cuda_stream(),
    ) || !std::sync::Arc::ptr_eq(
        &input_storage.device().cuda_stream(),
        &norm_storage.device().cuda_stream(),
    ) {
        candle_core::bail!("post-finalize requires one CUDA stream");
    }
    let (gate_start, gate_end) = gate_layout.contiguous_offsets().ok_or_else(|| {
        candle_core::Error::Msg("Qwen3.5 post-finalize gate is not contiguous".into())
    })?;
    let (norm_start, norm_end) = norm_layout.contiguous_offsets().ok_or_else(|| {
        candle_core::Error::Msg("Qwen3.5 post-finalize norm is not contiguous".into())
    })?;
    let source = input_storage
        .as_cuda_slice::<f32>()?
        .slice(input_start..input_end);
    let gate = gate_storage
        .as_cuda_slice::<bf16>()?
        .slice(gate_start..gate_end);
    let norm = norm_storage
        .as_cuda_slice::<bf16>()?
        .slice(norm_start..norm_end);
    let device = input_storage.device().clone();
    let count = input.elem_count();
    let rows = u32::try_from(count / columns)
        .map_err(|_| candle_core::Error::Msg("post-finalize row count overflow".into()))?;
    if count > i32::MAX as usize {
        candle_core::bail!("post-finalize index overflow");
    }
    let mut output = unsafe { device.alloc::<bf16>(count)? };
    let function =
        device.get_or_load_custom_func("qwen35_post_finalize_strided", "sys1_qwen35", CUDA_PTX)?;
    let config = LaunchConfig {
        grid_dim: (rows, 1, 1),
        block_dim: (128, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut launch = function.builder();
    launch.arg(&source);
    launch.arg(&gate);
    launch.arg(&norm);
    launch.arg(&mut output);
    let columns = columns_arg;
    let epsilon = epsilon as f32;
    launch.arg(&columns);
    launch.arg(&epsilon);
    launch.arg(&length_arg);
    launch.arg(&heads_arg);
    launch.arg(&strides[0]);
    launch.arg(&strides[1]);
    launch.arg(&strides[2]);
    unsafe { launch.launch(config) }.w()?;
    Ok(Tensor::from_storage(
        Storage::Cuda(CudaStorage::wrap_cuda_slice(output, device)),
        input.shape().clone(),
        BackpropOp::none(),
        false,
    ))
}

pub(crate) fn cuda_post_finalize_bf16_input(
    input: &Tensor,
    gate: &Tensor,
    norm: &Tensor,
    epsilon: f64,
) -> Result<Tensor> {
    let columns = input.dim(D::Minus1)?;
    if input.dtype() != DType::BF16
        || gate.dtype() != DType::BF16
        || norm.dtype() != DType::BF16
        || input.dims() != gate.dims()
        || norm.dims() != [columns]
    {
        candle_core::bail!("Unsupported Qwen3.5 CUDA post-finalize shape or dtype")
    }
    let (batch, length, heads, width) = input.dims4()?;
    if width == 0 || batch == 0 || length == 0 || heads == 0 || input.stride()[3] != 1 {
        candle_core::bail!("unsupported strided post-finalize layout");
    }
    let columns_arg = i32::try_from(columns)
        .map_err(|_| candle_core::Error::Msg("post-finalize width overflow".into()))?;
    let length_arg =
        u64::try_from(length).map_err(|_| candle_core::Error::Msg("length overflow".into()))?;
    let heads_arg =
        u64::try_from(heads).map_err(|_| candle_core::Error::Msg("heads overflow".into()))?;
    let strides: Vec<u64> = input.stride()[..3]
        .iter()
        .map(|&s| u64::try_from(s).map_err(|_| candle_core::Error::Msg("stride overflow".into())))
        .collect::<Result<_>>()?;
    let gate = gate.contiguous()?;
    let norm = norm.contiguous()?;
    let (input_storage, input_layout) = input.storage_and_layout();
    let (gate_storage, gate_layout) = gate.storage_and_layout();
    let (norm_storage, norm_layout) = norm.storage_and_layout();
    let (Storage::Cuda(input_storage), Storage::Cuda(gate_storage), Storage::Cuda(norm_storage)) =
        (&*input_storage, &*gate_storage, &*norm_storage)
    else {
        candle_core::bail!("Qwen3.5 post-finalize requires CUDA storage")
    };
    let input_start = input_layout.start_offset();
    let span =
        input
            .dims()
            .iter()
            .zip(input.stride())
            .try_fold(1usize, |span, (&dim, &stride)| {
                dim.checked_sub(1)
                    .and_then(|n| n.checked_mul(stride))
                    .and_then(|n| span.checked_add(n))
                    .ok_or_else(|| candle_core::Error::Msg("post-finalize span overflow".into()))
            })?;
    let input_end = input_start
        .checked_add(span)
        .ok_or_else(|| candle_core::Error::Msg("post-finalize offset overflow".into()))?;
    if input_end > input_storage.as_cuda_slice::<bf16>()?.len() {
        candle_core::bail!("post-finalize input span exceeds allocation");
    }
    if !std::sync::Arc::ptr_eq(
        &input_storage.device().cuda_stream(),
        &gate_storage.device().cuda_stream(),
    ) || !std::sync::Arc::ptr_eq(
        &input_storage.device().cuda_stream(),
        &norm_storage.device().cuda_stream(),
    ) {
        candle_core::bail!("post-finalize requires one CUDA stream");
    }
    let (gate_start, gate_end) = gate_layout.contiguous_offsets().ok_or_else(|| {
        candle_core::Error::Msg("Qwen3.5 post-finalize gate is not contiguous".into())
    })?;
    let (norm_start, norm_end) = norm_layout.contiguous_offsets().ok_or_else(|| {
        candle_core::Error::Msg("Qwen3.5 post-finalize norm is not contiguous".into())
    })?;
    let source = input_storage
        .as_cuda_slice::<bf16>()?
        .slice(input_start..input_end);
    let gate = gate_storage
        .as_cuda_slice::<bf16>()?
        .slice(gate_start..gate_end);
    let norm = norm_storage
        .as_cuda_slice::<bf16>()?
        .slice(norm_start..norm_end);
    let device = input_storage.device().clone();
    let count = input.elem_count();
    let rows = u32::try_from(count / columns)
        .map_err(|_| candle_core::Error::Msg("post-finalize row count overflow".into()))?;
    if count > i32::MAX as usize {
        candle_core::bail!("post-finalize index overflow");
    }
    let mut output = unsafe { device.alloc::<bf16>(count)? };
    let function = device.get_or_load_custom_func(
        "qwen35_post_finalize_bf16_input",
        "sys1_qwen35",
        CUDA_PTX,
    )?;
    let config = LaunchConfig {
        grid_dim: (rows, 1, 1),
        block_dim: (128, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut launch = function.builder();
    launch.arg(&source);
    launch.arg(&gate);
    launch.arg(&norm);
    launch.arg(&mut output);
    let columns = columns_arg;
    let epsilon = epsilon as f32;
    launch.arg(&columns);
    launch.arg(&epsilon);
    launch.arg(&length_arg);
    launch.arg(&heads_arg);
    launch.arg(&strides[0]);
    launch.arg(&strides[1]);
    launch.arg(&strides[2]);
    unsafe { launch.launch(config) }.w()?;
    Ok(Tensor::from_storage(
        Storage::Cuda(CudaStorage::wrap_cuda_slice(output, device)),
        input.shape().clone(),
        BackpropOp::none(),
        false,
    ))
}

pub(crate) struct Bf16PostF32Lookup {
    pub(crate) stream: std::sync::Arc<candle_core::cuda_backend::cudarc::driver::CudaStream>,
    pub(crate) table: candle_core::cuda_backend::cudarc::driver::CudaSlice<f32>,
}

pub(crate) static BF16_POST_F32_LOOKUP: std::sync::Mutex<Option<Bf16PostF32Lookup>> =
    std::sync::Mutex::new(None);

#[cfg(test)]
pub(crate) static BF16_POST_F32_LOOKUP_CALLS: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

pub(crate) fn cuda_post_finalize_bf16_warp(
    input: &Tensor,
    gate: &Tensor,
    norm: &Tensor,
    epsilon: f64,
) -> Result<Tensor> {
    cuda_post_finalize_bf16_warp_impl(
        input,
        gate,
        norm,
        epsilon,
        option_env!("SYS1_CUDA_COMPUTE_CAP") == Some("90"),
    )
}

pub(crate) fn cuda_post_finalize_bf16_warp_impl(
    input: &Tensor,
    gate: &Tensor,
    norm: &Tensor,
    epsilon: f64,
    use_lookup: bool,
) -> Result<Tensor> {
    let columns = input.dim(D::Minus1)?;
    if columns != 128 {
        return cuda_post_finalize_bf16_input(input, gate, norm, epsilon);
    }

    let (batch, length, heads, width) = input.dims4()?;
    let merged_gate = use_lookup
        && heads == 32
        && width == 128
        && gate.dims() == [batch, length, 4096]
        && gate.stride() == [length * 12288, 12288, 1];
    if input.dtype() != DType::BF16
        || gate.dtype() != DType::BF16
        || norm.dtype() != DType::BF16
        || (input.dims() != gate.dims() && !merged_gate)
        || norm.dims() != [columns]
    {
        candle_core::bail!("Unsupported Qwen3.5 CUDA post-finalize shape or dtype")
    }
    if width == 0 || batch == 0 || length == 0 || heads == 0 || input.stride()[3] != 1 {
        candle_core::bail!("unsupported strided post-finalize layout");
    }
    let columns_arg = i32::try_from(columns)
        .map_err(|_| candle_core::Error::Msg("post-finalize width overflow".into()))?;
    let length_arg =
        u64::try_from(length).map_err(|_| candle_core::Error::Msg("length overflow".into()))?;
    let heads_arg =
        u64::try_from(heads).map_err(|_| candle_core::Error::Msg("heads overflow".into()))?;
    let strides: Vec<u64> = input.stride()[..3]
        .iter()
        .map(|&s| u64::try_from(s).map_err(|_| candle_core::Error::Msg("stride overflow".into())))
        .collect::<Result<_>>()?;
    let gate = if merged_gate {
        gate.clone()
    } else {
        gate.contiguous()?
    };
    let norm = norm.contiguous()?;
    let (input_storage, input_layout) = input.storage_and_layout();
    let (gate_storage, gate_layout) = gate.storage_and_layout();
    let (norm_storage, norm_layout) = norm.storage_and_layout();
    let (Storage::Cuda(input_storage), Storage::Cuda(gate_storage), Storage::Cuda(norm_storage)) =
        (&*input_storage, &*gate_storage, &*norm_storage)
    else {
        candle_core::bail!("Qwen3.5 post-finalize requires CUDA storage")
    };
    let input_start = input_layout.start_offset();
    let span =
        input
            .dims()
            .iter()
            .zip(input.stride())
            .try_fold(1usize, |span, (&dim, &stride)| {
                dim.checked_sub(1)
                    .and_then(|n| n.checked_mul(stride))
                    .and_then(|n| span.checked_add(n))
                    .ok_or_else(|| candle_core::Error::Msg("post-finalize span overflow".into()))
            })?;
    let input_end = input_start
        .checked_add(span)
        .ok_or_else(|| candle_core::Error::Msg("post-finalize offset overflow".into()))?;
    if input_end > input_storage.as_cuda_slice::<bf16>()?.len() {
        candle_core::bail!("post-finalize input span exceeds allocation");
    }
    if !std::sync::Arc::ptr_eq(
        &input_storage.device().cuda_stream(),
        &gate_storage.device().cuda_stream(),
    ) || !std::sync::Arc::ptr_eq(
        &input_storage.device().cuda_stream(),
        &norm_storage.device().cuda_stream(),
    ) {
        candle_core::bail!("post-finalize requires one CUDA stream");
    }
    let (gate_start, gate_end) = if merged_gate {
        let start = gate_layout.start_offset();
        let end = (batch - 1)
            .checked_mul(length * 12288)
            .and_then(|n| n.checked_add((length - 1) * 12288))
            .and_then(|n| n.checked_add(4096))
            .and_then(|n| start.checked_add(n))
            .ok_or_else(|| candle_core::Error::Msg("merged gate span overflow".into()))?;
        if end > gate_storage.as_cuda_slice::<bf16>()?.len() {
            candle_core::bail!("merged gate span exceeds allocation")
        }
        (start, end)
    } else {
        gate_layout.contiguous_offsets().ok_or_else(|| {
            candle_core::Error::Msg("Qwen3.5 post-finalize gate is not contiguous".into())
        })?
    };
    let (norm_start, norm_end) = norm_layout.contiguous_offsets().ok_or_else(|| {
        candle_core::Error::Msg("Qwen3.5 post-finalize norm is not contiguous".into())
    })?;
    let source = input_storage
        .as_cuda_slice::<bf16>()?
        .slice(input_start..input_end);
    let gate = gate_storage
        .as_cuda_slice::<bf16>()?
        .slice(gate_start..gate_end);
    let norm = norm_storage
        .as_cuda_slice::<bf16>()?
        .slice(norm_start..norm_end);
    let device = input_storage.device().clone();
    let count = input.elem_count();
    let rows = u32::try_from(count / columns)
        .map_err(|_| candle_core::Error::Msg("post-finalize row count overflow".into()))?;
    if count > i32::MAX as usize {
        candle_core::bail!("post-finalize index overflow");
    }
    let mut output = unsafe { device.alloc::<bf16>(count)? };
    let mut lookup_slot = if use_lookup {
        Some(
            BF16_POST_F32_LOOKUP
                .lock()
                .map_err(|_| candle_core::Error::Msg("poisoned post SiLU lookup".into()))?,
        )
    } else {
        None
    };
    if let Some(slot) = lookup_slot.as_mut() {
        let stream = device.cuda_stream();
        if slot
            .as_ref()
            .is_none_or(|entry| !std::sync::Arc::ptr_eq(&entry.stream, &stream))
        {
            let mut table = unsafe { device.alloc::<f32>(65536)? };
            let init = device.get_or_load_custom_func(
                "qwen35_build_bfpost_silu_f32_lookup",
                "sys1_qwen35",
                CUDA_PTX,
            )?;
            let mut builder = init.builder();
            builder.arg(&mut table);
            unsafe { builder.launch(LaunchConfig::for_num_elems(65536)) }.w()?;
            **slot = Some(Bf16PostF32Lookup { stream, table });
        }
    }
    let function_name = if merged_gate {
        "hopper_qkvz_post_lookup_strided_gate"
    } else if use_lookup {
        "qwen35_bfpostwarp128_lookup"
    } else {
        "qwen35_bfpostwarp128"
    };
    let function = device.get_or_load_custom_func(function_name, "sys1_qwen35", CUDA_PTX)?;
    let config = LaunchConfig {
        grid_dim: (rows.div_ceil(8), 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut launch = function.builder();
    launch.arg(&source);
    launch.arg(&gate);
    if let Some(slot) = lookup_slot.as_ref() {
        let entry = slot
            .as_ref()
            .ok_or_else(|| candle_core::Error::Msg("missing post SiLU lookup".into()))?;
        launch.arg(&entry.table);
    }
    launch.arg(&norm);
    launch.arg(&mut output);
    let columns = columns_arg;
    let epsilon = epsilon as f32;
    launch.arg(&columns);
    launch.arg(&epsilon);
    launch.arg(&length_arg);
    launch.arg(&heads_arg);
    launch.arg(&strides[0]);
    launch.arg(&strides[1]);
    launch.arg(&strides[2]);
    let rows_arg = u64::from(rows);
    launch.arg(&rows_arg);
    let gate_strides = [(length as u64) * 12288, 12288u64, 128u64];
    if merged_gate {
        launch
            .arg(&gate_strides[0])
            .arg(&gate_strides[1])
            .arg(&gate_strides[2]);
    }
    unsafe { launch.launch(config) }.w()?;
    #[cfg(test)]
    if use_lookup {
        BF16_POST_F32_LOOKUP_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
    #[cfg(test)]
    if merged_gate {
        qkvz::STRIDED_POST.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
    Ok(Tensor::from_storage(
        Storage::Cuda(CudaStorage::wrap_cuda_slice(output, device)),
        input.shape().clone(),
        BackpropOp::none(),
        false,
    ))
}

pub(crate) fn cuda_gate_gather_small(
    input: &Tensor,
    linear: &Linear,
    tensor_core: bool,
    simt: SimtProjection,
    sequences: &PackedSequences,
) -> Result<Tensor> {
    let (batch, length, inner) = input.dims3()?;
    if batch != 1
        || length != sequences.total
        || !input
            .device()
            .same_device(sequences.gate_long_starts.device())
        || !input
            .device()
            .same_device(sequences.gate_long_ends.device())
        || !input
            .device()
            .same_device(sequences.gate_short_rows.device())
        || linear.bias().is_some()
    {
        candle_core::bail!("segmented narrow-gate shape/metadata mismatch");
    }
    let small_tile = 32usize;

    let (columns, weight_inner) = linear.weight().dims2()?;
    let rows = batch
        .checked_mul(length)
        .ok_or_else(|| candle_core::Error::Msg("Gate projection row count overflow".into()))?;
    let count = rows
        .checked_mul(columns)
        .ok_or_else(|| candle_core::Error::Msg("Gate projection output size overflow".into()))?;
    if input.dtype() != DType::BF16
        || linear.weight().dtype() != DType::BF16
        || (linear.bias().is_some()
            && !tensor_core
            && !matches!(
                simt,
                SimtProjection::Compensated | SimtProjection::TiledCompensated
            ))
        || inner != weight_inner
        || rows == 0
        || columns == 0
        || inner == 0
        || !input.device().same_device(linear.weight().device())
        || input.elem_count() > i32::MAX as usize
        || linear.weight().elem_count() > i32::MAX as usize
        || count > i32::MAX as usize
    {
        candle_core::bail!("Unsupported stable BF16 gate projection shape, dtype or device")
    }
    if let Some(bias) = linear.bias()
        && (bias.dtype() != DType::BF16
            || bias.dims() != [columns]
            || !input.device().same_device(bias.device()))
    {
        candle_core::bail!("Stable BF16 projection requires matching bias dtype, shape and device")
    }
    let input = input.contiguous()?;
    let weight = linear.weight().contiguous()?;
    let bias = linear.bias().map(Tensor::contiguous).transpose()?;
    let bias_storage = bias.as_ref().map(Tensor::storage_and_layout);
    let (xs, xl) = input.storage_and_layout();
    let (ws, wl) = weight.storage_and_layout();
    let (Storage::Cuda(xs), Storage::Cuda(ws)) = (&*xs, &*ws) else {
        candle_core::bail!("Stable gate projection requires CUDA storage")
    };
    let offsets = |layout: &candle_core::Layout| {
        layout.contiguous_offsets().ok_or_else(|| {
            candle_core::Error::Msg("Gate projection storage is not contiguous".into())
        })
    };
    let (xstart, xend) = offsets(xl)?;
    let (wstart, wend) = offsets(wl)?;
    let x = xs.as_cuda_slice::<bf16>()?.slice(xstart..xend);
    let w = ws.as_cuda_slice::<bf16>()?.slice(wstart..wend);
    let b = if let Some((storage, layout)) = &bias_storage {
        let Storage::Cuda(storage) = &**storage else {
            candle_core::bail!("Stable BF16 projection requires CUDA bias storage")
        };
        let (start, end) = offsets(layout)?;
        Some(storage.as_cuda_slice::<bf16>()?.slice(start..end))
    } else {
        None
    };
    let (starts_storage, starts_layout) = sequences.gate_long_starts.storage_and_layout();
    let (ends_storage, ends_layout) = sequences.gate_long_ends.storage_and_layout();
    let (ids_storage, ids_layout) = sequences.gate_short_rows.storage_and_layout();
    let (Storage::Cuda(starts_storage), Storage::Cuda(ends_storage), Storage::Cuda(ids_storage)) =
        (&*starts_storage, &*ends_storage, &*ids_storage)
    else {
        candle_core::bail!("CUDA gate metadata required")
    };
    let (ss, se) = offsets(starts_layout)?;
    let (es, ee) = offsets(ends_layout)?;
    let (is, ie) = offsets(ids_layout)?;
    let tile_starts = starts_storage.as_cuda_slice::<u32>()?.slice(ss..se);
    let tile_ends = ends_storage.as_cuda_slice::<u32>()?.slice(es..ee);
    let short_ids = ids_storage.as_cuda_slice::<u32>()?.slice(is..ie);
    let device = xs.device().clone();
    let mut output = unsafe { device.alloc::<bf16>(count)? };

    let tile_rows = 32usize;

    let kernel = if !tensor_core && matches!(simt, SimtProjection::TiledCompensated) {
        if b.is_some() {
            "qwen35_biased_tiled_compensated_projection_bf16"
        } else {
            "qwen35_tiled_compensated_projection_bf16"
        }
    } else if b.is_some() && !tensor_core {
        "qwen35_biased_compensated_projection_bf16"
    } else if b.is_some() && tile_rows == 128 {
        "qwen35_biased_projection_rows128_bf16"
    } else if b.is_some() && tile_rows == 32 {
        "qwen35_biased_projection_rows32_bf16"
    } else if b.is_some() {
        "qwen35_biased_projection_bf16"
    } else if tensor_core {
        "qwen35_gate_nativevec_packed_bf16"
    } else {
        match simt {
            SimtProjection::Compensated => "qwen35_gate_projection_bf16",
            SimtProjection::TiledCompensated => "qwen35_tiled_compensated_projection_bf16",
            SimtProjection::GuardedDouble => "qwen35_guarded_double_projection_bf16",
            SimtProjection::GuardedTensorDouble => "qwen35_guarded_tensor_double_projection_bf16",
            SimtProjection::GuardedTensorDoubleRows16 => {
                "qwen35_guarded_tensor_double_rows16_projection_bf16"
            }
        }
    };
    let function = device.get_or_load_custom_func(kernel, "sys1_qwen35", CUDA_PTX)?;
    let config = LaunchConfig {
        grid_dim: if tensor_core {
            (
                sequences.gate_long_count as u32,
                columns.div_ceil(64) as u32,
                1,
            )
        } else if matches!(simt, SimtProjection::TiledCompensated) {
            (rows.div_ceil(4) as u32, columns.div_ceil(4) as u32, 1)
        } else if simt.is_tensor_double() {
            let row_tile = if matches!(simt, SimtProjection::GuardedTensorDoubleRows16) {
                16
            } else {
                8
            };
            (
                rows.div_ceil(row_tile) as u32,
                columns.div_ceil(32) as u32,
                1,
            )
        } else {
            (rows as u32, columns.div_ceil(8) as u32, 1)
        },
        block_dim: (
            if tensor_core {
                tile_rows.min(32) as u32 * 8
            } else if simt.is_tensor_double() || matches!(simt, SimtProjection::TiledCompensated) {
                128
            } else {
                256
            },
            1,
            1,
        ),
        shared_mem_bytes: 0,
    };
    let mut launch = function.builder();
    launch.arg(&x);
    launch.arg(&w);
    if let Some(bias) = &b {
        launch.arg(bias);
    }
    launch.arg(&mut output);
    let rows = rows as i32;
    if tensor_core || simt.is_tensor_double() || matches!(simt, SimtProjection::TiledCompensated) {
        launch.arg(&rows);
    }
    let inner = inner as i32;
    let columns_arg = columns as i32;
    let radius_coefficient = if !tensor_core
        && !matches!(
            simt,
            SimtProjection::Compensated | SimtProjection::TiledCompensated
        ) {
        Some(guarded_double_radius_coefficient_impl(
            inner as usize,
            simt.is_tensor_double(),
        ))
    } else {
        None
    };
    launch.arg(&inner);
    launch.arg(&columns_arg);
    launch.arg(&tile_starts);
    launch.arg(&tile_ends);
    if let Some(coefficient) = &radius_coefficient {
        launch.arg(coefficient);
    }
    if sequences.gate_long_count > 0 {
        unsafe { launch.launch(config) }.w()?;
    }
    if sequences.gate_short_count > 0 {
        let function = device.get_or_load_custom_func(
            "qwen35_gate_gathervector_small_32",
            "sys1_qwen35",
            CUDA_PTX,
        )?;
        let mut launch = function.builder();
        let short_count = sequences.gate_short_count as i32;
        launch
            .arg(&x)
            .arg(&w)
            .arg(&mut output)
            .arg(&short_count)
            .arg(&inner)
            .arg(&columns_arg)
            .arg(&short_ids);
        unsafe {
            launch.launch(LaunchConfig {
                grid_dim: (
                    sequences.gate_short_count.div_ceil(small_tile) as u32,
                    columns.div_ceil(64) as u32,
                    1,
                ),
                block_dim: ((small_tile * 8) as u32, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .w()?;
    }

    Ok(Tensor::from_storage(
        Storage::Cuda(CudaStorage::wrap_cuda_slice(output, device)),
        (batch, length, columns),
        BackpropOp::none(),
        false,
    ))
}

pub(crate) fn cuda_short_value_reuse_selected(
    device: &candle_core::CudaDevice,
    kind: usize,
    heads: usize,
    value_dim: usize,
    tokens: u128,
    longest: usize,
) -> Result<bool> {
    if option_env!("SYS1_CUDA_COMPUTE_CAP") != Some("120") || longest == 0 {
        return Ok(false);
    }
    let sm_count = cuda_c64_sm_count(device)?;
    let blocks = (heads as u128) * (value_dim.div_ceil(32) as u128);
    if blocks * tokens < (sm_count as u128) * 2 * (longest as u128) {
        return Ok(false);
    }
    type Context = std::sync::Arc<candle_core::cuda_backend::cudarc::driver::CudaContext>;
    static CACHE: std::sync::Mutex<Option<(Context, [usize; 3])>> = std::sync::Mutex::new(None);
    let context = device.cuda_stream().context().clone();
    let mut cache = CACHE
        .lock()
        .map_err(|_| candle_core::Error::Msg("poisoned short value residency".into()))?;
    if !cache
        .as_ref()
        .is_some_and(|(old, _)| std::sync::Arc::ptr_eq(old, &context))
    {
        *cache = Some((context, [0; 3]));
    }
    let (_, resident) = cache.as_mut().unwrap();
    if resident[kind] == 0 {
        let name = [
            "qwen35_shortvalues_0_bf16",
            "qwen35_shortvalues_1_bf16",
            "qwen35_shortvalues_2_bf16",
        ][kind];
        let f = device.get_or_load_custom_func(name, "sys1_short_value_reuse", CUDA_PTX)?;
        resident[kind] = f
            .occupancy_max_active_blocks_per_multiprocessor(128, 0, None)
            .w()? as usize;
        if resident[kind] == 0 {
            candle_core::bail!("zero short value residency");
        }
        tracing::info!(
            sm_count,
            kind,
            resident_blocks_per_sm = resident[kind],
            value_columns_per_lane_group = 4,
            "Queried exact short recurrence operand-reuse residency"
        );
    }

    Ok(blocks * tokens
        >= (sm_count as u128) * (resident[kind].div_ceil(2) as u128) * (longest as u128))
}

pub(crate) fn cuda_vstore_short_selected(
    device: &candle_core::CudaDevice,
    kind: usize,
    heads: usize,
    value_dim: usize,
    tokens: u128,
    longest: usize,
) -> Result<bool> {
    if option_env!("SYS1_CUDA_COMPUTE_CAP") != Some("120") || longest == 0 {
        return Ok(false);
    }
    let sm_count = cuda_c64_sm_count(device)?;
    let blocks = (heads as u128) * (value_dim.div_ceil(32) as u128);
    if blocks * tokens < (sm_count as u128) * 2 * (longest as u128) {
        return Ok(false);
    }
    type Context = std::sync::Arc<candle_core::cuda_backend::cudarc::driver::CudaContext>;
    static CACHE: std::sync::Mutex<Option<(Context, [usize; 3])>> = std::sync::Mutex::new(None);
    let context = device.cuda_stream().context().clone();
    let mut cache = CACHE
        .lock()
        .map_err(|_| candle_core::Error::Msg("poisoned short value residency".into()))?;
    if !cache
        .as_ref()
        .is_some_and(|(old, _)| std::sync::Arc::ptr_eq(old, &context))
    {
        *cache = Some((context, [0; 3]));
    }
    let (_, resident) = cache.as_mut().unwrap();
    if resident[kind] == 0 {
        let name = [
            "qwen35_vstoreshort_0_bf16",
            "qwen35_vstoreshort_1_bf16",
            "qwen35_vstoreshort_2_bf16",
        ][kind];
        let f = device.get_or_load_custom_func(name, "sys1_short_value_reuse", CUDA_PTX)?;
        resident[kind] = f
            .occupancy_max_active_blocks_per_multiprocessor(128, 0, None)
            .w()? as usize;
        if resident[kind] == 0 {
            candle_core::bail!("zero short value residency");
        }
        tracing::info!(
            sm_count,
            kind,
            resident_blocks_per_sm = resident[kind],
            value_columns_per_lane_group = 4,
            "Queried exact short recurrence operand-reuse residency"
        );
    }

    Ok(blocks * tokens
        >= (sm_count as u128) * (resident[kind].div_ceil(2) as u128) * (longest as u128))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::architectures::qwen35_text::tests::supported_rotary_test_dtypes;

    #[test]
    fn right_sized_rmsnorm_launch_is_bitwise_equal_to_legacy() -> Result<()> {
        let device = test_cuda_device()?;
        for columns in [16, 32, 64, 128, 256, 512, 1024, 2560, 5120] {
            let input = Tensor::from_vec(
                (0..2 * 7 * columns)
                    .map(|i| ((i * 17 % 1009) as f32 - 504.) * 0.003)
                    .collect::<Vec<_>>(),
                (2, 7, columns),
                &device,
            )?
            .to_dtype(DType::BF16)?;
            let weight = Tensor::from_vec(
                (0..columns)
                    .map(|i| ((i % 37) as f32 - 18.) * 0.002)
                    .collect::<Vec<_>>(),
                columns,
                &device,
            )?
            .to_dtype(DType::BF16)?;
            let bits = |tensor: Tensor| -> Result<Vec<u32>> {
                Ok(tensor
                    .to_dtype(DType::F32)?
                    .flatten_all()?
                    .to_vec1::<f32>()?
                    .into_iter()
                    .map(f32::to_bits)
                    .collect())
            };
            assert_eq!(
                bits(cuda_rms_norm_delta(&input, &weight, 1e-6)?)?,
                bits(cuda_rms_norm_delta_with_block_size(
                    &input, &weight, 1e-6, 1024
                )?)?,
                "RMSNorm launch changes values for width {columns}",
            );
        }
        Ok(())
    }

    #[test]
    fn fused_residual_rmsnorm_is_bitwise_equal_to_separate_kernels() -> Result<()> {
        let device = test_cuda_device()?;

        if !supported_rotary_test_dtypes(&device)?.contains(&DType::BF16) {
            return Ok(());
        }
        for (rows, columns) in [(1, 64), (7, 4096), (33, 2560)] {
            let values = |offset: usize| {
                (0..rows * columns)
                    .map(|i| ((i * 17 + offset) % 1009) as f32 * 0.003 - 1.5)
                    .collect::<Vec<_>>()
            };
            let input =
                Tensor::from_vec(values(0), (1, rows, columns), &device)?.to_dtype(DType::BF16)?;
            let mixed = Tensor::from_vec(values(503), (1, rows, columns), &device)?
                .to_dtype(DType::BF16)?;
            let weight = Tensor::from_vec(
                (0..columns)
                    .map(|i| ((i % 37) as f32 - 18.) * 0.002)
                    .collect(),
                columns,
                &device,
            )?
            .to_dtype(DType::BF16)?;
            let expected_residual = (&input + &mixed)?;
            let expected_normalized = cuda_rms_norm_delta(&expected_residual, &weight, 1e-6)?;
            let (actual_residual, actual_normalized) =
                cuda_residual_rms_norm_delta(&input, &mixed, &weight, 1e-6)?;
            let bits = |tensor: Tensor| -> Result<Vec<u32>> {
                Ok(tensor
                    .to_dtype(DType::F32)?
                    .flatten_all()?
                    .to_vec1::<f32>()?
                    .into_iter()
                    .map(f32::to_bits)
                    .collect())
            };
            assert_eq!(
                bits(actual_residual)?,
                bits(expected_residual)?,
                "fused residual differs at rows={rows}, columns={columns}"
            );
            assert_eq!(
                bits(actual_normalized)?,
                bits(expected_normalized)?,
                "fused RMSNorm differs at rows={rows}, columns={columns}"
            );
        }
        Ok(())
    }

    #[test]
    fn post_finalize_cuda_matches_tensor_reference() {
        let shape = (2, 3, 4, 128);
        let count = shape.0 * shape.1 * shape.2 * shape.3;
        let cpu = candle_core::Device::Cpu;
        let input = Tensor::from_vec(
            (0..count)
                .map(|index| ((index * 17 % 101) as f32 - 50.) * 0.003)
                .collect::<Vec<_>>(),
            shape,
            &cpu,
        )
        .unwrap();
        let gate = Tensor::from_vec(
            (0..count)
                .map(|index| ((index * 11 % 67) as f32 - 33.) * 0.02)
                .collect::<Vec<_>>(),
            shape,
            &cpu,
        )
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap();
        let norm = Tensor::from_vec(
            (0..128)
                .map(|index| 0.8 + (index % 13) as f32 * 0.01)
                .collect::<Vec<_>>(),
            128,
            &cpu,
        )
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap();
        let epsilon = 1e-6;
        let reference = gated_norm_reference(&input, &gate, &norm, epsilon, DType::BF16)
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let cuda = test_cuda_device().unwrap();
        let actual = cuda_post_finalize_bf16(
            &input.to_device(&cuda).unwrap(),
            &gate.to_device(&cuda).unwrap(),
            &norm.to_device(&cuda).unwrap(),
            epsilon,
        )
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();

        let rtol = 1.6e-2f32;
        let atol = 1e-5f32;
        for (index, (expected, actual)) in reference.iter().zip(actual).enumerate() {
            let error = (expected - actual).abs();
            let allowed = atol + rtol * expected.abs();
            assert!(
                error <= allowed,
                "Post-finalize mismatch at {index}: expected {expected}, got {actual}, error {error} > {allowed}"
            );
        }
    }
}

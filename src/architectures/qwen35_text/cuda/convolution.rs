use super::*;

pub(crate) fn cuda_conv_silu(input: &Tensor, weights: &Tensor) -> Result<Tensor> {
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
            let function =
                device.get_or_load_custom_func("qwen35_conv_bf16", "sys1_qwen35", CUDA_PTX)?;
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
            let function =
                device.get_or_load_custom_func("qwen35_conv_f32", "sys1_qwen35", CUDA_PTX)?;
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

pub(crate) fn cuda_conv_silu_packed_sequences(
    input: &Tensor,
    weights: &Tensor,
    sequences: &PackedSequences,
) -> Result<Tensor> {
    let (batch, length, channels) = input.dims3()?;
    if batch != 1
        || length != sequences.total
        || length
            .checked_mul(channels)
            .is_none_or(|n| n > i32::MAX as usize)
    {
        candle_core::bail!("packed convolution shape/index overflow")
    }
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
    let (starts_storage, starts_layout) = sequences.starts.storage_and_layout();
    let Storage::Cuda(starts_storage) = &*starts_storage else {
        candle_core::bail!("packed metadata must be CUDA")
    };
    let (start, end) = starts_layout
        .contiguous_offsets()
        .ok_or_else(|| candle_core::Error::Msg("noncontiguous starts".into()))?;
    let starts = starts_storage.as_cuda_slice::<u32>()?.slice(start..end);
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
            let function = device.get_or_load_custom_func(
                "qwen35_packed_conv_bf16",
                "sys1_qwen35",
                CUDA_PTX,
            )?;
            let mut launch = function.builder();
            launch.arg(&source);
            launch.arg(&filters);
            launch.arg(&output);
            launch.arg(&starts);
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
            let function = device.get_or_load_custom_func(
                "qwen35_packed_conv_f32",
                "sys1_qwen35",
                CUDA_PTX,
            )?;
            let mut launch = function.builder();
            launch.arg(&source);
            launch.arg(&filters);
            launch.arg(&output);
            launch.arg(&starts);
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

pub(crate) fn cuda_conv_qkv(
    input: &Tensor,
    weights: &Tensor,
    key_heads: usize,
    value_heads: usize,
    sequences: Option<&PackedSequences>,
) -> Result<(Tensor, Tensor, Tensor)> {
    Ok(cuda_conv_qkv_impl(input, weights, key_heads, value_heads, sequences, false)?.0)
}

pub(crate) fn cuda_conv_qkv_prepared(
    input: &Tensor,
    weights: &Tensor,
    key_heads: usize,
    value_heads: usize,
    sequences: &PackedSequences,
) -> Result<((Tensor, Tensor, Tensor), Option<gated_delta::PreparedQv>)> {
    cuda_conv_qkv_impl(
        input,
        weights,
        key_heads,
        value_heads,
        Some(sequences),
        true,
    )
}

pub(crate) fn cuda_conv_qkv_impl(
    input: &Tensor,
    weights: &Tensor,
    key_heads: usize,
    value_heads: usize,
    sequences: Option<&PackedSequences>,
    prepare_qv: bool,
) -> Result<((Tensor, Tensor, Tensor), Option<gated_delta::PreparedQv>)> {
    let (batch, length, channels) = input.dims3()?;
    let (wc, groups, kernel) = weights.dims3()?;
    if batch == 0
        || length == 0
        || key_heads == 0
        || value_heads == 0
        || !value_heads.is_multiple_of(key_heads)
        || input.dtype() != DType::BF16
        || weights.dtype() != DType::BF16
        || !input.device().same_device(weights.device())
        || input.elem_count() > i32::MAX as usize
        || wc != channels
        || groups != 1
        || kernel == 0
        || kernel > i32::MAX as usize
        || key_heads
            .checked_mul(2)
            .and_then(|n| n.checked_add(value_heads))
            .and_then(|n| n.checked_mul(128))
            != Some(channels)
    {
        candle_core::bail!("unsupported fused convolution/QKV shape");
    }
    if let Some(s) = sequences
        && (batch != 1 || length != s.total || !input.device().same_device(s.starts.device()))
    {
        candle_core::bail!("fused convolution packed boundary mismatch");
    }
    let input = input.contiguous()?;
    let weights = weights.contiguous()?;
    let (xs, xl) = input.storage_and_layout();
    let (ws, wl) = weights.storage_and_layout();
    let (Storage::Cuda(xs), Storage::Cuda(ws)) = (&*xs, &*ws) else {
        candle_core::bail!("fused convolution requires CUDA");
    };
    let (x0, x1) = xl
        .contiguous_offsets()
        .ok_or_else(|| candle_core::Error::Msg("convolution input offsets".into()))?;
    let (w0, w1) = wl
        .contiguous_offsets()
        .ok_or_else(|| candle_core::Error::Msg("convolution weight offsets".into()))?;
    let source = xs.as_cuda_slice::<bf16>()?.slice(x0..x1);
    let filters = ws.as_cuda_slice::<bf16>()?.slice(w0..w1);
    let device = xs.device().clone();
    let q_count = batch * length * key_heads * 128;
    let v_count = batch * length * value_heads * 128;
    let mut q = unsafe { device.alloc::<f32>(q_count)? };
    let mut k = unsafe { device.alloc::<f32>(q_count)? };
    let mut v = unsafe { device.alloc::<f32>(v_count)? };
    let rows = (q_count / 128) as u32;
    let kh = key_heads as i32;
    let vh = value_heads as i32;
    let scale = 128f64.powf(-0.5) as f32;
    let length_arg = length as i32;
    let kernel_arg = kernel as i32;

    let use_lookup = option_env!("SYS1_CUDA_COMPUTE_CAP") == Some("90")
        && kernel == 4
        && key_heads == 16
        && value_heads == 32
        && x0 % 4 == 0
        && w0 % 8 == 0;
    let mut prepared = if prepare_qv && use_lookup {
        if let Some(sequences) = sequences {
            gated_delta::prepare_qv(input.device(), &sequences.lengths)?
        } else {
            None
        }
    } else {
        None
    };
    let name = if prepared.is_some() {
        "qwen35_convcast_qkv_packed"
    } else {
        match (use_lookup, sequences.is_some()) {
            (true, true) => "qwen35_conv_lookup_packed",
            (true, false) => "qwen35_conv_lookup_uniform",
            (false, true) => "qwen35_conv_qkv_packed",
            (false, false) => "qwen35_conv_qkv_uniform",
        }
    };
    let function = device.get_or_load_custom_func(name, "sys1_conv_qkv", CUDA_PTX)?;
    let config = LaunchConfig {
        grid_dim: (rows.div_ceil(4), 1, 1),
        block_dim: (128, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut run =
        |table: Option<&candle_core::cuda_backend::cudarc::driver::CudaSlice<bf16>>| -> Result<()> {
            let prepared_limit = prepared.as_ref().map_or(0., gated_delta::PreparedQv::limit);
            let mut launch = function.builder();
            launch.arg(&source);
            launch.arg(&filters);
            launch.arg(&mut q);
            launch.arg(&mut k);
            launch.arg(&mut v);
            macro_rules! tail {
                () => {{
                    launch.arg(&rows);
                    launch.arg(&kh);
                    launch.arg(&vh);
                    launch.arg(&scale);
                    launch.arg(&length_arg);
                    launch.arg(&kernel_arg);
                    if let Some(table) = table {
                        launch.arg(table);
                    }
                    if let Some(ticket) = prepared.as_mut() {
                        let (qhalf, khalf, vhalf, _, bad) = ticket.outputs();
                        launch
                            .arg(qhalf)
                            .arg(khalf)
                            .arg(vhalf)
                            .arg(&prepared_limit)
                            .arg(bad);
                    }
                }};
            }

            if let Some(s) = sequences {
                let (ss, sl) = s.starts.storage_and_layout();
                let Storage::Cuda(ss) = &*ss else {
                    candle_core::bail!("starts require CUDA");
                };
                let (s0, s1) = sl
                    .contiguous_offsets()
                    .ok_or_else(|| candle_core::Error::Msg("start offsets".into()))?;
                let starts = ss.as_cuda_slice::<u32>()?.slice(s0..s1);
                launch.arg(&starts);
                tail!();
                unsafe { launch.launch(config) }.w()?;
            } else {
                tail!();
                unsafe { launch.launch(config) }.w()?;
            }
            Ok(())
        };
    if use_lookup {
        #[cfg(test)]
        CONV_LOOKUP_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        crate::kernels::cuda::activation::with_sigmoid_lookup(&device, |table| run(Some(table)))?;
    } else {
        run(None)?;
    }
    let wrap = |data, heads| {
        Tensor::from_storage(
            Storage::Cuda(CudaStorage::wrap_cuda_slice(data, device.clone())),
            (batch, length, heads, 128),
            BackpropOp::none(),
            false,
        )
    };
    let query = wrap(q, key_heads);
    let key = wrap(k, key_heads);
    let value = wrap(v, value_heads);
    if let Some(ticket) = prepared.as_mut() {
        ticket.bind(&query, &key, &value);
        #[cfg(test)]
        CONV_PREPARED_QV_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
    Ok(((query, key, value), prepared))
}

pub(crate) fn cuda_conv_vstore(
    input: &Tensor,
    weights: &Tensor,
    key_heads: usize,
    value_heads: usize,
    sequences: Option<&PackedSequences>,
) -> Result<(Tensor, Tensor, Tensor)> {
    let (batch, length, channels) = input.dims3()?;
    let (wc, groups, kernel) = weights.dims3()?;
    if batch == 0
        || length == 0
        || key_heads == 0
        || value_heads == 0
        || !value_heads.is_multiple_of(key_heads)
        || input.dtype() != DType::BF16
        || weights.dtype() != DType::BF16
        || !input.device().same_device(weights.device())
        || input.elem_count() > i32::MAX as usize
        || wc != channels
        || groups != 1
        || kernel == 0
        || kernel > i32::MAX as usize
        || key_heads
            .checked_mul(2)
            .and_then(|n| n.checked_add(value_heads))
            .and_then(|n| n.checked_mul(128))
            != Some(channels)
    {
        candle_core::bail!("unsupported fused convolution/QKV shape");
    }
    if let Some(s) = sequences
        && (batch != 1 || length != s.total || !input.device().same_device(s.starts.device()))
    {
        candle_core::bail!("fused convolution packed boundary mismatch");
    }
    let input = input.contiguous()?;
    let weights = weights.contiguous()?;
    let (xs, xl) = input.storage_and_layout();
    let (ws, wl) = weights.storage_and_layout();
    let (Storage::Cuda(xs), Storage::Cuda(ws)) = (&*xs, &*ws) else {
        candle_core::bail!("fused convolution requires CUDA");
    };
    let (x0, x1) = xl
        .contiguous_offsets()
        .ok_or_else(|| candle_core::Error::Msg("convolution input offsets".into()))?;
    let (w0, w1) = wl
        .contiguous_offsets()
        .ok_or_else(|| candle_core::Error::Msg("convolution weight offsets".into()))?;
    let source = xs.as_cuda_slice::<bf16>()?.slice(x0..x1);
    let filters = ws.as_cuda_slice::<bf16>()?.slice(w0..w1);
    let device = xs.device().clone();
    let q_count = batch * length * key_heads * 128;
    let v_count = batch * length * value_heads * 128;
    let mut q = unsafe { device.alloc::<f32>(q_count)? };
    let mut k = unsafe { device.alloc::<f32>(q_count)? };
    let mut v = unsafe { device.alloc::<bf16>(v_count)? };
    let rows = (q_count / 128) as u32;
    let kh = key_heads as i32;
    let vh = value_heads as i32;
    let scale = 128f64.powf(-0.5) as f32;
    let length_arg = length as i32;
    let kernel_arg = kernel as i32;
    let name = if sequences.is_some() {
        "qwen35_conv_vstore_packed"
    } else {
        "qwen35_conv_vstore_uniform"
    };
    let function = device.get_or_load_custom_func(name, "sys1_conv_qkv", CUDA_PTX)?;
    let mut launch = function.builder();
    launch.arg(&source);
    launch.arg(&filters);
    launch.arg(&mut q);
    launch.arg(&mut k);
    launch.arg(&mut v);
    let config = LaunchConfig {
        grid_dim: (rows.div_ceil(4), 1, 1),
        block_dim: (128, 1, 1),
        shared_mem_bytes: 0,
    };
    macro_rules! tail {
        () => {{
            launch.arg(&rows);
            launch.arg(&kh);
            launch.arg(&vh);
            launch.arg(&scale);
            launch.arg(&length_arg);
            launch.arg(&kernel_arg);
        }};
    }
    if let Some(s) = sequences {
        let (ss, sl) = s.starts.storage_and_layout();
        let Storage::Cuda(ss) = &*ss else {
            candle_core::bail!("starts require CUDA");
        };
        let (s0, s1) = sl
            .contiguous_offsets()
            .ok_or_else(|| candle_core::Error::Msg("start offsets".into()))?;
        let starts = ss.as_cuda_slice::<u32>()?.slice(s0..s1);
        launch.arg(&starts);
        tail!();
        unsafe { launch.launch(config) }.w()?;
    } else {
        tail!();
        unsafe { launch.launch(config) }.w()?;
    }
    let wrap = |data, heads| {
        Tensor::from_storage(
            Storage::Cuda(CudaStorage::wrap_cuda_slice(data, device.clone())),
            (batch, length, heads, 128),
            BackpropOp::none(),
            false,
        )
    };
    let v = Tensor::from_storage(
        Storage::Cuda(CudaStorage::wrap_cuda_slice(v, device.clone())),
        (batch, length, value_heads, 128),
        BackpropOp::none(),
        false,
    );
    Ok((wrap(q, key_heads), wrap(k, key_heads), v))
}

#[cfg(test)]
pub(crate) static CONV_LOOKUP_CALLS: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

#[cfg(test)]
pub(crate) static CONV_PREPARED_QV_CALLS: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn conv_cuda_matches_candle_across_batches_and_padding() {
        let (batch, length, channels, kernel) = (2, 7, 31, 4);
        let cpu = candle_core::Device::Cpu;
        let projected = Tensor::from_vec(
            (0..batch * length * channels)
                .map(|index| ((index * 17 % 41) as f32 - 20.) * 0.04)
                .collect::<Vec<_>>(),
            (batch, length, channels),
            &cpu,
        )
        .unwrap();
        let weights = Tensor::from_vec(
            (0..channels * kernel)
                .map(|index| ((index * 11 % 23) as f32 - 11.) * 0.02)
                .collect::<Vec<_>>(),
            (channels, 1, kernel),
            &cpu,
        )
        .unwrap();
        let cuda = test_cuda_device().unwrap();
        for dtype in [DType::BF16, DType::F32] {
            let projected = projected.to_dtype(dtype).unwrap();
            let weights = weights.to_dtype(dtype).unwrap();
            let expected = causal_conv_silu(&projected, &weights, kernel)
                .unwrap()
                .to_dtype(DType::F32)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            let actual = cuda_conv_silu(
                &projected.to_device(&cuda).unwrap(),
                &weights.to_device(&cuda).unwrap(),
            )
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
            let max_error = expected
                .iter()
                .zip(actual)
                .map(|(expected, actual)| (expected - actual).abs())
                .fold(0f32, f32::max);
            assert!(
                max_error < 0.001,
                "{dtype:?} convolution error: {max_error}"
            );
        }
    }
}

use super::*;

pub(crate) fn cuda_full_norm_rope(
    input: &Tensor,
    weight: &Tensor,
    rotary: &RotaryTensors,
    epsilon: f64,
) -> Result<Tensor> {
    let (batch, length, heads, width) = input.dims4()?;
    let strides = input.stride();
    let row_stride = strides[2];
    if input.dtype() != DType::BF16
        || weight.dtype() != DType::BF16
        || weight.dims() != [width]
        || rotary.cos.dtype() != DType::BF16
        || rotary.sin.dtype() != DType::BF16
        || !(32..=1024).contains(&width)
        || rotary.dimension > width
        || !rotary.dimension.is_multiple_of(2)
        || rotary.cos.dims() != [1, length, 1, rotary.dimension]
        || rotary.sin.dims() != rotary.cos.dims()
        || strides[3] != 1
        || row_stride < width
        || strides[1] != heads * row_stride
        || strides[0] != length * heads * row_stride
        || batch == 0
        || length == 0
        || heads == 0
        || !input.device().is_cuda()
        || !input.device().same_device(weight.device())
        || !input.device().same_device(rotary.cos.device())
        || !input.device().same_device(rotary.sin.device())
    {
        candle_core::bail!("unsupported full attention fused norm/rotary layout");
    }
    let weight = weight.contiguous()?;
    let cosine = rotary.cos.contiguous()?;
    let sine = rotary.sin.contiguous()?;
    let (x_storage, x_layout) = input.storage_and_layout();
    let (w_storage, w_layout) = weight.storage_and_layout();
    let (c_storage, c_layout) = cosine.storage_and_layout();
    let (s_storage, s_layout) = sine.storage_and_layout();
    let (Storage::Cuda(x), Storage::Cuda(w), Storage::Cuda(c), Storage::Cuda(s)) =
        (&*x_storage, &*w_storage, &*c_storage, &*s_storage)
    else {
        candle_core::bail!("full norm/rotary CUDA storage required")
    };
    let rows = batch
        .checked_mul(length)
        .and_then(|n| n.checked_mul(heads))
        .ok_or_else(|| candle_core::Error::Msg("full norm/rotary rows overflow".into()))?;
    let span = (rows - 1)
        .checked_mul(row_stride)
        .and_then(|n| n.checked_add(width))
        .ok_or_else(|| candle_core::Error::Msg("full norm/rotary source span overflow".into()))?;
    let offset = x_layout.start_offset();
    let end = offset
        .checked_add(span)
        .ok_or_else(|| candle_core::Error::Msg("full norm/rotary offset overflow".into()))?;
    let input = x.as_cuda_slice::<bf16>()?.slice(offset..end);
    let contiguous = |layout: &candle_core::Layout| {
        layout.contiguous_offsets().ok_or_else(|| {
            candle_core::Error::Msg("noncontiguous full norm/rotary parameter".into())
        })
    };
    let (a, b) = contiguous(w_layout)?;
    let weight = w.as_cuda_slice::<bf16>()?.slice(a..b);
    let (a, b) = contiguous(c_layout)?;
    let cosine = c.as_cuda_slice::<bf16>()?.slice(a..b);
    let (a, b) = contiguous(s_layout)?;
    let sine = s.as_cuda_slice::<bf16>()?.slice(a..b);
    let device = x.device().clone();
    let count = rows
        .checked_mul(width)
        .ok_or_else(|| candle_core::Error::Msg("full norm/rotary output overflow".into()))?;
    let output = unsafe { device.alloc::<bf16>(count)? };
    let function =
        device.get_or_load_custom_func("qwen35_full_norm_rope_bf16", "sys1_qwen35", CUDA_PTX)?;
    let integer = |n| {
        i32::try_from(n)
            .map_err(|_| candle_core::Error::Msg("full norm/rotary launch index overflow".into()))
    };
    let width_arg = integer(width)?;
    let rotary_arg = integer(rotary.dimension)?;
    let heads_arg = integer(heads)?;
    let length_arg = integer(length)?;
    let stride_arg = integer(row_stride)?;
    let epsilon = epsilon as f32;
    let mut launch = function.builder();
    launch.arg(&input);
    launch.arg(&weight);
    launch.arg(&cosine);
    launch.arg(&sine);
    launch.arg(&output);
    launch.arg(&width_arg);
    launch.arg(&rotary_arg);
    launch.arg(&heads_arg);
    launch.arg(&length_arg);
    launch.arg(&stride_arg);
    launch.arg(&epsilon);
    unsafe {
        launch.launch(LaunchConfig {
            grid_dim: (
                u32::try_from(rows).map_err(|_| {
                    candle_core::Error::Msg("full norm/rotary grid overflow".into())
                })?,
                1,
                1,
            ),
            block_dim: (width.next_power_of_two().clamp(32, 1024) as u32, 1, 1),
            shared_mem_bytes: 0,
        })
    }
    .w()?;
    Ok(Tensor::from_storage(
        Storage::Cuda(CudaStorage::wrap_cuda_slice(output, device)),
        (batch, length, heads, width),
        BackpropOp::none(),
        false,
    ))
}

pub(crate) fn cuda_full_qkv_finish(
    projected: &Tensor,
    separate: Option<(&Tensor, &Tensor)>,
    norm_weights: (&Tensor, &Tensor),
    rotary: &RotaryTensors,
    heads: usize,
    kv_heads: usize,
    epsilon: f64,
) -> Result<(Tensor, Tensor, Tensor, Tensor)> {
    let (qw, kw) = norm_weights;
    let (batch, length, columns) = projected.dims3()?;
    let width = qw.dim(0)?;
    let merged = separate.is_none();
    let expected = (2 * heads + if merged { 2 * kv_heads } else { 0 }) * width;
    if batch == 0
        || length == 0
        || heads == 0
        || kv_heads == 0
        || columns != expected
        || kw.dims() != [width]
        || !(32..=1024).contains(&width)
        || rotary.dimension > width
        || !rotary.dimension.is_multiple_of(2)
        || rotary.cos.dims() != [1, length, 1, rotary.dimension]
        || rotary.sin.dims() != rotary.cos.dims()
    {
        candle_core::bail!("unsupported fused QKV geometry");
    }
    let k = separate.map(|(k, _)| k).unwrap_or(projected);
    let v = separate.map(|(_, v)| v).unwrap_or(projected);
    if !merged && (k.dims() != [batch, length, kv_heads * width] || v.dims() != k.dims()) {
        candle_core::bail!("separate fused QKV input geometry");
    }
    for value in [projected, k, v, qw, kw, &rotary.cos, &rotary.sin] {
        if value.dtype() != DType::BF16
            || !value.device().same_device(projected.device())
            || !value.is_contiguous()
        {
            candle_core::bail!("fused QKV requires colocated contiguous BF16");
        }
    }
    let (x_store, x_layout) = projected.storage_and_layout();
    let (k_store, k_layout) = k.storage_and_layout();
    let (v_store, v_layout) = v.storage_and_layout();
    let (qw_store, qw_layout) = qw.storage_and_layout();
    let (kw_store, kw_layout) = kw.storage_and_layout();
    let (c_store, c_layout) = rotary.cos.storage_and_layout();
    let (s_store, s_layout) = rotary.sin.storage_and_layout();
    let (
        Storage::Cuda(x),
        Storage::Cuda(k),
        Storage::Cuda(vs),
        Storage::Cuda(qw),
        Storage::Cuda(kw),
        Storage::Cuda(c),
        Storage::Cuda(s),
    ) = (
        &*x_store, &*k_store, &*v_store, &*qw_store, &*kw_store, &*c_store, &*s_store,
    )
    else {
        candle_core::bail!("fused QKV requires CUDA storage")
    };
    let offsets = |l: &candle_core::Layout| {
        l.contiguous_offsets()
            .ok_or_else(|| candle_core::Error::Msg("fused QKV layout".into()))
    };
    let device = x.device().clone();
    let (a, b) = offsets(x_layout)?;
    let x = x.as_cuda_slice::<bf16>()?.slice(a..b);
    let (a, b) = offsets(k_layout)?;
    let keys = k.as_cuda_slice::<bf16>()?.slice(a..b);
    let (a, b) = offsets(v_layout)?;
    let values = vs.as_cuda_slice::<bf16>()?.slice(a..b);
    let (a, b) = offsets(qw_layout)?;
    let qw = qw.as_cuda_slice::<bf16>()?.slice(a..b);
    let (a, b) = offsets(kw_layout)?;
    let kw = kw.as_cuda_slice::<bf16>()?.slice(a..b);
    let (a, b) = offsets(c_layout)?;
    let cosine = c.as_cuda_slice::<bf16>()?.slice(a..b);
    let (a, b) = offsets(s_layout)?;
    let sine = s.as_cuda_slice::<bf16>()?.slice(a..b);
    let tokens = batch
        .checked_mul(length)
        .ok_or_else(|| candle_core::Error::Msg("fused QKV token overflow".into()))?;
    let qcount = tokens
        .checked_mul(heads)
        .and_then(|x| x.checked_mul(width))
        .ok_or_else(|| candle_core::Error::Msg("fused QKV Q overflow".into()))?;
    let kcount = tokens
        .checked_mul(kv_heads)
        .and_then(|x| x.checked_mul(width))
        .ok_or_else(|| candle_core::Error::Msg("fused QKV K overflow".into()))?;
    let count = qcount
        .checked_mul(2)
        .and_then(|x| x.checked_add(kcount.checked_mul(if merged { 2 } else { 1 })?))
        .ok_or_else(|| candle_core::Error::Msg("fused QKV allocation overflow".into()))?;
    let output = unsafe { device.alloc::<bf16>(count)? };
    let integer = |x| {
        i32::try_from(x).map_err(|_| candle_core::Error::Msg("fused QKV index overflow".into()))
    };
    let d = integer(width)?;
    let r = integer(rotary.dimension)?;
    let h = integer(heads)?;
    let kh = integer(kv_heads)?;
    let l = integer(length)?;
    let eps = epsilon as f32;
    let grid = u32::try_from(
        tokens
            .checked_mul(heads + kv_heads)
            .ok_or_else(|| candle_core::Error::Msg("fused QKV grid overflow".into()))?,
    )
    .map_err(|_| candle_core::Error::Msg("fused QKV grid u32 overflow".into()))?;
    let warp_finish = width == 256
        && rotary.dimension == 64
        && heads == 16
        && kv_heads == 4
        && matches!(
            device.cuda_stream().context().compute_capability().w()?,
            (8, 0) | (9, 0)
        );
    let name = if warp_finish {
        if merged {
            "qwen35_full_finish_merged_warp128"
        } else {
            "qwen35_full_finish_warp127"
        }
    } else if merged {
        "qwen35_full_qkv_finish_merged_bf16"
    } else {
        "qwen35_full_qkv_finish_separate_bf16"
    };
    let function = device.get_or_load_custom_func(name, "sys1_qwen35", CUDA_PTX)?;
    crate::kernels::cuda::activation::with_sigmoid_lookup(&device, |table| {
        let mut launch = function.builder();
        launch.arg(&x);
        launch.arg(&keys);
        launch.arg(&values);
        launch.arg(&qw);
        launch.arg(&kw);
        launch.arg(&cosine);
        launch.arg(&sine);
        launch.arg(table);
        launch.arg(&output);
        launch.arg(&d);
        launch.arg(&r);
        launch.arg(&h);
        launch.arg(&kh);
        launch.arg(&l);
        launch.arg(&eps);
        let config = if warp_finish {
            LaunchConfig {
                grid_dim: (tokens as u32, 5, 1),
                block_dim: (128, 1, 1),
                shared_mem_bytes: 0,
            }
        } else {
            LaunchConfig {
                grid_dim: (grid, 1, 1),
                block_dim: (width.next_power_of_two().clamp(32, 1024) as u32, 1, 1),
                shared_mem_bytes: 0,
            }
        };
        unsafe { launch.launch(config) }.w()?;
        Ok(())
    })?;
    let output = Tensor::from_storage(
        Storage::Cuda(CudaStorage::wrap_cuda_slice(output, device)),
        count,
        BackpropOp::none(),
        false,
    );
    let q = output
        .narrow(0, 0, qcount)?
        .reshape((batch, length, heads, width))?;
    let k = output
        .narrow(0, qcount, kcount)?
        .reshape((batch, length, kv_heads, width))?;
    let gate =
        output
            .narrow(0, qcount + kcount, qcount)?
            .reshape((batch, length, heads * width))?;
    let v = if merged {
        output
            .narrow(0, 2 * qcount + kcount, kcount)?
            .reshape((batch, length, kv_heads, width))?
    } else {
        v.reshape((batch, length, kv_heads, width))?
    };
    Ok((q, k, v, gate))
}

impl FullAttention {
    pub(crate) fn forward_packed(
        &self,
        x: &Tensor,
        rotary: &RotaryTensors,
        sequences: &PackedSequences,
    ) -> Result<Tensor> {
        let (batch, length, _) = x.dims3()?;
        let (q, k, v, gate) = self.project_finish(x, rotary)?;
        let scale = (self.head_dim as f32).powf(-0.5);
        #[cfg(feature = "metal")]
        if x.device().is_metal()
            && x.dtype() != DType::BF16
            && self.implementation != AttentionImplementation::Eager
        {
            let attention = candle_nn::ops::sdpa(
                &q.transpose(1, 2)?.contiguous()?,
                &k.transpose(1, 2)?.contiguous()?,
                &v.transpose(1, 2)?.contiguous()?,
                None,
                self.causal,
                scale,
                1.0,
            )?
            .transpose(1, 2)?;

            let gated = full_attention_gate_mul(
                &attention,
                &gate,
                batch,
                length,
                self.heads * self.head_dim,
            )?;

            return gated.apply(&self.output);
        }
        let attention = crate::kernels::cuda::attention::flash_attn_varlen(
            &q.squeeze(0)?.contiguous()?,
            &k.squeeze(0)?.contiguous()?,
            &v.squeeze(0)?.contiguous()?,
            &sequences.cumulative,
            &sequences.cumulative,
            *sequences.lengths.iter().max().unwrap(),
            *sequences.lengths.iter().max().unwrap(),
            scale,
            self.causal,
        )?
        .unsqueeze(0)?;

        let gated =
            full_attention_gate_mul(&attention, &gate, batch, length, self.heads * self.head_dim)?;

        linear_attention_projection_forward(&gated, &self.output)
    }
}

use super::*;

pub(crate) fn cuda_gate_projection_bf16(input: &Tensor, linear: &Linear) -> Result<Tensor> {
    cuda_bf16_projection(input, linear, false)
}

#[cfg(test)]
pub(crate) fn fixed_bf16_projection(input: &Tensor, linear: &Linear) -> Result<Tensor> {
    head_bf16_projection(input, linear, true)
}

pub(crate) fn compensated_bf16_projection(input: &Tensor, linear: &CandleLinear) -> Result<Tensor> {
    head_bf16_projection(input, linear, false)
}

pub(crate) fn head_bf16_projection(
    input: &Tensor,
    linear: &CandleLinear,
    tensor_core: bool,
) -> Result<Tensor> {
    let inner = input.dim(D::Minus1)?;
    if inner == 0 || input.elem_count() == 0 {
        return input.apply(linear);
    }
    let rows = input.elem_count() / inner;
    let mut shape = input.dims().to_vec();
    *shape.last_mut().unwrap() = linear.weight().dim(0)?;
    cuda_bf16_projection(
        &input.contiguous()?.reshape((1, rows, inner))?,
        linear,
        tensor_core,
    )?
    .reshape(shape)
}

pub(crate) fn cuda_bf16_projection(
    input: &Tensor,
    linear: &CandleLinear,
    tensor_core: bool,
) -> Result<Tensor> {
    let inner = input.dim(D::Minus1)?;
    let columns = linear.weight().dim(0)?;
    let rows = input.elem_count() / inner.max(1);

    if !tensor_core
        && input.device().is_cuda()
        && input.dtype() == DType::BF16
        && linear.bias().is_none()
        && option_env!("SYS1_CUDA_COMPUTE_CAP") == Some("90")
        && inner == 12288
        && columns == 4096
        && (32768..=65536).contains(&rows)
    {
        return crate::kernels::cuda::blaslt::project_hopper_down_cooperative(input, linear);
    }

    if !tensor_core
        && input.dtype() == DType::BF16
        && linear.bias().is_none()
        && matches!(option_env!("SYS1_CUDA_COMPUTE_CAP"), Some("80" | "90"))
        && inner == 4096
        && matches!(columns, 32 | 64)
        && (1..=65536).contains(&rows)
    {
        return crate::kernels::cuda::head_blaslt::project_gdn_ba(input, linear);
    }

    #[cfg(sys1_cuda_fp64_emulation)]
    if !tensor_core
        && input.dtype() == DType::BF16
        && linear.bias().is_some()
        && option_env!("SYS1_CUDA_COMPUTE_CAP") == Some("86")
        && (16385..=32768).contains(&rows)
        && inner == 1024
        && columns == 2048
    {
        let (batch, length, _) = input.dims3()?;
        let flattened = input.contiguous()?.reshape((1, rows, inner))?;
        let chunk_rows = rows.div_ceil(rows.div_ceil(16384));
        let mut parts = Vec::with_capacity(rows.div_ceil(chunk_rows));
        for start in (0..rows).step_by(chunk_rows) {
            let count = chunk_rows.min(rows - start);
            parts.push(cuda_bf16_projection(
                &flattened.narrow(1, start, count)?,
                linear,
                false,
            )?);
        }
        return Tensor::cat(&parts, 1)?.reshape((batch, length, columns));
    }

    #[cfg(sys1_cuda_fp64_emulation)]
    if !tensor_core
        && input.dtype() == DType::BF16
        && linear.bias().is_some()
        && option_env!("SYS1_CUDA_COMPUTE_CAP") == Some("86")
        && (8192..=16384).contains(&rows)
        && inner == 1024
        && columns == 2048
    {
        return crate::kernels::cuda::guarded_dgemm::project(
            input,
            linear,
            guarded_double_radius_coefficient_impl(inner, true),
        );
    }

    if !tensor_core
        && input.dtype() == DType::BF16
        && linear.bias().is_some()
        && option_env!("SYS1_CUDA_COMPUTE_CAP") == Some("120")
        && rows >= 128
        && inner == 1024
        && columns == 2048
    {
        return cuda_bf16_projection_impl(input, linear, true, SimtProjection::Compensated);
    }

    if !tensor_core
        && input.dtype() == DType::BF16
        && linear.bias().is_some()
        && option_env!("SYS1_CUDA_COMPUTE_CAP") == Some("80")
        && (64..=8192).contains(&rows)
        && inner == 1024
        && matches!(columns, 1024 | 3072 | 4096)
    {
        #[cfg(test)]
        crate::kernels::cuda::guarded_dgemm::record_head_route163();
        return crate::kernels::cuda::guarded_dgemm::project(
            input,
            linear,
            guarded_double_radius_coefficient_impl(inner, true),
        );
    }

    if !tensor_core
        && input.dtype() == DType::BF16
        && linear.bias().is_some()
        && option_env!("SYS1_CUDA_COMPUTE_CAP") == Some("90")
        && (64..=65536).contains(&rows)
        && ((inner == 1024 && matches!(columns, 1024 | 2048 | 3072 | 4096))
            || (inner == 4096 && columns == 1024))
    {
        #[cfg(test)]
        if !(inner == 1024 && columns == 2048 && rows >= 128) {
            crate::kernels::cuda::guarded_dgemm::record_head_route163();
        }
        return crate::kernels::cuda::guarded_dgemm::project(
            input,
            linear,
            guarded_double_radius_coefficient_impl(inner, true),
        );
    }

    if !tensor_core
        && input.dtype() == DType::BF16
        && linear.bias().is_some()
        && option_env!("SYS1_CUDA_COMPUTE_CAP") == Some("90")
        && rows >= 128
        && inner == 1024
        && columns == 2048
    {
        return crate::kernels::cuda::guarded_dgemm::project(
            input,
            linear,
            guarded_double_radius_coefficient_impl(inner, true),
        );
    }

    if !tensor_core
        && input.dtype() == DType::BF16
        && linear.bias().is_some()
        && option_env!("SYS1_CUDA_COMPUTE_CAP") == Some("80")
        && rows >= 128
        && inner == 1024
        && columns == 2048
    {
        return crate::kernels::cuda::guarded_dgemm::project(
            input,
            linear,
            guarded_double_radius_coefficient_impl(inner, true),
        );
    }

    if !tensor_core
        && input.dtype() == DType::BF16
        && linear.bias().is_some()
        && matches!(option_env!("SYS1_CUDA_COMPUTE_CAP"), Some("80" | "90"))
        && rows >= 128
        && inner == 1024
        && columns == 2048
    {
        return crate::kernels::cuda::biased_projection::project(
            input,
            linear,
            guarded_double_radius_coefficient_impl(inner, true),
        );
    }

    if !tensor_core
        && input.dtype() == DType::BF16
        && linear.bias().is_none()
        && option_env!("SYS1_CUDA_COMPUTE_CAP") == Some("90")
        && rows >= 1536
        && inner == 4096
        && columns == 64
    {
        return crate::kernels::cuda::guarded_dgemm::project(
            input,
            linear,
            guarded_double_radius_coefficient_impl(inner, true),
        );
    }

    if !tensor_core
        && input.dtype() == DType::BF16
        && linear.bias().is_none()
        && matches!(option_env!("SYS1_CUDA_COMPUTE_CAP"), Some("80" | "90"))
        && rows >= 1536
        && inner == 4096
        && columns == 64
    {
        return cuda_bf16_projection_impl(
            input,
            linear,
            false,
            SimtProjection::GuardedTensorDoubleRows16,
        );
    }

    if !tensor_core
        && input.dtype() == DType::BF16
        && linear.bias().is_none()
        && option_env!("SYS1_CUDA_COMPUTE_CAP") == Some("120")
        && rows >= 2048
        && inner == 4096
        && columns == 64
    {
        return cuda_bf16_projection_impl(input, linear, false, SimtProjection::TiledCompensated);
    }
    if !tensor_core
        && linear.bias().is_none()
        && long_down_actual_m(
            rows,
            inner,
            columns,
            option_env!("SYS1_CUDA_COMPUTE_CAP") == Some("120"),
        )
    {
        return crate::kernels::cuda::blaslt::project_unblocked(input, linear);
    }

    if !tensor_core
        && linear.bias().is_none()
        && option_env!("SYS1_CUDA_COMPUTE_CAP") == Some("90")
        && rows >= 513
        && inner == 4096
        && columns == 12288
    {
        return crate::kernels::cuda::blaslt::project_unblocked(input, linear);
    }
    if !tensor_core
        && linear.bias().is_none()
        && columns >= 2048
        && columns >= inner.saturating_mul(2)
        && (rows <= 512 || columns == inner.saturating_mul(2))
    {
        return crate::kernels::cuda::blaslt::project_unblocked(input, linear);
    }

    if !tensor_core
        && input.dtype() == DType::BF16
        && linear.bias().is_none()
        && matches!(option_env!("SYS1_CUDA_COMPUTE_CAP"), Some("80" | "120"))
        && rows >= 8193
        && columns >= 256
    {
        return crate::kernels::cuda::blaslt::project_unblocked(input, linear);
    }

    if !tensor_core
        && input.dtype() == DType::BF16
        && linear.bias().is_none()
        && option_env!("SYS1_CUDA_COMPUTE_CAP") == Some("89")
        && (513..=1024).contains(&rows)
        && inner == 4096
        && columns == 12288
    {
        return crate::kernels::cuda::blaslt::project_pinned2048(input, linear);
    }

    if !tensor_core
        && input.dtype() == DType::BF16
        && linear.bias().is_none()
        && option_env!("SYS1_CUDA_COMPUTE_CAP") == Some("86")
        && (513..=4096).contains(&rows)
        && inner == 12288
        && columns == 4096
    {
        return crate::kernels::cuda::blaslt::project_pinned2048(input, linear);
    }

    if !tensor_core
        && input.dtype() == DType::BF16
        && linear.bias().is_none()
        && option_env!("SYS1_CUDA_COMPUTE_CAP") == Some("80")
        && inner == 12288
        && columns == 4096
        && (3730..=16384).contains(&rows)
    {
        return crate::kernels::cuda::blaslt::project_unblocked(input, linear);
    }

    if !tensor_core
        && linear.bias().is_none()
        && linear.weight().dims().first().copied().unwrap_or(0) >= 256
    {
        return crate::kernels::cuda::blaslt::project_pinned512(input, linear);
    }
    cuda_bf16_projection_impl(input, linear, tensor_core, SimtProjection::Compensated)
}

#[derive(Clone, Copy)]
pub(crate) enum SimtProjection {
    Compensated,
    TiledCompensated,

    #[allow(dead_code)]
    GuardedDouble,
    #[allow(dead_code)]
    GuardedTensorDouble,
    #[allow(dead_code)]
    GuardedTensorDoubleRows16,
}

pub(crate) fn cuda_bf16_projection_impl(
    input: &Tensor,
    linear: &CandleLinear,
    tensor_core: bool,
    simt: SimtProjection,
) -> Result<Tensor> {
    let (batch, length, inner) = input.dims3()?;
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
    let device = xs.device().clone();
    let mut output = unsafe { device.alloc::<bf16>(count)? };

    let tile_rows = if tensor_core {
        projection_tile_rows(
            rows,
            columns,
            option_env!("SYS1_CUDA_COMPUTE_CAP") == Some("120"),
        )
    } else {
        16
    };

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
    } else if tensor_core && tile_rows == 128 {
        "qwen35_kv_projection_rows128_bf16"
    } else if tensor_core && tile_rows == 32 {
        "qwen35_kv_projection_rows32_bf16"
    } else if tensor_core {
        "qwen35_kv_projection_bf16"
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
                rows.div_ceil(tile_rows) as u32,
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
    if let Some(coefficient) = &radius_coefficient {
        launch.arg(coefficient);
    }
    unsafe { launch.launch(config) }.w()?;
    Ok(Tensor::from_storage(
        Storage::Cuda(CudaStorage::wrap_cuda_slice(output, device)),
        (batch, length, columns),
        BackpropOp::none(),
        false,
    ))
}

pub(crate) fn cuda_gate_tensor16_rows32(
    input: &Tensor,
    linear: &Linear,
    tensor_core: bool,
    simt: SimtProjection,
) -> Result<Tensor> {
    let (batch, length, inner) = input.dims3()?;
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
        "qwen35_gate_nativevec_rows32_bf16"
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
                rows.div_ceil(tile_rows) as u32,
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
    if let Some(coefficient) = &radius_coefficient {
        launch.arg(coefficient);
    }
    unsafe { launch.launch(config) }.w()?;
    Ok(Tensor::from_storage(
        Storage::Cuda(CudaStorage::wrap_cuda_slice(output, device)),
        (batch, length, columns),
        BackpropOp::none(),
        false,
    ))
}

pub(crate) fn cuda_gate_tensor16_original_packed(
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
    if sequences.gate_short_count == sequences.total {
        return cuda_gate_projection_bf16(input, linear);
    }

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
    unsafe { launch.launch(config) }.w()?;
    if sequences.gate_short_count > 0 {
        let function = device.get_or_load_custom_func(
            "qwen35_gate_tensor16_short_bf16",
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
                    sequences.gate_short_count.div_ceil(4) as u32,
                    columns.div_ceil(8) as u32,
                    1,
                ),
                block_dim: (128, 1, 1),
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

pub(crate) fn cuda_gate_tensor16_packed(
    input: &Tensor,
    linear: &Linear,
    tensor_core: bool,
    simt: SimtProjection,
    sequences: &PackedSequences,
) -> Result<Tensor> {
    if option_env!("SYS1_CUDA_COMPUTE_CAP") == Some("120")
        && tensor_core
        && input.device().is_cuda()
        && input.dtype() == DType::BF16
        && linear.weight().dtype() == DType::BF16
        && linear.bias().is_none()
        && linear.weight().dims() == [64, 4096]
        && sequences.gate_short_count >= 4096
    {
        return cuda_gate_gather_small(input, linear, tensor_core, simt, sequences);
    }
    cuda_gate_tensor16_original_packed(input, linear, tensor_core, simt, sequences)
}

pub(crate) fn nn_geometry_available(inner: usize, columns: usize) -> bool {
    matches!(
        (inner, columns),
        (4096, 12288) | (12288, 4096) | (4096, 8192) | (4096, 4096)
    )
}

pub(crate) fn nn_layout_selected(rows: usize, inner: usize, columns: usize) -> bool {
    match (inner, columns) {
        (4096, 12288) | (12288, 4096) => rows >= 4096,
        (4096, 8192) => rows >= 1536,
        (4096, 4096) => rows >= 8193,
        _ => false,
    }
}

pub(crate) fn try_nn_projection(input: &Tensor, linear: &Linear) -> Result<Option<Tensor>> {
    #[cfg(test)]
    let enabled = QWEN_NN_REPLAY.with(std::cell::Cell::get);
    #[cfg(not(test))]
    let enabled = true;
    if !enabled
        || linear.bias().is_some()
        || !input.device().is_cuda()
        || input.dtype() != DType::BF16
        || option_env!("SYS1_CUDA_COMPUTE_CAP") != Some("120")
    {
        return Ok(None);
    }
    let inner = input.dim(D::Minus1)?;
    let columns = linear.weight().dim(0)?;
    let rows = input.elem_count() / inner.max(1);
    if !nn_layout_selected(rows, inner, columns) {
        return Ok(None);
    }
    let Some(weight) = linear.cached_nn()? else {
        return Ok(None);
    };
    let actual_m = long_down_actual_m(rows, inner, columns, true)
        || (columns >= 2048
            && columns >= inner.saturating_mul(2)
            && (rows <= 512 || columns == inner.saturating_mul(2)))
        || (rows >= 8193 && columns >= 256);
    #[cfg(test)]
    QWEN_NN_CALLS.with(|slot| slot.set(slot.get() + 1));
    Ok(Some(
        crate::kernels::cuda::blaslt::project_transposed_weight(input, &weight, !actual_m)?,
    ))
}

pub(crate) fn nn_capacity_preserved(
    free: usize,
    extra: usize,
    bytes_per_token: usize,
    target_tokens: usize,
) -> bool {
    match free.checked_sub(extra) {
        Some(after) => after.saturating_sub(after / 4) / bytes_per_token.max(1) >= target_tokens,
        None => false,
    }
}

impl SimtProjection {
    pub(crate) fn is_tensor_double(self) -> bool {
        matches!(
            self,
            Self::GuardedTensorDouble | Self::GuardedTensorDoubleRows16
        )
    }
}

#[cfg(sys1_cuda_fused_ffn)]
impl Mlp {
    pub(crate) fn fused_geometry_available(&self) -> bool {
        self.gate.bias().is_none()
            && self.up.bias().is_none()
            && crate::kernels::cuda::fused_gate_up::available(self.gate.weight(), self.up.weight())
    }
    pub(crate) fn cached_fused(&self) -> Result<Option<Tensor>> {
        Ok(self
            .fused_weight
            .lock()
            .map_err(|_| candle_core::Error::Msg("poisoned model-owned fused MLP weight".into()))?
            .clone())
    }
    pub(crate) fn prepared_fused(&self) -> Result<Tensor> {
        let mut slot = self
            .fused_weight
            .lock()
            .map_err(|_| candle_core::Error::Msg("poisoned model-owned fused MLP weight".into()))?;
        if slot.is_none() {
            *slot = Some(crate::kernels::cuda::fused_gate_up::prepare(
                self.gate.weight(),
                self.up.weight(),
            )?);
        }
        Ok(slot.as_ref().unwrap().clone())
    }
}

impl Mlp {
    #[cfg(all(feature = "cuda", sys1_cuda_direct_ffn))]
    pub(crate) fn direct_product(
        &self,
        x: &Tensor,
        capture_index: usize,
    ) -> Result<Option<Tensor>> {
        let rows = x.elem_count() / x.dim(D::Minus1)?.max(1);
        if !x.device().is_cuda() || x.dtype() != DType::BF16 || !(2048..=32768).contains(&rows) {
            return Ok(None);
        }
        let Some(weight) = self.direct_weight.as_ref() else {
            return Ok(None);
        };
        let result = crate::kernels::cuda::sm86::fused_gate_up::forward(
            x,
            weight,
            crate::numerics::enabled(),
        )?;
        #[cfg(test)]
        DIRECT_MLP_CALLS.with(|slot| slot.set(slot.get() + 1));
        if let Some((gate, up)) = result.projections {
            crate::numerics::layer_capture(capture_index, "mlp-gate-projected", &gate)?;
            crate::numerics::layer_capture(capture_index, "mlp-up-projected", &up)?;
            crate::numerics::layer_capture(
                capture_index,
                "mlp-gate-activated",
                &crate::kernels::activation::silu(&gate)?,
            )?;
        }
        profiling::mark(x.device(), capture_index, "mlp_direct_gate_up_product")?;
        Ok(Some(result.product))
    }
}

impl Linear {
    pub(crate) fn cached_nn(&self) -> Result<Option<Tensor>> {
        Ok(self
            .nn_weight
            .lock()
            .map_err(|_| candle_core::Error::Msg("poisoned model-owned NN weight".into()))?
            .clone())
    }

    pub(crate) fn prepared_nn(&self) -> Result<Tensor> {
        let mut slot = self
            .nn_weight
            .lock()
            .map_err(|_| candle_core::Error::Msg("poisoned model-owned NN weight".into()))?;
        if slot.is_none() {
            let prepared = self.weight().t()?.contiguous()?;
            #[cfg(test)]
            QWEN_NN_WEIGHT_BYTES.with(|slot| slot.set(slot.get() + self.weight().elem_count() * 2));
            *slot = Some(prepared);
        }
        Ok(slot.as_ref().unwrap().clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::architectures::qwen35_text::tests::supported_rotary_test_dtypes;

    #[test]
    fn stable_gate_projection_cuda_matches_fp64_cast_bitwise() -> Result<()> {
        let device = test_cuda_device()?;
        if !supported_rotary_test_dtypes(&device)?.contains(&DType::BF16) {
            return Ok(());
        }
        for (batch, length, inner, columns) in [
            (1, 3, 33, 9),
            (2, 5, 129, 17),
            (1, 17, 4096, 32),
            (1, 2, 6144, 64),
        ] {
            let xs: Vec<f32> = (0..batch * length * inner)
                .map(|i| ((i * 17 % 255) as f32 - 127.) / 64.)
                .collect();
            let ws: Vec<f32> = (0..columns * inner)
                .map(|i| ((i * 29 % 127) as f32 - 63.) / 512.)
                .collect();
            let x = Tensor::from_vec(xs.clone(), (batch, length, inner), &device)?
                .to_dtype(DType::BF16)?;
            let w =
                Tensor::from_vec(ws.clone(), (columns, inner), &device)?.to_dtype(DType::BF16)?;
            let linear = Linear::new(w, None);
            let actual = cuda_gate_projection_bf16(&x, &linear)?
                .to_dtype(DType::F32)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            let repeated = cuda_gate_projection_bf16(&x, &linear)?
                .to_dtype(DType::F32)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            let mut expected = Vec::with_capacity(actual.len());
            for row in 0..batch * length {
                for column in 0..columns {
                    let sum: f64 = (0..inner)
                        .map(|k| f64::from(xs[row * inner + k]) * f64::from(ws[column * inner + k]))
                        .sum();
                    expected.push(bf16::from_f32(sum as f32).to_f32());
                }
            }
            assert_eq!(
                actual, expected,
                "fixed dyadic FP64 reference {inner}/{columns}"
            );
            assert_eq!(actual, repeated, "actual repeated CUDA gate projection");
            let biased = Linear::new(
                linear.weight().clone(),
                Some(Tensor::zeros(columns, DType::BF16, &device)?),
            );
            let with_zero_bias = cuda_gate_projection_bf16(&x, &biased)?
                .to_dtype(DType::F32)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            assert_eq!(actual, with_zero_bias, "zero-bias compensated projection");
        }
        Ok(())
    }

    #[test]
    fn compensated_head_projection_cuda_matches_bias_cast_and_storage_offsets() -> Result<()> {
        let device = test_cuda_device()?;
        if !supported_rotary_test_dtypes(&device)?.contains(&DType::BF16) {
            return Ok(());
        }
        let bits = |tensor: &Tensor| -> Result<Vec<f32>> {
            tensor.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()
        };
        for (rows, inner, columns) in [(3, 33, 9), (5, 129, 17), (2, 2048, 64)] {
            let xs: Vec<f32> = (0..(rows + 1) * inner)
                .map(|i| ((i * 17 % 255) as f32 - 127.) / 64.)
                .collect();
            let ws: Vec<f32> = (0..(columns + 1) * inner)
                .map(|i| ((i * 29 % 127) as f32 - 63.) / 512.)
                .collect();
            let bs: Vec<f32> = (0..columns + 1).map(|i| (i as f32 - 8.) / 128.).collect();
            let x = Tensor::from_vec(xs.clone(), (rows + 1, inner), &device)?
                .to_dtype(DType::BF16)?
                .narrow(0, 1, rows)?;
            let w = Tensor::from_vec(ws.clone(), (columns + 1, inner), &device)?
                .to_dtype(DType::BF16)?
                .narrow(0, 1, columns)?;
            let b = Tensor::from_vec(bs.clone(), columns + 1, &device)?
                .to_dtype(DType::BF16)?
                .narrow(0, 1, columns)?;
            let layer = Linear::new(w, Some(b));
            let actual = compensated_bf16_projection(&x, &layer)?;
            assert_eq!(actual.dims(), [rows, columns]);
            let mut expected = Vec::with_capacity(rows * columns);
            for row in 0..rows {
                for column in 0..columns {
                    let sum: f64 = (0..inner)
                        .map(|k| {
                            f64::from(xs[(row + 1) * inner + k])
                                * f64::from(ws[(column + 1) * inner + k])
                        })
                        .sum();
                    expected.push(bf16::from_f32(sum as f32 + bs[column + 1]).to_f32());
                }
            }
            assert_eq!(bits(&actual)?, expected, "FP64 dot / FP32 bias / BF16 cast");
            assert_eq!(bits(&compensated_bf16_projection(&x, &layer)?)?, expected);
            let batch = Tensor::stack(&[&x, &x], 0)?;
            let batch_actual = compensated_bf16_projection(&batch, &layer)?;
            assert_eq!(batch_actual.dims(), [2, rows, columns]);
            assert_eq!(
                bits(&batch_actual)?,
                [expected.clone(), expected.clone()].concat()
            );
            let padded = Tensor::cat(&[&x, &Tensor::zeros((19, inner), DType::BF16, &device)?], 0)?;
            assert_eq!(
                bits(&compensated_bf16_projection(&padded, &layer)?.narrow(0, 0, rows)?)?,
                expected
            );
        }

        let x = Tensor::ones((1, 1), DType::BF16, &device)?;
        let layer = Linear::new(
            Tensor::from_vec(vec![1_f32, -1_f32], (2, 1), &device)?.to_dtype(DType::BF16)?,
            Some(
                Tensor::from_vec(vec![1_f32 / 256., -1_f32 / 256.], 2, &device)?
                    .to_dtype(DType::BF16)?,
            ),
        );
        assert_eq!(
            bits(&compensated_bf16_projection(&x, &layer)?)?,
            vec![1., -1.]
        );
        Ok(())
    }

    #[test]
    fn stable_gate_projection_cuda_preserves_cancellation_and_exceptional_products() -> Result<()> {
        let device = test_cuda_device()?;
        if !supported_rotary_test_dtypes(&device)?.contains(&DType::BF16) {
            return Ok(());
        }
        let mut weights = vec![2_f32.powi(-16); 129];
        weights[0] = 1024.;
        weights[128] = -1024.;
        let cases = [
            (vec![1.; 129], weights),
            (
                vec![2_f32.powi(100); 2],
                vec![2_f32.powi(100), -2_f32.powi(100)],
            ),
            (vec![2_f32.powi(-100); 2], vec![2_f32.powi(-100); 2]),
        ];
        for (xs, ws) in cases {
            let expected = bf16::from_f32(
                xs.iter()
                    .zip(&ws)
                    .map(|(&x, &w)| f64::from(x) * f64::from(w))
                    .sum::<f64>() as f32,
            )
            .to_f32();
            let x =
                Tensor::from_vec(xs.clone(), (1, 1, xs.len()), &device)?.to_dtype(DType::BF16)?;
            let linear = Linear::new(
                Tensor::from_vec(ws, (1, xs.len()), &device)?.to_dtype(DType::BF16)?,
                None,
            );
            let actual = cuda_gate_projection_bf16(&x, &linear)?
                .to_dtype(DType::F32)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            assert_eq!(
                actual,
                vec![expected],
                "cancellation/exceptional product fixture"
            );
        }
        Ok(())
    }

    #[test]
    fn guarded_tensor_double_cuda_matches_compensated_for_normal_tile_tails() -> Result<()> {
        let device = test_cuda_device()?;
        if !supported_rotary_test_dtypes(&device)?.contains(&DType::BF16) {
            return Ok(());
        }
        for (rows, inner, columns) in [(3, 7, 9), (17, 65, 33), (2, 513, 17)] {
            let x = Tensor::from_vec(
                (0..rows * inner)
                    .map(|i| ((i * 17 % 255) as f32 - 127.) / 64.)
                    .collect::<Vec<_>>(),
                (1, rows, inner),
                &device,
            )?
            .to_dtype(DType::BF16)?;
            let linear = Linear::new(
                Tensor::from_vec(
                    (0..columns * inner)
                        .map(|i| ((i * 29 % 127) as f32 - 63.) / 512.)
                        .collect::<Vec<_>>(),
                    (columns, inner),
                    &device,
                )?
                .to_dtype(DType::BF16)?,
                None,
            );
            let bits = |mode| -> Result<Vec<u32>> {
                Ok(cuda_bf16_projection_impl(&x, &linear, false, mode)?
                    .to_dtype(DType::F32)?
                    .flatten_all()?
                    .to_vec1::<f32>()?
                    .into_iter()
                    .map(f32::to_bits)
                    .collect())
            };
            assert_eq!(
                bits(SimtProjection::Compensated)?,
                bits(SimtProjection::GuardedTensorDouble)?
            );
            assert_eq!(
                bits(SimtProjection::Compensated)?,
                bits(SimtProjection::GuardedTensorDoubleRows16)?
            );
        }
        Ok(())
    }

    #[test]
    fn guarded_double_projection_cuda_matches_compensated_bits_across_tails() -> Result<()> {
        let device = test_cuda_device()?;
        if !supported_rotary_test_dtypes(&device)?.contains(&DType::BF16) {
            return Ok(());
        }
        let mut state = 0x6a09_e667_u32;
        let mut values = |count| {
            (0..count)
                .map(|_| {
                    state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                    let sign = (state >> 16) as u16 & 0x8000;
                    let exponent = ((state >> 24) % 128 + 63) as u16;
                    bf16::from_bits(sign | (exponent << 7) | ((state >> 17) as u16 & 127)).to_f32()
                })
                .collect::<Vec<_>>()
        };
        for (rows, inner, columns) in [
            (1, 1, 1),
            (3, 33, 9),
            (2, 65, 17),
            (17, 129, 65),
            (2, 4096, 65),
            (1, 18432, 9),
        ] {
            let x = Tensor::from_vec(values(rows * inner), (1, rows, inner), &device)?
                .to_dtype(DType::BF16)?;
            let linear = Linear::new(
                Tensor::from_vec(values(columns * inner), (columns, inner), &device)?
                    .to_dtype(DType::BF16)?,
                None,
            );
            let bits = |variant| -> Result<Vec<u32>> {
                Ok(cuda_bf16_projection_impl(&x, &linear, false, variant)?
                    .to_dtype(DType::F32)?
                    .flatten_all()?
                    .to_vec1::<f32>()?
                    .into_iter()
                    .map(f32::to_bits)
                    .collect())
            };
            for candidate in [
                SimtProjection::TiledCompensated,
                SimtProjection::GuardedDouble,
                SimtProjection::GuardedTensorDouble,
                SimtProjection::GuardedTensorDoubleRows16,
            ] {
                assert_eq!(
                    bits(SimtProjection::Compensated)?,
                    bits(candidate)?,
                    "guarded double must preserve exact original bits {rows}/{inner}/{columns}"
                );
            }
        }
        Ok(())
    }

    #[test]
    fn guarded_double_projection_cuda_preserves_midpoints_cancellation_and_exceptional_bits()
    -> Result<()> {
        let device = test_cuda_device()?;
        if !supported_rotary_test_dtypes(&device)?.contains(&DType::BF16) {
            return Ok(());
        }
        let mut cancellation = vec![0.; 65];
        cancellation[0] = 2_f32.powi(60);
        cancellation[32] = 1.;
        cancellation[64] = -2_f32.powi(60);
        let cases = [
            (vec![1.; 65], cancellation, Some(1_f32.to_bits())),
            (vec![1.; 3], vec![1., 2_f32.powi(-8), 2_f32.powi(-20)], None),
            (
                vec![1.; 3],
                vec![1., 2_f32.powi(-8), -2_f32.powi(-20)],
                None,
            ),
            (
                vec![2_f32.powi(100); 2],
                vec![2_f32.powi(100), -2_f32.powi(100)],
                None,
            ),
            (vec![2_f32.powi(-100); 2], vec![2_f32.powi(-100); 2], None),
            (vec![-0.; 33], vec![1.; 33], None),
            (vec![f32::INFINITY; 2], vec![1., -1.], None),
        ];
        for (xs, ws, exact) in cases {
            let inner = xs.len();
            let x = Tensor::from_vec(xs, (1, 1, inner), &device)?.to_dtype(DType::BF16)?;
            let linear = Linear::new(
                Tensor::from_vec(ws, (1, inner), &device)?.to_dtype(DType::BF16)?,
                None,
            );
            let bits = |variant| -> Result<u32> {
                Ok(cuda_bf16_projection_impl(&x, &linear, false, variant)?
                    .to_dtype(DType::F32)?
                    .flatten_all()?
                    .to_vec1::<f32>()?[0]
                    .to_bits())
            };
            let retained = bits(SimtProjection::Compensated)?;
            assert_eq!(
                retained,
                bits(SimtProjection::TiledCompensated)?,
                "tiled compensated exceptional bits"
            );
            assert_eq!(
                retained,
                bits(SimtProjection::GuardedDouble)?,
                "including NaN/signed-zero bit patterns"
            );
            assert_eq!(
                retained,
                bits(SimtProjection::GuardedTensorDouble)?,
                "tensor-double exceptional bits"
            );
            assert_eq!(
                retained,
                bits(SimtProjection::GuardedTensorDoubleRows16)?,
                "rows16 tensor-double exceptional bits"
            );
            if let Some(expected) = exact {
                assert_eq!(
                    retained, expected,
                    "plain FP64 loses this cancellation residual; guard must fall back"
                );
            }
        }
        Ok(())
    }

    #[test]
    fn tiled_compensated_projection_cuda_preserves_bias_offsets_batch_and_padding() -> Result<()> {
        let device = test_cuda_device()?;
        if !supported_rotary_test_dtypes(&device)?.contains(&DType::BF16) {
            return Ok(());
        }
        let bits = |tensor: &Tensor| -> Result<Vec<u32>> {
            Ok(tensor
                .to_dtype(DType::F32)?
                .flatten_all()?
                .to_vec1::<f32>()?
                .into_iter()
                .map(f32::to_bits)
                .collect())
        };
        for (rows, inner, columns) in [(1, 1, 2), (3, 33, 9), (5, 129, 17), (2, 2048, 64)] {
            let xs: Vec<f32> = (0..(rows + 1) * inner)
                .map(|i| ((i * 17 % 255) as f32 - 127.) / 64.)
                .collect();
            let ws: Vec<f32> = (0..(columns + 1) * inner)
                .map(|i| ((i * 29 % 127) as f32 - 63.) / 512.)
                .collect();
            let bs: Vec<f32> = (0..columns + 1).map(|i| (i as f32 - 8.) / 128.).collect();
            let x = Tensor::from_vec(xs, (rows + 1, inner), &device)?
                .to_dtype(DType::BF16)?
                .narrow(0, 1, rows)?
                .unsqueeze(0)?;
            let w = Tensor::from_vec(ws, (columns + 1, inner), &device)?
                .to_dtype(DType::BF16)?
                .narrow(0, 1, columns)?;
            let b = Tensor::from_vec(bs, columns + 1, &device)?
                .to_dtype(DType::BF16)?
                .narrow(0, 1, columns)?;
            let layer = Linear::new(w, Some(b));
            let run = |input, mode| cuda_bf16_projection_impl(input, &layer, false, mode);
            let expected = bits(&run(&x, SimtProjection::Compensated)?)?;
            assert_eq!(bits(&run(&x, SimtProjection::TiledCompensated)?)?, expected);
            assert_eq!(bits(&run(&x, SimtProjection::TiledCompensated)?)?, expected);
            let batch = Tensor::cat(&[&x, &x], 0)?;
            assert_eq!(
                bits(&run(&batch, SimtProjection::TiledCompensated)?)?,
                [expected.clone(), expected.clone()].concat()
            );
            let padded = Tensor::cat(
                &[&x, &Tensor::zeros((1, 19, inner), DType::BF16, &device)?],
                1,
            )?;
            assert_eq!(
                bits(&run(&padded, SimtProjection::TiledCompensated)?.narrow(1, 0, rows)?)?,
                expected
            );
        }

        let x = Tensor::ones((1, 1, 1), DType::BF16, &device)?;
        let layer = Linear::new(
            Tensor::from_vec(vec![1_f32, -1_f32], (2, 1), &device)?.to_dtype(DType::BF16)?,
            Some(
                Tensor::from_vec(vec![1_f32 / 256., -1_f32 / 256.], 2, &device)?
                    .to_dtype(DType::BF16)?,
            ),
        );
        assert_eq!(
            bits(&cuda_bf16_projection_impl(
                &x,
                &layer,
                false,
                SimtProjection::TiledCompensated
            )?)?,
            vec![1_f32.to_bits(), (-1_f32).to_bits()]
        );
        Ok(())
    }

    #[test]
    fn pinned_blaslt_vs_compensated_cuda_microprofile() -> Result<()> {
        let device = test_cuda_device()?;
        if !supported_rotary_test_dtypes(&device)?.contains(&DType::BF16) {
            return Ok(());
        }

        for (rows, inner, columns) in [
            (128, 2560, 4096),
            (512, 2560, 4096),
            (128, 2560, 9216),
            (128, 9216, 2560),
        ] {
            let x = Tensor::from_vec(
                (0..rows * inner)
                    .map(|i| ((i * 17 % 255) as f32 - 127.) / 64.)
                    .collect::<Vec<_>>(),
                (1, rows, inner),
                &device,
            )?
            .to_dtype(DType::BF16)?;
            let linear = Linear::new(
                Tensor::from_vec(
                    (0..columns * inner)
                        .map(|i| ((i * 29 % 127) as f32 - 63.) / 512.)
                        .collect::<Vec<_>>(),
                    (columns, inner),
                    &device,
                )?
                .to_dtype(DType::BF16)?,
                None,
            );
            let run = |index| match index {
                0 => crate::kernels::cuda::blaslt::project_pinned512(&x, &linear),
                1 => cuda_bf16_projection_impl(&x, &linear, false, SimtProjection::Compensated),
                _ => {
                    cuda_bf16_projection_impl(&x, &linear, false, SimtProjection::TiledCompensated)
                }
            };
            let baseline = run(0)?
                .to_dtype(DType::F32)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            for candidate in [1, 2] {
                let values = run(candidate)?
                    .to_dtype(DType::F32)?
                    .flatten_all()?
                    .to_vec1::<f32>()?;
                let different = baseline
                    .iter()
                    .zip(&values)
                    .filter(|(a, b)| a.to_bits() != b.to_bits())
                    .count();
                let max_abs = baseline
                    .iter()
                    .zip(&values)
                    .map(|(a, b)| (*a - *b).abs())
                    .fold(0f32, f32::max);
                for _ in 0..2 {
                    let _ = run(0)?;
                    let _ = run(candidate)?;
                }
                let mut samples = [Vec::new(), Vec::new()];
                for iteration in 0..6 {
                    for index in if iteration % 2 == 0 { [0, 1] } else { [1, 0] } {
                        device.synchronize()?;
                        let started = Instant::now();
                        let _ = run(if index == 0 { 0 } else { candidate })?;
                        device.synchronize()?;
                        samples[index].push(started.elapsed().as_secs_f64() * 1000.);
                    }
                }
                let median = |values: &mut Vec<f64>| {
                    values.sort_by(f64::total_cmp);
                    (values[2] + values[3]) / 2.
                };
                let old_ms = median(&mut samples[0]);
                let candidate_ms = median(&mut samples[1]);
                println!(
                    "PINNED_BLASLT_PROJECTION_PROFILE rows={rows} inner={inner} columns={columns} candidate={candidate} old_ms={old_ms:.9} candidate_ms={candidate_ms:.9} ratio={:.9} different_elements={different} max_abs={max_abs} scope=seeded_operator_allocation_launch_and_barriers_not_http",
                    candidate_ms / old_ms
                );
            }
        }
        Ok(())
    }

    #[test]
    fn pinned_blaslt_combined_gate_up_cuda_microprofile() -> Result<()> {
        let device = test_cuda_device()?;
        if !supported_rotary_test_dtypes(&device)?.contains(&DType::BF16) {
            return Ok(());
        }

        for (rows, inner, columns) in [
            (32, 4096, 12288),
            (128, 4096, 12288),
            (512, 4096, 12288),
            (128, 5120, 17408),
        ] {
            let tensor = |shape: &[usize], seed: u32| -> Result<Tensor> {
                let mut state = seed;
                let values = (0..shape.iter().product::<usize>())
                    .map(|_| {
                        state = state.wrapping_mul(1664525).wrapping_add(1013904223);
                        ((state >> 16) as f32 / 65536. - 0.5) / 8.
                    })
                    .collect::<Vec<_>>();
                Tensor::from_vec(values, shape, &device)?.to_dtype(DType::BF16)
            };
            let x = tensor(&[1, rows, inner], 123)?;
            let gate = Linear::new(tensor(&[columns, inner], 456)?, None);
            let up = Linear::new(tensor(&[columns, inner], 789)?, None);
            let packed = Linear::new(Tensor::cat(&[gate.weight(), up.weight()], 0)?, None);
            let run = |combined: bool| -> Result<(Tensor, Tensor)> {
                if combined {
                    let output = crate::kernels::cuda::blaslt::project_pinned512(&x, &packed)?;
                    Ok((
                        output.narrow(2, 0, columns)?.contiguous()?,
                        output.narrow(2, columns, columns)?.contiguous()?,
                    ))
                } else {
                    Ok((
                        crate::kernels::cuda::blaslt::project_pinned512(&x, &gate)?,
                        crate::kernels::cuda::blaslt::project_pinned512(&x, &up)?,
                    ))
                }
            };
            let host = |pair: (Tensor, Tensor)| -> Result<Vec<f32>> {
                Tensor::cat(&[&pair.0, &pair.1], 2)?
                    .to_dtype(DType::F32)?
                    .flatten_all()?
                    .to_vec1()
            };
            let baseline = host(run(false)?)?;
            let candidate = host(run(true)?)?;
            let different = baseline
                .iter()
                .zip(&candidate)
                .filter(|(a, b)| a.to_bits() != b.to_bits())
                .count();
            let max_abs = baseline
                .iter()
                .zip(&candidate)
                .map(|(a, b)| (*a - *b).abs())
                .fold(0f32, f32::max);
            for _ in 0..2 {
                let _ = run(false)?;
                let _ = run(true)?;
            }
            let mut samples = [Vec::new(), Vec::new()];
            for iteration in 0..6 {
                for index in if iteration % 2 == 0 { [0, 1] } else { [1, 0] } {
                    device.synchronize()?;
                    let started = Instant::now();
                    let _ = run(index == 1)?;
                    device.synchronize()?;
                    samples[index].push(started.elapsed().as_secs_f64() * 1000.);
                }
            }
            let median = |values: &mut Vec<f64>| {
                values.sort_by(f64::total_cmp);
                (values[2] + values[3]) / 2.
            };
            let old_ms = median(&mut samples[0]);
            let candidate_ms = median(&mut samples[1]);
            println!(
                "PINNED_BLASLT_GATE_UP_PROFILE rows={rows} inner={inner} columns={columns} old_ms={old_ms:.9} candidate_ms={candidate_ms:.9} ratio={:.9} different_elements={different} max_abs={max_abs} scope=seeded_operator_including_output_copies_not_http",
                candidate_ms / old_ms
            );
            assert_eq!(
                different, 0,
                "packing changes seeded gate/up projection bits"
            );
        }
        Ok(())
    }

    #[test]
    fn guarded_double_projection_cuda_paired_synchronized_microprofile() -> Result<()> {
        let device = test_cuda_device()?;
        if !supported_rotary_test_dtypes(&device)?.contains(&DType::BF16) {
            return Ok(());
        }
        for (rows, inner, columns) in [
            (17, 4096, 32),
            (128, 2560, 4096),
            (512, 2560, 4096),
            (128, 2560, 9216),
            (128, 9216, 2560),
        ] {
            let xs = (0..rows * inner)
                .map(|i| ((i * 17 % 255) as f32 - 127.) / 64.)
                .collect::<Vec<_>>();
            let ws = (0..columns * inner)
                .map(|i| ((i * 29 % 127) as f32 - 63.) / 512.)
                .collect::<Vec<_>>();
            let x = Tensor::from_vec(xs, (1, rows, inner), &device)?.to_dtype(DType::BF16)?;
            let linear = Linear::new(
                Tensor::from_vec(ws, (columns, inner), &device)?.to_dtype(DType::BF16)?,
                None,
            );
            let run = |variant| cuda_bf16_projection_impl(&x, &linear, false, variant);
            for candidate in [
                SimtProjection::TiledCompensated,
                SimtProjection::GuardedDouble,
                SimtProjection::GuardedTensorDouble,
                SimtProjection::GuardedTensorDoubleRows16,
            ] {
                let retained = run(SimtProjection::Compensated)?;
                let guarded = run(candidate)?;
                assert_eq!(
                    retained
                        .to_dtype(DType::F32)?
                        .flatten_all()?
                        .to_vec1::<f32>()?,
                    guarded
                        .to_dtype(DType::F32)?
                        .flatten_all()?
                        .to_vec1::<f32>()?
                );
                for _ in 0..2 {
                    let _ = run(SimtProjection::Compensated)?;
                    let _ = run(candidate)?;
                }
                device.synchronize()?;
                let mut samples = [Vec::new(), Vec::new()];
                for iteration in 0..6 {
                    for index in if iteration % 2 == 0 { [0, 1] } else { [1, 0] } {
                        device.synchronize()?;
                        let started = Instant::now();
                        let _ = run(if index == 0 {
                            SimtProjection::Compensated
                        } else {
                            candidate
                        })?;
                        device.synchronize()?;
                        samples[index].push(started.elapsed().as_secs_f64() * 1000.);
                    }
                }
                let median = |values: &mut Vec<f64>| {
                    values.sort_by(f64::total_cmp);
                    (values[2] + values[3]) / 2.
                };
                let old_ms = median(&mut samples[0]);
                let candidate_ms = median(&mut samples[1]);
                let profile_kind = if matches!(candidate, SimtProjection::TiledCompensated) {
                    "TILED_COMPENSATED_PROFILE"
                } else if matches!(candidate, SimtProjection::GuardedTensorDoubleRows16) {
                    "GUARDED_TENSOR_DOUBLE_ROWS16_PROFILE"
                } else if matches!(candidate, SimtProjection::GuardedTensorDouble) {
                    "GUARDED_TENSOR_DOUBLE_PROFILE"
                } else {
                    "GUARDED_DOUBLE_PROFILE"
                };
                println!(
                    "{profile_kind} rows={rows} inner={inner} columns={columns} old_ms={old_ms:.9} candidate_ms={candidate_ms:.9} ratio={:.9} scope=allocation_launch_and_barriers_not_http",
                    candidate_ms / old_ms
                );
            }
        }
        Ok(())
    }

    #[test]
    fn full_attention_kv_cuda_preserves_exceptional_products() -> Result<()> {
        let device = test_cuda_device()?;
        if !supported_rotary_test_dtypes(&device)?.contains(&DType::BF16) {
            return Ok(());
        }
        for (xs, ws) in [
            (
                vec![2_f32.powi(100); 2],
                vec![2_f32.powi(100), -2_f32.powi(100)],
            ),
            (vec![2_f32.powi(-70); 129], vec![2_f32.powi(-70); 129]),
        ] {
            let expected = bf16::from_f32(
                xs.iter()
                    .zip(&ws)
                    .map(|(&x, &w)| f64::from(x) * f64::from(w))
                    .sum::<f64>() as f32,
            )
            .to_f32();
            let x =
                Tensor::from_vec(xs.clone(), (1, 1, xs.len()), &device)?.to_dtype(DType::BF16)?;
            let layer = Linear::new(
                Tensor::from_vec(ws, (1, xs.len()), &device)?.to_dtype(DType::BF16)?,
                None,
            );
            let actual = full_attention_kv_forward(&x, &layer)?
                .to_dtype(DType::F32)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            assert_eq!(
                actual,
                vec![expected],
                "full K/V exceptional-product recovery"
            );
        }
        Ok(())
    }
}

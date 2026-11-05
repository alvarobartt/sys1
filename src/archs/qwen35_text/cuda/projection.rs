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

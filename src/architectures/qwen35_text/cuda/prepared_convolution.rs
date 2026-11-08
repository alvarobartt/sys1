use super::*;
use candle_core::cuda_backend::{
    WrapErr,
    cudarc::driver::{LaunchConfig, PushKernelArg},
};

pub(crate) fn prepare(
    input: &Tensor,
    weights: &Tensor,
    key_heads: usize,
    value_heads: usize,
    sequences: &PackedSequences,
) -> Result<Option<gated_delta::PreparedQv>> {
    if !matches!(
        option_env!("SYS1_CUDA_COMPUTE_CAP"),
        Some("80" | "86" | "89" | "90")
    ) || !input.device().is_cuda()
        || input.dtype() != DType::BF16
        || weights.dtype() != DType::BF16
        || !input.device().same_device(weights.device())
        || key_heads != 16
        || value_heads != 32
        || input.dims() != [1, sequences.total, 8192]
        || weights.dims() != [8192, 1, 4]
        || sequences.total < 513
        || input.elem_count() > i32::MAX as usize
        || !input.device().same_device(sequences.starts.device())
    {
        return Ok(None);
    }
    let producer_input = input;
    let producer_weights = weights;
    let merged_input = input.stride() == [sequences.total * 12288, 12288, 1];
    let input = if merged_input {
        input.clone()
    } else {
        input.contiguous()?
    };
    let weights = weights.contiguous()?;
    let (xs, xl) = input.storage_and_layout();
    let (ws, wl) = weights.storage_and_layout();
    let (ss, sl) = sequences.starts.storage_and_layout();
    let (Storage::Cuda(xs), Storage::Cuda(ws), Storage::Cuda(ss)) = (&*xs, &*ws, &*ss) else {
        candle_core::bail!("prepared-only producer requires CUDA storage")
    };
    let offsets = |layout: &candle_core::Layout| {
        layout.contiguous_offsets().ok_or_else(|| {
            candle_core::Error::Msg("prepared-only producer requires contiguous offsets".into())
        })
    };
    let (x0, x1) = if merged_input {
        let start = xl.start_offset();
        let span = (sequences.total - 1)
            .checked_mul(12288)
            .and_then(|n| n.checked_add(8192))
            .and_then(|n| start.checked_add(n))
            .ok_or_else(|| candle_core::Error::Msg("merged prepared span overflow".into()))?;
        if span > xs.as_cuda_slice::<bf16>()?.len() {
            candle_core::bail!("merged prepared span exceeds allocation")
        }
        (start, span)
    } else {
        offsets(xl)?
    };
    let (w0, w1) = offsets(wl)?;
    let (s0, s1) = offsets(sl)?;

    if x0 % 4 != 0 || w0 % 8 != 0 {
        return Ok(None);
    }
    let source = xs.as_cuda_slice::<bf16>()?.slice(x0..x1);
    let filters = ws.as_cuda_slice::<bf16>()?.slice(w0..w1);
    let starts = ss.as_cuda_slice::<u32>()?.slice(s0..s1);
    let device = xs.device().clone();
    let Some(mut ticket) = gated_delta::prepare_qv(input.device(), &sequences.lengths)? else {
        return Ok(None);
    };
    ticket.bind_producer(producer_input, producer_weights)?;
    let function = device.get_or_load_custom_func(
        if merged_input {
            "hopper_qkvz_conv_prepared_strided"
        } else {
            "halfonly_convcast_qkv_packed"
        },
        "sys1_conv_prepared_only",
        CUDA_PTX,
    )?;
    let rows = (sequences.total * key_heads) as u32;
    let kh = key_heads as i32;
    let vh = value_heads as i32;
    let scale = 128f64.powf(-0.5) as f32;
    let length = sequences.total as i32;
    let kernel = 4i32;

    let null = 0u64;
    crate::kernels::cuda::activation::with_sigmoid_lookup(&device, |table| {
        let (qhalf, khalf, vhalf, limit, bad) = ticket.outputs();
        let mut launch = function.builder();
        launch.arg(&source).arg(&filters);
        launch.arg(&null).arg(&null).arg(&null).arg(&starts);
        launch.arg(&rows).arg(&kh).arg(&vh).arg(&scale);
        launch.arg(&length).arg(&kernel).arg(table);
        launch.arg(qhalf).arg(khalf).arg(vhalf).arg(&limit).arg(bad);
        unsafe {
            launch.launch(LaunchConfig {
                grid_dim: (rows.div_ceil(4), 1, 1),
                block_dim: (128, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .w()?;
        Ok(())
    })?;
    #[cfg(test)]
    if merged_input {
        super::qkvz::STRIDED_PREPARED.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
    #[cfg(test)]
    CONV_PREPARED_QV_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    Ok(Some(ticket))
}

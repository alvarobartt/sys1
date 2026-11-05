pub(crate) use candle_flash_attn::{flash_attn_varlen_windowed, flash_attn_windowed};
#[cfg(sys1_cuda_fa4_sm90)]
mod fa4;

pub(crate) fn flash_attn(
    q: &candle_core::Tensor,
    k: &candle_core::Tensor,
    v: &candle_core::Tensor,
    scale: f32,
    causal: bool,
) -> candle_core::Result<candle_core::Tensor> {
    #[cfg(sys1_cuda_fa4_sm90)]
    if let Some(output) = fa4::uniform(q, k, v, scale, causal)? {
        return Ok(output);
    }
    candle_flash_attn::flash_attn(q, k, v, scale, causal)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn flash_attn_varlen(
    q: &candle_core::Tensor,
    k: &candle_core::Tensor,
    v: &candle_core::Tensor,
    cuq: &candle_core::Tensor,
    cuk: &candle_core::Tensor,
    maximum_q: usize,
    maximum_k: usize,
    scale: f32,
    causal: bool,
) -> candle_core::Result<candle_core::Tensor> {
    #[cfg(sys1_cuda_fa4_sm90)]
    if let Some(output) = fa4::varlen(q, k, v, cuq, cuk, maximum_q, maximum_k, scale, causal)? {
        return Ok(output);
    }
    candle_flash_attn::flash_attn_varlen(q, k, v, cuq, cuk, maximum_q, maximum_k, scale, causal)
}

#[cfg(test)]
pub(crate) fn supported_test_device(device: &candle_core::Device) -> candle_core::Result<bool> {
    let candle_core::Device::Cuda(cuda) = device else {
        return Ok(false);
    };
    let capability = cuda
        .cuda_stream()
        .context()
        .compute_capability()
        .map_err(|error| candle_core::Error::Msg(error.to_string()))?;
    Ok((8..=9).contains(&capability.0) || capability == (12, 0))
}

pub(crate) fn ordinary_reference(
    q: &candle_core::Tensor,
    k: &candle_core::Tensor,
    v: &candle_core::Tensor,
    scale: f32,
    causal: bool,
) -> candle_core::Result<candle_core::Tensor> {
    use candle_core::{D, DType, Tensor};
    let (batch, length, heads, width) = q.dims4()?;
    let (_, keys, kv_heads, _) = k.dims4()?;
    let expand = |x: &Tensor| {
        x.unsqueeze(3)?
            .broadcast_as((batch, keys, kv_heads, heads / kv_heads, width))?
            .reshape((batch, keys, heads, width))?
            .transpose(1, 2)
    };
    let query = q.transpose(1, 2)?;
    let key = expand(k)?.transpose(D::Minus2, D::Minus1)?.contiguous()?;
    let value = expand(v)?.contiguous()?;
    let mut chunks = Vec::new();
    for start in (0..length).step_by(128) {
        let count = (length - start).min(128);
        let mut scores =
            (query.narrow(2, start, count)?.contiguous()?.matmul(&key)? * scale as f64)?;
        if causal {
            let mask = Tensor::from_vec(
                (start..start + count)
                    .flat_map(|row| {
                        (0..keys).map(move |col| if col > row { f32::NEG_INFINITY } else { 0. })
                    })
                    .collect::<Vec<_>>(),
                (1, 1, count, keys),
                q.device(),
            )?
            .to_dtype(q.dtype())?;
            scores = scores.broadcast_add(&mask)?;
        }
        let probabilities =
            candle_nn::ops::softmax_last_dim(&scores.to_dtype(DType::F32)?.contiguous()?)?
                .to_dtype(q.dtype())?;
        chunks.push(probabilities.contiguous()?.matmul(&value)?);
    }
    Tensor::cat(&chunks.iter().collect::<Vec<_>>(), 2)?.transpose(1, 2)
}

#[cfg(test)]
mod reference_tests {
    use candle_core::{DType, Device, Result, Tensor};

    fn input(
        shape: (usize, usize, usize, usize),
        seed: u64,
        magnitude: f32,
        dtype: DType,
        device: &Device,
    ) -> Result<Tensor> {
        let mut state = seed;
        let values = (0..shape.0 * shape.1 * shape.2 * shape.3)
            .map(|_| {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                (((state >> 40) as f32 / (1u32 << 24) as f32) - 0.5) * magnitude
            })
            .collect::<Vec<_>>();
        Tensor::from_vec(values, shape, device)?.to_dtype(dtype)
    }

    fn ordinary(q: &Tensor, k: &Tensor, v: &Tensor, scale: f32, causal: bool) -> Result<Tensor> {
        super::ordinary_reference(q, k, v, scale, causal)
    }

    fn errors(actual: &Tensor, reference: &Tensor) -> Result<(f32, f64)> {
        let actual = actual
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let reference = reference.flatten_all()?.to_vec1::<f32>()?;
        let mut max = 0f32;
        let mut total = 0f64;
        for (actual, reference) in actual.iter().zip(reference.iter()) {
            assert!(actual.is_finite() && reference.is_finite());
            let delta = (actual - reference).abs();
            max = max.max(delta);
            total += delta as f64;
        }
        Ok((max, total / actual.len() as f64))
    }

    #[test]
    fn flash_forward_error_is_bounded_by_twice_low_precision_eager_error() -> Result<()> {
        let device =
            crate::device::load().map_err(|error| candle_core::Error::Msg(error.to_string()))?;
        if !super::supported_test_device(&device)? {
            println!("FA_REFERENCE_SKIPPED unsupported CUDA architecture");
            return Ok(());
        }

        for (name, queries, keys, heads, kv_heads, width, causal, magnitude) in [
            ("qwen128", 128, 128, 16, 4, 256, true, 2.),
            ("qwen262", 262, 262, 16, 4, 256, true, 2.),
            ("clef27b_gqa24_4", 262, 262, 24, 4, 256, true, 2.),
            (
                "clef27b_gqa24_4_high_scores",
                513,
                513,
                24,
                4,
                256,
                true,
                8.,
            ),
            ("qwen513_high_scores", 513, 513, 16, 4, 256, true, 8.),
            ("head_single", 1, 1, 16, 16, 64, false, 2.),
            ("head_self129", 129, 129, 16, 16, 64, false, 2.),
            ("head_cross32_257", 32, 257, 16, 16, 64, false, 2.),
        ] {
            for dtype in [DType::F16, DType::BF16] {
                let q = input(
                    (1, queries, heads, width),
                    0x6a09e667f3bcc909,
                    magnitude,
                    dtype,
                    &device,
                )?;
                let k = input(
                    (1, keys, kv_heads, width),
                    0xbb67ae8584caa73b,
                    magnitude,
                    dtype,
                    &device,
                )?;
                let v = input(
                    (1, keys, kv_heads, width),
                    0x3c6ef372fe94f82b,
                    2.,
                    dtype,
                    &device,
                )?;
                let scale = (width as f32).powf(-0.5);
                let reference = ordinary(
                    &q.to_dtype(DType::F32)?,
                    &k.to_dtype(DType::F32)?,
                    &v.to_dtype(DType::F32)?,
                    scale,
                    causal,
                )?;
                let eager = ordinary(&q, &k, &v, scale, causal)?;
                let flash = super::flash_attn(&q, &k, &v, scale, causal)?;
                let (eager_max, eager_mean) = errors(&eager, &reference)?;
                let (flash_max, flash_mean) = errors(&flash, &reference)?;
                let allowed = 2. * eager_max;
                println!(
                    "FLASH_REFERENCE case={name} dtype={dtype:?} eager_max={eager_max:.9} flash_max={flash_max:.9} allowed={allowed:.9} eager_mean={eager_mean:.9} flash_mean={flash_mean:.9}"
                );

                assert!(
                    flash_max <= allowed,
                    "{name} {dtype:?}: {flash_max} > {allowed}"
                );
            }
        }
        Ok(())
    }
}

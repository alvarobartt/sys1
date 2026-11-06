use candle_core::cuda_backend::{
    WrapErr,
    cudarc::driver::{LaunchConfig, PushKernelArg},
};
use candle_core::{
    CudaStorage, DType, Result, Storage, Tensor, backend::BackendStorage, op::BackpropOp,
};
use half::bf16;

const CUDA_PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/cuda.ptx"));

pub(crate) fn forward(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    scale: f32,
    causal: bool,
) -> Result<Tensor> {
    forward_impl(q, k, v, scale, causal, false)
}

pub(crate) fn forward_shared(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    scale: f32,
    causal: bool,
) -> Result<Tensor> {
    forward_impl(q, k, v, scale, causal, true)
}

fn shared_shape(heads: usize, kv_heads: usize, causal: bool) -> bool {
    causal
        && kv_heads > 0
        && heads.is_multiple_of(kv_heads)
        && heads.is_multiple_of(4)
        && heads / kv_heads >= 4
        && (heads / kv_heads).is_multiple_of(4)
}

fn forward_impl(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    scale: f32,
    causal: bool,
    shared: bool,
) -> Result<Tensor> {
    let (batch, queries, heads, width) = q.dims4()?;
    let (key_batch, keys, kv_heads, key_width) = k.dims4()?;
    if batch == 0
        || queries == 0
        || keys == 0
        || heads == 0
        || kv_heads == 0
        || !matches!(width, 32 | 64 | 128 | 256)
        || key_width != width
        || batch != key_batch
        || heads % kv_heads != 0
        || k.dims() != v.dims()
        || (causal && queries != keys)
        || !scale.is_finite()
        || !q.device().is_cuda()
        || !q.device().same_device(k.device())
        || !q.device().same_device(v.device())
        || [q, k, v]
            .iter()
            .any(|x| x.dtype() != DType::BF16 || x.elem_count() > i32::MAX as usize)
    {
        candle_core::bail!("Unsupported fixed-order BF16 attention shape, dtype, scale or device")
    }
    let q = q.contiguous()?;
    let k = k.contiguous()?;
    let v = v.contiguous()?;
    let (qs, ql) = q.storage_and_layout();
    let (ks, kl) = k.storage_and_layout();
    let (vs, vl) = v.storage_and_layout();
    let (Storage::Cuda(qs), Storage::Cuda(ks), Storage::Cuda(vs)) = (&*qs, &*ks, &*vs) else {
        candle_core::bail!("Fixed-order attention requires CUDA storage")
    };
    let offsets = |layout: &candle_core::Layout| {
        layout
            .contiguous_offsets()
            .ok_or_else(|| candle_core::Error::Msg("Noncontiguous attention storage".into()))
    };
    let (start, end) = offsets(ql)?;
    let q_slice = qs.as_cuda_slice::<bf16>()?.slice(start..end);
    let (start, end) = offsets(kl)?;
    let k_slice = ks.as_cuda_slice::<bf16>()?.slice(start..end);
    let (start, end) = offsets(vl)?;
    let v_slice = vs.as_cuda_slice::<bf16>()?.slice(start..end);
    let device = qs.device().clone();
    let mut output = unsafe { device.alloc::<bf16>(q.elem_count())? };
    let name = if shared && shared_shape(heads, kv_heads, causal) {
        "qwen35_online_attention_shared_bf16"
    } else {
        "qwen35_online_attention_bf16"
    };
    let function = device.get_or_load_custom_func(name, "sys1_qwen35", CUDA_PTX)?;
    let query_heads = (batch * queries * heads) as i32;
    let queries_arg = queries as i32;
    let keys_arg = keys as i32;
    let heads_arg = heads as i32;
    let kv_heads_arg = kv_heads as i32;
    let width_arg = width as i32;
    let causal_arg = i32::from(causal);
    let mut launch = function.builder();
    launch
        .arg(&q_slice)
        .arg(&k_slice)
        .arg(&v_slice)
        .arg(&mut output)
        .arg(&query_heads)
        .arg(&queries_arg)
        .arg(&keys_arg)
        .arg(&heads_arg)
        .arg(&kv_heads_arg)
        .arg(&width_arg)
        .arg(&scale)
        .arg(&causal_arg);
    unsafe {
        launch.launch(LaunchConfig {
            grid_dim: ((query_heads as u32).div_ceil(4), 1, 1),
            block_dim: (128, 1, 1),
            shared_mem_bytes: 0,
        })
    }
    .w()?;
    Ok(Tensor::from_storage(
        Storage::Cuda(CudaStorage::wrap_cuda_slice(output, device)),
        q.shape().clone(),
        BackpropOp::none(),
        false,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn shared_attention_shape_requires_block_uniform_kv_and_causality() {
        for (heads, kv_heads) in [(4, 1), (8, 1), (16, 2), (32, 8)] {
            assert!(shared_shape(heads, kv_heads, true));
            assert!(!shared_shape(heads, kv_heads, false));
        }
        for (heads, kv_heads) in [(2, 1), (4, 2), (8, 8), (12, 2), (8, 0), (8, 3)] {
            assert!(!shared_shape(heads, kv_heads, true));
        }
    }

    #[test]
    fn shared_attention_cuda_is_bitwise_scalar_with_tiles_tails_batch_and_padding() -> Result<()> {
        let device = crate::device::load().map_err(|e| candle_core::Error::Msg(e.to_string()))?;
        if let Device::Cuda(cuda) = &device {
            if cuda.cuda_stream().context().compute_capability().w()?.0 < 8 {
                return Ok(());
            }
        } else {
            return Ok(());
        }
        let values =
            |x: &Tensor| -> Result<Vec<f32>> { x.to_dtype(DType::F32)?.flatten_all()?.to_vec1() };
        for (length, heads, kv_heads, width) in [
            (1, 4, 1, 32),
            (17, 8, 1, 64),
            (33, 4, 1, 128),
            (65, 16, 2, 256),
            (133, 32, 8, 128),
            (511, 4, 1, 256),
            (33, 4, 2, 64),
        ] {
            let q = input((2, length, heads, width), 991, &device)?;
            let k = input((2, length, kv_heads, width), 992, &device)?;
            let v = input((2, length, kv_heads, width), 993, &device)?;
            let scale = (width as f32).sqrt().recip();
            let scalar = forward(&q, &k, &v, scale, true)?;
            let shared = forward_shared(&q, &k, &v, scale, true)?;
            assert_eq!(
                values(&scalar)?,
                values(&shared)?,
                "shared/scalar {length}/{width}"
            );
            assert_eq!(
                values(&shared)?,
                values(&forward_shared(&q, &k, &v, scale, true)?)?
            );
            let pad = |x: &Tensor, nheads: usize| -> Result<Tensor> {
                Tensor::cat(
                    &[
                        x,
                        &Tensor::zeros((2, 3, nheads, width), DType::BF16, &device)?,
                    ],
                    1,
                )
            };
            let padded = forward_shared(
                &pad(&q, heads)?,
                &pad(&k, kv_heads)?,
                &pad(&v, kv_heads)?,
                scale,
                true,
            )?;
            assert_eq!(
                values(&shared)?,
                values(&padded.narrow(1, 0, length)?)?,
                "shared padding"
            );
        }
        Ok(())
    }

    fn input(shape: (usize, usize, usize, usize), seed: u32, device: &Device) -> Result<Tensor> {
        let mut state = seed;
        let values = (0..shape.0 * shape.1 * shape.2 * shape.3)
            .map(|_| {
                state = state.wrapping_mul(1664525).wrapping_add(1013904223);
                ((state >> 16) as f32 / 65536.0 - 0.5) * 2.0
            })
            .collect::<Vec<_>>();
        Tensor::from_vec(values, shape, device)?.to_dtype(DType::BF16)
    }

    #[test]
    fn fixed_order_attention_cuda_cross_attention_unequal_lengths_and_gqa() -> Result<()> {
        let device = crate::device::load().map_err(|e| candle_core::Error::Msg(e.to_string()))?;
        if let Device::Cuda(cuda) = &device {
            if cuda.cuda_stream().context().compute_capability().w()?.0 < 8 {
                return Ok(());
            }
        } else {
            return Ok(());
        }
        let values =
            |x: &Tensor| -> Result<Vec<f32>> { x.to_dtype(DType::F32)?.flatten_all()?.to_vec1() };

        for (queries, keys, heads, kv_heads, width) in [
            (7, 1185, 4, 2, 64),
            (35, 17, 2, 2, 128),
            (3, 401, 2, 1, 256),
        ] {
            let q = input((2, queries, heads, width), 321, &device)?;
            let k = input((2, keys, kv_heads, width), 654, &device)?;
            let v = input((2, keys, kv_heads, width), 987, &device)?;
            let scale = (width as f32).sqrt().recip();
            let actual = forward(&q, &k, &v, scale, false)?;
            assert_eq!(actual.dims(), q.dims());
            assert_eq!(
                values(&actual)?,
                values(&forward(&q, &k, &v, scale, false)?)?
            );
            for slot in 0..2 {
                let single = forward(
                    &q.narrow(0, slot, 1)?,
                    &k.narrow(0, slot, 1)?,
                    &v.narrow(0, slot, 1)?,
                    scale,
                    false,
                )?;
                assert_eq!(values(&actual.narrow(0, slot, 1)?)?, values(&single)?);
            }
            let reference =
                crate::architectures::qwen35_text::eager_attention(&q, &k, &v, scale, false)?;
            for (a, b) in values(&actual)?.iter().zip(values(&reference)?) {
                assert!(a.is_finite() && b.is_finite());
                assert!(
                    (a - b).abs() <= 1e-5 + 1.6e-2 * b.abs(),
                    "BF16 cross-attention operator {a}/{b}"
                );
            }
            {
                let oracle = crate::kernels::cuda::attention::ordinary_reference(
                    &q.to_dtype(DType::F32)?,
                    &k.to_dtype(DType::F32)?,
                    &v.to_dtype(DType::F32)?,
                    scale,
                    false,
                )?;
                let ordinary =
                    crate::kernels::cuda::attention::ordinary_reference(&q, &k, &v, scale, false)?;
                let error = |candidate: &Tensor| -> Result<f32> {
                    Ok(values(candidate)?
                        .iter()
                        .zip(values(&oracle)?)
                        .map(|(a, b)| (a - b).abs())
                        .fold(0., f32::max))
                };
                assert!(
                    error(&actual)? <= 2. * error(&ordinary)?,
                    "upstream FA cross-attention operator error rule"
                );
            }
            assert!(
                forward(&q, &k, &v, scale, true).is_err(),
                "causal unequal lengths must not invent an offset"
            );
        }
        Ok(())
    }

    #[test]
    fn fixed_order_attention_cuda_repeats_padding_gqa_and_f32_reference() -> Result<()> {
        let device = crate::device::load().map_err(|e| candle_core::Error::Msg(e.to_string()))?;
        if !device.is_cuda() {
            return Ok(());
        }
        if let Device::Cuda(cuda) = &device
            && cuda.cuda_stream().context().compute_capability().w()?.0 < 8
        {
            return Ok(());
        }
        for (length, heads, kv_heads, width) in [
            (1, 2, 1, 32),
            (17, 4, 2, 64),
            (33, 4, 1, 128),
            (65, 4, 2, 256),
        ] {
            let q = input((1, length, heads, width), 123, &device)?;
            let k = input((1, length, kv_heads, width), 456, &device)?;
            let v = input((1, length, kv_heads, width), 789, &device)?;
            let scale = (width as f32).sqrt().recip();
            let values = |x: &Tensor| -> Result<Vec<f32>> {
                x.to_dtype(DType::F32)?.flatten_all()?.to_vec1()
            };
            let actual = forward(&q, &k, &v, scale, true)?;
            let repeated = forward(&q, &k, &v, scale, true)?;
            assert_eq!(
                values(&actual)?,
                values(&repeated)?,
                "repeat {length}/{width}"
            );
            let pad = |x: &Tensor, h| -> Result<Tensor> {
                Tensor::cat(
                    &[x, &Tensor::zeros((1, 19, h, width), DType::BF16, &device)?],
                    1,
                )
            };
            let padded = forward(
                &pad(&q, heads)?,
                &pad(&k, kv_heads)?,
                &pad(&v, kv_heads)?,
                scale,
                true,
            )?
            .narrow(1, 0, length)?;
            assert_eq!(
                values(&actual)?,
                values(&padded)?,
                "causal padding {length}/{width}"
            );
            let doubled = |x: &Tensor| Tensor::cat(&[x, x], 0);
            let packed = forward(&doubled(&q)?, &doubled(&k)?, &doubled(&v)?, scale, true)?;
            for slot in 0..2 {
                assert_eq!(
                    values(&actual)?,
                    values(&packed.narrow(0, slot, 1)?)?,
                    "batch {length}/{width}"
                );
            }
            for causal in [true, false] {
                let actual = forward(&q, &k, &v, scale, causal)?;
                let reference =
                    crate::architectures::qwen35_text::eager_attention(&q, &k, &v, scale, causal)?;

                for (a, b) in values(&actual)?.iter().zip(values(&reference)?) {
                    assert!(a.is_finite() && b.is_finite());
                    assert!(
                        (a - b).abs() <= 1e-5 + 1.6e-2 * b.abs(),
                        "attention operator {a}/{b}"
                    );
                }
                {
                    let oracle = crate::kernels::cuda::attention::ordinary_reference(
                        &q.to_dtype(DType::F32)?,
                        &k.to_dtype(DType::F32)?,
                        &v.to_dtype(DType::F32)?,
                        scale,
                        causal,
                    )?;
                    let low_precision = crate::kernels::cuda::attention::ordinary_reference(
                        &q, &k, &v, scale, causal,
                    )?;
                    let error = |candidate: &Tensor| -> Result<f32> {
                        Ok(values(candidate)?
                            .iter()
                            .zip(values(&oracle)?)
                            .map(|(a, b)| (a - b).abs())
                            .fold(0., f32::max))
                    };
                    let actual_error = error(&actual)?;
                    let eager_error = error(&low_precision)?;
                    assert!(
                        actual_error <= 2. * eager_error,
                        "upstream FA operator bound {length}/{width}/{causal}: {actual_error} > 2*{eager_error}"
                    );
                }
            }
        }
        Ok(())
    }
}

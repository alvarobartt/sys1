
#[cfg(test)]
mod eager_native_reference_tests {
    use super::*;

    #[test]
    #[ignore = "external reference data"]
    fn eager_opmath_matches_native_sdpa_fixtures() -> Result<()> {
        let root = crate::assets::fixtures("sdpa")?;
        let device = crate::device::load()
            .map_err(|error| candle_core::Error::Msg(error.to_string()))?;
        for (dtype, rtol) in [("bf16", 0.016f32), ("f16", 0.001), ("f32", 1.3e-6)] {
            #[cfg(feature = "cuda")]
            if dtype == "bf16"
                && let Device::Cuda(cuda) = &device {
                    let (major, _) = cuda.cuda_stream().context().compute_capability()
                        .map_err(|error| candle_core::Error::Msg(error.to_string()))?;
                    if major < 8 {
                        continue;
                    }
                }
            for mask in ["masked", "unmasked"] {
                let data = candle_core::safetensors::load(
                    root.join(format!("b2-h3-l11-d64-{mask}-{dtype}.safetensors")), &device)?;
                let actual = eager_attention(&data["q"], &data["k"], &data["v"],
                    0.125, Some(&data["mask"]))?;
                assert_eq!(actual.dtype(), data["expected"].dtype());
                assert_eq!(actual.dims(), data["expected"].dims());
                let actual = actual.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
                let expected = data["expected"].to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
                for (index, (actual, expected)) in actual.iter().zip(&expected).enumerate() {
                    assert!(actual.is_finite() && (*actual - expected).abs() <= 1e-5 + rtol * expected.abs(),
                        "{mask}/{dtype} native math SDPA at{index}: {actual} vs {expected}");
                }
            }
        }
        Ok(())
    }
}

#[cfg(all(test, feature = "cuda"))]
mod flash_attention_reference_tests {
    use super::*;

    #[derive(Clone, Copy)]
    struct Case {
        name: &'static str,
        batch: usize,
        length: usize,
        lengths: &'static [usize],
        window: Option<usize>,
    }

    fn deterministic_tensor(
        shape: (usize, usize, usize, usize),
        seed: u64,
        device: &Device,
    ) -> Result<Tensor> {
        let count = shape.0 * shape.1 * shape.2 * shape.3;
        let mut state = seed;
        let values = (0..count)
            .map(|_| {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                let unit = ((state >> 40) as u32) as f32 / ((1u32 << 24) as f32);
                (unit - 0.5) * 2.0
            })
            .collect::<Vec<_>>();
        Tensor::from_vec(values, shape, device)?.to_dtype(DType::F16)
    }

    fn attention_mask(
        lengths: &[usize],
        length: usize,
        window: Option<usize>,
        dtype: DType,
        device: &Device,
    ) -> Result<Tensor> {
        let padding = lengths
            .iter()
            .flat_map(|&valid| (0..length).map(move |column| (column < valid) as u32 as f32))
            .collect::<Vec<_>>();
        let padding = Tensor::from_vec(padding, (lengths.len(), length), device)?;
        let global = global_attention_mask(&padding, length, dtype)?;
        match window {
            Some(window) => {
                global.broadcast_add(&local_attention_mask(length, window, dtype, device)?)
            }
            None => Ok(global),
        }
    }

    fn valid_error(actual: &Tensor, reference: &Tensor, lengths: &[usize]) -> Result<(f32, f64)> {
        let (batch_size, heads, length, width) = actual.dims4()?;
        let actual = actual
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let reference = reference.flatten_all()?.to_vec1::<f32>()?;
        let mut maximum = 0f32;
        let mut total = 0f64;
        let mut count = 0usize;
        assert_eq!(lengths.len(), batch_size);
        for (batch, &valid_length) in lengths.iter().enumerate() {
            for head in 0..heads {
                for token in 0..valid_length {
                    for column in 0..width {
                        let index = ((batch * heads + head) * length + token) * width + column;
                        let delta = (actual[index] - reference[index]).abs();
                        maximum = maximum.max(delta);
                        total += delta as f64;
                        count += 1;
                    }
                }
            }
        }
        Ok((maximum, total / count as f64))
    }

    fn run(case: Case, device: &Device) -> Result<()> {
        const HEADS: usize = 12;
        const HEAD_SIZE: usize = 64;
        assert_eq!(case.batch, case.lengths.len());
        let shape = (case.batch, HEADS, case.length, HEAD_SIZE);
        let q = deterministic_tensor(shape, 0x6a09_e667_f3bc_c909, device)?;
        let k = deterministic_tensor(shape, 0xbb67_ae85_84ca_a73b, device)?;
        let v = deterministic_tensor(shape, 0x3c6e_f372_fe94_f82b, device)?;
        let scale = (HEAD_SIZE as f64).powf(-0.5);

        let q32 = q.to_dtype(DType::F32)?;
        let k32 = k.to_dtype(DType::F32)?;
        let v32 = v.to_dtype(DType::F32)?;
        let mask32 = attention_mask(case.lengths, case.length, case.window, DType::F32, device)?;
        let reference = eager_attention_full(&q32, &k32, &v32, scale, Some(&mask32))?;

        let mask16 = attention_mask(case.lengths, case.length, case.window, DType::F16, device)?;
        let eager = eager_attention_full(&q, &k, &v, scale, Some(&mask16))?;
        let flash = scaled_dot_product_attention(
            &q,
            &k,
            &v,
            scale,
            AttentionOptions {
                mask: None,
                implementation: crate::models::attention_for_device(DType::F16, device)
                    .map_err(|error| candle_core::Error::Msg(error.to_string()))?,
                lengths: case.lengths,
                window: case.window,
            },
        )?;

        let (eager_max, eager_mean) = valid_error(&eager, &reference, case.lengths)?;
        let (flash_max, flash_mean) = valid_error(&flash, &reference, case.lengths)?;
        let allowed = 2.0 * eager_max;
        println!(
            "MODERNBERT_FA2_REFERENCE case={} batch={} length={} lengths={:?} window={:?} eager_max={:.9} flash_max={:.9} allowed={:.9} eager_mean={:.9} flash_mean={:.9} pass={}",
            case.name,
            case.batch,
            case.length,
            case.lengths,
            case.window,
            eager_max,
            flash_max,
            allowed,
            eager_mean,
            flash_mean,
            flash_max <= allowed,
        );
        assert!(
            flash_max <= allowed,
            "{}: FA2 max error {flash_max} exceeds 2x eager F16 error {allowed}",
            case.name,
        );
        Ok(())
    }

    #[test]
    fn fa2_forward_error_is_bounded_by_twice_f16_eager_error() -> Result<()> {
        let device = crate::device::load().map_err(|error| candle_core::Error::Msg(error.to_string()))?;
        if !crate::kernels::cuda::attention::supported_test_device(&device)? {
            println!("MODERNBERT_FA_REFERENCE_SKIPPED unsupported CUDA architecture");
            return Ok(());
        }
        for case in [
            Case {
                name: "dense_128",
                batch: 1,
                length: 128,
                lengths: &[128],
                window: None,
            },
            Case {
                name: "local_257",
                batch: 1,
                length: 257,
                lengths: &[257],
                window: Some(64),
            },
            Case {
                name: "varlen_dense_129",
                batch: 3,
                length: 129,
                lengths: &[129, 97, 33],
                window: None,
            },
            Case {
                name: "varlen_local_513",
                batch: 3,
                length: 513,
                lengths: &[513, 383, 127],
                window: Some(128),
            },
        ] {
            run(case, &device)?;
        }
        Ok(())
    }
}

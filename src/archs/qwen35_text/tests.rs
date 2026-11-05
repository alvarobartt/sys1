#[cfg(feature = "cuda")]
mod cuda;
#[cfg(feature = "cuda")]
use cuda::test_cuda_device;

use super::*;

#[test]
fn guarded_double_radius_coefficient_bounds_exact_rational_values() {
    for (count, nearest_exact) in [
        (1, 4.884981811276547e-15_f64),
        (80, 2.2747687422185558e-11),
        (288, 2.9471959623355573e-10),
        (576, 1.178851243061726e-9),
        (4_194_303, 0.11111104093424891),
    ] {
        let coefficient = guarded_double_radius_coefficient(count * 32);
        assert!(
            coefficient >= nearest_exact.next_up(),
            "must exceed the exact rational bound"
        );
        assert!(
            coefficient <= nearest_exact * (1. + 64. * f64::EPSILON),
            "rounding outward must not replace the derived bound with an arbitrary large radius"
        );
        assert_eq!(
            coefficient,
            guarded_double_radius_coefficient(count * 32 - 31),
            "same ceil(K/32) bound for the tail"
        );
    }
    assert!(guarded_double_radius_coefficient(0).is_infinite());
    assert!(guarded_double_radius_coefficient(4_194_304 * 32).is_infinite());
    assert!(guarded_double_radius_coefficient(usize::MAX).is_infinite());
}

#[test]
fn guarded_tensor_double_radius_bounds_exact_rational_values() {
    for (count, nearest_exact) in [
        (1, 7.771561675301967e-15_f64),
        (80, 2.3022467620780365e-11),
        (288, 2.957102482384298e-10),
        (576, 1.1808331021829885e-9),
    ] {
        let coefficient = guarded_double_radius_coefficient_impl(count * 32, true);
        assert!(coefficient >= nearest_exact.next_up());
        assert!(coefficient <= nearest_exact * (1. + 64. * f64::EPSILON));
        assert!(coefficient >= guarded_double_radius_coefficient(count * 32));
    }
    assert!(guarded_double_radius_coefficient_impl(0, true).is_infinite());
    assert!(guarded_double_radius_coefficient_impl(usize::MAX, true).is_infinite());
}

#[test]
fn projection_geometry_restricts_larger_tiles_to_measured_architecture() {
    for sm120 in [false, true] {
        assert_eq!(projection_tile_rows(4096, 2047, sm120), 16);
        assert_eq!(projection_tile_rows(63, 2048, sm120), 16);
        assert_eq!(projection_tile_rows(64, 2048, sm120), 32);
        assert_eq!(projection_tile_rows(127, 2048, sm120), 32);
    }
    assert_eq!(projection_tile_rows(128, 2048, true), 128);
    assert_eq!(projection_tile_rows(129, 2049, true), 128);
    assert_eq!(projection_tile_rows(128, 2048, false), 32);
    assert_eq!(projection_tile_rows(4096, 12288, false), 32);
}

#[test]
fn blocked_projection_preserves_rows_storage_and_tail_shape() -> Result<()> {
    let device = candle_core::Device::Cpu;
    let weights = Tensor::from_vec(
        (0..15).map(|i| (i % 7) as f32 * 0.125).collect::<Vec<_>>(),
        (3, 5),
        &device,
    )?;
    let linear = Linear::new(weights, None);
    for (batch, length) in [(1, 1), (1, 16), (1, 17), (2, 19)] {
        let input = Tensor::from_vec(
            (0..batch * length * 5)
                .map(|i| (i % 8) as f32 * 0.25)
                .collect::<Vec<_>>(),
            (batch, length, 5),
            &device,
        )?;
        let actual = blocked_projection(&input, &linear, 16)?;
        let expected = input.apply(&linear)?;
        assert_eq!(actual.dims(), expected.dims());
        assert_eq!(actual.dtype(), input.dtype());
        assert_eq!(
            actual.flatten_all()?.to_vec1::<f32>()?,
            expected.flatten_all()?.to_vec1::<f32>()?
        );
        assert!(blocked_projection(&input, &linear, 0).is_err());
    }
    Ok(())
}

#[test]
fn full_attention_kv_fixed_reduction_preserves_storage_and_cast_boundary() -> Result<()> {
    let devices = vec![candle_core::Device::Cpu];
    #[cfg(feature = "cuda")]
    let devices = {
        let mut devices = devices;
        devices.push(test_cuda_device()?);
        devices
    };
    for device in devices {
        for dtype in supported_rotary_test_dtypes(&device)? {
            for (batch, length, inner, columns) in [
                (1, 2, 33, 9),
                (2, 9, 129, 17),
                (1, 33, 65, 65),
                (2, 1, 4096, 1024),
                (1, 2, 6144, 1024),
                (1, 2, 12288, 65),
                (1, 2, 18432, 65),
                (1, 33, 257, 2049),
                (1, 65, 65, 2049),
                (2, 33, 129, 2049),
                (1, 129, 65, 2049),
                (1, 129, 257, 2049),
            ] {
                let rows = batch * length;
                let xs: Vec<f32> = (0..rows * inner)
                    .map(|i| ((i * 17 % 255) as f32 - 127.) / 64.)
                    .collect();
                let ws: Vec<f32> = (0..columns * inner)
                    .map(|i| ((i * 29 % 127) as f32 - 63.) / 256.)
                    .collect();
                let weights =
                    Tensor::from_vec(ws.clone(), (columns, inner), &device)?.to_dtype(dtype)?;
                let original = Linear::new(weights.clone(), None);
                let layer = Linear::new(weights, None);
                assert_eq!(layer.weight().dtype(), dtype);
                let x = Tensor::from_vec(xs.clone(), (batch, length, inner), &device)?
                    .to_dtype(dtype)?;
                #[cfg(target_os = "macos")]
                if device.is_cpu() && dtype == DType::F16 {
                    assert!(full_attention_kv_forward(&x, &layer).is_err());
                    assert!(linear_attention_projection_forward(&x, &layer).is_err());
                    continue;
                }
                let projected = full_attention_kv_forward(&x, &layer)?;
                let linear_qkv = linear_attention_projection_forward(&x, &layer)?;
                assert_eq!(
                    projected
                        .to_dtype(DType::F32)?
                        .flatten_all()?
                        .to_vec1::<f32>()?,
                    linear_qkv
                        .to_dtype(DType::F32)?
                        .flatten_all()?
                        .to_vec1::<f32>()?,
                    "QKV candidate preserves projection dtype and cast boundary"
                );
                assert_eq!(projected.dtype(), dtype, "round back BEFORE attention/norm");
                let actual = projected
                    .to_dtype(DType::F32)?
                    .flatten_all()?
                    .to_vec1::<f32>()?;
                if device.is_cuda() && dtype == DType::BF16 {
                    let repeated = full_attention_kv_forward(&x, &layer)?
                        .to_dtype(DType::F32)?
                        .flatten_all()?
                        .to_vec1::<f32>()?;
                    assert_eq!(
                        actual, repeated,
                        "actual repeated fixed-reduction K/V projection"
                    );
                    #[cfg(feature = "cuda")]
                    {
                        for shaped in [x.clone(), x.reshape((rows, inner))?] {
                            let head = fixed_bf16_projection(&shaped, &layer)?;
                            assert_eq!(head.dtype(), dtype);
                            assert_eq!(head.dim(D::Minus1)?, columns);
                            assert_eq!(
                                actual,
                                head.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?,
                                "rank-two/three head adapter preserves validated projection"
                            );
                        }
                    }
                }
                if dtype != DType::F32 && !(device.is_cuda() && dtype == DType::BF16) {
                    let retained = if device.is_cpu() && dtype == DType::BF16 {
                        x.to_dtype(DType::F32)?
                            .apply(&Linear::new(original.weight().to_dtype(DType::F32)?, None))?
                            .to_dtype(dtype)?
                    } else {
                        x.apply(&original)?
                    }
                    .to_dtype(DType::F32)?
                    .flatten_all()?
                    .to_vec1::<f32>()?;
                    assert_eq!(actual, retained, "unchanged backend/dtype path");
                    continue;
                }
                let expected: Vec<f32> = (0..rows)
                    .flat_map(|row| {
                        let xs = &xs;
                        let ws = &ws;
                        (0..columns).map(move |column| {
                            (0..inner)
                                .map(|k| {
                                    f64::from(xs[row * inner + k])
                                        * f64::from(ws[column * inner + k])
                                })
                                .sum::<f64>() as f32
                        })
                    })
                    .collect();
                #[cfg(feature = "cuda")]
                if device.is_cuda() && dtype == DType::BF16 {
                    let biases: Vec<f32> = (0..columns)
                        .map(|column| ((column % 17) as f32 - 8.) / 256.)
                        .collect();
                    let biased = Linear::new(
                        layer.weight().clone(),
                        Some(Tensor::from_vec(biases.clone(), columns, &device)?.to_dtype(dtype)?),
                    );
                    let expected_biased: Vec<f32> = expected
                        .iter()
                        .enumerate()
                        .map(|(i, value)| *value + biases[i % columns])
                        .collect();
                    let expected_biased =
                        Tensor::from_vec(expected_biased, actual.len(), &candle_core::Device::Cpu)?
                            .to_dtype(dtype)?
                            .to_dtype(DType::F32)?
                            .to_vec1::<f32>()?;
                    for shaped in [x.clone(), x.reshape((rows, inner))?] {
                        let output = fixed_bf16_projection(&shaped, &biased)?;
                        assert_eq!(output.dim(D::Minus1)?, columns);
                        assert_eq!(
                            output
                                .to_dtype(DType::F32)?
                                .flatten_all()?
                                .to_vec1::<f32>()?,
                            expected_biased,
                            "biased rank-two/three projection adds FP32 bias before sole BF16 cast"
                        );
                    }
                }
                let expected = Tensor::from_vec(expected, actual.len(), &candle_core::Device::Cpu)?
                    .to_dtype(dtype)?
                    .to_dtype(DType::F32)?
                    .to_vec1::<f32>()?;
                assert_eq!(
                    actual, expected,
                    "exact dyadic reference {dtype:?}/{inner}/{columns}"
                );
            }
        }
    }
    Ok(())
}

#[test]
fn mlp_diagnostic_boundaries_preserve_original_expression() -> Result<()> {
    let device = candle_core::Device::Cpu;
    for dtype in [DType::F32, DType::F16, DType::BF16] {
        let weights = |output, input| -> Result<Linear> {
            let values: Vec<f32> = (0..output * input)
                .map(|i| ((i * 13 % 31) as f32 - 15.) / 128.)
                .collect();
            Ok(Linear::new(
                Tensor::from_vec(values, (output, input), &device)?.to_dtype(dtype)?,
                None,
            ))
        };
        let layer = Mlp {
            #[cfg(all(feature = "cuda", sys1_cuda_direct_ffn))]
            direct_weight: None,
            #[cfg(all(feature = "cuda", sys1_cuda_fused_ffn))]
            fused_weight: Default::default(),
            gate: weights(17, 33)?,
            up: weights(17, 33)?,
            down: weights(9, 17)?,
        };
        let values: Vec<f32> = (0..2 * 3 * 33)
            .map(|i| ((i * 7 % 29) as f32 - 14.) / 64.)
            .collect();
        let x = Tensor::from_vec(values, (2, 3, 33), &device)?.to_dtype(dtype)?;
        #[cfg(target_os = "macos")]
        if dtype == DType::F16 {
            assert!(layer.forward(&x, 0).is_err());
            continue;
        }
        let reference = |input: &Tensor, linear: &Linear| -> Result<Tensor> {
            if dtype == DType::BF16 {
                input
                    .to_dtype(DType::F32)?
                    .apply(&Linear::new(linear.weight().to_dtype(DType::F32)?, None))?
                    .to_dtype(dtype)
            } else {
                input.apply(linear)
            }
        };
        let original = reference(
            &(reference(&x, &layer.gate)?.silu()? * reference(&x, &layer.up)?)?,
            &layer.down,
        )?;
        let actual = layer.forward(&x, 0)?;
        assert_eq!(actual.dtype(), dtype);
        let values = |x: &Tensor| x.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>();
        assert_eq!(
            values(&actual)?,
            values(&original)?,
            "diagnostic-only MLP ordering {dtype:?}"
        );
    }
    Ok(())
}

fn supported_rotary_test_dtypes(_device: &candle_core::Device) -> Result<Vec<DType>> {
    let mut dtypes = vec![DType::F32, DType::F16];
    #[cfg(feature = "cuda")]
    if let candle_core::Device::Cuda(cuda) = _device {
        let (major, _) = cuda
            .cuda_stream()
            .context()
            .compute_capability()
            .map_err(|error| candle_core::Error::Msg(error.to_string()))?;
        if major >= 8 {
            dtypes.push(DType::BF16);
        }
        return Ok(dtypes);
    }
    dtypes.push(DType::BF16);
    Ok(dtypes)
}

#[test]
fn eager_causal_gqa_matches_independent_f64_scalar_reference() -> Result<()> {
    let mut devices = vec![candle_core::Device::Cpu];
    #[cfg(feature = "cuda")]
    devices.push(test_cuda_device()?);
    let (batch, length, heads, kv_heads, width) = (2, 129, 4, 2, 4);
    let make = |count, factor| {
        (0..count)
            .map(|i| ((i * 13 % 97) as f32 - 48.) * factor)
            .collect::<Vec<_>>()
    };
    let q = make(batch * length * heads * width, 0.013);
    let k = make(batch * length * kv_heads * width, 0.017);
    let v = make(batch * length * kv_heads * width, 0.019);
    let mut expected = Vec::new();
    for b in 0..batch {
        for row in 0..length {
            for h in 0..heads {
                let kv = h / (heads / kv_heads);
                let scores = (0..=row)
                    .map(|col| {
                        (0..width)
                            .map(|d| {
                                q[((b * length + row) * heads + h) * width + d] as f64
                                    * k[((b * length + col) * kv_heads + kv) * width + d] as f64
                            })
                            .sum::<f64>()
                            * 0.5
                    })
                    .collect::<Vec<_>>();
                let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let weights = scores
                    .iter()
                    .map(|score| (score - max).exp())
                    .collect::<Vec<_>>();
                let total = weights.iter().sum::<f64>();
                for d in 0..width {
                    expected.push(
                        weights
                            .iter()
                            .enumerate()
                            .map(|(col, weight)| {
                                weight * v[((b * length + col) * kv_heads + kv) * width + d] as f64
                            })
                            .sum::<f64>()
                            / total,
                    );
                }
            }
        }
    }
    for device in devices.drain(..) {
        let actual = eager_attention(
            &Tensor::from_vec(q.clone(), (batch, length, heads, width), &device)?,
            &Tensor::from_vec(k.clone(), (batch, length, kv_heads, width), &device)?,
            &Tensor::from_vec(v.clone(), (batch, length, kv_heads, width), &device)?,
            0.5,
            true,
        )?
        .flatten_all()?
        .to_vec1::<f32>()?;
        for (actual, expected) in actual.into_iter().zip(&expected) {
            assert!((actual as f64 - expected).abs() <= 1e-5 + 1.3e-6 * expected.abs());
        }
    }
    Ok(())
}

#[test]
fn shared_rotary_is_bitwise_equal_to_legacy_per_head_preparation() -> Result<()> {
    let mut devices = vec![candle_core::Device::Cpu];
    #[cfg(feature = "cuda")]
    devices.push(test_cuda_device()?);
    for device in devices.drain(..) {
        for dtype in supported_rotary_test_dtypes(&device)? {
            for (dimension, sections, width) in [(64, [16, 8, 8], 64), (64, [16, 8, 8], 256)] {
                let theta = 1_000_000f64;
                let positions = (0..17)
                    .map(|i| [i * 1031, i * 7, i * 31])
                    .collect::<Vec<_>>();
                let prepared =
                    Rotary::new(dimension, theta, sections)?.prepare(&positions, dtype, &device)?;
                for heads in [4, 24] {
                    let x = Tensor::from_vec(
                        (0..2 * positions.len() * heads * width)
                            .map(|i| (i % 251) as f32 * 0.013 - 1.5)
                            .collect::<Vec<_>>(),
                        (2, positions.len(), heads, width),
                        &device,
                    )?
                    .to_dtype(dtype)?;

                    let phase = positions
                        .iter()
                        .flat_map(|position| {
                            (0..dimension / 2).map(move |i| {
                                let axis = if i < sections[1] * 3 && i % 3 == 1 {
                                    1
                                } else if i < sections[2] * 3 && i % 3 == 2 {
                                    2
                                } else {
                                    0
                                };
                                position[axis] as f32
                                    / theta.powf(2. * i as f64 / dimension as f64) as f32
                            })
                        })
                        .collect::<Vec<_>>();
                    let phase =
                        Tensor::from_vec(phase, (1, positions.len(), 1, dimension / 2), &device)?;
                    let cos =
                        Tensor::cat(&[&phase.cos()?, &phase.cos()?], D::Minus1)?.to_dtype(dtype)?;
                    let sin =
                        Tensor::cat(&[&phase.sin()?, &phase.sin()?], D::Minus1)?.to_dtype(dtype)?;
                    let active = x.narrow(D::Minus1, 0, dimension)?;
                    let first = active.narrow(D::Minus1, 0, dimension / 2)?;
                    let second = active.narrow(D::Minus1, dimension / 2, dimension / 2)?;
                    let rotated = Tensor::cat(&[&second.neg()?, &first], D::Minus1)?;
                    let legacy = Tensor::cat(
                        &[
                            &(active.broadcast_mul(&cos)? + rotated.broadcast_mul(&sin)?)?,
                            &x.narrow(D::Minus1, dimension, width - dimension)?,
                        ],
                        D::Minus1,
                    )?;
                    let values = |tensor: Tensor| -> Result<Vec<u32>> {
                        Ok(tensor
                            .to_dtype(DType::F32)?
                            .flatten_all()?
                            .to_vec1::<f32>()?
                            .into_iter()
                            .map(f32::to_bits)
                            .collect())
                    };
                    assert_eq!(
                        values(prepared.apply(&x)?)?,
                        values(legacy)?,
                        "{dtype:?}, heads={heads}, width={width}"
                    );
                }
            }
        }
    }
    Ok(())
}

#[test]
fn eager_attention_single_key_and_chunked_causal_are_finite() {
    let mut devices = vec![candle_core::Device::Cpu];
    #[cfg(feature = "cuda")]
    devices.push(test_cuda_device().unwrap());
    #[cfg(feature = "metal")]
    devices.push(candle_core::Device::new_metal(0).unwrap());
    for device in devices.drain(..) {
        let q = Tensor::ones((1, 1, 2, 4), DType::F32, &device).unwrap();
        let k = Tensor::ones((1, 1, 1, 4), DType::F32, &device).unwrap();
        let v = Tensor::from_vec(vec![1f32, -2., 3., -4.], (1, 1, 1, 4), &device).unwrap();
        let actual = eager_attention(&q, &k, &v, 0.5, false)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert_eq!(actual, vec![1., -2., 3., -4., 1., -2., 3., -4.]);

        let q = Tensor::zeros((1, 257, 2, 4), DType::F32, &device).unwrap();
        let k = Tensor::zeros((1, 257, 1, 4), DType::F32, &device).unwrap();
        let v = Tensor::from_vec(
            (0..257).flat_map(|i| [i as f32; 4]).collect::<Vec<_>>(),
            (1, 257, 1, 4),
            &device,
        )
        .unwrap();
        let actual = eager_attention(&q, &k, &v, 0.5, true)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        for (index, actual) in actual.into_iter().enumerate() {
            let expected = (index / 8) as f32 * 0.5;
            assert!(
                (actual - expected).abs() <= 1e-5 + expected.abs() * 1.3e-6,
                "{device:?}: index {index}, expected {expected}, got {actual}"
            );
        }
    }
}

#[test]
#[ignore = "external reference data"]
fn convolution_matches_frozen_native_torch_gpu_fixtures() -> Result<()> {
    let directory = crate::assets::fixtures("qwen35_conv")?;
    let mut fixtures = std::fs::read_dir(directory)?
        .filter_map(|entry| {
            let path = entry.ok()?.path();
            (path
                .extension()
                .is_some_and(|extension| extension == "safetensors"))
            .then_some(path)
        })
        .collect::<Vec<_>>();
    fixtures.sort();
    assert_eq!(
        fixtures.len(),
        8,
        "all pinned native BF16/FP32 cases are required"
    );
    for path in fixtures {
        let tensors = candle_core::safetensors::load(&path, &candle_core::Device::Cpu)?;
        let input = &tensors["input"];
        let weight = &tensors["weight"];
        let kernel = weight.dim(2)?;
        let expected_conv = &tensors["expected_conv"];
        let expected = &tensors["expected"];
        let assert_close = |actual: &Tensor, reference: &Tensor, name: &str| -> Result<()> {
            let actual = actual
                .to_dtype(DType::F32)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            let reference = reference
                .to_dtype(DType::F32)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            assert_eq!(actual.len(), reference.len());

            let rtol = if input.dtype() == DType::BF16 {
                1.6e-2
            } else {
                1.3e-6
            };
            for (index, (actual, reference)) in actual.into_iter().zip(reference).enumerate() {
                assert!(actual.is_finite() && reference.is_finite());
                assert!(
                    (actual - reference).abs() <= 1e-5 + rtol * reference.abs(),
                    "{} {name} element {index}: {actual} vs {reference}",
                    path.display()
                );
            }
            Ok(())
        };
        let convolution = causal_conv(input, weight, kernel)?;
        if input.dtype() == DType::BF16 {
            assert_eq!(
                convolution
                    .to_dtype(DType::F32)?
                    .flatten_all()?
                    .to_vec1::<f32>()?,
                expected_conv
                    .to_dtype(DType::F32)?
                    .flatten_all()?
                    .to_vec1::<f32>()?
            );
        }
        assert_close(&convolution, expected_conv, "CPU convolution")?;
        assert_close(
            &causal_conv_silu(input, weight, kernel)?,
            expected,
            "CPU SiLU",
        )?;
        #[cfg(feature = "cuda")]
        {
            let device = test_cuda_device()?;
            let capability = match &device {
                candle_core::Device::Cuda(cuda) => {
                    cuda.cuda_stream()
                        .context()
                        .compute_capability()
                        .map_err(|error| candle_core::Error::Msg(error.to_string()))?
                        .0
                }
                _ => unreachable!(),
            };
            if input.dtype() != DType::BF16 || capability >= 8 {
                let result =
                    cuda_conv_silu(&input.to_device(&device)?, &weight.to_device(&device)?)?;
                assert_close(&result, expected, "CUDA SiLU")?;
            }
        }
        #[cfg(feature = "metal")]
        {
            let device = candle_core::Device::new_metal(0)?;
            let result = crate::kernels::metal::conv_silu(
                &input.to_device(&device)?,
                &weight.to_device(&device)?,
            )?;

            let round = |value: f32| {
                if input.dtype() == DType::BF16 {
                    let bits = value.to_bits();
                    f32::from_bits(bits.wrapping_add(0x7fff + ((bits >> 16) & 1)) & 0xffff0000)
                } else {
                    value
                }
            };
            let (batch, length, channels) = input.dims3()?;
            let xs = input
                .to_dtype(DType::F32)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            let ws = weight
                .to_dtype(DType::F32)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            let mut reference = Vec::with_capacity(xs.len());
            for b in 0..batch {
                for token in 0..length {
                    for channel in 0..channels {
                        let mut sum = 0f32;
                        for tap in 0..kernel {
                            if token + tap + 1 >= kernel {
                                let source = token + tap + 1 - kernel;
                                let product = round(
                                    xs[(b * length + source) * channels + channel]
                                        * ws[channel * kernel + tap],
                                );
                                sum = round(sum + product);
                            }
                        }
                        reference.push(round(sum / (1. + (-sum).exp())));
                    }
                }
            }
            let reference =
                Tensor::from_vec(reference, input.shape().clone(), &candle_core::Device::Cpu)?
                    .to_dtype(input.dtype())?;
            assert_close(&result, &reference, "Metal retained per-tap SiLU")?;
        }
    }
    Ok(())
}

#[test]
#[ignore = "external reference data"]
fn delta_rule_matches_pinned_native_chunk_and_fp64_fixtures() -> Result<()> {
    let directory = crate::assets::fixtures("qwen35_delta")?;
    let device =
        crate::device::load().map_err(|error| candle_core::Error::Msg(error.to_string()))?;
    for preparation in ["f32", "bf16"] {
        let tensors = candle_core::safetensors::load(
            directory.join(format!("b2-l65-kh2-h4-d16-{preparation}.safetensors")),
            &device,
        )?;
        let q = &tensors["query_scaled_f32"];
        let k = &tensors["key_f32"];
        let v = &tensors["value_f32"];
        let g = &tensors["gate_f32"];
        let beta = &tensors["beta_f32"];
        #[allow(unused_mut)]
        let mut actuals = vec![(
            "tensor_chunk32",
            chunked_delta_rule_fallback(
                &q.transpose(1, 2)?,
                &k.transpose(1, 2)?,
                &v.transpose(1, 2)?,
                &g.transpose(1, 2)?,
                &beta.transpose(1, 2)?,
            )?,
        )];
        #[cfg(feature = "cuda")]
        if device.is_cuda() {
            actuals.push((
                "cuda_expanded",
                cuda_delta_rule_token_major(q, k, v, g, beta)?,
            ));
            let indices = Tensor::from_vec(vec![0u32, 2], 2, &device)?;
            actuals.push((
                "cuda_grouped",
                cuda_delta_rule_token_major(
                    &q.index_select(&indices, 2)?,
                    &k.index_select(&indices, 2)?,
                    v,
                    g,
                    beta,
                )?,
            ));
        }
        for (implementation, actual) in actuals {
            assert_eq!(actual.dims(), [2, 65, 4, 16]);
            let actual = actual.flatten_all()?.to_vec1::<f32>()?;
            for oracle in [
                "native_chunk_f32",
                "native_sequential_f32",
                "fp64_recurrence_rounded_f32",
            ] {
                let expected = tensors[oracle].flatten_all()?.to_vec1::<f32>()?;
                assert_eq!(actual.len(), expected.len());
                for (index, (actual, expected)) in actual.iter().zip(&expected).enumerate() {
                    assert!(
                        actual.is_finite()
                            && expected.is_finite()
                            && (actual - expected).abs() <= 1e-5 + 1.3e-6 * expected.abs(),
                        "{preparation} {implementation} vs {oracle}, element{index}: {actual} vs {expected}"
                    );
                }
            }
        }
    }
    Ok(())
}

#[test]
#[ignore = "external reference data"]
fn recurrent_rounding_matches_native_fixtures() -> Result<()> {
    let root = crate::assets::fixtures("qwen35_recurrent")?;
    let device =
        crate::device::load().map_err(|error| candle_core::Error::Msg(error.to_string()))?;
    for (name, dtype, rtol) in [
        ("bf16", DType::BF16, 1.6e-2f32),
        ("f16", DType::F16, 1e-3),
        ("f32", DType::F32, 1.3e-6),
    ] {
        #[cfg(feature = "cuda")]
        if dtype == DType::BF16
            && let candle_core::Device::Cuda(cuda) = &device
            && cuda
                .cuda_stream()
                .context()
                .compute_capability()
                .map_err(|error| candle_core::Error::Msg(error.to_string()))?
                .0
                < 8
        {
            continue;
        }
        let data = candle_core::safetensors::load(
            root.join(format!("post-finalize-{name}.safetensors")),
            &device,
        )?;
        let actual = gated_norm_reference(
            &data["recurrent_f32"],
            &data["gate"],
            &data["weight"],
            1e-6,
            dtype,
        )?;
        let values = |tensor: &Tensor| tensor.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>();
        let check = |actual: &Tensor, expected: &Tensor| -> Result<()> {
            for (index, (actual, expected)) in
                values(actual)?.iter().zip(values(expected)?).enumerate()
            {
                assert!(actual.is_finite() && expected.is_finite());
                assert!(
                    (actual - expected).abs() <= 1e-5 + rtol * expected.abs(),
                    "native {name} fixture index={index}: {actual} vs {expected}"
                );
            }
            Ok(())
        };
        check(&actual, &data["expected"])?;
        check(&rounded_beta(&data["beta_input"])?, &data["expected_beta"])?;
        #[cfg(feature = "cuda")]
        if device.is_cuda() && dtype == DType::BF16 {
            check(
                &cuda_post_finalize_bf16(
                    &data["recurrent_f32"],
                    &data["gate"],
                    &data["weight"],
                    1e-6,
                )?,
                &data["expected"],
            )?;
        }
    }
    Ok(())
}

#[test]
fn chunked_delta_rule_matches_scalar_recurrence_across_chunk_boundary() {
    let length = 70;
    let key_dim = 4;
    let value_dim = 3;
    let make = |width: usize, factor: f32| {
        (0..length * width)
            .map(|index| ((index * 13 % 29) as f32 - 14.) * factor)
            .collect::<Vec<_>>()
    };
    let q = make(key_dim, 0.01);
    let k = make(key_dim, 0.012);
    let v = make(value_dim, 0.02);
    let beta = vec![0.45f32; length];
    let g = vec![-0.1f32; length];
    let device = candle_core::Device::Cpu;
    let output = chunked_delta_rule(
        &Tensor::from_vec(q.clone(), (1, 1, length, key_dim), &device).unwrap(),
        &Tensor::from_vec(k.clone(), (1, 1, length, key_dim), &device).unwrap(),
        &Tensor::from_vec(v.clone(), (1, 1, length, value_dim), &device).unwrap(),
        &Tensor::from_vec(g.clone(), (1, 1, length), &device).unwrap(),
        &Tensor::from_vec(beta.clone(), (1, 1, length), &device).unwrap(),
    )
    .unwrap()
    .flatten_all()
    .unwrap()
    .to_vec1::<f32>()
    .unwrap();
    let mut state = vec![0f32; key_dim * value_dim];
    for step in 0..length {
        let decay = g[step].exp();
        for value in &mut state {
            *value *= decay;
        }
        for column in 0..value_dim {
            let remembered = (0..key_dim)
                .map(|row| k[step * key_dim + row] * state[row * value_dim + column])
                .sum::<f32>();
            let delta = (v[step * value_dim + column] - remembered) * beta[step];
            for row in 0..key_dim {
                state[row * value_dim + column] += k[step * key_dim + row] * delta;
            }
            let expected = (0..key_dim)
                .map(|row| q[step * key_dim + row] * state[row * value_dim + column])
                .sum::<f32>();
            assert!((output[step * value_dim + column] - expected).abs() < 1e-4);
        }
    }
}

#[test]
fn chunked_delta_rule_stays_finite_with_large_negative_gates() {
    let device = candle_core::Device::Cpu;
    let q = Tensor::ones((1, 1, 64, 4), DType::F32, &device).unwrap();
    let k = Tensor::ones((1, 1, 64, 4), DType::F32, &device).unwrap();
    let v = Tensor::ones((1, 1, 64, 3), DType::F32, &device).unwrap();
    let g = Tensor::new(-8f32, &device)
        .unwrap()
        .broadcast_as((1, 1, 64))
        .unwrap();
    let beta = Tensor::new(0.5f32, &device)
        .unwrap()
        .broadcast_as((1, 1, 64))
        .unwrap();
    let output = chunked_delta_rule(&q, &k, &v, &g, &beta)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    assert!(output.iter().all(|value| value.is_finite()));
}

#[cfg(feature = "metal")]
#[test]
fn chunked_delta_rule_runs_with_broadcast_gates_on_metal() {
    let device = candle_core::Device::new_metal(0).unwrap();
    let q = Tensor::ones((1, 1, 64, 4), DType::F32, &device).unwrap();
    let k = Tensor::ones((1, 1, 64, 4), DType::F32, &device).unwrap();
    let v = Tensor::ones((1, 1, 64, 3), DType::F32, &device).unwrap();
    let g = Tensor::new(-8f32, &device)
        .unwrap()
        .broadcast_as((1, 1, 64))
        .unwrap();
    let beta = Tensor::new(0.5f32, &device)
        .unwrap()
        .broadcast_as((1, 1, 64))
        .unwrap();
    let output = chunked_delta_rule(&q, &k, &v, &g, &beta)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    assert!(output.iter().all(|value| value.is_finite()));
}

use super::*;

#[cfg(feature = "cuda")]
use half::bf16;

#[cfg(feature = "cuda")]
pub(super) fn test_cuda_device() -> Result<candle_core::Device> {
    crate::device::load().map_err(|error| candle_core::Error::Msg(error.to_string()))
}

#[cfg(feature = "cuda")]
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
        let x =
            Tensor::from_vec(xs.clone(), (batch, length, inner), &device)?.to_dtype(DType::BF16)?;
        let w = Tensor::from_vec(ws.clone(), (columns, inner), &device)?.to_dtype(DType::BF16)?;
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

#[cfg(feature = "cuda")]
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

#[cfg(feature = "cuda")]
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
        let x = Tensor::from_vec(xs.clone(), (1, 1, xs.len()), &device)?.to_dtype(DType::BF16)?;
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

#[cfg(feature = "cuda")]
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

#[cfg(feature = "cuda")]
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

#[cfg(feature = "cuda")]
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

#[cfg(feature = "cuda")]
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

#[cfg(feature = "cuda")]
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
            _ => cuda_bf16_projection_impl(&x, &linear, false, SimtProjection::TiledCompensated),
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

#[cfg(feature = "cuda")]
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

#[cfg(feature = "cuda")]
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

#[cfg(feature = "cuda")]
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
        let x = Tensor::from_vec(xs.clone(), (1, 1, xs.len()), &device)?.to_dtype(DType::BF16)?;
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

#[cfg(feature = "cuda")]
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

#[cfg(feature = "cuda")]
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
        let mixed =
            Tensor::from_vec(values(503), (1, rows, columns), &device)?.to_dtype(DType::BF16)?;
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

#[cfg(feature = "cuda")]
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

#[cfg(feature = "cuda")]
#[test]
fn delta_cuda_matches_chunked_reference() {
    let batch = 2;
    let heads = 2;
    let length = 290;
    let key_dim = 128;
    let value_dim = 16;
    let values = |width: usize, factor: f32| {
        (0..batch * heads * length * width)
            .map(|index| ((index * 13 % 29) as f32 - 14.) * factor)
            .collect::<Vec<_>>()
    };
    let cpu = candle_core::Device::Cpu;
    let q = Tensor::from_vec(
        values(key_dim, 0.001),
        (batch, heads, length, key_dim),
        &cpu,
    )
    .unwrap();
    let k = Tensor::from_vec(
        values(key_dim, 0.006),
        (batch, heads, length, key_dim),
        &cpu,
    )
    .unwrap();
    let v = Tensor::from_vec(
        values(value_dim, 0.008),
        (batch, heads, length, value_dim),
        &cpu,
    )
    .unwrap();
    let g = Tensor::from_vec(
        vec![-0.002f32; batch * heads * length],
        (batch, heads, length),
        &cpu,
    )
    .unwrap();
    let beta = Tensor::from_vec(
        vec![0.5f32; batch * heads * length],
        (batch, heads, length),
        &cpu,
    )
    .unwrap();
    let reference = chunked_delta_rule(&q, &k, &v, &g, &beta)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    let cuda = test_cuda_device().unwrap();
    let actual = cuda_delta_rule_token_major(
        &q.transpose(1, 2).unwrap().to_device(&cuda).unwrap(),
        &k.transpose(1, 2).unwrap().to_device(&cuda).unwrap(),
        &v.transpose(1, 2).unwrap().to_device(&cuda).unwrap(),
        &g.transpose(1, 2).unwrap().to_device(&cuda).unwrap(),
        &beta.transpose(1, 2).unwrap().to_device(&cuda).unwrap(),
    )
    .unwrap()
    .flatten_all()
    .unwrap()
    .to_vec1::<f32>()
    .unwrap();

    let rtol = 1.3e-6f32;
    let atol = 1e-5f32;
    for (index, (expected, actual)) in reference.iter().zip(actual).enumerate() {
        let error = (expected - actual).abs();
        let allowed = atol + rtol * expected.abs();
        assert!(
            error <= allowed,
            "Delta-rule mismatch at {index}: expected {expected}, got {actual}, error {error} > {allowed}"
        );
    }
}

#[cfg(feature = "cuda")]
#[test]
fn subwarp16_cuda_is_bitwise_equal_to_uncached_recurrence() -> Result<()> {
    let device = test_cuda_device()?;
    for (batch, length, value_dim) in [
        (1usize, 17usize, 128usize),
        (2, 290, 128),
        (1, 1024, 128),
        (1, 17, 7),
    ] {
        let key_heads = 2usize;
        let heads = 4usize;
        let key_dim = 128usize;
        let data = |count: usize, factor: f32| {
            (0..count)
                .map(|i| ((i * 13 % 29) as f32 - 14.) * factor)
                .collect::<Vec<_>>()
        };
        let q = Tensor::from_vec(
            data(batch * length * key_heads * key_dim, 0.001),
            (batch, length, key_heads, key_dim),
            &device,
        )?;
        let k = Tensor::from_vec(
            data(batch * length * key_heads * key_dim, 0.006),
            (batch, length, key_heads, key_dim),
            &device,
        )?;
        let v = Tensor::from_vec(
            data(batch * length * heads * value_dim, 0.008),
            (batch, length, heads, value_dim),
            &device,
        )?;
        let gate = Tensor::from_vec(
            (0..batch * length * heads)
                .map(|i| -0.001 * (1 + (i % 29)) as f32)
                .collect::<Vec<_>>(),
            (batch, length, heads),
            &device,
        )?;
        let beta = Tensor::from_vec(
            vec![0.5f32; batch * length * heads],
            (batch, length, heads),
            &device,
        )?;
        let old = cuda_delta_rule_token_major_impl(&q, &k, &v, &gate, &beta, false)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let new = cuda_delta_rule_token_major(&q, &k, &v, &gate, &beta)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        assert_eq!(old.len(), new.len());
        for (i, (a, b)) in old.iter().zip(&new).enumerate() {
            assert!(a.is_finite() && b.is_finite());
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "subwarp delta mismatch at {batch}/{length}/{i}"
            );
        }
    }
    println!("SUBWARP_DELTA_PARITY exact_recurrence=true grouped_heads=true batch_shapes=3");
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn grouped_delta_cuda_is_bitwise_equal_to_expanded_preparation() {
    let device = test_cuda_device().unwrap();
    for (batch, length, key_heads, heads, value_dim) in
        [(1, 1, 1, 4, 5), (2, 33, 2, 8, 16), (2, 290, 4, 16, 128)]
    {
        for dtype in supported_rotary_test_dtypes(&device).unwrap() {
            let input = |width: usize, factor: f32, count_heads: usize| {
                Tensor::from_vec(
                    (0..batch * length * count_heads * width)
                        .map(|i| ((i * 17 % 101) as f32 - 50.) * factor)
                        .collect::<Vec<_>>(),
                    (batch, length, count_heads, width),
                    &device,
                )
                .unwrap()
                .to_dtype(dtype)
                .unwrap()
            };
            let expand = |x: &Tensor| {
                x.unsqueeze(3)
                    .unwrap()
                    .broadcast_as((batch, length, key_heads, heads / key_heads, 128))
                    .unwrap()
                    .reshape((batch, length, heads, 128))
                    .unwrap()
            };
            let normalize = |x: &Tensor, scale: f64| {
                let x = x.to_dtype(DType::F32).unwrap();
                let denominator = (x.sqr().unwrap().sum_keepdim(D::Minus1).unwrap() + 1e-6)
                    .unwrap()
                    .sqrt()
                    .unwrap();
                (x.broadcast_div(&denominator).unwrap() * scale).unwrap()
            };
            let raw_q = input(128, 0.003, key_heads);
            let raw_k = input(128, 0.007, key_heads);
            let q = normalize(&raw_q, 128f64.powf(-0.5));
            let k = normalize(&raw_k, 1.);
            let old_q = normalize(&expand(&raw_q), 128f64.powf(-0.5));
            let old_k = normalize(&expand(&raw_k), 1.);
            let read = |x: &Tensor| x.flatten_all().unwrap().to_vec1::<f32>().unwrap();

            let assert_bits = |a: &Tensor, b: &Tensor| {
                let a = read(a);
                let b = read(b);
                assert_eq!(a.len(), b.len());
                for (i, (a, b)) in a.iter().zip(&b).enumerate() {
                    assert!(a.is_finite() && b.is_finite());
                    assert_eq!(
                        a.to_bits(),
                        b.to_bits(),
                        "{dtype:?} batch={batch} length={length} heads={heads} value_dim={value_dim} element={i}"
                    );
                }
            };
            assert_bits(&expand(&q), &old_q);
            assert_bits(&expand(&k), &old_k);
            let v = input(value_dim, 0.008, heads).to_dtype(DType::F32).unwrap();
            let gate = Tensor::full(-0.002f32, (batch, length, heads), &device).unwrap();
            let beta = Tensor::full(0.5f32, (batch, length, heads), &device).unwrap();
            let grouped = cuda_delta_rule_token_major(&q, &k, &v, &gate, &beta).unwrap();
            let expanded = cuda_delta_rule_token_major(&old_q, &old_k, &v, &gate, &beta).unwrap();
            assert_bits(&grouped, &expanded);
        }
    }
}

#[cfg(feature = "cuda")]
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

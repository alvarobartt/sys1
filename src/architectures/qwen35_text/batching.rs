const LENGTH_RATIO_NUMERATOR: usize = 5;
const LENGTH_RATIO_DENOMINATOR: usize = 4;
#[cfg(any(feature = "cuda", test))]
pub(crate) fn activation_bytes_per_token(
    dtype_bytes: usize,
    hidden: usize,
    intermediate: usize,
    key_width: usize,
    value_width: usize,
    value_heads: usize,
) -> usize {
    let qkv_width = key_width.saturating_mul(2).saturating_add(value_width);

    let linear_f32 = 4usize.saturating_mul(
        value_width
            .saturating_mul(4)
            .saturating_add(value_heads.saturating_mul(2)),
    );
    let linear_low_precision = dtype_bytes.saturating_mul(
        qkv_width
            .saturating_mul(2)
            .saturating_add(value_width)
            .saturating_add(hidden.saturating_mul(4)),
    );

    let mlp = dtype_bytes.saturating_mul(
        intermediate
            .saturating_mul(3)
            .saturating_add(hidden.saturating_mul(4)),
    );
    linear_f32.saturating_add(linear_low_precision).max(mlp)
}

pub(crate) fn cuda_token_budget(
    free_bytes: usize,
    bytes_per_token: usize,
    max_length: usize,
) -> usize {
    estimated_memory_tokens(free_bytes, bytes_per_token).clamp(1, max_length)
}

#[allow(dead_code)]
pub(crate) fn cuda_token_budget_for_attention(
    free_bytes: usize,
    bytes_per_token: usize,
    max_length: usize,
    flash: bool,
) -> usize {
    if flash {
        estimated_memory_tokens(free_bytes, bytes_per_token).max(1)
    } else {
        cuda_token_budget(free_bytes, bytes_per_token, max_length)
    }
}

pub(crate) fn estimated_memory_tokens(free_bytes: usize, bytes_per_token: usize) -> usize {
    free_bytes.saturating_sub(free_bytes / 4) / bytes_per_token.max(1)
}

pub(crate) fn estimated_active_rows(
    token_budget: usize,
    max_length: usize,
    row_length: usize,
) -> usize {
    if row_length == 0 || row_length > max_length {
        return 0;
    }
    let attention_rows = (max_length as u128).pow(2) / (row_length as u128).pow(2);
    (token_budget / row_length).min(attention_rows.min(usize::MAX as u128) as usize)
}

#[allow(dead_code)]
pub(crate) fn estimated_active_rows_for_attention(
    token_budget: usize,
    max_length: usize,
    row_length: usize,
    flash: bool,
) -> usize {
    if row_length == 0 || row_length > max_length {
        return 0;
    }
    if flash {
        token_budget / row_length
    } else {
        estimated_active_rows(token_budget, max_length, row_length)
    }
}

#[cfg(any(feature = "cuda", test))]
pub(crate) fn refreshed_token_budget(current: usize, live: usize) -> usize {
    current.min(live).max(1)
}

pub(crate) fn token_budget_after_oom(current: usize, failed_padded_tokens: usize) -> usize {
    current
        .min(failed_padded_tokens.saturating_sub(1) / 2)
        .max(1)
}

pub(crate) fn buckets(
    lengths: &[(usize, usize)],
    max_length: usize,
    token_budget: usize,
) -> Vec<Vec<usize>> {
    buckets_for_attention(lengths, max_length, token_budget, false)
}

#[allow(dead_code)]
pub(crate) fn buckets_for_attention(
    lengths: &[(usize, usize)],
    max_length: usize,
    token_budget: usize,
    flash: bool,
) -> Vec<Vec<usize>> {
    let mut sorted = lengths.to_vec();
    sorted.sort_by_key(|&(slot, length)| (length, slot));
    let mut result = Vec::new();
    let mut current: Vec<usize> = Vec::new();
    let mut shortest = 0usize;

    for (slot, length) in sorted {
        let candidate_shortest = if current.is_empty() { length } else { shortest };
        let candidate_longest = length;
        let candidate_batch = current.len() + 1;
        let similar = candidate_longest.saturating_mul(LENGTH_RATIO_DENOMINATOR)
            <= candidate_shortest.saturating_mul(LENGTH_RATIO_NUMERATOR);
        let token_safe = candidate_batch.saturating_mul(candidate_longest) <= token_budget;
        let attention_safe = if flash {
            candidate_longest <= max_length
        } else {
            (candidate_batch as u128)
                .saturating_mul(candidate_longest as u128)
                .saturating_mul(candidate_longest as u128)
                <= (max_length as u128).saturating_mul(max_length as u128)
        };

        if !current.is_empty() && !(similar && token_safe && attention_safe) {
            result.push(std::mem::take(&mut current));
            shortest = length;
        } else if current.is_empty() {
            shortest = length;
        }
        current.push(slot);
        debug_assert!(current.len().saturating_mul(length) <= token_budget || current.len() == 1);
    }
    if !current.is_empty() {
        result.push(current);
    }
    result
}

pub(crate) fn packed_buckets(
    lengths: &[(usize, usize)],
    max_length: usize,
    budget: usize,
) -> Vec<Vec<usize>> {
    if option_env!("SYS1_CUDA_COMPUTE_CAP") == Some("90") {
        return packed_buckets_long_work(lengths, max_length, budget, 128);
    }
    if option_env!("SYS1_CUDA_COMPUTE_CAP") == Some("80") {
        return packed_buckets_long_work(lengths, max_length, budget, 128);
    }
    let mut sorted = lengths.to_vec();
    sorted.sort_by_key(|&(slot, length)| (length, slot));
    let mut buckets = Vec::new();
    let mut current = Vec::new();
    let mut tokens = 0usize;
    for (slot, length) in sorted {
        let safe = tokens.checked_add(length).is_some_and(|n| n <= budget) && length <= max_length;
        if !current.is_empty() && !safe {
            buckets.push(std::mem::take(&mut current));
            tokens = 0;
        }
        current.push(slot);
        tokens = tokens.saturating_add(length);
        if length > max_length {
            buckets.push(std::mem::take(&mut current));
            tokens = 0;
        }
    }
    if !current.is_empty() {
        buckets.push(current);
    }
    buckets
}

#[cfg(feature = "cuda")]
pub(crate) fn c64_scratch_bytes_per_token(key_dim: usize, value_dim: usize, heads: usize) -> usize {
    let base = heads
        .saturating_mul(key_dim.saturating_add(value_dim).saturating_add(64 + 2))
        .saturating_mul(4);
    let prepack = if matches!(option_env!("SYS1_CUDA_COMPUTE_CAP"), Some("80" | "90"))
        && key_dim == 128
        && value_dim <= 128
    {
        heads.saturating_mul(28672 * 2 + 4).div_ceil(64)
    } else {
        0
    };

    base.saturating_add(base.div_ceil(
        if matches!(option_env!("SYS1_CUDA_COMPUTE_CAP"), Some("80" | "90")) {
            2
        } else {
            8
        },
    ))
    .saturating_add(64)
    .saturating_add(prepack)
    .saturating_add(prepack.div_ceil(2))
}

pub(crate) fn aggregate_work_multiplier() -> usize {
    if matches!(option_env!("SYS1_CUDA_COMPUTE_CAP"), Some("80" | "90")) {
        4
    } else {
        2
    }
}

pub(crate) fn long_work_budget(
    max_length: usize,
    memory_budget: usize,
    row_length: usize,
    long_threshold: usize,
) -> usize {
    memory_budget.min(if row_length >= long_threshold {
        max_length.saturating_mul(aggregate_work_multiplier())
    } else {
        max_length
    })
}

pub(crate) fn packed_buckets_long_work(
    lengths: &[(usize, usize)],
    max_length: usize,
    memory_budget: usize,
    long_threshold: usize,
) -> Vec<Vec<usize>> {
    let mut sorted = lengths.to_vec();
    if option_env!("SYS1_CUDA_COMPUTE_CAP") == Some("90") {
        sorted.sort_by_key(|&(slot, length)| {
            let long = length >= long_threshold;
            (long, if long { length.div_ceil(64) } else { length }, slot)
        });
    } else {
        sorted.sort_by_key(|&(slot, length)| (length, slot));
    }
    let mut result = Vec::new();
    let mut current = Vec::new();
    let mut tokens = 0usize;
    let mut current_long = false;
    for (slot, length) in sorted {
        let is_long = length >= long_threshold;

        if !current.is_empty() && current_long != is_long {
            result.push(std::mem::take(&mut current));
            tokens = 0;
        }
        let budget = long_work_budget(max_length, memory_budget, length, long_threshold);
        let safe = tokens.checked_add(length).is_some_and(|n| n <= budget) && length <= max_length;
        if !current.is_empty() && !safe {
            result.push(std::mem::take(&mut current));
            tokens = 0;
        }
        current_long = is_long;
        current.push(slot);
        tokens = tokens.saturating_add(length);

        if length > max_length {
            result.push(std::mem::take(&mut current));
            tokens = 0;
        }
    }
    if !current.is_empty() {
        result.push(current);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device, Tensor};

    #[test]
    fn flash_admission_uses_memory_not_single_request_context_sum() {
        let bytes = 139_520;
        let free = 3 * 1024usize.pow(3);
        let budget = cuda_token_budget_for_attention(free, bytes, 4096, true);
        assert_eq!(budget, 17_315);
        assert_eq!(
            cuda_token_budget_for_attention(free, bytes, 4096, false),
            4096
        );
        assert_eq!(
            estimated_active_rows_for_attention(budget, 4096, 4096, true),
            4
        );
        assert_eq!(
            estimated_active_rows_for_attention(budget, 4096, 4096, false),
            1
        );
        assert_eq!(
            estimated_active_rows_for_attention(budget, 4096, 4097, true),
            0
        );
        assert_eq!(
            estimated_active_rows_for_attention(budget, 4096, 0, true),
            0
        );
        let rows = [(0, 4096), (1, 4096), (2, 4096), (3, 4096), (4, 4096)];
        assert_eq!(
            buckets_for_attention(&rows, 4096, budget, true),
            vec![vec![0, 1, 2, 3], vec![4]]
        );
        assert_eq!(
            buckets_for_attention(&rows, 4096, budget, false),
            vec![vec![0], vec![1], vec![2], vec![3], vec![4]]
        );
    }

    #[test]
    fn flash_admission_preserves_padding_and_oom_limits() {
        assert_eq!(
            buckets_for_attention(&[(0, 200), (1, 260)], 1000, 10000, true),
            vec![vec![0], vec![1]]
        );
        assert_eq!(
            buckets_for_attention(&[(0, 1500), (1, 1500)], 1000, 10000, true),
            vec![vec![0], vec![1]]
        );
        assert_eq!(cuda_token_budget_for_attention(0, 100, 4096, true), 1);
        assert_eq!(refreshed_token_budget(17000, 12000), 12000);
        let corrected = token_budget_after_oom(17000, 16000);
        assert_eq!(
            buckets_for_attention(&[(0, 4096), (1, 4096)], 4096, corrected, true),
            vec![vec![0], vec![1]]
        );
    }

    #[test]
    fn buckets_are_stable_length_bounded_and_cover_every_slot() {
        let lengths = [(0, 100), (1, 105), (2, 400), (3, 110), (4, 390)];
        let groups = buckets(&lengths, 1_000, 1_000);
        assert_eq!(groups, vec![vec![0, 1, 3], vec![4, 2]]);
        let mut covered: Vec<_> = groups.into_iter().flatten().collect();
        covered.sort_unstable();
        assert_eq!(covered, vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn buckets_split_on_padding_and_attention_budgets() {
        assert_eq!(
            buckets(&[(0, 300), (1, 300), (2, 300), (3, 300)], 1_000, 1_000),
            vec![vec![0, 1, 2], vec![3]]
        );
        assert_eq!(
            buckets(&[(0, 200), (1, 260)], 1_000, 1_000),
            vec![vec![0], vec![1]]
        );
        assert_eq!(
            buckets(&[(0, 1_500), (1, 1_500)], 1_000, 1_000),
            vec![vec![0], vec![1]]
        );
    }

    #[test]
    fn buckets_accept_empty_input_and_keep_one_long_row() {
        assert!(buckets(&[], 1_000, 1_000).is_empty());
        assert_eq!(buckets(&[(7, 1_500)], 1_000, 1_000), vec![vec![7]]);
    }

    #[test]
    fn cuda_budget_uses_architecture_bytes_and_reserves_headroom() {
        let bytes = activation_bytes_per_token(2, 4096, 12288, 2048, 4096, 32);
        assert_eq!(bytes, 139_520);
        let gib = 1024usize.pow(3);
        assert_eq!(cuda_token_budget(3 * gib, bytes, 32_768), 17_315);
        assert_eq!(cuda_token_budget(256 << 20, bytes, 32_768), 1_442);
        assert_eq!(cuda_token_budget(96 * gib, bytes, 32_768), 32_768);
        assert!(estimated_memory_tokens(96 * gib, bytes) > 32_768);
        assert_eq!(estimated_memory_tokens(0, bytes), 0);
    }

    #[test]
    fn reported_capacity_obeys_token_context_and_attention_limits() {
        assert_eq!(estimated_active_rows(16_384, 16_384, 512), 32);
        assert_eq!(estimated_active_rows(16_384, 16_384, 2_048), 8);
        assert_eq!(estimated_active_rows(16_384, 16_384, 8_192), 2);
        assert_eq!(estimated_active_rows(16_384, 16_384, 16_384), 1);
        assert_eq!(estimated_active_rows(100_000, 16_384, 8_192), 4);
        assert_eq!(estimated_active_rows(1, 16_384, 512), 0);
        assert_eq!(estimated_active_rows(16_384, 16_384, 16_385), 0);
        assert_eq!(estimated_active_rows(16_384, 16_384, 0), 0);
    }

    #[test]
    fn token_budget_splits_otherwise_compatible_rows_but_keeps_singletons() {
        assert_eq!(
            buckets(&[(0, 1_000), (1, 1_000), (2, 1_000)], 32_768, 2_100),
            vec![vec![0, 1], vec![2]]
        );
        assert_eq!(
            buckets(&[(0, 3_000), (1, 3_000)], 32_768, 2_100),
            vec![vec![0], vec![1]]
        );
    }

    #[test]
    fn refreshed_and_oom_budgets_only_shrink() {
        assert_eq!(refreshed_token_budget(5_700, 6_000), 5_700);
        assert_eq!(refreshed_token_budget(5_700, 4_800), 4_800);
        assert_eq!(token_budget_after_oom(5_700, 5_600), 2_799);
        assert_eq!(token_budget_after_oom(2_000, 5_600), 2_000);
        assert_eq!(token_budget_after_oom(5_700, 1), 1);
    }

    #[test]
    fn right_padding_cannot_change_valid_causal_attention_prefix() {
        #[allow(unused_mut)]
        let mut devices = vec![Device::Cpu];
        #[cfg(feature = "cuda")]
        devices.push(Device::new_cuda(0).unwrap());
        for device in devices {
            let row = |values: &[f32], length| {
                Tensor::from_vec(values.to_vec(), (1, length, 1, 2), &device).unwrap()
            };
            let short_q = row(&[1., 0., 0., 1.], 2);
            let short_k = row(&[1., 0., 0., 1.], 2);
            let short_v = row(&[2., 3., 5., 7.], 2);
            let padded_q = row(&[1., 0., 0., 1., 50., -50.], 3);
            let padded_k = row(&[1., 0., 0., 1., -80., 80.], 3);
            let padded_v = row(&[2., 3., 5., 7., 10_000., -10_000.], 3);
            let long_q = row(&[0., 1., 1., 1., 1., 0.], 3);
            let long_k = row(&[0., 1., 1., 1., 1., 0.], 3);
            let long_v = row(&[1., 2., 3., 4., 5., 6.], 3);
            let q = Tensor::cat(&[&padded_q, &long_q], 0).unwrap();
            let k = Tensor::cat(&[&padded_k, &long_k], 0).unwrap();
            let v = Tensor::cat(&[&padded_v, &long_v], 0).unwrap();

            let singleton = crate::architectures::qwen35_text::eager_attention(
                &short_q, &short_k, &short_v, 0.5, true,
            )
            .unwrap();
            let batched = crate::architectures::qwen35_text::eager_attention(&q, &k, &v, 0.5, true)
                .unwrap()
                .narrow(0, 0, 1)
                .unwrap()
                .narrow(1, 0, 2)
                .unwrap();
            let delta = (&singleton.to_dtype(DType::F32).unwrap()
                - &batched.to_dtype(DType::F32).unwrap())
                .unwrap()
                .abs()
                .unwrap()
                .max_all()
                .unwrap()
                .to_scalar::<f32>()
                .unwrap();
            assert!(
                delta <= 1e-6,
                "right padding changed a valid prefix by {delta} on {device:?}"
            );
        }
    }
}

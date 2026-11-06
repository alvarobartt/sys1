pub(super) fn bytes(rows: usize, length: usize, linear: usize, quadratic: usize) -> usize {
    rows.saturating_mul(length)
        .saturating_mul(linear.saturating_add(length.saturating_mul(quadratic)))
}

pub(super) fn buckets(
    lengths: &[usize],
    budget: usize,
    linear: usize,
    quadratic: usize,
) -> Vec<Vec<usize>> {
    let mut slots: Vec<_> = (0..lengths.len()).collect();
    slots.sort_by_key(|&slot| (lengths[slot], slot));
    let mut groups = Vec::new();
    let mut group = Vec::new();
    let mut shortest = 0usize;
    for slot in slots {
        let length = lengths[slot];
        let similar = length.saturating_mul(4) <= shortest.saturating_mul(5);
        if !group.is_empty()
            && (!similar || bytes(group.len() + 1, length, linear, quadratic) > budget)
        {
            groups.push(std::mem::take(&mut group));
        }
        if group.is_empty() {
            shortest = length;
        }
        group.push(slot);
    }
    if !group.is_empty() {
        groups.push(group);
    }
    groups
}

pub(super) fn is_cuda_oom(error: &anyhow::Error) -> bool {
    format!("{error:#}").contains("CUDA_ERROR_OUT_OF_MEMORY")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn covers_slots_and_respects_padding_memory_and_length_ratio() {
        let lengths = [100, 110, 400, 100, 390, 10_000];
        let groups = buckets(&lengths, 100_000, 10, 1);
        let mut covered: Vec<_> = groups.iter().flatten().copied().collect();
        covered.sort_unstable();
        assert_eq!(covered, (0..lengths.len()).collect::<Vec<_>>());
        for group in groups {
            let maximum = group.iter().map(|&slot| lengths[slot]).max().unwrap();
            let minimum = group.iter().map(|&slot| lengths[slot]).min().unwrap();
            assert!(group.len() == 1 || bytes(group.len(), maximum, 10, 1) <= 100_000);
            assert!(group.len() == 1 || maximum * 4 <= minimum * 5);
        }
    }

    #[test]
    fn eager_quadratic_workspace_limits_batches_and_flash_does_not_pay_it() {
        let lengths = [1000; 4];
        assert_eq!(
            buckets(&lengths, 2_100_000, 100, 1),
            vec![vec![0], vec![1], vec![2], vec![3]]
        );
        assert_eq!(buckets(&lengths, 2_100_000, 100, 0), vec![vec![0, 1, 2, 3]]);
        assert!(buckets(&[], 0, 0, 0).is_empty());
        assert_eq!(bytes(usize::MAX, usize::MAX, 10, 10), usize::MAX);
    }

    #[test]
    fn only_cuda_allocation_failures_trigger_retries() {
        let error = anyhow::anyhow!("DriverError(CUDA_ERROR_OUT_OF_MEMORY, out of memory)")
            .context("encoder forward");
        assert!(is_cuda_oom(&error));
        assert!(!is_cuda_oom(&anyhow::anyhow!("shape mismatch")));
        assert!(!is_cuda_oom(&anyhow::anyhow!("CUDA_ERROR_ILLEGAL_ADDRESS")));
    }
}

use candle_core::cuda_backend::{
    CudaDevice,
    cudarc::driver::{result, sys},
};

fn read(pool: sys::CUmemoryPool, attr: sys::CUmemPool_attribute) -> anyhow::Result<u64> {
    let mut value = 0u64;
    unsafe { result::mem_pool::get_attribute(pool, attr, (&mut value as *mut u64).cast()) }?;
    Ok(value)
}

pub(crate) fn configure(cuda: &CudaDevice) {
    if option_env!("SYS1_CUDA_COMPUTE_CAP") != Some("90") {
        return;
    }
    let attempt = || -> anyhow::Result<_> {
        let ctx = cuda.cuda_stream().context().clone();
        ctx.bind_to_thread()?;
        let (_, total) = ctx.mem_get_info()?;
        let pool = unsafe { result::device::get_mem_pool(ctx.cu_device()) }?;
        let attr = sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_RELEASE_THRESHOLD;
        let previous = read(pool, attr)?;

        let mut selected = previous.max((64u64 << 30).min(total as u64 / 2));
        unsafe { result::mem_pool::set_attribute(pool, attr, (&mut selected as *mut u64).cast()) }?;
        anyhow::ensure!(
            read(pool, attr)? == selected,
            "CUDA pool policy was not applied"
        );
        Ok((previous, selected))
    };
    match attempt() {
        Ok((previous, selected)) => tracing::info!(
            previous_release_threshold_bytes = previous,
            release_threshold_bytes = selected,
            "CUDA stream-ordered allocation reuse (not KV cache)"
        ),
        Err(error) => {
            tracing::warn!(%error,"CUDA pool reuse unavailable; retaining allocator default")
        }
    }
}

fn available(free: usize, total: usize, reserved: u64, used: u64) -> usize {
    (free as u64)
        .saturating_add(reserved.saturating_sub(used))
        .min(total as u64) as usize
}

pub(crate) fn snapshot(cuda: &CudaDevice) -> anyhow::Result<(usize, usize)> {
    let ctx = cuda.cuda_stream().context().clone();
    ctx.bind_to_thread()?;
    let (free, total) = ctx.mem_get_info()?;
    if !matches!(option_env!("SYS1_CUDA_COMPUTE_CAP"), Some("86" | "90")) {
        return Ok((free, total));
    }
    let credit = || -> anyhow::Result<_> {
        let pool = unsafe { result::device::get_mem_pool(ctx.cu_device()) }?;
        let reserved = read(
            pool,
            sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_RESERVED_MEM_CURRENT,
        )?;
        let used = read(
            pool,
            sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_USED_MEM_CURRENT,
        )?;
        Ok((
            available(free, total, reserved, used),
            reserved.saturating_sub(used),
        ))
    };
    match credit() {
        Ok((capacity, reusable)) => {
            #[cfg(test)]
            println!(
                "CUDA_POOL_SNAPSHOT {}",
                serde_json::json!({
                    "system_free_bytes":free,"total_bytes":total,
                    "own_pool_reusable_bytes":reusable,"available_activation_bytes":capacity})
            );
            tracing::debug!(
                system_free_bytes = free,
                own_pool_reusable_bytes = reusable,
                available_activation_bytes = capacity,
                "CUDA admission includes reusable own-pool memory"
            );
            Ok((capacity, total))
        }

        Err(_) => Ok((free, total)),
    }
}

#[cfg(test)]
mod tests {
    use super::available;
    #[test]
    fn only_unused_own_pool_is_credited() {
        assert_eq!(available(100, 1000, 200, 150), 150);
    }
    #[test]
    fn snapshots_saturate_and_never_exceed_total() {
        assert_eq!(available(100, 1000, 100, 200), 100);
        assert_eq!(available(950, 1000, 500, 100), 1000);
        assert_eq!(available(100, 1000, u64::MAX, 0), 1000);
    }
}

use super::{Decoder, TextConfig};
use crate::architectures::qwen35_text::batching;
use anyhow::Context;
use candle_core::{CudaDevice, DType};

pub(in crate::architectures::qwen35_text) fn initialize_admission(
    decoder: &Decoder,
    config: &TextConfig,
    dtype: DType,
    max_length: usize,
    cuda: &CudaDevice,
) -> anyhow::Result<(usize, usize)> {
    let key_width = config.linear_num_key_heads * config.linear_key_head_dim;
    let value_width = config.linear_num_value_heads * config.linear_value_head_dim;
    let bytes_per_token = batching::activation_bytes_per_token(
        dtype.size_in_bytes(),
        config.hidden_size,
        config.intermediate_size,
        key_width,
        value_width,
        config.linear_num_value_heads,
    );
    let packed_admission = dtype == DType::BF16
        && matches!(
            option_env!("SYS1_CUDA_COMPUTE_CAP"),
            Some("80" | "86" | "89" | "90" | "120")
        )
        && decoder.supports_packed();
    if packed_admission {
        crate::kernels::cuda::memory::configure(cuda);
    }
    let c64_scratch_bytes_per_token = if packed_admission
        && matches!(
            option_env!("SYS1_CUDA_COMPUTE_CAP"),
            Some("80" | "90" | "120")
        )
        && config.linear_value_head_dim <= 128
    {
        batching::c64_scratch_bytes_per_token(
            config.linear_key_head_dim,
            config.linear_value_head_dim,
            config.linear_num_value_heads,
        )
    } else {
        0
    };
    let base_activation_bytes_per_token = bytes_per_token;
    let bytes_per_token = bytes_per_token.saturating_add(c64_scratch_bytes_per_token);
    let (free_bytes, _) = crate::kernels::cuda::memory::snapshot(cuda)
        .context("Failed to query CUDA memory after loading Qwen3.5")?;
    let maximum_projection_width = config
        .intermediate_size
        .max(config.hidden_size)
        .max(key_width.saturating_mul(2).saturating_add(value_width))
        .max(
            config
                .num_attention_heads
                .saturating_mul(config.head_dim)
                .saturating_mul(2),
        );
    let index_token_capacity = (i32::MAX as usize) / maximum_projection_width.max(1);
    let token_budget = |free_bytes| {
        if packed_admission {
            let multiplier = match option_env!("SYS1_CUDA_COMPUTE_CAP") {
                Some("80" | "90") => 4,
                Some("86" | "120") => 2,
                _ => 1,
            };
            batching::estimated_memory_tokens(free_bytes, bytes_per_token)
                .max(1)
                .min(max_length.saturating_mul(multiplier))
                .min(index_token_capacity.max(1))
        } else {
            batching::cuda_token_budget(free_bytes, bytes_per_token, max_length)
        }
    };
    let (_, _, head_cache_growth_reserve_bytes) =
        crate::kernels::cuda::guarded_dgemm::head_cache_memory();
    let free_bytes = free_bytes.saturating_sub(head_cache_growth_reserve_bytes);
    let before_layout_tokens = token_budget(free_bytes);
    let nn_weight_bytes = decoder.prepare_nn_weights(before_layout_tokens, bytes_per_token)?;
    let (free_bytes, total_bytes) = crate::kernels::cuda::memory::snapshot(cuda)
        .context("Failed to query CUDA memory after preparing optional layouts")?;
    let (head_cache_entries, head_cache_resident_bytes, head_cache_growth_reserve_bytes) =
        crate::kernels::cuda::guarded_dgemm::head_cache_memory();
    let memory_available_before_head_reserve = free_bytes;
    let free_bytes = free_bytes.saturating_sub(head_cache_growth_reserve_bytes);
    let budget = token_budget(free_bytes);
    let active_rows = |row_length| {
        let work_budget = if packed_admission && option_env!("SYS1_CUDA_COMPUTE_CAP") == Some("90")
        {
            batching::long_work_budget(max_length, budget, row_length, 128)
        } else {
            budget
        };
        batching::estimated_active_rows_for_attention(
            work_budget,
            max_length,
            row_length,
            packed_admission,
        )
    };
    tracing::info!(
        free_bytes,
        total_bytes,
        occupied_device_bytes = total_bytes.saturating_sub(memory_available_before_head_reserve),
        optional_weight_layout_bytes = nn_weight_bytes,
        memory_available_before_head_reserve,
        head_cache_entries,
        head_cache_resident_bytes,
        head_cache_growth_reserve_bytes,
        reserved_headroom_bytes = free_bytes / 4,
        usable_activation_bytes = free_bytes.saturating_sub(free_bytes / 4),
        bytes_per_token,
        base_activation_bytes_per_token,
        c64_scratch_bytes_per_token,
        estimated_memory_tokens = batching::estimated_memory_tokens(free_bytes, bytes_per_token),
        batch_token_budget = budget,
        short_work_token_budget = if packed_admission
            && matches!(option_env!("SYS1_CUDA_COMPUTE_CAP"), Some("80" | "90"))
        {
            budget.min(max_length)
        } else {
            budget
        },
        long_work_token_budget = budget,
        long_work_threshold_tokens = if packed_admission {
            match option_env!("SYS1_CUDA_COMPUTE_CAP") {
                Some("80") => 128usize,
                Some("90") => 128usize,
                _ => 0,
            }
        } else {
            0
        },
        packed_token_admission = packed_admission,
        max_context_tokens = max_length,
        active_rows_at_512_tokens = active_rows(512),
        active_rows_at_2048_tokens = active_rows(2048),
        active_rows_at_8192_tokens = active_rows(8192),
        active_rows_at_16384_tokens = active_rows(16384),
        "Qwen3.5 CUDA activation capacity estimate (not KV cache; rows also limited by server batch cap; singleton OOM remains possible)"
    );
    Ok((bytes_per_token, budget))
}

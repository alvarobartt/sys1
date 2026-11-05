#[cfg(feature = "metal")]
use crate::architectures::metal::QwenDecoder as ModelDecoder;
#[cfg(not(feature = "metal"))]
use crate::architectures::qwen35_text::Decoder as ModelDecoder;

use crate::models::AttentionImplementation;
use crate::{
    architectures::{qwen35_text as text, qwen35_vision as vision},
    device,
    schema::ApiError,
    tokenizer,
};
use anyhow::Context;
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use std::{
    fs,
    path::Path,
    sync::atomic::{AtomicUsize, Ordering},
};

use super::batching;

pub(crate) fn cuda_capability(_device: &Device) -> anyhow::Result<Option<(i32, i32)>> {
    #[cfg(feature = "cuda")]
    if let Device::Cuda(cuda) = _device {
        return Ok(Some(
            cuda.cuda_stream()
                .context()
                .compute_capability()
                .context("Failed to query CUDA compute capability")?,
        ));
    }
    Ok(None)
}

fn select_attention(
    requested: AttentionImplementation,
    dtype: DType,
    capability: Option<(i32, i32)>,
) -> AttentionImplementation {
    if requested != AttentionImplementation::Auto {
        return requested;
    }
    crate::models::automatic_attention(dtype, capability)
}

fn validate_flash_device(
    attention: AttentionImplementation,
    capability: Option<(i32, i32)>,
) -> anyhow::Result<()> {
    let supported = match attention {
        AttentionImplementation::Auto | AttentionImplementation::Eager => return Ok(()),
        AttentionImplementation::FlashAttention2 => {
            capability.is_some_and(|(major, _)| (8..=9).contains(&major))
        }
        AttentionImplementation::FlashAttentionBlackwell => capability == Some((12, 0)),
    };
    anyhow::ensure!(
        capability.is_some(),
        "Attention {} requires a CUDA device",
        attention.cli_name()
    );
    let (major, minor) = capability.unwrap();
    anyhow::ensure!(
        supported,
        "Attention {} does not support CUDA compute capability {major}.{minor}",
        attention.cli_name()
    );
    Ok(())
}

fn validate_declared_dtypes(
    declared: Option<&str>,
    text_dtype: Option<&str>,
    vision_dtype: Option<&str>,
    requested: DType,
) -> anyhow::Result<()> {
    use crate::models::dtype::parse;

    let mut source = declared.map(parse).transpose()?;
    for (section, value) in [("text_config", text_dtype), ("vision_config", vision_dtype)] {
        if let Some(value) = value {
            let dtype = parse(value)?;
            if let Some(source) = source {
                anyhow::ensure!(
                    dtype == source,
                    "Qwen3.5 {section}.dtype {value:?} differs from checkpoint dtype {source:?}"
                );
            } else {
                source = Some(dtype);
            }
        }
    }
    anyhow::ensure!(
        requested != DType::F16 || source.is_none_or(|source| source == DType::F16),
        "Unsafe Qwen3.5 dtype conversion from {source:?} to f16: FP16 has a narrower exponent range; use --dtype bf16 or --dtype f32"
    );
    tracing::info!(checkpoint_dtype = declared, inference_dtype = ?requested, "Qwen3.5 dtype validated");
    Ok(())
}

pub(crate) struct Backbone {
    pub(crate) context: crate::protocols::Context,
    pub(crate) causal: bool,
    pub(crate) decoder: ModelDecoder,
    pub(crate) vision: vision::VisionTower,
    pub(crate) device: Device,
    pub(crate) dtype: DType,
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    pub(crate) batch_bytes_per_token: usize,
    pub(crate) batch_token_budget: AtomicUsize,
}

impl std::ops::Deref for Backbone {
    type Target = crate::protocols::Context;
    fn deref(&self) -> &Self::Target {
        &self.context
    }
}

impl Backbone {
    pub(crate) fn load<H>(
        path: &Path,
        dtype: DType,
        max_model_len: Option<usize>,
        attention: AttentionImplementation,
        causal: bool,
        max_pixels: usize,
        load_head: impl FnOnce(
            &text::Config,
            VarBuilder,
            AttentionImplementation,
            &tokenizer::Tokenizer,
        ) -> anyhow::Result<H>,
    ) -> anyhow::Result<(Self, H)> {
        let dtype =
            crate::models::dtype::resolve(path, crate::models::Architecture::Qwen35, Some(dtype))?;
        let mut config = text::Config::load(&path.join("config.json"))?;
        validate_declared_dtypes(
            config.dtype.as_deref(),
            config.text_config.dtype.as_deref(),
            config.vision_config.dtype.as_deref(),
            dtype,
        )?;
        config.text_config.full_attention_causal = causal;
        vision::images::validate_config_with_limit(
            &path.join("processor_config.json"),
            &config.vision_config,
            max_pixels,
        )?;
        vision::video::validate_config(&path.join("processor_config.json"), &config.vision_config)?;
        let device = device::load()?;
        anyhow::ensure!(
            !device.is_cpu() || dtype == DType::F32,
            "Qwen3.5 on CPU requires f32; use --dtype f32 explicitly"
        );
        let capability = if attention == AttentionImplementation::Eager
            || (attention == AttentionImplementation::Auto && dtype == DType::F32)
        {
            None
        } else {
            cuda_capability(&device)?
        };
        let attention = if attention == AttentionImplementation::Auto && device.is_metal() {
            AttentionImplementation::Auto
        } else {
            select_attention(attention, dtype, capability)
        };
        attention.validate(dtype)?;
        validate_flash_device(attention, capability)?;
        let attention_name = attention.cli_name();
        tracing::info!(attention = attention_name, "Qwen3.5 attention selected");
        let max_length = max_model_len.unwrap_or(16384);
        anyhow::ensure!(
            max_length > 0 && max_length <= 262144,
            "Invalid Qwen3.5 model length"
        );
        let tokenizer = tokenizer::from_qwen_json(&fs::read(path.join("tokenizer.json"))?)
            .context("Qwen tokenizer")?;
        let mut shards: Vec<_> = fs::read_dir(path)?
            .filter_map(|entry| entry.ok().map(|entry| entry.path()))
            .filter(|file| {
                file.file_name()
                    .and_then(|name| name.to_str())
                    .is_some_and(|name| {
                        name == "model.safetensors"
                            || (name.starts_with("model-") && name.ends_with(".safetensors"))
                    })
            })
            .collect();
        shards.sort();
        anyhow::ensure!(!shards.is_empty(), "No Qwen3.5 model safetensors found");
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&shards, dtype, &device)? };
        let prefix = if vb.contains_tensor("model.language_model.embed_tokens.weight") {
            "model."
        } else {
            ""
        };
        let decoder = ModelDecoder::load(
            &config.text_config,
            vb.pp(format!("{prefix}language_model")),
            attention,
        )
        .context("Qwen3.5 decoder")?;
        let vision = vision::VisionTower::load(
            &config.vision_config,
            vb.pp(format!("{prefix}visual")),
            attention,
        )
        .context("Qwen3.5 vision tower")?;
        let head = load_head(&config, vb, attention, &tokenizer)?;
        #[cfg(feature = "cuda")]
        let (batch_bytes_per_token, batch_token_budget) = if let Device::Cuda(cuda) = &device {
            let key_width =
                config.text_config.linear_num_key_heads * config.text_config.linear_key_head_dim;
            let value_width = config.text_config.linear_num_value_heads
                * config.text_config.linear_value_head_dim;
            let bytes_per_token = batching::activation_bytes_per_token(
                dtype.size_in_bytes(),
                config.text_config.hidden_size,
                config.text_config.intermediate_size,
                key_width,
                value_width,
                config.text_config.linear_num_value_heads,
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
                && config.text_config.linear_value_head_dim <= 128
            {
                batching::c64_scratch_bytes_per_token(
                    config.text_config.linear_key_head_dim,
                    config.text_config.linear_value_head_dim,
                    config.text_config.linear_num_value_heads,
                )
            } else {
                0
            };
            let base_activation_bytes_per_token = bytes_per_token;
            let bytes_per_token = bytes_per_token.saturating_add(c64_scratch_bytes_per_token);
            let (free_bytes, _) = crate::kernels::cuda::memory::snapshot(cuda)
                .context("Failed to query CUDA memory after loading Qwen3.5")?;
            let maximum_projection_width = config
                .text_config
                .intermediate_size
                .max(config.text_config.hidden_size)
                .max(key_width.saturating_mul(2).saturating_add(value_width))
                .max(
                    config
                        .text_config
                        .num_attention_heads
                        .saturating_mul(config.text_config.head_dim)
                        .saturating_mul(2),
                );
            let index_token_capacity = (i32::MAX as usize) / maximum_projection_width.max(1);
            let (_, _, head_cache_growth_reserve_bytes) =
                crate::kernels::cuda::guarded_dgemm::head_cache163_memory();
            let free_bytes = free_bytes.saturating_sub(head_cache_growth_reserve_bytes);
            let before_layout_tokens = if packed_admission {
                batching::estimated_memory_tokens(free_bytes, bytes_per_token)
                    .max(1)
                    .min(max_length.saturating_mul(
                        if matches!(option_env!("SYS1_CUDA_COMPUTE_CAP"), Some("80" | "90")) {
                            4
                        } else if matches!(option_env!("SYS1_CUDA_COMPUTE_CAP"), Some("86" | "120"))
                        {
                            2
                        } else {
                            1
                        },
                    ))
                    .min(index_token_capacity.max(1))
            } else {
                batching::cuda_token_budget(free_bytes, bytes_per_token, max_length)
            };
            let nn_weight_bytes =
                decoder.prepare_nn_weights(before_layout_tokens, bytes_per_token)?;
            let (free_bytes, total_bytes) = crate::kernels::cuda::memory::snapshot(cuda)
                .context("Failed to query CUDA memory after preparing optional layouts")?;
            let (head_cache_entries, head_cache_resident_bytes, head_cache_growth_reserve_bytes) =
                crate::kernels::cuda::guarded_dgemm::head_cache163_memory();
            let memory_available_before_head_reserve = free_bytes;
            let free_bytes = free_bytes.saturating_sub(head_cache_growth_reserve_bytes);
            let budget = if packed_admission {
                batching::estimated_memory_tokens(free_bytes, bytes_per_token)
                    .max(1)
                    .min(max_length.saturating_mul(
                        if matches!(option_env!("SYS1_CUDA_COMPUTE_CAP"), Some("80" | "90")) {
                            4
                        } else if matches!(
                            option_env!("SYS1_CUDA_COMPUTE_CAP"),
                            Some("86" | "90" | "120")
                        ) {
                            2
                        } else {
                            1
                        },
                    ))
                    .min(index_token_capacity.max(1))
            } else {
                batching::cuda_token_budget(free_bytes, bytes_per_token, max_length)
            };
            tracing::info!(
                free_bytes,
                total_bytes,
                occupied_device_bytes =
                    total_bytes.saturating_sub(memory_available_before_head_reserve),
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
                estimated_memory_tokens =
                    batching::estimated_memory_tokens(free_bytes, bytes_per_token),
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
                active_rows_at_512_tokens = batching::estimated_active_rows_for_attention(
                    if packed_admission && option_env!("SYS1_CUDA_COMPUTE_CAP") == Some("90") {
                        batching::long_work_budget(max_length, budget, 512, 128)
                    } else {
                        budget
                    },
                    max_length,
                    512,
                    packed_admission
                ),
                active_rows_at_2048_tokens = batching::estimated_active_rows_for_attention(
                    if packed_admission && option_env!("SYS1_CUDA_COMPUTE_CAP") == Some("90") {
                        batching::long_work_budget(max_length, budget, 2048, 128)
                    } else {
                        budget
                    },
                    max_length,
                    2048,
                    packed_admission
                ),
                active_rows_at_8192_tokens = batching::estimated_active_rows_for_attention(
                    if packed_admission && option_env!("SYS1_CUDA_COMPUTE_CAP") == Some("90") {
                        batching::long_work_budget(max_length, budget, 8192, 128)
                    } else {
                        budget
                    },
                    max_length,
                    8192,
                    packed_admission
                ),
                active_rows_at_16384_tokens = batching::estimated_active_rows_for_attention(
                    if packed_admission && option_env!("SYS1_CUDA_COMPUTE_CAP") == Some("90") {
                        batching::long_work_budget(max_length, budget, 16384, 128)
                    } else {
                        budget
                    },
                    max_length,
                    16384,
                    packed_admission
                ),
                "Qwen3.5 CUDA activation capacity estimate (not KV cache; rows also limited by server batch cap; singleton OOM remains possible)"
            );
            (bytes_per_token, budget)
        } else {
            (1, max_length)
        };
        #[cfg(not(feature = "cuda"))]
        let (batch_bytes_per_token, batch_token_budget) = (1, max_length);
        Ok((
            Self {
                context: crate::protocols::Context {
                    tokenizer,
                    vision_config: config.vision_config,
                    image_token_id: config.image_token_id,
                    video_token_id: config.video_token_id,
                    max_length,
                },
                causal,
                decoder,
                vision,
                device,
                dtype,
                batch_bytes_per_token,
                batch_token_budget: AtomicUsize::new(batch_token_budget),
            },
            head,
        ))
    }

    pub(crate) fn visual_features(
        &self,
        patches: &[f32],
        grids: &[[usize; 3]],
    ) -> Result<Tensor, ApiError> {
        let patch_width = self.vision_config.in_channels
            * self.vision_config.temporal_patch_size
            * self.vision_config.patch_size.pow(2);
        let patches = Tensor::from_vec(
            patches.to_vec(),
            (patches.len() / patch_width, patch_width),
            &self.device,
        )
        .and_then(|tensor| tensor.to_dtype(self.dtype))
        .map_err(|error| ApiError::internal(error.to_string()))?;
        self.vision
            .forward(&patches, grids)
            .map_err(|error| ApiError::internal(error.to_string()))
    }

    pub(crate) fn refresh_batch_token_budget(&self) -> usize {
        let current = self.batch_token_budget.load(Ordering::Relaxed);
        #[cfg(feature = "cuda")]
        if let Device::Cuda(cuda) = &self.device {
            match crate::kernels::cuda::memory::snapshot(cuda) {
                Ok((free_bytes, total_bytes)) => {
                    let (_, _, head_cache_growth_reserve_bytes) =
                        crate::kernels::cuda::guarded_dgemm::head_cache163_memory();
                    let free_bytes = free_bytes.saturating_sub(head_cache_growth_reserve_bytes);
                    let live = if self.packed_text_supported() {
                        batching::estimated_memory_tokens(free_bytes, self.batch_bytes_per_token)
                            .max(1)
                            .min(
                                self.max_length
                                    .saturating_mul(batching::aggregate_work_multiplier()),
                            )
                    } else {
                        batching::cuda_token_budget(
                            free_bytes,
                            self.batch_bytes_per_token,
                            self.max_length,
                        )
                    };
                    let selected = batching::refreshed_token_budget(current, live);
                    self.batch_token_budget
                        .fetch_min(selected, Ordering::Relaxed);
                    if selected < current {
                        tracing::info!(
                            free_bytes,
                            total_bytes,
                            previous_batch_token_budget = current,
                            batch_token_budget = selected,
                            "Qwen3.5 CUDA activation admission budget reduced after live memory refresh"
                        );
                    }
                    tracing::debug!(
                        free_bytes,
                        total_bytes,
                        bytes_per_token = self.batch_bytes_per_token,
                        previous_batch_token_budget = current,
                        batch_token_budget = selected,
                        "Qwen3.5 CUDA batch activation budget refreshed"
                    );
                    return selected;
                }
                Err(error) => tracing::warn!(
                    %error,
                    batch_token_budget = current,
                    "Failed to refresh Qwen3.5 CUDA memory; retaining conservative batch budget"
                ),
            }
        }
        current
    }

    pub(crate) fn record_batch_oom(&self, failed_padded_tokens: usize) -> usize {
        let current = self.batch_token_budget.load(Ordering::Relaxed);
        let reduced = batching::token_budget_after_oom(current, failed_padded_tokens);
        self.batch_token_budget
            .fetch_min(reduced, Ordering::Relaxed);
        reduced
    }
    pub(crate) fn packed_text_supported(&self) -> bool {
        #[cfg(feature = "cuda")]
        {
            self.device.is_cuda()
                && self.dtype == DType::BF16
                && matches!(
                    option_env!("SYS1_CUDA_COMPUTE_CAP"),
                    Some("80" | "86" | "89" | "90" | "120")
                )
                && self.decoder.supports_packed()
        }
        #[cfg(not(feature = "cuda"))]
        {
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn checkpoint_dtype_rejects_unsafe_fp16_conversion() {
        for source in ["bfloat16", "float32"] {
            let error =
                validate_declared_dtypes(Some(source), Some(source), Some(source), DType::F16)
                    .unwrap_err();
            assert!(
                error
                    .to_string()
                    .contains("Unsafe Qwen3.5 dtype conversion")
            );
        }
        for requested in [DType::BF16, DType::F32] {
            validate_declared_dtypes(
                Some("bfloat16"),
                Some("bfloat16"),
                Some("bfloat16"),
                requested,
            )
            .unwrap();
        }
        validate_declared_dtypes(
            Some("float16"),
            Some("float16"),
            Some("float16"),
            DType::F16,
        )
        .unwrap();
        validate_declared_dtypes(None, None, None, DType::F16).unwrap();
        assert!(
            validate_declared_dtypes(
                Some("bfloat16"),
                Some("float16"),
                Some("bfloat16"),
                DType::BF16,
            )
            .is_err()
        );
    }

    #[test]
    fn auto_attention_respects_dtype_and_cuda_capability() {
        assert_eq!(
            select_attention(AttentionImplementation::Auto, DType::F32, Some((8, 9))),
            AttentionImplementation::Eager
        );
        assert_eq!(
            select_attention(AttentionImplementation::Auto, DType::BF16, None),
            AttentionImplementation::Eager
        );
        #[cfg(feature = "cuda")]
        {
            assert_eq!(
                select_attention(AttentionImplementation::Auto, DType::BF16, Some((8, 9))),
                AttentionImplementation::FlashAttention2
            );
            assert_eq!(
                select_attention(AttentionImplementation::Auto, DType::BF16, Some((7, 5))),
                AttentionImplementation::Eager
            );
        }
        #[cfg(feature = "cuda")]
        {
            assert_eq!(
                select_attention(AttentionImplementation::Auto, DType::BF16, Some((12, 0))),
                AttentionImplementation::FlashAttentionBlackwell
            );
            assert_eq!(
                select_attention(AttentionImplementation::Auto, DType::BF16, Some((10, 0))),
                AttentionImplementation::Eager
            );
        }
    }
}

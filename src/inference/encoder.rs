#[cfg(feature = "metal")]
use crate::architectures::metal::ModernBertEncoder;
#[cfg(not(feature = "metal"))]
use crate::architectures::modernbert::Encoder as ModernBertEncoder;
use crate::models::AttentionImplementation;
use crate::models::DecisionModel;
use crate::schema::{ApiError, DecisionRequest, DecisionResponse};
use std::collections::HashMap;

use anyhow::Context;
use candle_core::{D, DType, Device, IndexOp, Tensor};
use candle_nn::{Embedding, LayerNorm, Linear};
use std::sync::atomic::{AtomicUsize, Ordering};

use super::encoder_batching as batching;
use crate::{
    architectures::heads::encoder::HeadLayer,
    protocols::ordinal::{Item, OrdinalProtocol, RequestItems},
};
#[derive(Eq, Hash, PartialEq)]
struct ItemKey<'a> {
    kind: usize,
    ids: &'a [u32],
    markers: &'a [usize],
}

pub(crate) struct TokenClassifier {
    pub(crate) protocol: OrdinalProtocol,
    pub(crate) activation_budget: AtomicUsize,
    pub(crate) activation_bytes_per_token: usize,
    pub(crate) attention_bytes_per_cell: usize,
    pub(crate) encoder: ModernBertEncoder,
    pub(crate) head: Vec<HeadLayer>,
    pub(crate) type_embedding: Embedding,
    pub(crate) scorer_norm: LayerNorm,
    pub(crate) scorer_in: Linear,
    pub(crate) scorer_out: Linear,
    pub(crate) device: Device,
    pub(crate) compute_dtype: DType,
}

impl TokenClassifier {
    pub(crate) fn forward(&self, prepared: &[RequestItems]) -> anyhow::Result<Vec<Vec<f32>>> {
        let items: Vec<_> = prepared.iter().flat_map(|request| &request.items).collect();
        let (unique, indices) = deduplicate(&items);
        if unique.is_empty() {
            return Ok(Vec::new());
        }
        let output = if self.device.is_cuda() {
            let lengths: Vec<_> = unique.iter().map(|item| item.ids.len()).collect();
            let groups = batching::buckets(
                &lengths,
                self.refresh_activation_budget(),
                self.activation_bytes_per_token,
                self.attention_bytes_per_cell,
            );
            let mut ordered = vec![None; unique.len()];
            for group in groups {
                let subset: Vec<_> = group.iter().map(|&slot| unique[slot]).collect();
                let results = self.forward_admitted(&subset)?;
                anyhow::ensure!(
                    results.len() == group.len(),
                    "Laya admitted result count differs"
                );
                for (slot, result) in group.into_iter().zip(results) {
                    ordered[slot] = Some(result);
                }
            }
            ordered
                .into_iter()
                .map(|row| row.context("Missing Laya admitted result"))
                .collect::<anyhow::Result<Vec<_>>>()?
        } else {
            self.forward_unique(&unique)?
        };
        Ok(indices
            .into_iter()
            .map(|index| output[index].clone())
            .collect())
    }

    pub(crate) fn refresh_activation_budget(&self) -> usize {
        #[cfg(feature = "cuda")]
        if let Device::Cuda(cuda) = &self.device
            && let Ok((free_bytes, total_bytes)) = cuda.cuda_stream().context().mem_get_info()
        {
            let live = free_bytes.saturating_sub(free_bytes / 4).max(1);
            let previous = self.activation_budget.fetch_min(live, Ordering::Relaxed);
            if live < previous {
                tracing::info!(
                    free_bytes,
                    total_bytes,
                    reserved_headroom_bytes = free_bytes / 4,
                    activation_budget_bytes = live,
                    activation_bytes_per_token = self.activation_bytes_per_token,
                    attention_bytes_per_cell = self.attention_bytes_per_cell,
                    estimated_linear_tokens = live / self.activation_bytes_per_token.max(1),
                    capacity_kind = "estimated activations, not KV cache; eager attention adds quadratic workspace",
                    "Laya CUDA admission budget updated"
                );
            }
        }
        self.activation_budget.load(Ordering::Relaxed)
    }

    fn forward_admitted(&self, items: &[&Item]) -> anyhow::Result<Vec<Vec<f32>>> {
        match self.forward_unique(items) {
            Err(error) if items.len() > 1 && batching::is_cuda_oom(&error) => {
                let length = items.iter().map(|item| item.ids.len()).max().unwrap_or(0);
                let estimate = batching::bytes(
                    items.len(),
                    length,
                    self.activation_bytes_per_token,
                    self.attention_bytes_per_cell,
                );
                self.activation_budget
                    .fetch_min((estimate / 2).max(1), Ordering::Relaxed);
                tracing::warn!(
                    rows = items.len(),
                    padded_length = length,
                    activation_budget_bytes = self.activation_budget.load(Ordering::Relaxed),
                    "Laya CUDA OOM: retrying smaller internal batches"
                );
                let (left, right) = items.split_at(items.len() / 2);
                let mut output = self.forward_admitted(left)?;
                output.extend(self.forward_admitted(right)?);
                Ok(output)
            }
            result => result,
        }
    }

    fn forward_unique(&self, unique: &[&Item]) -> anyhow::Result<Vec<Vec<f32>>> {
        let length = unique.iter().map(|item| item.ids.len()).max().unwrap_or(0);
        self.forward_unique_at_length(unique, length)
    }

    fn forward_unique_at_length(
        &self,
        unique: &[&Item],
        length: usize,
    ) -> anyhow::Result<Vec<Vec<f32>>> {
        anyhow::ensure!(
            unique.iter().all(|item| item.ids.len() <= length),
            "padded length is shorter than an input"
        );
        let batch = unique.len();
        let mut ids = vec![self.protocol.pad_id; batch * length];
        let mut mask = vec![0f32; batch * length];
        let mut kinds = Vec::with_capacity(batch);
        for (row, item) in unique.iter().enumerate() {
            let start = row * length;
            ids[start..start + item.ids.len()].copy_from_slice(&item.ids);
            mask[start..start + item.ids.len()].fill(1.0);
            kinds.push(item.question.kind as u32);
        }
        let ids = Tensor::from_vec(ids, (batch, length), &self.device)?;
        let head_mask: Vec<_> = mask
            .iter()
            .map(|value| {
                if *value == 0.0 {
                    f32::NEG_INFINITY
                } else {
                    0.0
                }
            })
            .collect();
        let attention_mask =
            Tensor::from_vec(mask, (batch, length), &self.device)?.to_dtype(self.compute_dtype)?;
        let head_mask = Tensor::from_vec(head_mask, (batch, 1, 1, length), &self.device)?
            .to_dtype(self.compute_dtype)?;
        let type_ids = Tensor::from_vec(kinds, batch, &self.device)?;
        let has_padding = unique.iter().any(|item| item.ids.len() != length);
        let lengths: Vec<_> = unique.iter().map(|item| item.ids.len()).collect();
        let mut hidden = self
            .encoder
            .forward(&ids, &attention_mask, &lengths, has_padding)
            .context("encoder forward")?;
        hidden = hidden.broadcast_add(&type_ids.apply(&self.type_embedding)?.unsqueeze(1)?)?;
        for (index, layer) in self.head.iter().enumerate() {
            hidden = layer
                .forward(&hidden, &head_mask, &lengths)
                .with_context(|| format!("decision head layer {index}"))?;
        }
        let output = if !self.device.is_cpu() {
            self.score_batched(&hidden, unique, length)?
        } else {
            self.score_rows(&hidden, unique)?
        };
        Ok(output)
    }

    fn score_rows(&self, hidden: &Tensor, items: &[&Item]) -> anyhow::Result<Vec<Vec<f32>>> {
        let mut output = Vec::with_capacity(items.len());
        for (row, item) in items.iter().enumerate() {
            let markers = Tensor::from_vec(
                item.markers
                    .iter()
                    .map(|value| *value as u32)
                    .collect::<Vec<_>>(),
                item.markers.len(),
                &self.device,
            )?;
            let selected = hidden.i(row)?.index_select(&markers, 0)?;
            output.push(self.score(&selected)?);
        }
        Ok(output)
    }

    fn score_batched(
        &self,
        hidden: &Tensor,
        items: &[&Item],
        length: usize,
    ) -> anyhow::Result<Vec<Vec<f32>>> {
        let hidden_size = hidden.dim(D::Minus1)?;
        let counts: Vec<_> = items.iter().map(|item| item.markers.len()).collect();
        let marker_count = counts.iter().sum();
        let mut markers = Vec::with_capacity(marker_count);
        for (row, item) in items.iter().enumerate() {
            markers.extend(
                item.markers
                    .iter()
                    .map(|position| (row * length + position) as u32),
            );
        }
        let markers = Tensor::from_vec(markers, marker_count, &self.device)?;
        let selected = hidden
            .reshape((items.len() * length, hidden_size))?
            .index_select(&markers, 0)?;
        let logits = self.score(&selected)?;
        let mut offset = 0;
        Ok(counts
            .into_iter()
            .map(|count| {
                let values = logits[offset..offset + count].to_vec();
                offset += count;
                values
            })
            .collect())
    }

    fn score(&self, selected: &Tensor) -> anyhow::Result<Vec<f32>> {
        let normalized = selected
            .apply(&self.scorer_norm)?
            .to_dtype(self.compute_dtype)?;
        let projected = crate::kernels::linear::forward(&normalized, &self.scorer_in)?;
        let intermediate = crate::kernels::activation::gelu_erf(&projected)?;
        Ok(
            crate::kernels::linear::forward(&intermediate, &self.scorer_out)?
                .squeeze(1)?
                .to_dtype(DType::F32)?
                .to_vec1::<f32>()?,
        )
    }
}
pub(crate) fn execution_dtypes(requested: DType, is_cuda: bool) -> (DType, DType) {
    let model = if requested == DType::F16 && is_cuda {
        DType::F32
    } else {
        requested
    };
    (model, requested)
}

pub(crate) fn validate_attention(
    attention: AttentionImplementation,
    _device: &Device,
    compute_dtype: DType,
) -> anyhow::Result<()> {
    attention.validate(compute_dtype)?;
    #[cfg(feature = "cuda")]
    match attention {
        AttentionImplementation::Auto | AttentionImplementation::Eager => {}
        implementation => {
            anyhow::ensure!(
                _device.is_cuda(),
                "{} requires a CUDA device",
                implementation.cli_name()
            );
            let (major, minor) = match _device {
                Device::Cuda(cuda) => cuda
                    .cuda_stream()
                    .context()
                    .compute_capability()
                    .context("failed to query CUDA compute capability")?,
                _ => unreachable!(),
            };
            validate_flash_capability(implementation, major, minor)?;
        }
    }
    Ok(())
}

#[cfg(any(feature = "cuda", test))]
pub(crate) fn validate_flash_capability(
    attention: AttentionImplementation,
    major: i32,
    minor: i32,
) -> anyhow::Result<()> {
    let supported = match attention {
        AttentionImplementation::Auto => true,
        AttentionImplementation::Eager => true,
        AttentionImplementation::FlashAttention2 => (8..=9).contains(&major),
        AttentionImplementation::FlashAttentionBlackwell => (major, minor) == (12, 0),
    };
    anyhow::ensure!(
        supported,
        "{} does not support CUDA compute capability {major}.{minor}",
        attention.cli_name()
    );
    Ok(())
}

fn deduplicate<'a>(items: &[&'a Item]) -> (Vec<&'a Item>, Vec<usize>) {
    let mut unique = Vec::with_capacity(items.len());
    let mut indices = Vec::with_capacity(items.len());
    let mut seen = HashMap::with_capacity(items.len());
    for item in items {
        let key = ItemKey {
            kind: item.question.kind,
            ids: &item.ids,
            markers: &item.markers,
        };
        let index = *seen.entry(key).or_insert_with(|| {
            let index = unique.len();
            unique.push(*item);
            index
        });
        indices.push(index);
    }
    (unique, indices)
}

impl DecisionModel for TokenClassifier {
    fn predict_batch(
        &self,
        requests: Vec<DecisionRequest>,
    ) -> Vec<Result<DecisionResponse, ApiError>> {
        let mut slots = Vec::with_capacity(requests.len());
        let mut prepared = Vec::new();
        for request in requests {
            match self.protocol.prepare(request) {
                Ok(request) => {
                    slots.push(Ok(prepared.len()));
                    prepared.push(request);
                }
                Err(error) => slots.push(Err(error)),
            }
        }
        let outputs = match self.forward(&prepared) {
            Ok(outputs) => outputs,
            Err(error) => {
                let message = format!("{error:#}");
                return slots
                    .into_iter()
                    .map(|slot| slot.and_then(|_| Err(ApiError::new(message.clone()))))
                    .collect();
            }
        };
        let mut outputs = outputs.into_iter();
        let responses: Vec<_> = prepared
            .into_iter()
            .map(|request| Ok(self.protocol.response(request, &mut outputs)))
            .collect();
        slots
            .into_iter()
            .map(|slot| slot.and_then(|index| responses[index].clone()))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocols::ordinal::Question;
    use serde_json::Value;
    fn item(id: &str, kind: usize, ids: &[u32]) -> Item {
        Item {
            question: Question {
                id: id.to_owned(),
                kind,
                instructions: String::new(),
                criteria: Value::Null,
                labels: Vec::new(),
                options: Vec::new(),
            },
            ids: ids.to_vec(),
            markers: vec![1],
        }
    }

    #[test]
    fn deduplicates_identical_model_inputs_without_losing_order() {
        let first = item("first", 0, &[1, 2, 3]);
        let duplicate = item("duplicate", 0, &[1, 2, 3]);
        let other_kind = item("other-kind", 1, &[1, 2, 3]);
        let items = vec![&first, &duplicate, &other_kind, &first];

        let (unique, indices) = deduplicate(&items);

        assert_eq!(unique.len(), 2);
        assert_eq!(indices, vec![0, 0, 1, 0]);
    }

    #[test]
    fn cuda_f16_uses_autocast_style_storage_and_compute_dtypes() {
        assert_eq!(execution_dtypes(DType::F16, true), (DType::F32, DType::F16));
        assert_eq!(
            execution_dtypes(DType::F16, false),
            (DType::F16, DType::F16)
        );
        assert_eq!(execution_dtypes(DType::F32, true), (DType::F32, DType::F32));
    }

    #[test]
    fn rejects_flash_attention_without_a_compatible_backend() {
        assert!(
            validate_attention(
                AttentionImplementation::FlashAttention2,
                &Device::Cpu,
                DType::BF16,
            )
            .is_err()
        );
        validate_attention(AttentionImplementation::Eager, &Device::Cpu, DType::F32).unwrap();
    }

    #[test]
    fn validates_flash_attention_compute_capabilities() {
        validate_flash_capability(AttentionImplementation::FlashAttention2, 8, 9).unwrap();
        validate_flash_capability(AttentionImplementation::FlashAttention2, 9, 0).unwrap();

        assert!(validate_flash_capability(AttentionImplementation::FlashAttention2, 7, 5).is_err());
        assert!(
            validate_flash_capability(AttentionImplementation::FlashAttention2, 10, 0).is_err()
        );
        validate_flash_capability(AttentionImplementation::FlashAttentionBlackwell, 12, 0).unwrap();
        assert!(
            validate_flash_capability(AttentionImplementation::FlashAttentionBlackwell, 10, 0)
                .is_err()
        );
    }
}

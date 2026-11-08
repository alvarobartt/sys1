use crate::{
    architectures::qwen35_text::{batching, inference::Backbone},
    models::DecisionModel,
    protocols::Encoded,
    schema::{ApiError, DecisionRequest, DecisionResponse},
};
use candle_core::Tensor;

pub(crate) trait Adapter: Send + Sync {
    fn encode(&self, backbone: &Backbone, request: &DecisionRequest) -> Result<Encoded, ApiError>;
    fn score(
        &self,
        encoded: &Encoded,
        ids: &Tensor,
        hidden: &Tensor,
    ) -> candle_core::Result<Vec<Vec<f32>>>;
    fn response(
        &self,
        encoded: &Encoded,
        logits: Vec<Vec<f32>>,
    ) -> Result<DecisionResponse, ApiError>;
    fn supports_videos(&self) -> bool {
        false
    }
    fn needs_confirmation(&self, _response: &DecisionResponse) -> bool {
        false
    }
    fn confirm_batches(&self) -> bool {
        true
    }
    #[cfg(feature = "cuda")]
    fn score_packed(
        &self,
        encoded: &[&Encoded],
        hidden: &Tensor,
    ) -> candle_core::Result<Vec<Vec<Vec<f32>>>> {
        let mut offset = 0;
        encoded
            .iter()
            .map(|item| {
                let length = item.ids.len();
                let ids = Tensor::from_vec(item.ids.clone(), length, hidden.device())?;
                let features = hidden.squeeze(0)?.narrow(0, offset, length)?;
                offset += length;
                self.score(item, &ids, &features)
            })
            .collect()
    }
}

pub(crate) struct Qwen35<A> {
    backbone: Backbone,
    adapter: A,
}
impl<A: Adapter> Qwen35<A> {
    pub(crate) fn new(backbone: Backbone, adapter: A) -> Self {
        Self { backbone, adapter }
    }
    fn encode(&self, request: &DecisionRequest) -> Result<Encoded, ApiError> {
        self.adapter.encode(&self.backbone, request)
    }
    fn response_logits(
        &self,
        encoded: &Encoded,
        logits: Vec<Vec<f32>>,
    ) -> Result<DecisionResponse, ApiError> {
        self.adapter.response(encoded, logits)
    }
    fn predict_one(&self, request: DecisionRequest) -> Result<DecisionResponse, ApiError> {
        self.predict_encoded(&self.encode(&request)?)
    }
    fn predict_encoded(&self, encoded: &Encoded) -> Result<DecisionResponse, ApiError> {
        let ids = Tensor::from_vec(
            encoded.ids.clone(),
            (1, encoded.ids.len()),
            &self.backbone.device,
        )
        .map_err(|error| ApiError::internal(error.to_string()))?;

        let hidden = if encoded.images.is_some() || encoded.videos.is_some() {
            let image_features = encoded
                .images
                .as_ref()
                .map(|images| {
                    self.backbone
                        .visual_features(&images.patches, &images.grids)
                })
                .transpose()?;
            let video_features = encoded
                .videos
                .as_ref()
                .map(|videos| {
                    self.backbone
                        .visual_features(&videos.patches, &videos.grids)
                })
                .transpose()?;
            let embeddings = self
                .backbone
                .decoder
                .embed(&ids)
                .and_then(|tensor| tensor.squeeze(0))
                .map_err(|error| ApiError::internal(error.to_string()))?;
            let mut chunks = Vec::new();
            let mut token_offset = 0;
            let mut image_offset = 0;
            let mut video_offset = 0;
            for &(start, count, is_video) in &encoded.media_spans {
                if start > token_offset {
                    chunks.push(
                        embeddings
                            .narrow(0, token_offset, start - token_offset)
                            .map_err(|error| ApiError::internal(error.to_string()))?,
                    );
                }
                let (features, offset) = if is_video {
                    (video_features.as_ref().unwrap(), &mut video_offset)
                } else {
                    (image_features.as_ref().unwrap(), &mut image_offset)
                };
                chunks.push(
                    features
                        .narrow(0, *offset, count)
                        .map_err(|error| ApiError::internal(error.to_string()))?,
                );
                token_offset = start + count;
                *offset += count;
            }
            if token_offset < encoded.ids.len() {
                chunks.push(
                    embeddings
                        .narrow(0, token_offset, encoded.ids.len() - token_offset)
                        .map_err(|error| ApiError::internal(error.to_string()))?,
                );
            }
            let embeddings = Tensor::cat(&chunks.iter().collect::<Vec<_>>(), 0)
                .and_then(|tensor| tensor.unsqueeze(0))
                .map_err(|error| ApiError::internal(error.to_string()))?;
            self.backbone
                .decoder
                .forward_embeds(&embeddings, &encoded.positions)
                .map_err(|error| ApiError::internal(error.to_string()))?
        } else {
            self.backbone
                .decoder
                .forward(&ids)
                .map_err(|error| ApiError::internal(error.to_string()))?
        };
        let ids = ids
            .squeeze(0)
            .map_err(|error| ApiError::internal(error.to_string()))?;
        self.response(encoded, &ids, &hidden)
    }

    fn response(
        &self,
        encoded: &Encoded,
        ids: &Tensor,
        hidden: &Tensor,
    ) -> Result<DecisionResponse, ApiError> {
        let logits = self
            .adapter
            .score(encoded, ids, hidden)
            .map_err(|error| ApiError::internal(error.to_string()))?;
        self.response_logits(encoded, logits)
    }

    #[cfg(feature = "cuda")]
    fn predict_packed_text_bucket(
        &self,
        encoded: &[&Encoded],
    ) -> Vec<Result<DecisionResponse, ApiError>> {
        let run = || -> candle_core::Result<Vec<Vec<Vec<f32>>>> {
            let lengths = encoded.iter().map(|r| r.ids.len()).collect::<Vec<_>>();
            let flat = encoded
                .iter()
                .flat_map(|r| r.ids.iter().copied())
                .collect::<Vec<_>>();
            let total = flat.len();
            let ids = Tensor::from_vec(flat, (1, total), &self.backbone.device)?;
            let hidden = self.backbone.decoder.forward_packed(&ids, &lengths)?;
            self.adapter.score_packed(encoded, &hidden)
        };
        match run() {
            Ok(logits) => encoded
                .iter()
                .zip(logits)
                .map(|(item, scores)| self.response_logits(item, scores))
                .collect(),
            Err(error) => encoded
                .iter()
                .map(|_| Err(ApiError::internal(error.to_string())))
                .collect(),
        }
    }

    fn predict_text_bucket(&self, encoded: &[&Encoded]) -> Vec<Result<DecisionResponse, ApiError>> {
        if !self.backbone.causal && !self.backbone.packed_text_supported() && encoded.len() > 1 {
            return encoded
                .iter()
                .map(|item| self.predict_text_bucket(&[item]).remove(0))
                .collect();
        }
        #[cfg(feature = "cuda")]
        if !encoded.is_empty() && self.backbone.packed_text_supported() {
            return self.predict_packed_text_bucket(encoded);
        }

        let length = encoded.iter().map(|item| item.ids.len()).max().unwrap_or(0);
        let mut flattened = Vec::with_capacity(encoded.len() * length);
        for item in encoded {
            flattened.extend_from_slice(&item.ids);
            flattened.resize(flattened.len() + length - item.ids.len(), 0);
        }
        let ids = match Tensor::from_vec(flattened, (encoded.len(), length), &self.backbone.device)
        {
            Ok(ids) => ids,
            Err(error) => {
                return (0..encoded.len())
                    .map(|_| Err(ApiError::internal(error.to_string())))
                    .collect();
            }
        };
        let hidden = match self.backbone.decoder.forward(&ids) {
            Ok(hidden) => hidden,
            Err(error) => {
                return (0..encoded.len())
                    .map(|_| Err(ApiError::internal(error.to_string())))
                    .collect();
            }
        };
        encoded
            .iter()
            .enumerate()
            .map(|(batch_index, item)| {
                let item_hidden = hidden
                    .narrow(0, batch_index, 1)
                    .and_then(|value| value.narrow(1, 0, item.ids.len()))
                    .and_then(|value| value.squeeze(0))
                    .map_err(|error| ApiError::internal(error.to_string()))?;
                let item_ids =
                    Tensor::from_vec(item.ids.clone(), item.ids.len(), &self.backbone.device)
                        .map_err(|error| ApiError::internal(error.to_string()))?;
                self.response(item, &item_ids, &item_hidden)
            })
            .collect()
    }
}
impl<A: Adapter> DecisionModel for Qwen35<A> {
    fn supports_images(&self) -> bool {
        true
    }

    fn supports_videos(&self) -> bool {
        self.adapter.supports_videos()
    }

    fn predict_batch(
        &self,
        requests: Vec<DecisionRequest>,
    ) -> Vec<Result<DecisionResponse, ApiError>> {
        let mut results: Vec<Option<Result<DecisionResponse, ApiError>>> =
            (0..requests.len()).map(|_| None).collect();
        self.predict_batch_incremental(requests, &mut |slot, result| {
            results[slot] = Some(result);
        });
        results
            .into_iter()
            .map(|result| {
                result.unwrap_or_else(|| Err(ApiError::internal("Qwen3.5 batch result missing")))
            })
            .collect()
    }

    fn predict_batch_incremental(
        &self,
        requests: Vec<DecisionRequest>,
        complete: &mut dyn FnMut(usize, Result<DecisionResponse, ApiError>),
    ) {
        if requests.len() <= 1 || !self.backbone.device.is_cuda() {
            for (slot, request) in requests.into_iter().enumerate() {
                complete(slot, self.predict_one(request));
            }
            return;
        }

        let use_parallel = option_env!("SYS1_CUDA_COMPUTE_CAP") == Some("90")
            && requests.len() >= 64
            && requests
                .iter()
                .all(|r| r.images.is_empty() && r.videos.is_empty());
        let mut prepared = Vec::new();
        if use_parallel {
            let available = std::thread::available_parallelism()
                .map(usize::from)
                .unwrap_or(1);
            let count = super::preparation::workers(requests.len(), available);
            for (slot, item) in super::preparation::map(&requests, count, |r| self.encode(r))
                .into_iter()
                .enumerate()
            {
                match item {
                    Ok(encoded) => prepared.push((slot, encoded)),
                    Err(error) => {
                        complete(slot, Err(error));
                    }
                }
            }
            drop(requests);
        } else {
            for (slot, request) in requests.into_iter().enumerate() {
                if !request.images.is_empty() || !request.videos.is_empty() {
                    complete(slot, self.predict_one(request));
                    continue;
                }
                match self.encode(&request) {
                    Ok(encoded) => prepared.push((slot, encoded)),
                    Err(error) => {
                        complete(slot, Err(error));
                    }
                }
            }
        }

        let lengths: Vec<_> = prepared
            .iter()
            .enumerate()
            .map(|(index, (_, encoded))| (index, encoded.ids.len()))
            .collect();
        let mut remaining = lengths;
        while !remaining.is_empty() {
            let batch_token_budget = self.backbone.refresh_batch_token_budget();
            let planned = if self.backbone.packed_text_supported() {
                batching::packed_buckets(
                    &remaining,
                    self.backbone.context.max_length,
                    batch_token_budget,
                )
            } else {
                batching::buckets(
                    &remaining,
                    self.backbone.context.max_length,
                    batch_token_budget,
                )
            };
            let bucket = planned
                .into_iter()
                .next()
                .expect("non-empty lengths must produce a bucket");
            remaining.retain(|(index, _)| !bucket.contains(index));
            let inputs: Vec<_> = bucket.iter().map(|&index| &prepared[index].1).collect();
            let mut responses = self.predict_text_bucket(&inputs);
            let failed_padded_tokens = if self.backbone.packed_text_supported() {
                inputs.iter().map(|r| r.ids.len()).sum()
            } else {
                inputs
                    .iter()
                    .map(|r| r.ids.len())
                    .max()
                    .unwrap_or(0)
                    .saturating_mul(inputs.len())
            };
            let batch_oom = responses
                .iter()
                .any(|response| response.as_ref().is_err_and(is_cuda_out_of_memory));
            let reduced_budget =
                batch_oom.then(|| self.backbone.record_batch_oom(failed_padded_tokens));
            for (position, response) in responses.iter_mut().enumerate() {
                let out_of_memory =
                    inputs.len() > 1 && response.as_ref().is_err_and(is_cuda_out_of_memory);
                if out_of_memory {
                    tracing::warn!(
                        batch_size = inputs.len(),
                        input_tokens = inputs[position].ids.len(),
                        failed_padded_tokens,
                        reduced_batch_token_budget = reduced_budget.unwrap(),
                        "Qwen3.5 CUDA batch exhausted memory; retrying request as a singleton"
                    );
                    *response = self
                        .predict_text_bucket(&[inputs[position]])
                        .into_iter()
                        .next()
                        .unwrap();
                } else if inputs.len() > 1
                    && !(self.backbone.packed_text_supported() && !self.adapter.confirm_batches())
                    && response
                        .as_ref()
                        .is_ok_and(|r| self.adapter.needs_confirmation(r))
                {
                    tracing::debug!(
                        batch_size = inputs.len(),
                        input_tokens = inputs[position].ids.len(),
                        "Qwen3.5 confirming numerically ambiguous batched decision as a singleton"
                    );
                    *response = self
                        .predict_text_bucket(&[inputs[position]])
                        .into_iter()
                        .next()
                        .unwrap();
                }
            }
            for (index, response) in bucket.into_iter().zip(responses) {
                complete(prepared[index].0, response);
            }
        }
    }
}

fn is_cuda_out_of_memory(error: &ApiError) -> bool {
    error.error.contains("CUDA_ERROR_OUT_OF_MEMORY")
        || error.error.to_ascii_lowercase().contains("out of memory")
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn cuda_oom_fallback_only_matches_memory_exhaustion() {
        assert!(is_cuda_out_of_memory(&ApiError::internal(
            "DriverError(CUDA_ERROR_OUT_OF_MEMORY, \"out of memory\")"
        )));
        assert!(!is_cuda_out_of_memory(&ApiError::internal(
            "CUDA invalid argument"
        )));
    }
}

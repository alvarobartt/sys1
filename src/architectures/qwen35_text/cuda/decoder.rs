use super::*;

impl Layer {
    pub(in super::super) fn forward_deferred(
        &self,
        residual: Tensor,
        update: Option<Tensor>,
        rotary: Option<&RotaryTensors>,
        sequences: Option<&PackedSequences>,
    ) -> Result<(Tensor, Tensor)> {
        let x = &residual;
        let profile = layer_profiling_enabled();
        if profile {
            x.device().synchronize()?;
        }
        profiling::mark(x.device(), self.index, "start")?;
        let started = Instant::now();
        let had_update = update.is_some();
        let (x, normalized) = match update {
            Some(mlp) => self.input_norm.forward_residual(x, &mlp)?,
            None => (x.clone(), self.input_norm.forward(x)?),
        };

        drop(residual);
        if had_update && self.index > 0 {
            crate::numerics::capture(&format!("layer-{:02}-output", self.index - 1), &x)?;
        }
        crate::numerics::capture(&format!("layer-{:02}-normalized", self.index), &normalized)?;
        profiling::mark(x.device(), self.index, "norm")?;
        if profile {
            x.device().synchronize()?;
        }
        let norm_ms = started.elapsed().as_secs_f64() * 1_000.;
        let started = Instant::now();
        let mixed = match (&self.mixer, sequences) {
            (Mixer::Full(layer), Some(sequences)) => layer.forward_packed(
                &normalized,
                rotary.ok_or_else(|| {
                    candle_core::Error::Msg("Missing Qwen3.5 rotary tensors".into())
                })?,
                self.index,
                sequences,
            )?,
            (Mixer::Full(layer), None) => layer.forward(
                &normalized,
                rotary.ok_or_else(|| {
                    candle_core::Error::Msg("Missing Qwen3.5 rotary tensors".into())
                })?,
                self.index,
            )?,
            (Mixer::Linear(layer), Some(sequences)) => {
                layer.forward_packed(&normalized, self.index, sequences)?
            }
            (Mixer::Linear(layer), None) => layer.forward(&normalized, self.index)?,
        };
        crate::numerics::capture(&format!("layer-{:02}-mixed", self.index), &mixed)?;
        profiling::mark(x.device(), self.index, "mixer_tail")?;
        if profile {
            x.device().synchronize()?;
        }
        let mixer_ms = started.elapsed().as_secs_f64() * 1_000.;
        let started = Instant::now();
        let (x, post_normalized) = self.post_norm.forward_residual(&x, &mixed)?;
        profiling::mark(x.device(), self.index, "residual_and_post_norm")?;
        crate::numerics::capture(
            &format!("layer-{:02}-post-normalized", self.index),
            &post_normalized,
        )?;
        let mlp_output = self.mlp.forward(&post_normalized, self.index)?;
        crate::numerics::capture(&format!("layer-{:02}-mlp-output", self.index), &mlp_output)?;

        profiling::mark(x.device(), self.index, "mlp_residual_deferred")?;
        if profile {
            x.device().synchronize()?;
            tracing::info!(
                layer = self.index,
                mixer = match &self.mixer {
                    Mixer::Full(_) => "full",
                    Mixer::Linear(_) => "linear",
                },
                norm_ms,
                mixer_ms,
                mlp_ms = started.elapsed().as_secs_f64() * 1_000.,
                "Qwen3.5 layer profile"
            );
        }
        Ok((x, mlp_output))
    }
}

impl Decoder {
    pub(in super::super) fn forward_deferred(
        &self,
        embeds: &Tensor,
        rotary: Option<&RotaryTensors>,
        sequences: Option<&PackedSequences>,
    ) -> Result<Tensor> {
        let mut residual = embeds.clone();
        let mut update = None;
        for layer in &self.layers {
            let (next, mlp) = layer.forward_deferred(residual, update, rotary, sequences)?;
            residual = next;
            update = Some(mlp);
        }
        match update {
            Some(mlp) => {
                let (hidden, output) = self.norm.forward_residual(&residual, &mlp)?;
                if let Some(last) = self.layers.last() {
                    crate::numerics::capture(&format!("layer-{:02}-output", last.index), &hidden)?;
                }
                Ok(output)
            }
            None => self.norm.forward(&residual),
        }
    }
}

impl Decoder {
    pub(crate) fn prepare_nn_weights_base(
        &self,
        target_tokens: usize,
        bytes_per_token: usize,
    ) -> Result<usize> {
        if !matches!(option_env!("SYS1_CUDA_COMPUTE_CAP"), Some("80" | "120")) {
            return Ok(0);
        }
        let mut candidates: Vec<&Linear> = Vec::new();
        #[cfg(sys1_cuda_fused_ffn)]
        let mut fused_mlps: Vec<&Mlp> = Vec::new();
        #[cfg(not(sys1_cuda_fused_ffn))]
        let fused_mlps: Vec<&Mlp> = Vec::new();
        for layer in &self.layers {
            if option_env!("SYS1_CUDA_COMPUTE_CAP") == Some("80") {
                #[cfg(sys1_cuda_fused_ffn)]
                if layer.mlp.fused_geometry_available() && layer.mlp.cached_fused()?.is_none() {
                    fused_mlps.push(&layer.mlp);
                }
                continue;
            }
            #[cfg(sys1_cuda_fused_ffn)]
            let fused = layer.mlp.fused_geometry_available();
            #[cfg(not(sys1_cuda_fused_ffn))]
            let fused = false;
            if fused {
                #[cfg(sys1_cuda_fused_ffn)]
                if layer.mlp.cached_fused()?.is_none() {
                    fused_mlps.push(&layer.mlp);
                }
                candidates.push(&layer.mlp.down);
            } else {
                candidates.extend([&layer.mlp.gate, &layer.mlp.up, &layer.mlp.down]);
            }
            match &layer.mixer {
                Mixer::Linear(mixer) => candidates.extend([&mixer.qkv, &mixer.z, &mixer.output]),
                Mixer::Full(mixer) => candidates.extend([&mixer.q, &mixer.output]),
            }
        }
        let mut weights = Vec::new();
        let mut extra = 0usize;
        let mut fused_bytes = 0usize;
        for linear in candidates {
            if linear.bias().is_some() || linear.weight().dtype() != DType::BF16 {
                continue;
            }
            let (columns, inner) = linear.weight().dims2()?;
            if !nn_geometry_available(inner, columns) || linear.cached_nn()?.is_some() {
                continue;
            }
            extra = extra
                .checked_add(linear.weight().elem_count().checked_mul(2).ok_or_else(|| {
                    candle_core::Error::Msg("NN weight byte count overflow".into())
                })?)
                .ok_or_else(|| {
                    candle_core::Error::Msg("NN total weight byte count overflow".into())
                })?;
            weights.push(linear);
        }
        for mlp in &fused_mlps {
            fused_bytes =
                fused_bytes
                    .checked_add(mlp.gate.weight().elem_count().checked_mul(4).ok_or_else(
                        || candle_core::Error::Msg("paired MLP weight bytes overflow".into()),
                    )?)
                    .ok_or_else(|| {
                        candle_core::Error::Msg("paired total weight bytes overflow".into())
                    })?;
        }
        extra = extra
            .checked_add(fused_bytes)
            .ok_or_else(|| candle_core::Error::Msg("optional layout bytes overflow".into()))?;
        if weights.is_empty() && fused_mlps.is_empty() {
            return Ok(0);
        }
        let reference = weights
            .first()
            .map(|linear| linear.weight())
            .or_else(|| fused_mlps.first().map(|mlp| mlp.gate.weight()))
            .ok_or_else(|| candle_core::Error::Msg("missing layout reference".into()))?;
        let candle_core::Device::Cuda(cuda) = reference.device() else {
            return Ok(0);
        };
        let (free, _) = cuda
            .cuda_stream()
            .context()
            .mem_get_info()
            .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
        if !nn_capacity_preserved(free, extra, bytes_per_token, target_tokens) {
            tracing::info!(
                free_bytes = free,
                extra_weight_bytes = extra,
                fused_mlp_weight_bytes = fused_bytes,
                target_tokens,
                "Qwen optional physical layouts skipped to preserve activation capacity"
            );
            return Ok(0);
        }
        #[cfg(sys1_cuda_fused_ffn)]
        for mlp in fused_mlps {
            let _ = mlp.prepared_fused()?;
        }
        for linear in weights {
            let _ = linear.prepared_nn()?;
        }
        tracing::info!(
            extra_weight_bytes = extra,
            fused_mlp_weight_bytes = fused_bytes,
            target_tokens,
            "Qwen optional physical layouts prepared without reducing token capacity"
        );
        Ok(extra)
    }
}

impl Decoder {
    pub(crate) fn prepare_nn_weights(
        &self,
        target_tokens: usize,
        bytes_per_token: usize,
    ) -> Result<usize> {
        let base = self.prepare_nn_weights_base(target_tokens, bytes_per_token)?;
        #[cfg(sys1_cuda_fused_ffn)]
        {
            if option_env!("SYS1_CUDA_COMPUTE_CAP") != Some("120") {
                return Ok(base);
            }
            let mut candidates = Vec::new();
            let mut extra = 0usize;
            for layer in &self.layers {
                if let Mixer::Linear(mixer) = &layer.mixer {
                    if !mixer.qkvz_available() {
                        continue;
                    }
                    if mixer.qkv.cached_nn()?.is_none() || mixer.z.cached_nn()?.is_none() {
                        return Ok(base);
                    }
                    if mixer
                        .qkvz_weight
                        .lock()
                        .map_err(|_| {
                            candle_core::Error::Msg("poisoned model-owned QKV/Z weight".into())
                        })?
                        .is_some()
                    {
                        continue;
                    }
                    let bytes = mixer
                        .qkv
                        .weight()
                        .elem_count()
                        .checked_add(mixer.z.weight().elem_count())
                        .and_then(|n| n.checked_mul(2))
                        .ok_or_else(|| {
                            candle_core::Error::Msg("QKV/Z packed bytes overflow".into())
                        })?;
                    extra = extra.checked_add(bytes).ok_or_else(|| {
                        candle_core::Error::Msg("QKV/Z total bytes overflow".into())
                    })?;
                    candidates.push(mixer);
                }
            }
            let Some(first) = candidates.first() else {
                return Ok(base);
            };
            let candle_core::Device::Cuda(cuda) = first.qkv.weight().device() else {
                return Ok(base);
            };
            let (free, _) = cuda
                .cuda_stream()
                .context()
                .mem_get_info()
                .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
            if !nn_capacity_preserved(free, extra, bytes_per_token, target_tokens) {
                tracing::info!(
                    free_bytes = free,
                    extra_qkvz_weight_bytes = extra,
                    target_tokens,
                    "Optional direct QKV/Z layout skipped; parent layouts and token capacity retained"
                );
                return Ok(base);
            }
            for mixer in candidates {
                let packed = crate::kernels::cuda::fused_qkvz::prepare(
                    mixer.qkv.weight(),
                    mixer.z.weight(),
                )?;
                *mixer.qkvz_weight.lock().map_err(|_| {
                    candle_core::Error::Msg("poisoned model-owned QKV/Z weight".into())
                })? = Some(packed);
            }
            let total = base.checked_add(extra).ok_or_else(|| {
                candle_core::Error::Msg("total optional weight bytes overflow".into())
            })?;
            tracing::info!(
                parent_new_weight_bytes = base,
                extra_qkvz_weight_bytes = extra,
                total_new_weight_bytes = total,
                target_tokens,
                "Direct QKV/Z layouts prepared in addition to all original layouts"
            );
            Ok(total)
        }
        #[cfg(not(sys1_cuda_fused_ffn))]
        {
            Ok(base)
        }
    }
}

impl Decoder {
    pub(crate) fn supports_packed(&self) -> bool {
        self.layers.iter().all(|layer| match &layer.mixer {
            Mixer::Full(full) => matches!(
                full.implementation,
                AttentionImplementation::FlashAttention2
                    | AttentionImplementation::FlashAttentionBlackwell
            ),
            Mixer::Linear(linear) => linear.key_dim == 128 && linear.value_dim == 128,
        })
    }

    pub(crate) fn forward_packed(&self, ids: &Tensor, lengths: &[usize]) -> Result<Tensor> {
        if !self.supports_packed() || crate::numerics::enabled() {
            candle_core::bail!(
                "packed decoder requires supported Flash attention outside capture mode"
            )
        }
        let sequences = PackedSequences::new(lengths, ids.device())?;
        if ids.dims() != [1, sequences.total] {
            candle_core::bail!("packed ID shape mismatch")
        }
        let positions = lengths
            .iter()
            .flat_map(|&length| (0..length).map(|i| [i as u32; 3]))
            .collect::<Vec<_>>();
        let embeds = self.embed(ids)?;
        if embeds.dtype() != DType::BF16 {
            candle_core::bail!("packed decoder requires BF16")
        }
        let rotary = self
            .layers
            .iter()
            .find_map(|layer| match &layer.mixer {
                Mixer::Full(full) => Some(&full.rotary),
                Mixer::Linear(_) => None,
            })
            .map(|rotary| rotary.prepare(&positions, embeds.dtype(), embeds.device()))
            .transpose()?;
        profiling::begin(embeds.device())?;
        let output = self.forward_deferred(&embeds, rotary.as_ref(), Some(&sequences))?;
        profiling::finish(1, sequences.total)?;
        Ok(output)
    }
}

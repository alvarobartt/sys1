use super::*;

impl LinearAttention {
    pub(crate) fn forward_packed(
        &self,
        x: &Tensor,
        capture_index: usize,
        sequences: &PackedSequences,
    ) -> Result<Tensor> {
        let mut profile = if layer_profiling_enabled() {
            x.device().synchronize()?;
            Some(Instant::now())
        } else {
            None
        };
        if !x.device().is_cuda()
            || x.dtype() != DType::BF16
            || self.key_dim != 128
            || crate::numerics::enabled()
        {
            candle_core::bail!("unsupported packed recurrent architecture/capture mode")
        }
        let (batch, length, _) = x.dims3()?;
        let key_width = self.key_heads * self.key_dim;
        let value_width = self.value_heads * self.value_dim;
        let (projected, early_z) = self.project_qkvz(x)?;
        let mut prepared_qv = None;
        crate::numerics::layer_capture(capture_index, "linear-projected", &projected)?;
        linear_profile_checkpoint(x, capture_index, "qkv_projection", &mut profile)?;

        if matches!(option_env!("SYS1_CUDA_COMPUTE_CAP"), Some("80" | "90"))
            && self.value_dim == 128
            && let Some(ticket) = prepared_convolution::prepare(
                &projected,
                &self.conv,
                self.key_heads,
                self.value_heads,
                sequences,
            )?
        {
            linear_profile_checkpoint(x, capture_index, "convolution_qkv_fused", &mut profile)?;
            let (g, beta) = self.reference_gates_packed_tensor16(x, capture_index, sequences)?;
            linear_profile_checkpoint(x, capture_index, "qk_and_gates", &mut profile)?;
            if let Some(output) = gated_delta::try_forward_prepared_source(
                &projected,
                &self.conv,
                &g,
                &beta,
                &sequences.lengths,
                ticket,
            )? {
                crate::numerics::layer_capture(capture_index, "linear-delta", &output)?;
                linear_profile_checkpoint(x, capture_index, "delta_rule", &mut profile)?;
                let result = self.finish_delta(
                    x,
                    output,
                    batch,
                    length,
                    value_width,
                    capture_index,
                    early_z,
                )?;
                linear_profile_checkpoint(
                    x,
                    capture_index,
                    "finalize_and_projection",
                    &mut profile,
                )?;
                return Ok(result);
            }
        }

        let (q, k, v) = {
            if x.device().is_cuda()
                && x.dtype() == DType::BF16
                && matches!(
                    option_env!("SYS1_CUDA_COMPUTE_CAP"),
                    Some("89" | "90" | "120")
                )
                && batch.saturating_mul(length) >= 512
                && self.key_dim == 128
                && self.value_dim == 128
            {
                #[cfg(test)]
                let vstorage_enabled = VSTORAGE_MODEL_REPLAY.with(std::cell::Cell::get);
                #[cfg(not(test))]
                let vstorage_enabled = true;
                let values = if option_env!("SYS1_CUDA_COMPUTE_CAP") == Some("90") {
                    let (values, prepared) = cuda_conv_qkv_prepared(
                        &projected,
                        &self.conv,
                        self.key_heads,
                        self.value_heads,
                        sequences,
                    )?;
                    prepared_qv = prepared;
                    values
                } else if option_env!("SYS1_CUDA_COMPUTE_CAP") == Some("120") && vstorage_enabled {
                    #[cfg(test)]
                    VSTORAGE_MODEL_CALLS.with(|slot| slot.set(slot.get() + 1));
                    cuda_conv_vstore(
                        &projected,
                        &self.conv,
                        self.key_heads,
                        self.value_heads,
                        Some(sequences),
                    )?
                } else {
                    cuda_conv_qkv(
                        &projected,
                        &self.conv,
                        self.key_heads,
                        self.value_heads,
                        Some(sequences),
                    )?
                };

                if crate::numerics::enabled() {
                    let mixed = cuda_conv_silu_packed_sequences(&projected, &self.conv, sequences)?;
                    crate::numerics::layer_capture(capture_index, "linear-conv", &mixed)?;
                }
                linear_profile_checkpoint(x, capture_index, "convolution_qkv_fused", &mut profile)?;
                values
            } else {
                let mixed = if x.device().is_cuda() && matches!(x.dtype(), DType::BF16 | DType::F32)
                {
                    cuda_conv_silu_packed_sequences(&projected, &self.conv, sequences)?
                } else {
                    causal_conv_silu(&projected, &self.conv, self.kernel)?
                };
                #[cfg(feature = "metal")]
                let mixed = if x.device().is_metal()
                    && matches!(x.dtype(), DType::BF16 | DType::F16 | DType::F32)
                {
                    crate::kernels::metal::conv_silu(&projected, &self.conv)?
                } else {
                    causal_conv_silu(&projected, &self.conv, self.kernel)?
                };
                #[cfg(not(any(feature = "cuda", feature = "metal")))]
                let mixed = causal_conv_silu(&projected, &self.conv, self.kernel)?;
                crate::numerics::layer_capture(capture_index, "linear-conv", &mixed)?;
                linear_profile_checkpoint(x, capture_index, "convolution", &mut profile)?;
                let (q, k, v) = {
                    if x.device().is_cuda()
                        && x.dtype() == DType::BF16
                        && self.key_dim == 128
                        && self.value_dim == 128
                    {
                        cuda_fused_qkv_prepare(&mixed, self.key_heads, self.value_heads)?
                    } else {
                        let q = mixed.narrow(D::Minus1, 0, key_width)?.reshape((
                            batch,
                            length,
                            self.key_heads,
                            self.key_dim,
                        ))?;
                        let k = mixed.narrow(D::Minus1, key_width, key_width)?.reshape((
                            batch,
                            length,
                            self.key_heads,
                            self.key_dim,
                        ))?;
                        let v = mixed
                            .narrow(D::Minus1, key_width * 2, value_width)?
                            .reshape((batch, length, self.value_heads, self.value_dim))?;
                        let repeat = self.value_heads / self.key_heads;
                        let expand = |xs: Tensor| -> Result<Tensor> {
                            xs.unsqueeze(3)?
                                .broadcast_as((
                                    batch,
                                    length,
                                    self.key_heads,
                                    repeat,
                                    self.key_dim,
                                ))?
                                .reshape((batch, length, self.value_heads, self.key_dim))
                        };
                        let grouped_qk =
                            (cfg!(feature = "cuda") && x.device().is_cuda() && self.key_dim == 128)
                                || (cfg!(feature = "metal")
                                    && x.device().is_metal()
                                    && self.key_dim == 128
                                    && self.value_dim == 128);
                        let l2 = |xs: Tensor, scale: f64| -> Result<Tensor> {
                            if xs.device().is_cuda() && self.key_dim == 128 {
                                return cuda_l2_128(&xs, scale as f32);
                            }
                            let denom = (xs.sqr()?.sum_keepdim(D::Minus1)? + 1e-6)?.sqrt()?;
                            let normalized = xs.broadcast_div(&denom)?;
                            if scale == 1. {
                                Ok(normalized)
                            } else {
                                normalized * scale
                            }
                        };
                        let q = if grouped_qk { q } else { expand(q)? };
                        let q = l2(q.to_dtype(DType::F32)?, (self.key_dim as f64).powf(-0.5))?;
                        let k = if grouped_qk { k } else { expand(k)? };
                        let k = l2(k.to_dtype(DType::F32)?, 1.)?;
                        let v = v.to_dtype(DType::F32)?;
                        (q, k, v)
                    }
                    #[cfg(not(feature = "cuda"))]
                    {
                        let q = mixed.narrow(D::Minus1, 0, key_width)?.reshape((
                            batch,
                            length,
                            self.key_heads,
                            self.key_dim,
                        ))?;
                        let k = mixed.narrow(D::Minus1, key_width, key_width)?.reshape((
                            batch,
                            length,
                            self.key_heads,
                            self.key_dim,
                        ))?;
                        let v = mixed
                            .narrow(D::Minus1, key_width * 2, value_width)?
                            .reshape((batch, length, self.value_heads, self.value_dim))?;
                        let repeat = self.value_heads / self.key_heads;
                        let expand = |xs: Tensor| -> Result<Tensor> {
                            xs.unsqueeze(3)?
                                .broadcast_as((
                                    batch,
                                    length,
                                    self.key_heads,
                                    repeat,
                                    self.key_dim,
                                ))?
                                .reshape((batch, length, self.value_heads, self.key_dim))
                        };
                        let grouped_qk =
                            (cfg!(feature = "cuda") && x.device().is_cuda() && self.key_dim == 128)
                                || (cfg!(feature = "metal")
                                    && x.device().is_metal()
                                    && self.key_dim == 128
                                    && self.value_dim == 128);
                        let l2 = |xs: Tensor, scale: f64| -> Result<Tensor> {
                            if xs.device().is_cuda() && self.key_dim == 128 {
                                return cuda_l2_128(&xs, scale as f32);
                            }
                            let denom = (xs.sqr()?.sum_keepdim(D::Minus1)? + 1e-6)?.sqrt()?;
                            let normalized = xs.broadcast_div(&denom)?;
                            if scale == 1. {
                                Ok(normalized)
                            } else {
                                normalized * scale
                            }
                        };
                        let q = if grouped_qk { q } else { expand(q)? };
                        let q = l2(q.to_dtype(DType::F32)?, (self.key_dim as f64).powf(-0.5))?;
                        let k = if grouped_qk { k } else { expand(k)? };
                        let k = l2(k.to_dtype(DType::F32)?, 1.)?;
                        let v = v.to_dtype(DType::F32)?;
                        (q, k, v)
                    }
                };
                (q, k, v)
            }
            #[cfg(not(feature = "cuda"))]
            {
                let mixed = if x.device().is_cuda() && matches!(x.dtype(), DType::BF16 | DType::F32)
                {
                    cuda_conv_silu_packed_sequences(&projected, &self.conv, sequences)?
                } else {
                    causal_conv_silu(&projected, &self.conv, self.kernel)?
                };
                #[cfg(feature = "metal")]
                let mixed = if x.device().is_metal()
                    && matches!(x.dtype(), DType::BF16 | DType::F16 | DType::F32)
                {
                    crate::kernels::metal::conv_silu(&projected, &self.conv)?
                } else {
                    causal_conv_silu(&projected, &self.conv, self.kernel)?
                };
                #[cfg(not(any(feature = "cuda", feature = "metal")))]
                let mixed = causal_conv_silu(&projected, &self.conv, self.kernel)?;
                crate::numerics::layer_capture(capture_index, "linear-conv", &mixed)?;
                linear_profile_checkpoint(x, capture_index, "convolution", &mut profile)?;
                let (q, k, v) = {
                    if x.device().is_cuda()
                        && x.dtype() == DType::BF16
                        && self.key_dim == 128
                        && self.value_dim == 128
                    {
                        cuda_fused_qkv_prepare(&mixed, self.key_heads, self.value_heads)?
                    } else {
                        let q = mixed.narrow(D::Minus1, 0, key_width)?.reshape((
                            batch,
                            length,
                            self.key_heads,
                            self.key_dim,
                        ))?;
                        let k = mixed.narrow(D::Minus1, key_width, key_width)?.reshape((
                            batch,
                            length,
                            self.key_heads,
                            self.key_dim,
                        ))?;
                        let v = mixed
                            .narrow(D::Minus1, key_width * 2, value_width)?
                            .reshape((batch, length, self.value_heads, self.value_dim))?;
                        let repeat = self.value_heads / self.key_heads;
                        let expand = |xs: Tensor| -> Result<Tensor> {
                            xs.unsqueeze(3)?
                                .broadcast_as((
                                    batch,
                                    length,
                                    self.key_heads,
                                    repeat,
                                    self.key_dim,
                                ))?
                                .reshape((batch, length, self.value_heads, self.key_dim))
                        };
                        let grouped_qk =
                            (cfg!(feature = "cuda") && x.device().is_cuda() && self.key_dim == 128)
                                || (cfg!(feature = "metal")
                                    && x.device().is_metal()
                                    && self.key_dim == 128
                                    && self.value_dim == 128);
                        let l2 = |xs: Tensor, scale: f64| -> Result<Tensor> {
                            if xs.device().is_cuda() && self.key_dim == 128 {
                                return cuda_l2_128(&xs, scale as f32);
                            }
                            let denom = (xs.sqr()?.sum_keepdim(D::Minus1)? + 1e-6)?.sqrt()?;
                            let normalized = xs.broadcast_div(&denom)?;
                            if scale == 1. {
                                Ok(normalized)
                            } else {
                                normalized * scale
                            }
                        };
                        let q = if grouped_qk { q } else { expand(q)? };
                        let q = l2(q.to_dtype(DType::F32)?, (self.key_dim as f64).powf(-0.5))?;
                        let k = if grouped_qk { k } else { expand(k)? };
                        let k = l2(k.to_dtype(DType::F32)?, 1.)?;
                        let v = v.to_dtype(DType::F32)?;
                        (q, k, v)
                    }
                    #[cfg(not(feature = "cuda"))]
                    {
                        let q = mixed.narrow(D::Minus1, 0, key_width)?.reshape((
                            batch,
                            length,
                            self.key_heads,
                            self.key_dim,
                        ))?;
                        let k = mixed.narrow(D::Minus1, key_width, key_width)?.reshape((
                            batch,
                            length,
                            self.key_heads,
                            self.key_dim,
                        ))?;
                        let v = mixed
                            .narrow(D::Minus1, key_width * 2, value_width)?
                            .reshape((batch, length, self.value_heads, self.value_dim))?;
                        let repeat = self.value_heads / self.key_heads;
                        let expand = |xs: Tensor| -> Result<Tensor> {
                            xs.unsqueeze(3)?
                                .broadcast_as((
                                    batch,
                                    length,
                                    self.key_heads,
                                    repeat,
                                    self.key_dim,
                                ))?
                                .reshape((batch, length, self.value_heads, self.key_dim))
                        };
                        let grouped_qk =
                            (cfg!(feature = "cuda") && x.device().is_cuda() && self.key_dim == 128)
                                || (cfg!(feature = "metal")
                                    && x.device().is_metal()
                                    && self.key_dim == 128
                                    && self.value_dim == 128);
                        let l2 = |xs: Tensor, scale: f64| -> Result<Tensor> {
                            if xs.device().is_cuda() && self.key_dim == 128 {
                                return cuda_l2_128(&xs, scale as f32);
                            }
                            let denom = (xs.sqr()?.sum_keepdim(D::Minus1)? + 1e-6)?.sqrt()?;
                            let normalized = xs.broadcast_div(&denom)?;
                            if scale == 1. {
                                Ok(normalized)
                            } else {
                                normalized * scale
                            }
                        };
                        let q = if grouped_qk { q } else { expand(q)? };
                        let q = l2(q.to_dtype(DType::F32)?, (self.key_dim as f64).powf(-0.5))?;
                        let k = if grouped_qk { k } else { expand(k)? };
                        let k = l2(k.to_dtype(DType::F32)?, 1.)?;
                        let v = v.to_dtype(DType::F32)?;
                        (q, k, v)
                    }
                };
                (q, k, v)
            }
        };
        let (g, beta) = self.reference_gates_packed_tensor16(x, capture_index, sequences)?;
        crate::numerics::layer_capture(capture_index, "linear-q", &q)?;
        crate::numerics::layer_capture(capture_index, "linear-k", &k)?;
        crate::numerics::layer_capture(capture_index, "linear-v", &v)?;
        crate::numerics::layer_capture(capture_index, "linear-g", &g)?;
        crate::numerics::layer_capture(capture_index, "linear-beta", &beta)?;
        linear_profile_checkpoint(x, capture_index, "qk_and_gates", &mut profile)?;
        if x.device().is_cuda() && self.key_dim == 128 {
            let output = if x.dtype() == DType::BF16
                && matches!(option_env!("SYS1_CUDA_COMPUTE_CAP"), Some("80" | "90"))
                && self.value_dim == 128
            {
                cuda_c64bfout_varlen_prepared(
                    &q,
                    &k,
                    &v,
                    (&g, &beta),
                    sequences,
                    false,
                    prepared_qv.take(),
                )?
            } else if x.dtype() == DType::BF16
                && option_env!("SYS1_CUDA_COMPUTE_CAP") == Some("120")
                && self.value_dim == 128
            {
                if v.dtype() == DType::BF16 {
                    cuda_vstore_gdn_varlen(&q, &k, &v, &g, &beta, sequences, false)?
                } else {
                    cuda_c64bfout_varlen(&q, &k, &v, &g, &beta, sequences, false)?
                }
            } else if x.dtype() == DType::BF16
                && option_env!("SYS1_CUDA_COMPUTE_CAP") == Some("120")
                && self.value_dim <= 128
            {
                cuda_c64_varlen(&q, &k, &v, &g, &beta, sequences, false)?
            } else {
                cuda_delta_rule_packed_sequences(&q, &k, &v, &g, &beta, sequences, true)?
            };
            crate::numerics::layer_capture(capture_index, "linear-delta", &output)?;
            if crate::numerics::enabled() {
                crate::numerics::layer_capture(
                    capture_index,
                    "linear-delta-rounded",
                    &output.to_dtype(x.dtype())?,
                )?;
            }
            linear_profile_checkpoint(x, capture_index, "delta_rule", &mut profile)?;
            let result = self.finish_delta(
                x,
                output,
                batch,
                length,
                value_width,
                capture_index,
                early_z,
            )?;
            linear_profile_checkpoint(x, capture_index, "finalize_and_projection", &mut profile)?;
            return Ok(result);
        }
        let q = q.transpose(1, 2)?;
        let k = k.transpose(1, 2)?;
        let v = v.transpose(1, 2)?;
        let beta = beta.transpose(1, 2)?;
        let g = g.transpose(1, 2)?;
        let output = chunked_delta_rule(&q, &k, &v, &g, &beta)?;
        if crate::numerics::enabled() {
            crate::numerics::layer_capture(
                capture_index,
                "linear-delta-rounded",
                &output.to_dtype(x.dtype())?,
            )?;
        }
        linear_profile_checkpoint(x, capture_index, "delta_rule", &mut profile)?;
        let result = self.finish_delta(
            x,
            output,
            batch,
            length,
            value_width,
            capture_index,
            early_z,
        )?;
        linear_profile_checkpoint(x, capture_index, "finalize_and_projection", &mut profile)?;
        Ok(result)
    }

    pub(crate) fn reference_gates_packed_tensor16(
        &self,
        x: &Tensor,
        capture_index: usize,
        sequences: &PackedSequences,
    ) -> Result<(Tensor, Tensor)> {
        if option_env!("SYS1_CUDA_COMPUTE_CAP") == Some("120") && x.dtype() == DType::BF16 {
            let ba = self
                .ba
                .as_ref()
                .ok_or_else(|| candle_core::Error::Msg("missing packed BF16 gates".into()))?;
            if ba.weight().dims() == [64, 4096] {
                let projected =
                    cuda_gate_tensor16_packed(x, ba, true, SimtProjection::Compensated, sequences)?;
                return cuda_packed_ba_nonlinearity(
                    &projected,
                    &projected,
                    &self.dt_bias,
                    &self.a_log,
                );
            }
        }
        self.reference_gates(x, capture_index)
    }
}

impl LinearAttention {
    #[cfg(sys1_cuda_fused_ffn)]
    pub(crate) fn qkvz_available(&self) -> bool {
        self.qkv.bias().is_none()
            && self.z.bias().is_none()
            && crate::kernels::cuda::fused_qkvz::available(self.qkv.weight(), self.z.weight())
    }
}

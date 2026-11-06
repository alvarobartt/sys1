use super::*;

impl LinearAttention {
    pub(crate) fn forward_packed(&self, x: &Tensor, sequences: &PackedSequences) -> Result<Tensor> {
        if !x.device().is_cuda() || x.dtype() != DType::BF16 || self.key_dim != 128 {
            candle_core::bail!("unsupported packed recurrent architecture")
        }
        let (batch, length, _) = x.dims3()?;
        let key_width = self.key_heads * self.key_dim;
        let value_width = self.value_heads * self.value_dim;
        let (projected, early_z) = self.project_qkvz(x)?;
        let mut prepared_qv = None;

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
            let (g, beta) = self.reference_gates_packed_tensor16(x, sequences)?;

            if let Some(output) = gated_delta::try_forward_prepared_source(
                &projected,
                &self.conv,
                &g,
                &beta,
                &sequences.lengths,
                ticket,
            )? {
                let result = self.finish_delta(x, output, batch, length, value_width, early_z)?;

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
                if option_env!("SYS1_CUDA_COMPUTE_CAP") == Some("90") {
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
                }
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
        let (g, beta) = self.reference_gates_packed_tensor16(x, sequences)?;

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
                    prepared_qv.take(),
                )?
            } else if x.dtype() == DType::BF16
                && option_env!("SYS1_CUDA_COMPUTE_CAP") == Some("120")
                && self.value_dim == 128
            {
                if v.dtype() == DType::BF16 {
                    cuda_vstore_gdn_varlen(&q, &k, &v, &g, &beta, sequences)?
                } else {
                    cuda_c64bfout_varlen(&q, &k, &v, &g, &beta, sequences)?
                }
            } else if x.dtype() == DType::BF16
                && option_env!("SYS1_CUDA_COMPUTE_CAP") == Some("120")
                && self.value_dim <= 128
            {
                cuda_c64_varlen(&q, &k, &v, &g, &beta, sequences)?
            } else {
                cuda_delta_rule_packed_sequences(&q, &k, &v, &g, &beta, sequences, true)?
            };

            let result = self.finish_delta(x, output, batch, length, value_width, early_z)?;

            return Ok(result);
        }
        let q = q.transpose(1, 2)?;
        let k = k.transpose(1, 2)?;
        let v = v.transpose(1, 2)?;
        let beta = beta.transpose(1, 2)?;
        let g = g.transpose(1, 2)?;
        let output = chunked_delta_rule(&q, &k, &v, &g, &beta)?;

        let result = self.finish_delta(x, output, batch, length, value_width, early_z)?;

        Ok(result)
    }

    pub(crate) fn reference_gates_packed_tensor16(
        &self,
        x: &Tensor,
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
        self.reference_gates(x)
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

use crate::architectures::modernbert::AttentionOptions;
use crate::models::AttentionImplementation;
use candle_core::{DType, Tensor};
use candle_nn::{LayerNorm, Linear, VarBuilder};
pub(crate) fn layer_norm(size: usize, eps: f64, vb: VarBuilder) -> candle_core::Result<LayerNorm> {
    Ok(LayerNorm::new(
        vb.get_with_hints(size, "weight", candle_nn::Init::Const(1.))?,
        vb.get_with_hints(size, "bias", candle_nn::Init::Const(0.))?,
        eps,
    ))
}

pub(crate) fn linear_dtype(
    input: usize,
    output: usize,
    vb: VarBuilder,
    dtype: DType,
) -> candle_core::Result<Linear> {
    Ok(Linear::new(
        vb.get((output, input), "weight")?.to_dtype(dtype)?,
        Some(vb.get(output, "bias")?.to_dtype(dtype)?),
    ))
}

pub(crate) struct HeadLayer {
    qkv: Linear,
    projection: Linear,
    norm1: LayerNorm,
    norm2: LayerNorm,
    linear1: Linear,
    linear2: Linear,
    pub(crate) heads: usize,
    compute_dtype: DType,
    attention: AttentionImplementation,
}

impl HeadLayer {
    pub(crate) fn load(
        vb: VarBuilder,
        hidden: usize,
        compute_dtype: DType,
        attention: AttentionImplementation,
    ) -> candle_core::Result<Self> {
        Ok(Self {
            qkv: Linear::new(
                vb.get((hidden * 3, hidden), "self_attn.in_proj_weight")?
                    .to_dtype(compute_dtype)?,
                Some(
                    vb.get(hidden * 3, "self_attn.in_proj_bias")?
                        .to_dtype(compute_dtype)?,
                ),
            ),
            projection: linear_dtype(hidden, hidden, vb.pp("self_attn.out_proj"), compute_dtype)?,
            norm1: layer_norm(hidden, 1e-5, vb.pp("norm1"))?,
            norm2: layer_norm(hidden, 1e-5, vb.pp("norm2"))?,
            linear1: linear_dtype(hidden, hidden * 4, vb.pp("linear1"), compute_dtype)?,
            linear2: linear_dtype(hidden * 4, hidden, vb.pp("linear2"), compute_dtype)?,
            heads: hidden / 64,
            compute_dtype,
            attention,
        })
    }

    pub(crate) fn forward(
        &self,
        xs: &Tensor,
        mask: &Tensor,
        lengths: &[usize],
    ) -> candle_core::Result<Tensor> {
        let (batch, length, hidden) = xs.dims3()?;
        let size = hidden / self.heads;
        let normalized = xs.apply(&self.norm1)?.to_dtype(self.compute_dtype)?;
        let qkv = crate::kernels::linear::forward(&normalized, &self.qkv)?
            .reshape((batch, length, 3, self.heads, size))?
            .permute((2, 0, 3, 1, 4))?;
        let q = qkv.get(0)?;
        let k = qkv.get(1)?;
        let v = qkv.get(2)?;
        let scale = (size as f64).powf(-0.5);

        #[cfg(feature = "metal")]
        let attention = if xs.device().is_metal() && self.attention == AttentionImplementation::Auto
        {
            let mask = mask
                .broadcast_as((batch, self.heads, length, length))?
                .contiguous()?;
            candle_nn::ops::sdpa(&q, &k, &v, Some(&mask), false, scale as f32, 1.0)?
        } else {
            crate::architectures::modernbert::scaled_dot_product_attention(
                &q,
                &k,
                &v,
                scale,
                AttentionOptions {
                    mask: Some(mask),
                    implementation: self.attention,
                    lengths,
                    window: None,
                },
            )?
        };
        #[cfg(not(feature = "metal"))]
        let attention = crate::architectures::modernbert::scaled_dot_product_attention(
            &q,
            &k,
            &v,
            scale,
            AttentionOptions {
                mask: Some(mask),
                implementation: self.attention,
                lengths,
                window: None,
            },
        )?;
        #[cfg(feature = "metal")]
        let attention = if attention.device().is_metal() {
            crate::architectures::modernbert::metal::merge_heads(&attention, batch, length, hidden)?
        } else {
            attention
                .transpose(1, 2)?
                .reshape((batch, length, hidden))?
        };
        #[cfg(not(feature = "metal"))]
        let attention = attention
            .transpose(1, 2)?
            .reshape((batch, length, hidden))?;
        let attention =
            crate::kernels::linear::forward(&attention, &self.projection)?.to_dtype(xs.dtype())?;
        let xs = (xs + attention)?;
        let normalized = xs.apply(&self.norm2)?.to_dtype(self.compute_dtype)?;
        let intermediate = crate::kernels::linear::forward(&normalized, &self.linear1)?.relu()?;
        let feed_forward =
            crate::kernels::linear::forward(&intermediate, &self.linear2)?.to_dtype(xs.dtype())?;
        xs + feed_forward
    }
}

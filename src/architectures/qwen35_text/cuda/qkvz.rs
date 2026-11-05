use super::{Linear, LinearAttention};
use candle_core::{DType, Device, Result, Tensor, cuda_backend::WrapErr};

#[cfg(test)]
pub(crate) static STRIDED_PREPARED: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);
#[cfg(test)]
pub(crate) static STRIDED_POST: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

#[cfg(test)]
static SELECTIONS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

pub(crate) fn prepare(qkv: Linear, z: Linear) -> Result<(Linear, Linear, Option<Linear>)> {
    let supported = qkv.weight().dtype() == DType::BF16
        && z.weight().dtype() == DType::BF16
        && qkv.weight().dims() == [8192, 4096]
        && z.weight().dims() == [4096, 4096]
        && qkv.bias().is_none()
        && z.bias().is_none();
    let hopper = match qkv.weight().device() {
        Device::Cuda(device) => {
            supported && device.cuda_stream().context().compute_capability().w()? == (9, 0)
        }
        _ => false,
    };
    if !hopper {
        return Ok((qkv, z, None));
    }

    let packed = Tensor::cat(&[qkv.weight(), z.weight()], 0)?;
    let qkv = Linear::new(packed.narrow(0, 0, 8192)?, None);
    let z = Linear::new(packed.narrow(0, 8192, 4096)?, None);
    Ok((qkv, z, Some(Linear::new(packed, None))))
}

impl LinearAttention {
    pub(crate) fn project_merged_hopper_qkvz(
        &self,
        input: &Tensor,
    ) -> Result<Option<(Tensor, Option<Tensor>)>> {
        let Some(weight) = &self.merged_hopper_qkvz else {
            return Ok(None);
        };
        let (batch, length, inner) = input.dims3()?;
        let rows = batch
            .checked_mul(length)
            .ok_or_else(|| candle_core::Error::Msg("GDN merged projection row overflow".into()))?;
        if input.dtype() != DType::BF16
            || !input.device().is_cuda()
            || inner != 4096
            || !(16384..=65536).contains(&rows)
            || option_env!("SYS1_CUDA_COMPUTE_CAP") != Some("90")
        {
            return Ok(None);
        }

        let projected = crate::kernels::cuda::blaslt::project_unblocked(input, weight)?;

        let qkv = projected.narrow(candle_core::D::Minus1, 0, 8192)?;
        let z = projected.narrow(candle_core::D::Minus1, 8192, 4096)?;
        #[cfg(test)]
        SELECTIONS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Ok(Some((qkv, Some(z))))
    }
}

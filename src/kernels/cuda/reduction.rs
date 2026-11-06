use candle_core::{
    CudaStorage, DType, Result, Storage, Tensor,
    backend::BackendStorage,
    cuda_backend::{
        WrapErr,
        cudarc::driver::{LaunchConfig, PushKernelArg},
    },
    op::BackpropOp,
};
use half::bf16;
const CUDA_PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/cuda.ptx"));

pub(crate) fn weighted_summary(values: &Tensor, weights: &Tensor) -> Result<Tensor> {
    let (rows, columns) = values.dims2()?;
    if !(1..=32).contains(&rows)
        || columns == 0
        || columns > u32::MAX as usize / 32
        || values.dtype() != DType::BF16
        || weights.dtype() != DType::BF16
        || weights.dims() != [1, rows]
        || !values.device().same_device(weights.device())
    {
        candle_core::bail!("weighted summary requires colocated CUDA BF16 rows1..32");
    }
    let values = values.contiguous()?;
    let weights = weights.contiguous()?;
    let (vs, vl) = values.storage_and_layout();
    let (ws, wl) = weights.storage_and_layout();
    let (Storage::Cuda(vs), Storage::Cuda(ws)) = (&*vs, &*ws) else {
        candle_core::bail!("weighted summary requires CUDA storage");
    };
    let (vstart, vend) = vl
        .contiguous_offsets()
        .ok_or_else(|| candle_core::Error::Msg("summary values noncontiguous".into()))?;
    let (wstart, wend) = wl
        .contiguous_offsets()
        .ok_or_else(|| candle_core::Error::Msg("summary weights noncontiguous".into()))?;
    let v = vs.as_cuda_slice::<bf16>()?.slice(vstart..vend);
    let w = ws.as_cuda_slice::<bf16>()?.slice(wstart..wend);
    let device = vs.device().clone();
    let output = unsafe { device.alloc::<bf16>(columns)? };
    let function =
        device.get_or_load_custom_func("weighted_summary_bf16", "sys1_reduction", CUDA_PTX)?;
    let mut launch = function.builder();
    launch.arg(&v);
    launch.arg(&w);
    launch.arg(&output);
    let r = rows as u32;
    let c = columns as u32;
    launch.arg(&r);
    launch.arg(&c);
    unsafe { launch.launch(LaunchConfig::for_num_elems(c)) }.w()?;
    Ok(Tensor::from_storage(
        Storage::Cuda(CudaStorage::wrap_cuda_slice(output, device)),
        (columns,),
        BackpropOp::none(),
        false,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn weighted_summary_matches_independent_rounded_scalar_sum() -> Result<()> {
        let device = crate::device::load().map_err(|e| candle_core::Error::Msg(e.to_string()))?;
        let candle_core::Device::Cuda(cuda) = &device else {
            return Ok(());
        };
        if cuda
            .cuda_stream()
            .context()
            .compute_capability()
            .map_err(|e| candle_core::Error::Msg(e.to_string()))?
            .0
            < 8
        {
            return Ok(());
        }
        for rows in [1, 3, 11, 32] {
            for columns in [7, 128] {
                let vals: Vec<bf16> = (0..rows * columns)
                    .map(|i| bf16::from_f32(((i * 7919 % 65521) as f32 - 32760.) / 32768.))
                    .collect();
                let weights: Vec<bf16> = (0..rows)
                    .map(|i| bf16::from_f32((i + 1) as f32 / (rows * (rows + 1) / 2) as f32))
                    .collect();
                let expected: Vec<f32> = (0..columns)
                    .map(|col| {
                        let mut sum = 0f32;
                        for row in 0..rows {
                            sum += bf16::from_f32(
                                vals[row * columns + col].to_f32() * weights[row].to_f32(),
                            )
                            .to_f32();
                        }
                        bf16::from_f32(sum).to_f32()
                    })
                    .collect();
                let v = Tensor::from_vec(vals, (rows, columns), &device)?;
                let w = Tensor::from_vec(weights, (1, rows), &device)?;
                let actual = weighted_summary(&v, &w)?
                    .to_dtype(DType::F32)?
                    .to_vec1::<f32>()?;
                assert_eq!(actual, expected, "rows{rows} columns{columns}");
            }
        }
        Ok(())
    }
}

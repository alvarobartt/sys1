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
use std::collections::BTreeMap;
const CUDA_PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/cuda.ptx"));

pub(crate) fn span_means(source: &Tensor, spans: &[(usize, usize)]) -> Result<Tensor> {
    means(source, None, spans)
}

pub(crate) fn lexical_means(
    table: &Tensor,
    ids: &Tensor,
    spans: &[(usize, usize)],
) -> Result<Tensor> {
    means(table, Some(ids), spans)
}

fn means(source: &Tensor, ids: Option<&Tensor>, spans: &[(usize, usize)]) -> Result<Tensor> {
    let (source_rows, columns) = source.dims2()?;
    if source.dtype() != DType::BF16
        || !source.device().is_cuda()
        || source_rows > u32::MAX as usize
        || columns == 0
        || columns > u32::MAX as usize
        || spans.len() > u32::MAX as usize
    {
        candle_core::bail!("schema means require CUDA BF16 matrix and bounded geometry");
    }
    if let Some(ids) = ids
        && (ids.dtype() != DType::U32
            || ids.rank() != 1
            || !ids.device().same_device(source.device()))
    {
        candle_core::bail!("lexical means require colocated U32 token IDs");
    }
    let logical_rows = ids.map_or(source_rows, Tensor::elem_count);
    let mut groups: BTreeMap<u32, Vec<u32>> = BTreeMap::new();
    for (destination, &(start, end)) in spans.iter().enumerate() {
        if start >= end || end > logical_rows || end > u32::MAX as usize {
            candle_core::bail!("invalid schema span");
        }
        let length = end - start;
        let threads = length.min(1024).next_power_of_two() as u32;
        groups.entry(threads).or_default().extend([
            start as u32,
            length as u32,
            destination as u32,
        ]);
    }
    if spans.is_empty() {
        return Tensor::zeros((0, columns), DType::BF16, source.device());
    }
    let count = spans
        .len()
        .checked_mul(columns)
        .ok_or_else(|| candle_core::Error::Msg("schema mean allocation overflow".into()))?;
    let source = source.contiguous()?;
    let ids = ids.map(Tensor::contiguous).transpose()?;
    let (storage, layout) = source.storage_and_layout();
    let Storage::Cuda(storage) = &*storage else {
        candle_core::bail!("CUDA source required");
    };
    let (start, end) = layout
        .contiguous_offsets()
        .ok_or_else(|| candle_core::Error::Msg("schema source noncontiguous".into()))?;
    let input = storage.as_cuda_slice::<bf16>()?.slice(start..end);
    let id_storage = ids.as_ref().map(Tensor::storage_and_layout);
    let id_view = if let Some((storage, layout)) = &id_storage {
        let Storage::Cuda(storage) = &**storage else {
            candle_core::bail!("CUDA IDs required");
        };
        let (start, end) = layout
            .contiguous_offsets()
            .ok_or_else(|| candle_core::Error::Msg("schema IDs noncontiguous".into()))?;
        Some(storage.as_cuda_slice::<u32>()?.slice(start..end))
    } else {
        None
    };
    let device = storage.device().clone();
    let output = unsafe { device.alloc::<bf16>(count)? };
    let function = device.get_or_load_custom_func("span_means_bf16", "sys1_schema", CUDA_PTX)?;
    for (rows_per_block, packed) in groups {
        let group_count = packed.len() / 3;
        if group_count > 65535 {
            candle_core::bail!("schema span group exceeds CUDA grid limit");
        }
        let meta = Tensor::from_vec(packed, (group_count, 3), source.device())?;
        let (meta_storage, meta_layout) = meta.storage_and_layout();
        let Storage::Cuda(meta_storage) = &*meta_storage else {
            unreachable!()
        };
        let (start, end) = meta_layout.contiguous_offsets().unwrap();
        let meta = meta_storage.as_cuda_slice::<u32>()?.slice(start..end);
        let selected_ids = id_view.as_ref().unwrap_or(&meta);
        let column_tile = (1024 / rows_per_block).min(32);
        let source_rows = source_rows as u32;
        let columns = columns as u32;
        let lexical = i32::from(ids.is_some());
        let mut launch = function.builder();
        launch.arg(&input);
        launch.arg(selected_ids);
        launch.arg(&meta);
        launch.arg(&output);
        launch.arg(&source_rows);
        launch.arg(&columns);
        launch.arg(&lexical);
        unsafe {
            launch.launch(LaunchConfig {
                grid_dim: (columns.div_ceil(column_tile), group_count as u32, 1),
                block_dim: (column_tile, rows_per_block, 1),
                shared_mem_bytes: 0,
            })
        }
        .w()?;
    }
    Ok(Tensor::from_storage(
        Storage::Cuda(CudaStorage::wrap_cuda_slice(output, device)),
        (spans.len(), columns),
        BackpropOp::none(),
        false,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn schema_and_lexical_means_match_stock_tree_and_cast_bits() -> Result<()> {
        let device = crate::device::load().map_err(|e| candle_core::Error::Msg(e.to_string()))?;
        let candle_core::Device::Cuda(cuda) = &device else {
            return Ok(());
        };
        if cuda.cuda_stream().context().compute_capability().w()?.0 < 8 {
            return Ok(());
        }
        for width in [7, 128, 4096] {
            let source = Tensor::from_vec(
                (0..1120 * width)
                    .map(|i| bf16::from_f32(((i * 7919 % 65521) as f32 - 32760.) / 16384.))
                    .collect::<Vec<_>>(),
                (1120, width),
                &device,
            )?;
            let ids = Tensor::from_vec(
                (0..1120)
                    .map(|i| (i * 17 % 1120) as u32)
                    .collect::<Vec<_>>(),
                1120,
                &device,
            )?;
            let spans = [
                (1, 2),
                (3, 6),
                (11, 22),
                (19, 51),
                (33, 66),
                (63, 128),
                (500, 1013),
                (77, 1107),
            ];
            let mean = |xs: &Tensor| -> Result<Tensor> {
                (xs.to_dtype(DType::F32)?.sum(0)? / xs.dim(0)? as f64)?.to_dtype(DType::BF16)
            };
            let expected = Tensor::stack(
                &spans
                    .iter()
                    .map(|&(start, end)| mean(&source.narrow(0, start, end - start)?))
                    .collect::<Result<Vec<_>>>()?,
                0,
            )?;
            let expected_lex = Tensor::stack(
                &spans
                    .iter()
                    .map(|&(start, end)| {
                        mean(&source.index_select(&ids.narrow(0, start, end - start)?, 0)?)
                    })
                    .collect::<Result<Vec<_>>>()?,
                0,
            )?;
            let bits = |xs: Tensor| -> Result<Vec<u16>> {
                Ok(xs
                    .flatten_all()?
                    .to_vec1::<bf16>()?
                    .iter()
                    .map(|x| x.to_bits())
                    .collect())
            };
            assert_eq!(
                bits(span_means(&source, &spans)?)?,
                bits(expected)?,
                "schema width{width}"
            );
            assert_eq!(
                bits(lexical_means(&source, &ids, &spans)?)?,
                bits(expected_lex)?,
                "lexical width{width}"
            );
            assert!(span_means(&source, &[(1, 1)]).is_err());
            assert!(span_means(&source, &[(0, 1121)]).is_err());
        }
        Ok(())
    }
}

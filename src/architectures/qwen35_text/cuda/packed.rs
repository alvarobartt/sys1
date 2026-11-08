use super::*;

pub(crate) fn cuda_packed_ba_nonlinearity(
    a: &Tensor,
    b: &Tensor,
    bias: &Tensor,
    a_log: &Tensor,
) -> Result<(Tensor, Tensor)> {
    let heads = *a
        .dims()
        .last()
        .ok_or_else(|| candle_core::Error::Msg("gate rank".into()))?
        / 2;
    let count = a.elem_count() / 2;
    if a.elem_count() > i32::MAX as usize
        || a.dim(D::Minus1)? % 2 != 0
        || count == 0
        || count > i32::MAX as usize
        || heads == 0
        || a.dims() != b.dims()
        || a.dtype() != DType::BF16
        || b.dtype() != DType::BF16
        || bias.dtype() != DType::F32
        || a_log.dtype() != DType::F32
        || bias.dims() != [heads]
        || a_log.dims() != [heads]
        || !a.device().same_device(b.device())
        || !a.device().same_device(bias.device())
        || !a.device().same_device(a_log.device())
    {
        candle_core::bail!("unsupported fused gate tensors");
    }
    let a = a.contiguous()?;
    let b = b.contiguous()?;
    let bias = bias.contiguous()?;
    let a_log = a_log.contiguous()?;
    let (as_, al) = a.storage_and_layout();
    let (bs, bl) = b.storage_and_layout();
    let (ds, dl) = bias.storage_and_layout();
    let (ls, ll) = a_log.storage_and_layout();
    let (Storage::Cuda(as_), Storage::Cuda(bs), Storage::Cuda(ds), Storage::Cuda(ls)) =
        (&*as_, &*bs, &*ds, &*ls)
    else {
        candle_core::bail!("fused gates require CUDA");
    };
    let offsets = |l: &candle_core::Layout| {
        l.contiguous_offsets()
            .ok_or_else(|| candle_core::Error::Msg("noncontiguous gate storage".into()))
    };
    let (a0, a1) = offsets(al)?;
    let (b0, b1) = offsets(bl)?;
    let (d0, d1) = offsets(dl)?;
    let (l0, l1) = offsets(ll)?;
    let av = as_.as_cuda_slice::<bf16>()?.slice(a0..a1);
    let bv = bs.as_cuda_slice::<bf16>()?.slice(b0..b1);
    let dv = ds.as_cuda_slice::<f32>()?.slice(d0..d1);
    let lv = ls.as_cuda_slice::<f32>()?.slice(l0..l1);
    let device = as_.device().clone();
    let mut gate = unsafe { device.alloc::<f32>(count)? };
    let mut beta = unsafe { device.alloc::<f32>(count)? };
    let function = device.get_or_load_custom_func(
        "qwen35_packed_ba_nonlinearity_bf16",
        "sys1_fused_gates",
        CUDA_PTX,
    )?;
    let mut launch = function.builder();
    launch.arg(&av);
    launch.arg(&bv);
    launch.arg(&dv);
    launch.arg(&lv);
    launch.arg(&mut gate);
    launch.arg(&mut beta);
    let count_arg = count as i32;
    let heads_arg = heads as i32;
    launch.arg(&count_arg);
    launch.arg(&heads_arg);
    unsafe {
        launch.launch(LaunchConfig {
            grid_dim: (count.div_ceil(256) as u32, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        })
    }
    .w()?;
    let mut shape = a.dims().to_vec();
    *shape.last_mut().unwrap() = heads;
    let wrap = |data| {
        Tensor::from_storage(
            Storage::Cuda(CudaStorage::wrap_cuda_slice(data, device.clone())),
            shape.clone(),
            BackpropOp::none(),
            false,
        )
    };
    Ok((wrap(gate), wrap(beta)))
}

pub(crate) type C64GeometryCache = std::sync::Mutex<Option<((usize, usize, usize), bool)>>;

pub(crate) struct PackedSequences {
    pub(crate) lengths: Vec<usize>,
    pub(crate) cumulative: Tensor,
    pub(crate) starts: Tensor,
    pub(crate) total: usize,
    pub(crate) c64_chunks: Tensor,
    pub(crate) c64_long_ids: Tensor,
    pub(crate) c64_short_ids: Tensor,
    pub(crate) c64_chunk_count: usize,
    pub(crate) c64_long_count: usize,
    pub(crate) c64_short_count: usize,
    pub(crate) c64_geometry_cache: C64GeometryCache,
    pub(crate) gate_long_starts: Tensor,
    pub(crate) gate_long_ends: Tensor,
    pub(crate) gate_long_count: usize,
    pub(crate) gate_short_rows: Tensor,
    pub(crate) gate_short_count: usize,
}

impl PackedSequences {
    pub(crate) fn new(lengths: &[usize], device: &candle_core::Device) -> Result<Self> {
        if lengths.is_empty() || lengths.contains(&0) || !device.is_cuda() {
            candle_core::bail!("packed sequences require nonempty positive lengths and CUDA")
        }
        let total = lengths
            .iter()
            .try_fold(0usize, |sum, &n| sum.checked_add(n))
            .filter(|&n| n <= i32::MAX as usize)
            .ok_or_else(|| candle_core::Error::Msg("packed total overflow".into()))?;
        let mut offsets = vec![0u32];
        let mut starts = Vec::with_capacity(total);
        let mut begin = 0usize;
        for &n in lengths {
            starts.resize(starts.len() + n, begin as u32);
            begin += n;
            offsets.push(begin as u32);
        }
        let mut c64_offsets = vec![0u32];
        let mut long_ids = Vec::new();
        let mut short_ids = Vec::new();
        let mut chunk_count = 0usize;
        for (index, &n) in lengths.iter().enumerate() {
            if n >= (if matches!(option_env!("SYS1_CUDA_COMPUTE_CAP"), Some("80" | "90")) {
                128
            } else {
                512
            }) {
                long_ids.push(index as u32);
                chunk_count += n.div_ceil(64);
            } else {
                short_ids.push(index as u32);
            }
            c64_offsets.push(chunk_count as u32);
        }

        long_ids.sort_by_key(|&index| std::cmp::Reverse(lengths[index as usize].div_ceil(64)));
        let long_count = long_ids.len();
        let short_count = short_ids.len();
        if long_ids.is_empty() {
            long_ids.push(0);
        }
        if short_ids.is_empty() {
            short_ids.push(0);
        }
        let mut gate_long_starts = Vec::new();
        let mut gate_long_ends = Vec::new();
        let mut gate_short_rows = Vec::new();
        let mut gate_begin = 0usize;
        for &n in lengths {
            if n >= 1024 {
                for offset in (0..n).step_by(32) {
                    gate_long_starts.push((gate_begin + offset) as u32);
                    gate_long_ends.push((gate_begin + n) as u32);
                }
            } else {
                gate_short_rows.extend((gate_begin..gate_begin + n).map(|i| i as u32));
            }
            gate_begin += n;
        }
        let gate_long_count = gate_long_starts.len();
        if gate_long_starts.is_empty() {
            gate_long_starts.push(0);
            gate_long_ends.push(0);
        }
        let gate_short_count = gate_short_rows.len();
        if gate_short_rows.is_empty() {
            gate_short_rows.push(0);
        }
        Ok(Self {
            lengths: lengths.to_vec(),
            cumulative: Tensor::from_vec(offsets, lengths.len() + 1, device)?,
            starts: Tensor::from_vec(starts, total, device)?,
            total,
            c64_chunks: Tensor::from_vec(c64_offsets, lengths.len() + 1, device)?,
            c64_long_ids: Tensor::from_vec(long_ids, long_count.max(1), device)?,
            c64_short_ids: Tensor::from_vec(short_ids, short_count.max(1), device)?,
            c64_chunk_count: chunk_count,
            c64_long_count: long_count,
            c64_short_count: short_count,
            c64_geometry_cache: std::sync::Mutex::new(None),
            gate_long_starts: Tensor::from_vec(gate_long_starts, gate_long_count.max(1), device)?,
            gate_long_ends: Tensor::from_vec(gate_long_ends, gate_long_count.max(1), device)?,
            gate_long_count,
            gate_short_rows: Tensor::from_vec(gate_short_rows, gate_short_count.max(1), device)?,
            gate_short_count,
        })
    }
}

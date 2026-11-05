use candle_core::cuda_backend::{
    WrapErr,
    cudarc::{
        driver::{
            CudaFunction, CudaSlice, CudaStream, DeviceRepr, LaunchConfig, PushKernelArg,
            sys::{CUdevice_attribute, CUfunction_attribute},
        },
        nvrtc::Ptx,
    },
};
use candle_core::{CudaStorage, DType, Device, Result, Storage, Tensor, op::BackpropOp};
use std::sync::{Arc, Mutex, MutexGuard, OnceLock};

const NAMES: [&str; 5] = [
    "chunk_local_cumsum_scalar_kernel",
    "chunk_gated_delta_rule_fwd_kkt_solve_kernel",
    "recompute_w_u_fwd_kernel",
    "chunk_gated_delta_rule_fwd_kernel_h_blockdim64",
    "chunk_fwd_kernel_o",
];
const THREADS: [u32; 5] = [256, 32, 128, 128, 128];
const SHARED: [u32; 5] = [8, 9216, 32768, 73988, 49152];
#[cfg(sys1_cuda_gdn_sm90)]
const BINARIES: [&[u8]; 5] = [
    include_bytes!(concat!(env!("OUT_DIR"), "/qwen35_aot_assets_v91/0.cubin")),
    include_bytes!(concat!(env!("OUT_DIR"), "/qwen35_aot_assets_v91/1.cubin")),
    include_bytes!(concat!(env!("OUT_DIR"), "/qwen35_aot_assets_v91/2.cubin")),
    include_bytes!(concat!(env!("OUT_DIR"), "/qwen35_aot_assets_v91/3.cubin")),
    include_bytes!(concat!(env!("OUT_DIR"), "/qwen35_aot_assets_v91/4.cubin")),
];

const SM80_THREADS: [u32; 5] = [256, 64, 256, 128, 128];
const SM80_SHARED: [u32; 5] = [8, 9216, 32768, 41220, 34816];
#[cfg(sys1_cuda_gdn_sm80)]
const SM80_BINARIES: [&[u8]; 5] = [
    include_bytes!(concat!(env!("OUT_DIR"), "/qwen35_aot_assets_sm80/0.cubin")),
    include_bytes!(concat!(env!("OUT_DIR"), "/qwen35_aot_assets_sm80/1.cubin")),
    include_bytes!(concat!(env!("OUT_DIR"), "/qwen35_aot_assets_sm80/2.cubin")),
    include_bytes!(concat!(env!("OUT_DIR"), "/qwen35_aot_assets_sm80/3.cubin")),
    include_bytes!(concat!(env!("OUT_DIR"), "/qwen35_aot_assets_sm80/4.cubin")),
];
fn native_binary(i: usize) -> Result<&'static [u8]> {
    #[cfg(sys1_cuda_gdn_sm80)]
    {
        Ok(SM80_BINARIES[i])
    }
    #[cfg(sys1_cuda_gdn_sm90)]
    {
        Ok(BINARIES[i])
    }
    #[cfg(not(any(sys1_cuda_gdn_sm80, sys1_cuda_gdn_sm90)))]
    {
        let _ = i;
        candle_core::bail!("CUDA recurrent kernel image is unavailable for this build target")
    }
}
fn native_threads(i: usize) -> u32 {
    if option_env!("SYS1_CUDA_COMPUTE_CAP") == Some("80") {
        SM80_THREADS[i]
    } else {
        THREADS[i]
    }
}
fn native_shared(i: usize) -> u32 {
    if option_env!("SYS1_CUDA_COMPUTE_CAP") == Some("80") {
        SM80_SHARED[i]
    } else {
        SHARED[i]
    }
}

struct Metadata {
    lengths: Vec<usize>,
    cumulative: CudaSlice<i64>,
    chunks: CudaSlice<i64>,
    offsets: CudaSlice<i64>,
    initial_ids: CudaSlice<i64>,
    count: usize,
}
impl Metadata {
    fn new(stream: &Arc<CudaStream>, lengths: &[usize]) -> Result<Self> {
        let mut cumulative = vec![0i64];
        let mut offsets = vec![0i64];
        let mut chunks = Vec::new();
        for (sequence, &length) in lengths.iter().enumerate() {
            cumulative.push(cumulative.last().unwrap() + length as i64);
            let n = length.div_ceil(64);
            for chunk in 0..n {
                chunks.extend([sequence as i64, chunk as i64]);
            }
            offsets.push(offsets.last().unwrap() + n as i64);
        }
        Ok(Self {
            lengths: lengths.to_vec(),
            count: chunks.len() / 2,
            cumulative: stream.clone_htod(&cumulative).w()?,
            chunks: stream.clone_htod(&chunks).w()?,
            offsets: stream.clone_htod(&offsets).w()?,

            initial_ids: stream.clone_htod(&vec![0i64; lengths.len()]).w()?,
        })
    }
}

struct Workspace {
    stream: Arc<CudaStream>,
    functions: Vec<CudaFunction>,
    metadata: Option<Metadata>,
    q: Option<CudaSlice<u16>>,
    k: Option<CudaSlice<u16>>,
    v: Option<CudaSlice<u16>>,
    gate: Option<CudaSlice<f32>>,
    beta: Option<CudaSlice<f32>>,
    prefix: Option<CudaSlice<f32>>,
    a: Option<CudaSlice<u16>>,
    w: Option<CudaSlice<u16>>,
    u: Option<CudaSlice<u16>>,
    vnew: Option<CudaSlice<u16>>,
    h: Option<CudaSlice<u16>>,
    initial: CudaSlice<f32>,
    bad: CudaSlice<u32>,
    qv_bad: CudaSlice<u32>,
}
impl Workspace {
    fn new(stream: Arc<CudaStream>) -> Result<Self> {
        let mut functions = Vec::new();
        for (i, &name) in NAMES.iter().enumerate() {
            let module = stream
                .context()
                .load_module(Ptx::from_binary(native_binary(i)?.to_vec()))
                .w()?;
            let function = module.load_function(name).w()?;
            function
                .set_attribute(
                    CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                    native_shared(i) as i32,
                )
                .w()?;
            functions.push(function);
        }
        let initial = stream.alloc_zeros::<f32>(32 * 128 * 128).w()?;
        let bad = stream.alloc_zeros::<u32>(1).w()?;
        let qv_bad = stream.alloc_zeros::<u32>(1).w()?;
        Ok(Self {
            stream,
            functions,
            metadata: None,
            q: None,
            k: None,
            v: None,
            gate: None,
            beta: None,
            prefix: None,
            a: None,
            w: None,
            u: None,
            vnew: None,
            h: None,
            initial,
            bad,
            qv_bad,
        })
    }
}

static WORKSPACES: OnceLock<Vec<Mutex<Option<Workspace>>>> = OnceLock::new();

fn workspace_slot(stream: &Arc<CudaStream>) -> Result<&'static Mutex<Option<Workspace>>> {
    if WORKSPACES.get().is_none() {
        let count = candle_core::cuda_backend::cudarc::driver::CudaContext::device_count().w()?;
        let slots = (0..count).map(|_| Mutex::new(None)).collect();
        let _ = WORKSPACES.set(slots);
    }
    WORKSPACES
        .get()
        .and_then(|slots| slots.get(stream.context().ordinal()))
        .ok_or_else(|| {
            candle_core::Error::Msg("AOT CUDA device ordinal outside initialized cache".into())
        })
}

fn value_limit(lengths: &[usize]) -> f32 {
    (16384f32 * (0.75 / (*lengths.iter().max().unwrap() as f32)).sqrt()).min(1024.)
}

pub(crate) struct PreparedQv {
    slot: MutexGuard<'static, Option<Workspace>>,
    lengths: Vec<usize>,
    limit: f32,
    query: Option<Tensor>,
    key: Option<Tensor>,
    value: Option<Tensor>,
    producer: Option<(Tensor, Tensor)>,
    keep_cache: bool,
}

impl Drop for PreparedQv {
    fn drop(&mut self) {
        if !self.keep_cache {
            self.slot.take();
        }
    }
}

impl PreparedQv {
    pub(crate) fn limit(&self) -> f32 {
        self.limit
    }
    pub(crate) fn outputs(
        &mut self,
    ) -> (
        &mut CudaSlice<u16>,
        &mut CudaSlice<u16>,
        &mut CudaSlice<u16>,
        f32,
        &mut CudaSlice<u32>,
    ) {
        let workspace = self
            .slot
            .as_mut()
            .expect("prepared workspace remains owned");
        (
            workspace.q.as_mut().unwrap(),
            workspace.k.as_mut().unwrap(),
            workspace.v.as_mut().unwrap(),
            self.limit,
            &mut workspace.qv_bad,
        )
    }

    pub(crate) fn bind(&mut self, query: &Tensor, key: &Tensor, value: &Tensor) {
        self.query = Some(query.clone());
        self.key = Some(key.clone());
        self.value = Some(value.clone());
    }

    pub(crate) fn bind_producer(&mut self, source: &Tensor, weights: &Tensor) -> Result<()> {
        if self.query.is_some()
            || self.key.is_some()
            || self.value.is_some()
            || self.producer.is_some()
        {
            candle_core::bail!("prepared producer ticket already bound");
        }
        self.producer = Some((source.clone(), weights.clone()));
        Ok(())
    }

    fn matches_producer(&self, source: &Tensor, weights: &Tensor, lengths: &[usize]) -> bool {
        self.lengths == lengths
            && self.query.is_none()
            && self.key.is_none()
            && self.value.is_none()
            && self
                .producer
                .as_ref()
                .is_some_and(|(x, w)| x.id() == source.id() && w.id() == weights.id())
            && self.limit.to_bits() == value_limit(lengths).to_bits()
    }

    fn matches(&self, query: &Tensor, key: &Tensor, value: &Tensor, lengths: &[usize]) -> bool {
        self.lengths == lengths
            && self.query.as_ref().is_some_and(|q| q.id() == query.id())
            && self.key.as_ref().is_some_and(|k| k.id() == key.id())
            && self.value.as_ref().is_some_and(|v| v.id() == value.id())
            && self.limit.to_bits() == value_limit(lengths).to_bits()
    }
}

pub(crate) fn prepare_qv(device: &Device, lengths: &[usize]) -> Result<Option<PreparedQv>> {
    let Device::Cuda(device) = device else {
        return Ok(None);
    };
    if !matches!(option_env!("SYS1_CUDA_COMPUTE_CAP"), Some("80" | "90"))
        || lengths.is_empty()
        || lengths.contains(&0)
    {
        return Ok(None);
    }
    let Some(total) = lengths
        .iter()
        .try_fold(0usize, |sum, &n| sum.checked_add(n))
    else {
        return Ok(None);
    };
    if total == 0 || total > i32::MAX as usize / (32 * 128) {
        return Ok(None);
    }

    if option_env!("SYS1_CUDA_COMPUTE_CAP") == Some("80")
        && (total > 65536 || lengths.iter().any(|&n| n < 128))
    {
        return Ok(None);
    }
    let expected_major = if option_env!("SYS1_CUDA_COMPUTE_CAP") == Some("80") {
        8
    } else {
        9
    };
    let stream = device.cuda_stream();
    if stream
        .context()
        .attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)
        .w()?
        != expected_major
        || stream
            .context()
            .attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)
            .w()?
            != 0
    {
        return Ok(None);
    }
    let mut slot = workspace_slot(&stream)?
        .lock()
        .map_err(|_| candle_core::Error::Msg("poisoned AOT workspace".into()))?;
    if slot
        .as_ref()
        .is_none_or(|w| !Arc::ptr_eq(&w.stream, &stream))
    {
        match Workspace::new(stream.clone()) {
            Ok(workspace) => *slot = Some(workspace),
            Err(error) => {
                slot.take();
                return Err(error);
            }
        }
    }
    let allocation = (|| -> Result<()> {
        let workspace = slot.as_mut().unwrap();
        grow(&mut workspace.q, &stream, total * 16 * 128)?;
        grow(&mut workspace.k, &stream, total * 16 * 128)?;
        grow(&mut workspace.v, &stream, total * 32 * 128)?;
        stream.memset_zeros(&mut workspace.qv_bad).w()?;
        Ok(())
    })();
    if let Err(error) = allocation {
        slot.take();
        return Err(error);
    }
    Ok(Some(PreparedQv {
        slot,
        lengths: lengths.to_vec(),
        limit: value_limit(lengths),
        query: None,
        key: None,
        value: None,
        producer: None,
        keep_cache: false,
    }))
}

fn grow<'a, T: DeviceRepr>(
    slot: &'a mut Option<CudaSlice<T>>,
    stream: &Arc<CudaStream>,
    len: usize,
) -> Result<&'a mut CudaSlice<T>> {
    if slot.as_ref().is_none_or(|old| old.len() < len) {
        *slot = Some(unsafe { stream.alloc::<T>(len) }.w()?);
    }
    Ok(slot.as_mut().unwrap())
}

pub(crate) fn try_forward(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    g: &Tensor,
    beta: &Tensor,
    lengths: &[usize],
) -> Result<Option<Tensor>> {
    try_forward_impl(q, k, v, g, beta, lengths, None)
}

pub(crate) fn try_forward_prepared(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    g: &Tensor,
    beta: &Tensor,
    lengths: &[usize],
    prepared: PreparedQv,
) -> Result<Option<Tensor>> {
    try_forward_impl(q, k, v, g, beta, lengths, Some(prepared))
}

pub(crate) fn try_forward_prepared_source(
    source: &Tensor,
    weights: &Tensor,
    g: &Tensor,
    beta: &Tensor,
    lengths: &[usize],
    prepared: PreparedQv,
) -> Result<Option<Tensor>> {
    try_forward_inputs(
        None,
        g,
        beta,
        lengths,
        Some(prepared),
        Some((source, weights)),
    )
}

fn try_forward_impl(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    g: &Tensor,
    beta: &Tensor,
    lengths: &[usize],
    prepared: Option<PreparedQv>,
) -> Result<Option<Tensor>> {
    try_forward_inputs(Some((q, k, v)), g, beta, lengths, prepared, None)
}

fn try_forward_inputs(
    wide: Option<(&Tensor, &Tensor, &Tensor)>,
    g: &Tensor,
    beta: &Tensor,
    lengths: &[usize],
    mut prepared: Option<PreparedQv>,
    producer_inputs: Option<(&Tensor, &Tensor)>,
) -> Result<Option<Tensor>> {
    let Device::Cuda(device) = g.device() else {
        return Ok(None);
    };
    if g.dims().len() != 3
        || g.dtype() != DType::F32
        || beta.dtype() != DType::F32
        || !g.device().same_device(beta.device())
    {
        return Ok(None);
    }
    let total = g.dim(1)?;
    if total == 0
        || total > i32::MAX as usize / (32 * 128)
        || lengths.is_empty()
        || lengths.contains(&0)
        || lengths.iter().try_fold(0usize, |s, &n| s.checked_add(n)) != Some(total)
        || g.dims() != [1, total, 32]
        || beta.dims() != g.dims()
    {
        return Ok(None);
    }
    if let Some((q, k, v)) = wide {
        if [q, k, v]
            .iter()
            .any(|t| t.dtype() != DType::F32 || !g.device().same_device(t.device()))
            || q.dims() != [1, total, 16, 128]
            || k.dims() != q.dims()
            || v.dims() != [1, total, 32, 128]
        {
            return Ok(None);
        }
    } else {
        let Some((source, weights)) = producer_inputs else {
            return Ok(None);
        };
        if prepared.is_none()
            || source.dtype() != DType::BF16
            || weights.dtype() != DType::BF16
            || !g.device().same_device(source.device())
            || !g.device().same_device(weights.device())
            || source.dims() != [1, total, 8192]
            || weights.dims() != [8192, 1, 4]
        {
            return Ok(None);
        }
    }

    if option_env!("SYS1_CUDA_COMPUTE_CAP") == Some("80") {
        if total > 65536
            || lengths.iter().any(|&n| n < 128)
            || [g, beta]
                .iter()
                .any(|t| !t.layout().start_offset().is_multiple_of(4))
            || wide.is_some_and(|(q, k, v)| {
                [q, k, v]
                    .iter()
                    .any(|t| !t.layout().start_offset().is_multiple_of(4))
            })
        {
            return Ok(None);
        }
    } else if option_env!("SYS1_CUDA_COMPUTE_CAP") != Some("90") {
        return Ok(None);
    }
    let stream = device.cuda_stream();
    if let Some(ticket) = prepared.as_ref() {
        let bound = match (wide, producer_inputs) {
            (Some((q, k, v)), None) => ticket.matches(q, k, v, lengths),
            (None, Some((source, weights))) => ticket.matches_producer(source, weights, lengths),
            _ => false,
        };
        if !bound || !Arc::ptr_eq(&ticket.slot.as_ref().unwrap().stream, &stream) {
            candle_core::bail!("prepared Q/V tensor identity, stream or sequence mismatch");
        }
    } else if stream
        .context()
        .attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)
        .w()?
        != (if option_env!("SYS1_CUDA_COMPUTE_CAP") == Some("80") {
            8
        } else {
            9
        })
        || stream
            .context()
            .attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)
            .w()?
            != 0
    {
        return Ok(None);
    }
    let has_prepared = prepared.is_some();
    let result = (|| -> Result<Option<Tensor>> {
        let g = g.contiguous()?;
        let beta = beta.contiguous()?;
        let (gs, gl) = g.storage_and_layout();
        let (bs, bl) = beta.storage_and_layout();
        let (Storage::Cuda(gs), Storage::Cuda(bs)) = (&*gs, &*bs) else {
            candle_core::bail!("CUDA AOT gate storage required")
        };
        let offsets = |l: &candle_core::Layout| {
            l.contiguous_offsets()
                .ok_or_else(|| candle_core::Error::Msg("AOT input must be contiguous".into()))
        };
        let (a, b) = offsets(gl)?;
        let source_gate = gs.as_cuda_slice::<f32>()?.slice(a..b);
        let (a, b) = offsets(bl)?;
        let source_beta = bs.as_cuda_slice::<f32>()?.slice(a..b);
        let mut ordinary_slot;
        let workspace = if let Some(ticket) = prepared.as_mut() {
            ticket.slot.as_mut().unwrap()
        } else {
            ordinary_slot = workspace_slot(&stream)?
                .lock()
                .map_err(|_| candle_core::Error::Msg("poisoned AOT workspace".into()))?;
            if ordinary_slot
                .as_ref()
                .is_none_or(|w| !Arc::ptr_eq(&w.stream, &stream))
            {
                *ordinary_slot = Some(Workspace::new(stream.clone())?);
            }
            ordinary_slot.as_mut().unwrap()
        };
        if workspace
            .metadata
            .as_ref()
            .is_none_or(|m| m.lengths != lengths)
        {
            workspace.metadata = Some(Metadata::new(&stream, lengths)?);
        }
        let Workspace {
            functions,
            metadata,
            q: q16,
            k: k16,
            v: v16,
            gate,
            beta: beta16,
            prefix,
            a,
            w,
            u,
            vnew,
            h,
            initial,
            bad,
            qv_bad,
            ..
        } = workspace;
        let meta = metadata.as_ref().unwrap();
        let nt = meta.count;
        let qcount = total * 16 * 128;
        let vcount = total * 32 * 128;
        let gcount = total * 32;
        let q16 = grow(q16, &stream, qcount)?;
        let k16 = grow(k16, &stream, qcount)?;
        let v16 = grow(v16, &stream, vcount)?;
        let gate = grow(gate, &stream, gcount)?;
        let beta16 = grow(beta16, &stream, gcount)?;
        stream.memset_zeros(bad).w()?;
        let count_q = qcount as u64;
        let rows = (qcount / 128) as u64;
        let count_v = vcount as u64;

        let vector_cast =
            option_env!("SYS1_CUDA_COMPUTE_CAP") == Some("90") && (6080..=65536).contains(&total);
        #[cfg(test)]
        if vector_cast && !has_prepared {
            VECTOR_CAST_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }

        let limit = value_limit(lengths);
        if !has_prepared {
            let (q, k, v) = wide.ok_or_else(|| {
                candle_core::Error::Msg("ordinary AOT conversion requires wide inputs".into())
            })?;
            let q = q.contiguous()?;
            let k = k.contiguous()?;
            let v = v.contiguous()?;
            let (qs, ql) = q.storage_and_layout();
            let (ks, kl) = k.storage_and_layout();
            let (vs, vl) = v.storage_and_layout();
            let (Storage::Cuda(qs), Storage::Cuda(ks), Storage::Cuda(vs)) = (&*qs, &*ks, &*vs)
            else {
                candle_core::bail!("CUDA AOT input storage required")
            };
            let (a, b) = offsets(ql)?;
            let query = qs.as_cuda_slice::<f32>()?.slice(a..b);
            let (a, b) = offsets(kl)?;
            let key = ks.as_cuda_slice::<f32>()?.slice(a..b);
            let (a, b) = offsets(vl)?;
            let value = vs.as_cuda_slice::<f32>()?.slice(a..b);

            let f = device.get_or_load_custom_func(
                if vector_cast {
                    "qwen35_vector_cast_q134"
                } else {
                    "qwen35_aot91_cast_q"
                },
                "sys1_aot91_cast",
                super::CUDA_PTX,
            )?;
            let mut b = f.builder();
            b.arg(&query).arg(&mut *q16).arg(&count_q).arg(&mut *bad);
            unsafe {
                b.launch(LaunchConfig {
                    grid_dim: (
                        (qcount
                            .div_ceil(if vector_cast { 1024 } else { 256 })
                            .min(4096)) as u32,
                        1,
                        1,
                    ),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                })
            }
            .w()?;

            let f = device.get_or_load_custom_func(
                "qwen35_aot91_cast_k",
                "sys1_aot91_cast",
                super::CUDA_PTX,
            )?;
            let mut b = f.builder();
            b.arg(&key).arg(&mut *k16).arg(&rows).arg(&mut *bad);
            unsafe {
                b.launch(LaunchConfig {
                    grid_dim: (((qcount / 128).div_ceil(4).min(4096)) as u32, 1, 1),
                    block_dim: (128, 1, 1),
                    shared_mem_bytes: 0,
                })
            }
            .w()?;

            let f = device.get_or_load_custom_func(
                if vector_cast {
                    "qwen35_vector_cast"
                } else {
                    "qwen35_aot91_cast_v"
                },
                "sys1_aot91_cast",
                super::CUDA_PTX,
            )?;
            let mut b = f.builder();
            b.arg(&value)
                .arg(&source_gate)
                .arg(&source_beta)
                .arg(&mut *v16)
                .arg(&mut *gate)
                .arg(&mut *beta16)
                .arg(&count_v)
                .arg(&limit)
                .arg(&mut *bad);
            unsafe {
                b.launch(LaunchConfig {
                    grid_dim: (
                        (vcount
                            .div_ceil(if vector_cast { 1024 } else { 256 })
                            .min(4096)) as u32,
                        1,
                        1,
                    ),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                })
            }
            .w()?;
        } else {
            let f = device.get_or_load_custom_func(
                "qwen35_convcast_gbeta_merge",
                "sys1_conv_qv_prepared",
                super::CUDA_PTX,
            )?;
            let mut b = f.builder();
            let heads = gcount as u64;
            b.arg(&source_gate)
                .arg(&source_beta)
                .arg(&mut *gate)
                .arg(&mut *beta16)
                .arg(&heads)
                .arg(&*qv_bad)
                .arg(&mut *bad);
            unsafe {
                b.launch(LaunchConfig {
                    grid_dim: (gcount.div_ceil(256).min(4096) as u32, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                })
            }
            .w()?;
        }

        if stream.clone_dtoh(&*bad).w()?[0] != 0 {
            #[cfg(test)]
            if has_prepared {
                PREPARED_QV_REJECTED.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
            return Ok(None);
        }
        #[cfg(test)]
        if has_prepared {
            PREPARED_QV_ACCEPTED.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
        let prefix = grow(prefix, &stream, gcount)?;
        let a = grow(a, &stream, gcount * 64)?;
        let w = grow(w, &stream, vcount)?;
        let u = grow(u, &stream, vcount)?;
        let vnew = grow(vnew, &stream, vcount)?;
        let h = grow(h, &stream, nt * 32 * 128 * 128)?;
        stream.memset_zeros(&mut a.slice_mut(..gcount * 64)).w()?;
        let mut output = unsafe { stream.alloc::<half::bf16>(vcount) }.w()?;
        let t = total as i32;
        let zero = 0u64;
        let stride = 32 * 128 * 128;
        let track = 0i32;
        let scale = 1f32;
        macro_rules! go {($i:expr,$grid:expr,$($arg:expr),+)=>{{
            let mut b=stream.launch_builder(&functions[$i]);$(b.arg($arg);)+
            unsafe{b.launch(LaunchConfig{grid_dim:$grid,block_dim:(native_threads($i),1,1),shared_mem_bytes:native_shared($i)})}.w()?;
        }}}
        go!(
            0,
            (nt as u32, 32, 1),
            &*gate,
            &mut *prefix,
            &meta.cumulative,
            &meta.chunks,
            &t,
            &zero,
            &zero
        );
        go!(
            1,
            (nt as u32, 32, 1),
            &*k16,
            &*prefix,
            &*beta16,
            &mut *a,
            &meta.cumulative,
            &meta.chunks,
            &t,
            &zero,
            &zero
        );
        go!(
            2,
            (nt as u32, 32, 1),
            &*k16,
            &*v16,
            &*beta16,
            &mut *w,
            &mut *u,
            &*a,
            &*prefix,
            &meta.cumulative,
            &meta.chunks,
            &t,
            &zero,
            &zero
        );
        go!(
            3,
            (4, (lengths.len() * 32) as u32, 1),
            &*k16,
            &*u,
            &*w,
            &mut *vnew,
            &*prefix,
            &mut *h,
            &*initial,
            &meta.initial_ids,
            &stride,
            &meta.cumulative,
            &meta.offsets,
            &track,
            &t,
            &zero,
            &zero
        );
        go!(
            4,
            (2, nt as u32, 32),
            &*q16,
            &*k16,
            &*vnew,
            &*h,
            &*prefix,
            &mut output,
            &meta.cumulative,
            &meta.chunks,
            &scale,
            &t,
            &zero,
            &zero
        );
        let output = Tensor::from_storage(
            Storage::Cuda(CudaStorage::wrap_cuda_slice(output, device.clone())),
            (1, total, 32, 128),
            BackpropOp::none(),
            false,
        );
        Ok(Some(output))
    })();
    if matches!(&result, Ok(Some(_)))
        && let Some(ticket) = prepared.as_mut()
    {
        ticket.keep_cache = true;
    }

    drop(prepared);

    if !matches!(&result, Ok(Some(_)))
        && let Ok(mut slot) = workspace_slot(&stream)?.lock()
    {
        slot.take();
    }
    result
}

#[cfg(test)]
static VECTOR_CAST_CALLS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

#[cfg(test)]
static PREPARED_QV_ACCEPTED: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);
#[cfg(test)]
static PREPARED_QV_REJECTED: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

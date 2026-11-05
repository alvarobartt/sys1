use candle_core::cuda_backend::cudarc::{
    driver::{CudaContext, CudaFunction, CudaStream, sys},
    nvrtc::Ptx,
};
use candle_core::cuda_backend::{
    WrapErr,
    cudarc::driver::{DevicePtr, DevicePtrMut},
};
use candle_core::{CudaStorage, DType, Device, Result, Storage, Tensor, op::BackpropOp};
use std::ffi::c_void;
use std::sync::{Arc, Mutex, OnceLock};

mod abi;

const IMAGE: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/fa4_sm90_bf16_d256.fatbin"));
const KERNEL: &str = "kernel_cutlass_kernel_flash_attncuteflash_fwd_sm90FlashAttentionForwardSm90_object_at__tensor000o111012_tensor000o111012_tensor000o111012_tensor0000o1073741824214748364711101213_None_tens_0";

struct Loaded {
    context: Arc<CudaContext>,
    function: CudaFunction,
}

static MODULES: OnceLock<Vec<Mutex<Option<Loaded>>>> = OnceLock::new();

fn function(stream: &Arc<CudaStream>) -> Result<CudaFunction> {
    if MODULES.get().is_none() {
        let count = CudaContext::device_count().w()?;
        let _ = MODULES.set((0..count).map(|_| Mutex::new(None)).collect());
    }
    let context = stream.context();
    context.bind_to_thread().w()?;
    let mut slot = MODULES
        .get()
        .and_then(|s| s.get(context.ordinal()))
        .ok_or_else(|| {
            candle_core::Error::Msg("FA4 device outside initialized module cache".into())
        })?
        .lock()
        .map_err(|_| candle_core::Error::Msg("poisoned FA4 module cache".into()))?;
    if slot
        .as_ref()
        .is_none_or(|old| !Arc::ptr_eq(&old.context, context))
    {
        let module = context.load_module(Ptx::from_binary(IMAGE.to_vec())).w()?;
        let function = module.load_function(KERNEL).w()?;
        let optin = context
            .attribute(
                sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN,
            )
            .w()?;
        let fixed = function.shared_size_bytes().w()?;
        let maximum = optin.checked_sub(fixed).ok_or_else(|| {
            candle_core::Error::Msg("invalid FA4 shared-memory attributes".into())
        })?;
        if maximum < abi::SHARED_BYTES as i32 {
            candle_core::bail!(
                "FA4 device shared-memory capacity is below the original launch requirement"
            )
        }
        function
            .set_attribute(
                sys::CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                maximum,
            )
            .w()?;
        function
            .set_attribute(
                sys::CUfunction_attribute::CU_FUNC_ATTRIBUTE_NON_PORTABLE_CLUSTER_SIZE_ALLOWED,
                1,
            )
            .w()?;

        for (index, &(expected_offset, expected_size)) in abi::PARAM_LAYOUT.iter().enumerate() {
            let (mut offset, mut size) = (0usize, 0usize);
            unsafe {
                sys::cuFuncGetParamInfo(function.cu_function(), index, &mut offset, &mut size)
                    .result()
            }
            .w()?;
            if (offset, size) != (expected_offset, expected_size) {
                candle_core::bail!(
                    "FA4 device parameter {index} does not match the frozen host ABI"
                )
            }
        }
        *slot = Some(Loaded {
            context: context.clone(),
            function,
        });
    }
    Ok(slot.as_ref().unwrap().function.clone())
}

fn launch(stream: &Arc<CudaStream>, packed: &abi::Packed) -> Result<()> {
    let function = function(stream)?;
    stream.context().bind_to_thread().w()?;

    let mut attributes: [sys::CUlaunchAttribute; 2] = unsafe { std::mem::zeroed() };
    attributes[0].id =
        sys::CUlaunchAttributeID::CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION;
    attributes[1].id = sys::CUlaunchAttributeID::CU_LAUNCH_ATTRIBUTE_COOPERATIVE;
    let config = sys::CUlaunchConfig {
        gridDimX: packed.grid[0],
        gridDimY: packed.grid[1],
        gridDimZ: packed.grid[2],
        blockDimX: abi::BLOCK[0],
        blockDimY: abi::BLOCK[1],
        blockDimZ: abi::BLOCK[2],
        sharedMemBytes: abi::SHARED_BYTES,
        hStream: stream.cu_stream(),
        attrs: attributes.as_mut_ptr(),
        numAttrs: attributes.len() as u32,
    };
    let mut bytes = packed.bytes().len();
    let mut extra = [
        sys::CU_LAUNCH_PARAM_BUFFER_POINTER_AS_INT as usize as *mut c_void,
        packed.bytes().as_ptr() as *mut c_void,
        sys::CU_LAUNCH_PARAM_BUFFER_SIZE_AS_INT as usize as *mut c_void,
        (&mut bytes as *mut usize).cast::<c_void>(),
        sys::CU_LAUNCH_PARAM_END_AS_INT as usize as *mut c_void,
    ];

    unsafe {
        sys::cuLaunchKernelEx(
            &config,
            function.cu_function(),
            std::ptr::null_mut(),
            extra.as_mut_ptr(),
        )
        .result()
    }
    .w()?;
    Ok(())
}

fn eligible(q: &Tensor, k: &Tensor, v: &Tensor, scale: f32, causal: bool) -> Result<bool> {
    if !causal
        || !scale.is_finite()
        || scale != 0.0625
        || q.dtype() != DType::BF16
        || k.dtype() != DType::BF16
        || v.dtype() != DType::BF16
        || !q.device().same_device(k.device())
        || !q.device().same_device(v.device())
        || !q.is_contiguous()
        || !k.is_contiguous()
        || !v.is_contiguous()
    {
        return Ok(false);
    }
    let Device::Cuda(device) = q.device() else {
        return Ok(false);
    };
    let cap = device.cuda_stream().context().compute_capability().w()?;
    if cap != (9, 0) {
        return Ok(false);
    }
    let (tokens, heads, width) = q.dims3()?;
    if tokens == 0
        || tokens > i32::MAX as usize
        || heads != 16
        || width != 256
        || k.dims() != [tokens, 4, 256]
        || v.dims() != [tokens, 4, 256]
    {
        return Ok(false);
    }

    Ok([q, k, v].iter().all(|t| t.layout().start_offset() % 8 == 0))
}

pub(super) fn varlen(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    cuq: &Tensor,
    cuk: &Tensor,
    maximum_q: usize,
    maximum_k: usize,
    scale: f32,
    causal: bool,
) -> Result<Option<Tensor>> {
    if !eligible(q, k, v, scale, causal)?
        || cuq.id() != cuk.id()
        || maximum_q != maximum_k
        || !matches!(cuq.dtype(), DType::U32 | DType::I32)
        || !cuq.is_contiguous()
        || !q.device().same_device(cuq.device())
    {
        return Ok(None);
    }
    let count = cuq.dim(0)?;
    if cuq.rank() != 1 || count < 2 || count > i32::MAX as usize {
        return Ok(None);
    }
    let (qs, ql) = q.storage_and_layout();
    let (ks, kl) = k.storage_and_layout();
    let (vs, vl) = v.storage_and_layout();
    let (cs, cl) = cuq.storage_and_layout();
    let (Storage::Cuda(qs), Storage::Cuda(ks), Storage::Cuda(vs), Storage::Cuda(cs)) =
        (&*qs, &*ks, &*vs, &*cs)
    else {
        candle_core::bail!("native FA4 CUDA storage required")
    };
    let device = q.device().as_cuda_device()?;
    let stream = device.cuda_stream();
    let bounds = |l: &candle_core::Layout| {
        l.contiguous_offsets()
            .ok_or_else(|| candle_core::Error::Msg("native FA4 contiguous bounds required".into()))
    };
    let (start, end) = bounds(ql)?;
    let query = qs.as_cuda_slice::<half::bf16>()?.slice(start..end);
    let (start, end) = bounds(kl)?;
    let key = ks.as_cuda_slice::<half::bf16>()?.slice(start..end);
    let (start, end) = bounds(vl)?;
    let value = vs.as_cuda_slice::<half::bf16>()?.slice(start..end);
    let (qp, _qguard) = query.device_ptr(&stream);
    let (kp, _kguard) = key.device_ptr(&stream);
    let (vp, _vguard) = value.device_ptr(&stream);
    let mut output = unsafe { device.alloc::<half::bf16>(q.elem_count())? };
    let (op, _oguard) = output.device_ptr_mut(&stream);
    let (start, end) = bounds(cl)?;
    let invoke = |ptr: u64| -> Result<()> {
        let packed = abi::Packed::new(qp, kp, vp, op, ptr, q.dim(0)?, count - 1, scale)
            .map_err(|error| candle_core::Error::Msg(error.into()))?;
        launch(&stream, &packed)
    };
    match cuq.dtype() {
        DType::U32 => {
            let slice = cs.as_cuda_slice::<u32>()?.slice(start..end);
            let (ptr, _guard) = slice.device_ptr(&stream);
            invoke(ptr)?;
        }
        DType::I32 => {
            let slice = cs.as_cuda_slice::<i32>()?.slice(start..end);
            let (ptr, _guard) = slice.device_ptr(&stream);
            invoke(ptr)?;
        }
        _ => unreachable!(),
    }
    drop(_oguard);
    Ok(Some(Tensor::from_storage(
        Storage::Cuda(CudaStorage::wrap_cuda_slice(output, device.clone())),
        q.shape(),
        BackpropOp::none(),
        false,
    )))
}

struct Uniform {
    device: Device,
    batch: usize,
    length: usize,
    cumulative: Tensor,
}
static UNIFORM: OnceLock<Mutex<Option<Uniform>>> = OnceLock::new();

pub(super) fn uniform(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    scale: f32,
    causal: bool,
) -> Result<Option<Tensor>> {
    let (batch, length, heads, width) = q.dims4()?;
    if k.dims() != [batch, length, 4, 256]
        || v.dims() != [batch, length, 4, 256]
        || heads != 16
        || width != 256
        || !q.is_contiguous()
        || !k.is_contiguous()
        || !v.is_contiguous()
        || batch == 0
        || length == 0
        || batch
            .checked_mul(length)
            .is_none_or(|n| n > i32::MAX as usize)
    {
        return Ok(None);
    }
    let flat = q.reshape((batch * length, 16, 256))?;
    let fk = k.reshape((batch * length, 4, 256))?;
    let fv = v.reshape((batch * length, 4, 256))?;
    if !eligible(&flat, &fk, &fv, scale, causal)? {
        return Ok(None);
    }
    let cumulative = {
        let mut slot = UNIFORM
            .get_or_init(|| Mutex::new(None))
            .lock()
            .map_err(|_| candle_core::Error::Msg("poisoned FA4 uniform metadata cache".into()))?;
        if slot.as_ref().is_none_or(|s| {
            !s.device.same_device(q.device()) || s.batch != batch || s.length != length
        }) {
            *slot = Some(Uniform {
                device: q.device().clone(),
                batch,
                length,
                cumulative: Tensor::from_vec(
                    (0..=batch).map(|i| (i * length) as u32).collect::<Vec<_>>(),
                    batch + 1,
                    q.device(),
                )?,
            });
        }
        slot.as_ref().unwrap().cumulative.clone()
    };
    varlen(
        &flat,
        &fk,
        &fv,
        &cumulative,
        &cumulative,
        length,
        length,
        scale,
        causal,
    )?
    .map(|t| t.reshape(q.shape()))
    .transpose()
}

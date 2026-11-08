use candle_core::{DType, Result, Tensor};

pub(crate) fn cuda_bf16_activation_original(
    input: &Tensor,
    up: Option<&Tensor>,
    mode: i32,
) -> Result<Tensor> {
    use candle_core::{
        CudaStorage, Storage,
        backend::BackendStorage,
        cuda_backend::{
            WrapErr,
            cudarc::driver::{LaunchConfig, PushKernelArg},
        },
        op::BackpropOp,
    };
    use half::bf16;
    if input.dtype() != DType::BF16 || !(0..=2).contains(&mode) || (mode == 2) != up.is_some() {
        candle_core::bail!("unsupported BF16 activation mode/input");
    }
    if let Some(up) = up
        && (up.dims() != input.dims()
            || up.dtype() != input.dtype()
            || !up.device().same_device(input.device()))
    {
        candle_core::bail!("SiLU product requires identical colocated BF16 shapes");
    }
    if input.elem_count() == 0 {
        return Ok(input.clone());
    }
    let count = u32::try_from(input.elem_count())
        .map_err(|_| candle_core::Error::Msg("activation count overflow".into()))?;
    let input = input.contiguous()?;
    let up = up.unwrap_or(&input).contiguous()?;
    let (xs, xl) = input.storage_and_layout();
    let (us, ul) = up.storage_and_layout();
    let (Storage::Cuda(xs), Storage::Cuda(us)) = (&*xs, &*us) else {
        candle_core::bail!("BF16 activation requires CUDA storage");
    };
    let (xstart, xend) = xl
        .contiguous_offsets()
        .ok_or_else(|| candle_core::Error::Msg("activation input noncontiguous".into()))?;
    let (ustart, uend) = ul
        .contiguous_offsets()
        .ok_or_else(|| candle_core::Error::Msg("activation up noncontiguous".into()))?;
    let x = xs.as_cuda_slice::<bf16>()?.slice(xstart..xend);
    let u = us.as_cuda_slice::<bf16>()?.slice(ustart..uend);
    let device = xs.device().clone();
    let output = unsafe { device.alloc::<bf16>(count as usize)? };
    let function = device.get_or_load_custom_func(
        "activation_bf16",
        "sys1_activation",
        include_str!(concat!(env!("OUT_DIR"), "/cuda.ptx")),
    )?;
    let mut launch = function.builder();
    launch.arg(&x);
    launch.arg(&u);
    launch.arg(&output);
    launch.arg(&count);
    launch.arg(&mode);
    unsafe { launch.launch(LaunchConfig::for_num_elems(count)) }.w()?;
    Ok(Tensor::from_storage(
        Storage::Cuda(CudaStorage::wrap_cuda_slice(output, device)),
        input.shape().clone(),
        BackpropOp::none(),
        false,
    ))
}

pub(crate) fn cuda_bf16_activation_lookup(
    input: &Tensor,
    up: Option<&Tensor>,
    mode: i32,
) -> Result<Tensor> {
    use candle_core::{
        CudaStorage, Storage,
        backend::BackendStorage,
        cuda_backend::{
            WrapErr,
            cudarc::driver::{LaunchConfig, PushKernelArg},
        },
        op::BackpropOp,
    };
    use half::bf16;
    if input.dtype() != DType::BF16 || !(0..=2).contains(&mode) || (mode == 2) != up.is_some() {
        candle_core::bail!("unsupported BF16 activation mode/input");
    }
    if let Some(up) = up
        && (up.dims() != input.dims()
            || up.dtype() != input.dtype()
            || !up.device().same_device(input.device()))
    {
        candle_core::bail!("SiLU product requires identical colocated BF16 shapes");
    }
    if input.elem_count() == 0 {
        return Ok(input.clone());
    }
    let count = u32::try_from(input.elem_count())
        .map_err(|_| candle_core::Error::Msg("activation count overflow".into()))?;
    let input = input.contiguous()?;
    let up = up.unwrap_or(&input).contiguous()?;
    let (xs, xl) = input.storage_and_layout();
    let (us, ul) = up.storage_and_layout();
    let (Storage::Cuda(xs), Storage::Cuda(us)) = (&*xs, &*us) else {
        candle_core::bail!("BF16 activation requires CUDA storage");
    };
    let (xstart, xend) = xl
        .contiguous_offsets()
        .ok_or_else(|| candle_core::Error::Msg("activation input noncontiguous".into()))?;
    let (ustart, uend) = ul
        .contiguous_offsets()
        .ok_or_else(|| candle_core::Error::Msg("activation up noncontiguous".into()))?;
    let x = xs.as_cuda_slice::<bf16>()?.slice(xstart..xend);
    let u = us.as_cuda_slice::<bf16>()?.slice(ustart..uend);
    let device = xs.device().clone();
    let mut output = unsafe { device.alloc::<bf16>(count as usize)? };
    let stream = device.cuda_stream();
    let mut slot = ACTIVATION_LOOKUP
        .lock()
        .map_err(|_| candle_core::Error::Msg("poisoned activation lookup".into()))?;
    if slot
        .as_ref()
        .is_none_or(|old| !std::sync::Arc::ptr_eq(&old.stream, &stream))
    {
        let mut table = unsafe { device.alloc::<bf16>(131072)? };
        let build = device.get_or_load_custom_func(
            "build_activation_lookup_bf16",
            "sys1_activation_lookup",
            include_str!(concat!(env!("OUT_DIR"), "/cuda.ptx")),
        )?;
        let mut init = build.builder();
        init.arg(&mut table);
        unsafe { init.launch(LaunchConfig::for_num_elems(131072)) }.w()?;
        *slot = Some(ActivationLookup { stream, table });
    }
    let table = &slot
        .as_ref()
        .ok_or_else(|| candle_core::Error::Msg("missing lookup table".into()))?
        .table;
    let function = device.get_or_load_custom_func(
        "activation_lookup_bf16",
        "sys1_activation_lookup",
        include_str!(concat!(env!("OUT_DIR"), "/cuda.ptx")),
    )?;
    let mut launch = function.builder();
    launch.arg(&x);
    launch.arg(&u);
    launch.arg(table);
    launch.arg(&mut output);
    launch.arg(&count);
    launch.arg(&mode);
    unsafe { launch.launch(LaunchConfig::for_num_elems(count.div_ceil(8))) }.w()?;
    Ok(Tensor::from_storage(
        Storage::Cuda(CudaStorage::wrap_cuda_slice(output, device)),
        input.shape().clone(),
        BackpropOp::none(),
        false,
    ))
}

pub(crate) struct ActivationLookup {
    pub(crate) stream: std::sync::Arc<candle_core::cuda_backend::cudarc::driver::CudaStream>,
    pub(crate) table: candle_core::cuda_backend::cudarc::driver::CudaSlice<half::bf16>,
}

pub(crate) static ACTIVATION_LOOKUP: std::sync::Mutex<Option<ActivationLookup>> =
    std::sync::Mutex::new(None);

pub(crate) fn cuda_bf16_activation(
    input: &Tensor,
    up: Option<&Tensor>,
    mode: i32,
) -> Result<Tensor> {
    if matches!(
        option_env!("SYS1_CUDA_COMPUTE_CAP"),
        Some("80" | "89" | "90" | "120")
    ) {
        cuda_bf16_activation_lookup(input, up, mode)
    } else {
        cuda_bf16_activation_original(input, up, mode)
    }
}

pub(crate) fn with_sigmoid_lookup<R>(
    device: &candle_core::CudaDevice,
    run: impl FnOnce(&candle_core::cuda_backend::cudarc::driver::CudaSlice<half::bf16>) -> Result<R>,
) -> Result<R> {
    use candle_core::cuda_backend::{
        WrapErr,
        cudarc::driver::{LaunchConfig, PushKernelArg},
    };
    let stream = device.cuda_stream();
    let mut slot = ACTIVATION_LOOKUP
        .lock()
        .map_err(|_| candle_core::Error::Msg("poisoned activation lookup".into()))?;
    if slot
        .as_ref()
        .is_none_or(|old| !std::sync::Arc::ptr_eq(&old.stream, &stream))
    {
        let mut table = unsafe { device.alloc::<half::bf16>(131072)? };
        let build = device.get_or_load_custom_func(
            "build_activation_lookup_bf16",
            "sys1_activation_lookup",
            include_str!(concat!(env!("OUT_DIR"), "/cuda.ptx")),
        )?;
        let mut init = build.builder();
        init.arg(&mut table);
        unsafe { init.launch(LaunchConfig::for_num_elems(131072)) }.w()?;
        *slot = Some(ActivationLookup { stream, table });
    }
    run(&slot
        .as_ref()
        .ok_or_else(|| candle_core::Error::Msg("missing lookup table".into()))?
        .table)
}

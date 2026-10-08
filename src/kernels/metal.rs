use candle_core::{
    DType, Error, MetalDevice, MetalStorage, Result, Storage, Tensor, backend::BackendStorage,
    op::BackpropOp,
};
use candle_metal_kernels::metal::ComputePipeline;
use candle_metal_kernels::metal::Library;
use dispatch2::DispatchData;
use objc2_metal::MTLDevice;
use objc2_metal::MTLSize;
use std::{
    collections::HashMap,
    sync::{Mutex, OnceLock},
};

const LIBRARY_BYTES: &[u8] = include_bytes!(env!("SYS1_KERNEL_METALLIB_PATH"));
type PipelineCache = HashMap<(candle_core::metal_backend::DeviceId, &'static str), ComputePipeline>;
static PIPELINES: OnceLock<Mutex<PipelineCache>> = OnceLock::new();

fn pipeline(device: &MetalDevice, name: &'static str) -> Result<ComputePipeline> {
    let mut cache = PIPELINES
        .get_or_init(|| Mutex::new(HashMap::new()))
        .lock()
        .map_err(|e| Error::Msg(format!("Metal pipeline cache: {e}")))?;
    if let Some(pipeline) = cache.get(&(device.id(), name)) {
        return Ok(pipeline.clone());
    }
    let library = {
        let data = DispatchData::from_static_bytes(LIBRARY_BYTES);
        let raw = device
            .metal_device()
            .as_ref()
            .newLibraryWithData_error(&data)
            .map_err(|e| Error::Msg(format!("Loading Qwen3.5 metallib: {e}")))?;
        Library::new(raw)
    };
    let function = library.get_function(name, None).map_err(Error::wrap)?;
    let pipeline = device
        .metal_device()
        .new_compute_pipeline_state_with_function(&function)
        .map_err(Error::wrap)?;
    cache.insert((device.id(), name), pipeline.clone());
    Ok(pipeline)
}

fn contiguous_storage(tensor: &Tensor) -> Result<(Tensor, MetalStorage, usize)> {
    let tensor = tensor.contiguous()?;
    let (storage, layout) = tensor.storage_and_layout();
    let Storage::Metal(storage) = &*storage else {
        candle_core::bail!("Qwen3.5 Metal kernel requires Metal storage")
    };
    let (start, _) = layout
        .contiguous_offsets()
        .ok_or_else(|| Error::Msg("Qwen3.5 Metal tensor is not contiguous".into()))?;
    Ok((
        tensor.clone(),
        storage.clone(),
        start * tensor.dtype().size_in_bytes(),
    ))
}

pub(crate) fn conv_silu(input: &Tensor, weights: &Tensor) -> Result<Tensor> {
    let (batch, length, channels) = input.dims3()?;
    let (weight_channels, groups, taps) = weights.dims3()?;
    if channels != weight_channels || groups != 1 || taps == 0 || input.dtype() != weights.dtype() {
        candle_core::bail!("Unsupported Qwen3.5 Metal convolution shape or dtype")
    }
    let name = match input.dtype() {
        DType::F32 => "qwen35_conv_f32",
        DType::F16 => "qwen35_conv_f16",
        DType::BF16 => "qwen35_conv_bf16",
        dtype => candle_core::bail!("Unsupported Qwen3.5 Metal convolution dtype: {dtype:?}"),
    };
    let (_input, input_storage, input_offset) = contiguous_storage(input)?;
    let (_weights, weight_storage, weight_offset) = contiguous_storage(weights)?;
    let device = input_storage.device().clone();
    let count = batch * length * channels;
    let buffer = device.new_buffer(count, input.dtype(), "qwen35-conv")?;
    let pipeline = pipeline(&device, name)?;
    let encoder = device.command_encoder()?;
    encoder.set_label(name);
    encoder.set_compute_pipeline_state(&pipeline);
    encoder
        .as_ref()
        .set_input_buffer(0, Some(input_storage.buffer()), input_offset);
    encoder
        .as_ref()
        .set_input_buffer(1, Some(weight_storage.buffer()), weight_offset);
    encoder.as_ref().set_output_buffer(2, Some(&buffer), 0);
    for (index, value) in [count, length, channels, taps].into_iter().enumerate() {
        encoder.as_ref().set_bytes(index + 3, &(value as u32));
    }
    encoder.as_ref().dispatch_threads(
        MTLSize {
            width: count,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: 256,
            height: 1,
            depth: 1,
        },
    );
    drop(encoder);
    Ok(Tensor::from_storage(
        Storage::Metal(MetalStorage::new(buffer, device, count, input.dtype())),
        (batch, length, channels),
        BackpropOp::none(),
        false,
    ))
}

pub(crate) fn delta_rule(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    g: &Tensor,
    beta: &Tensor,
) -> Result<Tensor> {
    let (batch, heads, length, key_dim) = q.dims4()?;
    let value_dim = v.dim(3)?;
    if key_dim != 128
        || q.dtype() != DType::F32
        || k.dims() != q.dims()
        || v.dims() != [batch, heads, length, value_dim]
        || g.dims() != [batch, heads, length]
        || beta.dims() != [batch, heads, length]
    {
        candle_core::bail!("Unsupported Qwen3.5 Metal delta-rule shape or dtype")
    }
    let packed = if value_dim == 128 {
        pack_delta_f32(q, k, v, g, beta)?
    } else {
        Tensor::cat(&[q, k, v, &g.unsqueeze(3)?, &beta.unsqueeze(3)?], 3)?
    };
    delta_rule_packed(&packed, value_dim)
}

pub(crate) fn pack_delta_f32(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    g: &Tensor,
    beta: &Tensor,
) -> Result<Tensor> {
    let (batch, key_heads, length, key_dim) = q.dims4()?;
    let (_, heads, _, _) = v.dims4()?;
    if key_dim != 128
        || k.dims() != q.dims()
        || key_heads == 0
        || heads % key_heads != 0
        || v.dims() != [batch, heads, length, 128]
        || g.dims() != [batch, heads, length]
        || beta.dims() != g.dims()
        || [q, k, v, g, beta]
            .iter()
            .any(|tensor| tensor.dtype() != DType::F32 || !tensor.device().is_metal())
    {
        candle_core::bail!("Unsupported Qwen3.5 Metal delta packing shape or dtype")
    }
    let device = q.device().as_metal_device()?.clone();
    let count = batch * heads * length * 386;
    let buffer = device.new_buffer(count, DType::F32, "qwen35-delta-packed")?;
    let pipeline = pipeline(&device, "qwen35_pack_delta_f32")?;
    let encoder = device.command_encoder()?;
    encoder.set_label("qwen35_pack_delta_f32");
    encoder.set_compute_pipeline_state(&pipeline);
    for (index, tensor) in [q, k, v, g, beta].into_iter().enumerate() {
        let (storage, layout) = tensor.storage_and_layout();
        let Storage::Metal(storage) = &*storage else {
            candle_core::bail!("Qwen3.5 delta packing requires Metal storage")
        };
        encoder.as_ref().set_input_buffer(
            index,
            Some(storage.buffer()),
            layout.start_offset() * DType::F32.size_in_bytes(),
        );
        let strides = layout.stride();
        let strides = [
            strides[0] as u32,
            strides[1] as u32,
            strides[2] as u32,
            strides.get(3).copied().unwrap_or(0) as u32,
        ];
        encoder.as_ref().set_bytes(index + 7, &strides);
    }
    encoder.as_ref().set_output_buffer(5, Some(&buffer), 0);
    let dims = [batch as u32, heads as u32, length as u32, key_heads as u32];
    encoder.as_ref().set_bytes(6, &dims);
    encoder.as_ref().dispatch_thread_groups(
        MTLSize {
            width: batch * heads * length,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: 128,
            height: 1,
            depth: 1,
        },
    );
    drop(encoder);
    Ok(Tensor::from_storage(
        Storage::Metal(MetalStorage::new(buffer, device, count, DType::F32)),
        (batch, heads, length, 386),
        BackpropOp::none(),
        false,
    ))
}

pub(crate) fn delta_rule_packed(packed: &Tensor, value_dim: usize) -> Result<Tensor> {
    let (batch, heads, length, width) = packed.dims4()?;
    if packed.dtype() != DType::F32 || width != 258 + value_dim {
        candle_core::bail!("Unsupported Qwen3.5 packed delta-rule shape or dtype")
    }
    let (_packed, storage, offset) = contiguous_storage(&packed)?;
    let device = storage.device().clone();
    let count = batch * heads * length * value_dim;
    let buffer = device.new_buffer(count, DType::F32, "qwen35-delta")?;
    let pipeline = pipeline(&device, "qwen35_delta_f32")?;
    let encoder = device.command_encoder()?;
    encoder.set_label("qwen35_delta_f32");
    encoder.set_compute_pipeline_state(&pipeline);
    encoder
        .as_ref()
        .set_input_buffer(0, Some(storage.buffer()), offset);
    encoder.as_ref().set_output_buffer(1, Some(&buffer), 0);
    encoder.as_ref().set_bytes(2, &(length as u32));
    encoder.as_ref().set_bytes(3, &(value_dim as u32));
    encoder.as_ref().dispatch_thread_groups(
        MTLSize {
            width: batch * heads,
            height: value_dim.div_ceil(4),
            depth: 1,
        },
        MTLSize {
            width: 128,
            height: 1,
            depth: 1,
        },
    );
    drop(encoder);
    Tensor::from_storage(
        Storage::Metal(MetalStorage::new(buffer, device, count, DType::F32)),
        (batch, heads, length, value_dim),
        BackpropOp::none(),
        false,
    )
    .transpose(1, 2)
}

pub(crate) fn post_finalize_bf16(
    output: &Tensor,
    denominator: &Tensor,
    gate_silu: &Tensor,
    norm: &Tensor,
) -> Result<Tensor> {
    let (batch, length, heads, width) = output.dims4()?;
    if output.dtype() != DType::F32
        || width != 128
        || denominator.dims() != [batch, length, heads, 1]
        || gate_silu.dims() != output.dims()
        || norm.dims() != [width]
        || [denominator, gate_silu, norm]
            .iter()
            .any(|tensor| tensor.dtype() != DType::F32)
    {
        candle_core::bail!("Unsupported Qwen3.5 post-finalize shape or dtype")
    }
    let (output_storage, layout) = output.storage_and_layout();
    let Storage::Metal(output_storage) = &*output_storage else {
        candle_core::bail!("Qwen3.5 post-finalize requires Metal storage")
    };
    let (_denominator, denominator_storage, denominator_offset) = contiguous_storage(denominator)?;
    let (_gate_silu, gate_storage, gate_offset) = contiguous_storage(gate_silu)?;
    let (_norm, norm_storage, norm_offset) = contiguous_storage(norm)?;
    let device = output_storage.device().clone();
    let count = batch * length * heads * width;
    let buffer = device.new_buffer(count, DType::BF16, "qwen35-post-finalize")?;
    let pipeline = pipeline(&device, "qwen35_post_finalize_bf16")?;
    let encoder = device.command_encoder()?;
    encoder.set_label("qwen35_post_finalize_bf16");
    encoder.set_compute_pipeline_state(&pipeline);
    encoder.as_ref().set_input_buffer(
        0,
        Some(output_storage.buffer()),
        layout.start_offset() * DType::F32.size_in_bytes(),
    );
    for (index, (storage, offset)) in [
        (denominator_storage, denominator_offset),
        (gate_storage, gate_offset),
        (norm_storage, norm_offset),
    ]
    .into_iter()
    .enumerate()
    {
        encoder
            .as_ref()
            .set_input_buffer(index + 1, Some(storage.buffer()), offset);
    }
    encoder.as_ref().set_output_buffer(4, Some(&buffer), 0);
    let dims = [batch as u32, length as u32, heads as u32, width as u32];
    let strides = layout.stride();
    let strides = [
        strides[0] as u32,
        strides[1] as u32,
        strides[2] as u32,
        strides[3] as u32,
    ];
    encoder.as_ref().set_bytes(5, &dims);
    encoder.as_ref().set_bytes(6, &strides);
    encoder.as_ref().dispatch_threads(
        MTLSize {
            width: count,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: 256,
            height: 1,
            depth: 1,
        },
    );
    drop(encoder);
    Ok(Tensor::from_storage(
        Storage::Metal(MetalStorage::new(buffer, device, count, DType::BF16)),
        output.shape().clone(),
        BackpropOp::none(),
        false,
    ))
}

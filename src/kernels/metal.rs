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

const LIBRARY_BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/kernels.metallib"));
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
            .map_err(|e| Error::Msg(format!("Loading the sys1 metallib: {e}")))?;
        Library::new(raw)
    };
    let function = library.get_function(name, None).map_err(|error| {
        Error::Msg(format!(
            "Metal kernel {name:?} is missing from the embedded metallib ({error}). If this \
             target directory is shared with another checkout of this package, rebuild the \
             kernels with `touch src/kernels/*.metal`."
        ))
    })?;
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

pub(crate) fn layer_norm(input: &Tensor, weight: &Tensor, eps: f64) -> Result<Tensor> {
    let hidden = input.dim(candle_core::D::Minus1)?;
    if weight.dims() != [hidden] || weight.dtype() != input.dtype() || hidden == 0 {
        candle_core::bail!("Unsupported ModernBERT Metal layer-norm shape or dtype")
    }
    let name = match input.dtype() {
        DType::F32 => "modernbert_layer_norm_f32",
        DType::F16 => "modernbert_layer_norm_f16",
        DType::BF16 => "modernbert_layer_norm_bf16",
        dtype => candle_core::bail!("Unsupported ModernBERT Metal layer-norm dtype: {dtype:?}"),
    };
    let (_input, input_storage, input_offset) = contiguous_storage(input)?;
    let (_weight, weight_storage, weight_offset) = contiguous_storage(weight)?;
    let device = input_storage.device().clone();
    let count = input.elem_count();
    let rows = count / hidden;
    let buffer = device.new_buffer(count, input.dtype(), "modernbert-layer-norm")?;
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
    encoder.as_ref().set_bytes(3, &(hidden as u32));
    encoder.as_ref().set_bytes(4, &(eps as f32));
    // One threadgroup per row; the kernel's reduction assumes 256 threads.
    encoder.as_ref().dispatch_thread_groups(
        MTLSize {
            width: rows,
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
        input.shape().clone(),
        BackpropOp::none(),
        false,
    ))
}

pub(crate) fn geglu(input: &Tensor) -> Result<Tensor> {
    let mut dims = input.dims().to_vec();
    let Some(last) = dims.last_mut() else {
        candle_core::bail!("ModernBERT GeGLU input has no feature dimension")
    };
    if *last == 0 || *last % 2 != 0 {
        candle_core::bail!("ModernBERT GeGLU feature dimension must be positive and even")
    }
    *last /= 2;
    let inner = *last;
    let name = match input.dtype() {
        DType::F32 => "modernbert_geglu_f32",
        DType::F16 => "modernbert_geglu_f16",
        DType::BF16 => "modernbert_geglu_bf16",
        dtype => candle_core::bail!("Unsupported ModernBERT GeGLU dtype: {dtype:?}"),
    };
    let (_input, storage, offset) = contiguous_storage(input)?;
    let device = storage.device().clone();
    let count = input.elem_count() / 2;
    let buffer = device.new_buffer(count, input.dtype(), "modernbert-geglu")?;
    let pipeline = pipeline(&device, name)?;
    let encoder = device.command_encoder()?;
    encoder.set_label(name);
    encoder.set_compute_pipeline_state(&pipeline);
    encoder
        .as_ref()
        .set_input_buffer(0, Some(storage.buffer()), offset);
    encoder.as_ref().set_output_buffer(1, Some(&buffer), 0);
    encoder.as_ref().set_bytes(2, &(count as u32));
    encoder.as_ref().set_bytes(3, &(inner as u32));
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
        dims,
        BackpropOp::none(),
        false,
    ))
}

/// `cos`/`sin` must be f32 tables with at least `length` rows and `dim / 2` columns.
pub(crate) fn rope_qkv(qkv: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let (batch, length, three, heads, dim) = qkv.dims5()?;
    if three != 3 || dim == 0 || dim % 2 != 0 || heads == 0 || length == 0 {
        candle_core::bail!("Unsupported ModernBERT Metal fused RoPE shape")
    }
    if cos.dtype() != DType::F32
        || sin.dtype() != DType::F32
        || cos.dims() != sin.dims()
        || cos.dims() != [cos.dim(0)?, dim / 2]
        || cos.dim(0)? < length
    {
        candle_core::bail!("Unsupported ModernBERT Metal fused RoPE cosine/sine tables")
    }
    let name = match qkv.dtype() {
        DType::F32 => "modernbert_rope_qkv_f32",
        DType::F16 => "modernbert_rope_qkv_f16",
        DType::BF16 => "modernbert_rope_qkv_bf16",
        dtype => candle_core::bail!("Unsupported ModernBERT Metal fused RoPE dtype: {dtype:?}"),
    };
    let (_qkv, qkv_storage, qkv_offset) = contiguous_storage(qkv)?;
    let (_cos, cos_storage, cos_offset) = contiguous_storage(cos)?;
    let (_sin, sin_storage, sin_offset) = contiguous_storage(sin)?;
    let device = qkv_storage.device().clone();
    let pairs = batch * length * heads * (dim / 2);
    let count = pairs * 6;
    let buffer = device.new_buffer(count, qkv.dtype(), "modernbert-rope-qkv")?;
    let pipeline = pipeline(&device, name)?;
    let encoder = device.command_encoder()?;
    encoder.set_label(name);
    encoder.set_compute_pipeline_state(&pipeline);
    for (index, (storage, offset)) in [
        (&qkv_storage, qkv_offset),
        (&cos_storage, cos_offset),
        (&sin_storage, sin_offset),
    ]
    .into_iter()
    .enumerate()
    {
        encoder
            .as_ref()
            .set_input_buffer(index, Some(storage.buffer()), offset);
    }
    encoder.as_ref().set_output_buffer(3, Some(&buffer), 0);
    for (index, value) in [pairs, length, heads, dim].into_iter().enumerate() {
        encoder.as_ref().set_bytes(index + 4, &(value as u32));
    }
    encoder.as_ref().dispatch_threads(
        MTLSize {
            width: pairs,
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
        Storage::Metal(MetalStorage::new(buffer, device, count, qkv.dtype())),
        (3, batch, heads, length, dim),
        BackpropOp::none(),
        false,
    ))
}

pub(crate) fn merge_heads(attention: &Tensor) -> Result<Tensor> {
    let (batch, heads, length, dim) = attention.dims4()?;
    if heads == 0 || length == 0 || dim == 0 {
        candle_core::bail!("Unsupported ModernBERT Metal head-merge shape")
    }
    let name = match attention.dtype() {
        DType::F32 => "modernbert_merge_heads_f32",
        DType::F16 => "modernbert_merge_heads_f16",
        DType::BF16 => "modernbert_merge_heads_bf16",
        dtype => candle_core::bail!("Unsupported ModernBERT Metal head-merge dtype: {dtype:?}"),
    };
    let (_attention, storage, offset) = contiguous_storage(attention)?;
    let device = storage.device().clone();
    let count = batch * heads * length * dim;
    let buffer = device.new_buffer(count, attention.dtype(), "modernbert-merge-heads")?;
    let pipeline = pipeline(&device, name)?;
    let encoder = device.command_encoder()?;
    encoder.set_label(name);
    encoder.set_compute_pipeline_state(&pipeline);
    encoder
        .as_ref()
        .set_input_buffer(0, Some(storage.buffer()), offset);
    encoder.as_ref().set_output_buffer(1, Some(&buffer), 0);
    for (index, value) in [heads, length, dim].into_iter().enumerate() {
        encoder.as_ref().set_bytes(index + 2, &(value as u32));
    }

    encoder.as_ref().dispatch_threads(
        MTLSize {
            width: dim,
            height: length,
            depth: batch * heads,
        },
        MTLSize {
            width: dim.min(64),
            height: (256 / dim.min(64)).max(1),
            depth: 1,
        },
    );
    drop(encoder);
    Ok(Tensor::from_storage(
        Storage::Metal(MetalStorage::new(buffer, device, count, attention.dtype())),
        (batch, length, heads * dim),
        BackpropOp::none(),
        false,
    ))
}

pub(crate) fn swiglu(gate: &Tensor, up: &Tensor) -> Result<Tensor> {
    if gate.dims() != up.dims() || gate.dtype() != up.dtype() {
        candle_core::bail!("Qwen3.5 SwiGLU operands must match in shape and dtype")
    }
    let name = match gate.dtype() {
        DType::F32 => "qwen35_swiglu_f32",
        DType::F16 => "qwen35_swiglu_f16",
        DType::BF16 => "qwen35_swiglu_bf16",
        dtype => candle_core::bail!("Unsupported Qwen3.5 SwiGLU dtype: {dtype:?}"),
    };
    let (_gate, gate_storage, gate_offset) = contiguous_storage(gate)?;
    let (_up, up_storage, up_offset) = contiguous_storage(up)?;
    let device = gate_storage.device().clone();
    let count = gate.elem_count();
    let buffer = device.new_buffer(count, gate.dtype(), "qwen35-swiglu")?;
    let pipeline = pipeline(&device, name)?;
    let encoder = device.command_encoder()?;
    encoder.set_label(name);
    encoder.set_compute_pipeline_state(&pipeline);
    encoder
        .as_ref()
        .set_input_buffer(0, Some(gate_storage.buffer()), gate_offset);
    encoder
        .as_ref()
        .set_input_buffer(1, Some(up_storage.buffer()), up_offset);
    encoder.as_ref().set_output_buffer(2, Some(&buffer), 0);
    encoder.as_ref().set_bytes(3, &(count as u32));
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
        Storage::Metal(MetalStorage::new(buffer, device, count, gate.dtype())),
        gate.shape().clone(),
        BackpropOp::none(),
        false,
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

/// `mixed` is `[batch, length, 2 * key_width + value_width]`; the result is
/// `[batch, value_heads, length, 258 + value_dim]`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn pack_delta_inputs_bf16(
    mixed: &Tensor,
    beta_projection: &Tensor,
    gate_projection: &Tensor,
    dt_bias: &Tensor,
    decay: &Tensor,
    key_heads: usize,
    value_heads: usize,
    query_scale: f32,
) -> Result<Tensor> {
    let (batch, length, mixed_width) = mixed.dims3()?;
    let dim = 128;
    let key_width = key_heads * dim;
    if mixed_width != 2 * key_width + value_heads * dim
        || key_heads == 0
        || !value_heads.is_multiple_of(key_heads)
        || mixed.dtype() != DType::BF16
        || beta_projection.dims() != [batch, length, value_heads]
        || gate_projection.dims() != [batch, length, value_heads]
        || beta_projection.dtype() != DType::BF16
        || gate_projection.dtype() != DType::BF16
        || dt_bias.dims() != [value_heads]
        || decay.dims() != [value_heads]
        || dt_bias.dtype() != DType::F32
        || decay.dtype() != DType::F32
    {
        candle_core::bail!("Unsupported Qwen3.5 Metal delta-input shape or dtype")
    }
    let (_mixed, mixed_storage, mixed_offset) = contiguous_storage(mixed)?;
    let (_beta, beta_storage, beta_offset) = contiguous_storage(beta_projection)?;
    let (_gate, gate_storage, gate_offset) = contiguous_storage(gate_projection)?;
    let (_bias, bias_storage, bias_offset) = contiguous_storage(dt_bias)?;
    let (_decay, decay_storage, decay_offset) = contiguous_storage(decay)?;
    let device = mixed_storage.device().clone();
    let width = 258 + dim;
    let count = batch * value_heads * length * width;
    let buffer = device.new_buffer(count, DType::F32, "qwen35-delta-inputs")?;
    let pipeline = pipeline(&device, "qwen35_pack_delta_inputs_bf16")?;
    let encoder = device.command_encoder()?;
    encoder.set_label("qwen35_pack_delta_inputs_bf16");
    encoder.set_compute_pipeline_state(&pipeline);
    for (index, (storage, offset)) in [
        (&mixed_storage, mixed_offset),
        (&beta_storage, beta_offset),
        (&gate_storage, gate_offset),
        (&bias_storage, bias_offset),
        (&decay_storage, decay_offset),
    ]
    .into_iter()
    .enumerate()
    {
        encoder
            .as_ref()
            .set_input_buffer(index, Some(storage.buffer()), offset);
    }
    encoder.as_ref().set_output_buffer(5, Some(&buffer), 0);
    let dims = [
        batch as u32,
        length as u32,
        value_heads as u32,
        key_heads as u32,
    ];
    encoder.as_ref().set_bytes(6, &dims);
    encoder
        .as_ref()
        .set_bytes(7, &[key_width as u32, mixed_width as u32]);
    encoder.as_ref().set_bytes(8, &query_scale);
    encoder.as_ref().dispatch_thread_groups(
        MTLSize {
            width: batch * value_heads * length,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: dim,
            height: 1,
            depth: 1,
        },
    );
    drop(encoder);
    Ok(Tensor::from_storage(
        Storage::Metal(MetalStorage::new(buffer, device, count, DType::F32)),
        (batch, value_heads, length, width),
        BackpropOp::none(),
        false,
    ))
}

pub(crate) fn delta_rule_packed(packed: &Tensor, value_dim: usize) -> Result<Tensor> {
    let (batch, heads, length, width) = packed.dims4()?;
    if packed.dtype() != DType::F32 || width != 258 + value_dim {
        candle_core::bail!("Unsupported Qwen3.5 packed delta-rule shape or dtype")
    }
    let (_packed, storage, offset) = contiguous_storage(packed)?;
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
    gate: &Tensor,
    norm: &Tensor,
) -> Result<Tensor> {
    let (batch, length, heads, width) = output.dims4()?;
    if output.dtype() != DType::F32
        || width != 128
        || denominator.dims() != [batch, length, heads, 1]
        || gate.dims() != output.dims()
        || norm.dims() != [width]
        || gate.dtype() != DType::BF16
        || [denominator, norm]
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
    let (_gate, gate_storage, gate_offset) = contiguous_storage(gate)?;
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

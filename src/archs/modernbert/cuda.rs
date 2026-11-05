use super::*;

use candle_core::backend::BackendStorage;
use candle_core::cuda_backend::{
    WrapErr,
    cudarc::driver::{LaunchConfig, PushKernelArg},
};
use candle_core::{CpuStorage, CudaStorage, CustomOp1, CustomOp2, CustomOp3, Layout, Shape};
use half::{bf16, f16};

pub(crate) fn flash_attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    lengths: &[usize],
    scale: f32,
    window: Option<usize>,
    implementation: AttentionImplementation,
) -> Result<Tensor> {
    let (batch, heads, max_length, head_size) = q.dims4()?;
    if lengths.len() != batch || lengths.iter().any(|&length| length > max_length) {
        candle_core::bail!(
            "invalid Flash Attention sequence lengths {:?} for shape {:?}",
            lengths,
            q.shape()
        )
    }
    let q = q.transpose(1, 2)?.contiguous()?;
    let k = k.transpose(1, 2)?.contiguous()?;
    let v = v.transpose(1, 2)?.contiguous()?;
    let attention = if lengths.iter().all(|&length| length == max_length) {
        match implementation {
            AttentionImplementation::FlashAttention2
            | AttentionImplementation::FlashAttentionBlackwell => {
                crate::kernels::cuda::attention::flash_attn_windowed(
                    &q, &k, &v, scale, window, window,
                )?
            }
            _ => candle_core::bail!("{} support is not compiled in", implementation.cli_name()),
        }
    } else {
        let mut indices = Vec::with_capacity(lengths.iter().sum());
        let mut cumulative = Vec::with_capacity(batch + 1);
        cumulative.push(0u32);
        for (row, &length) in lengths.iter().enumerate() {
            indices.extend((0..length).map(|column| (row * max_length + column) as u32));
            cumulative.push(cumulative.last().copied().unwrap() + length as u32);
        }
        let indices = Tensor::from_vec(
            indices,
            cumulative.last().copied().unwrap() as usize,
            q.device(),
        )?;
        let cumulative = Tensor::from_vec(cumulative, batch + 1, q.device())?;
        let pack = |tensor: &Tensor| {
            tensor
                .reshape((batch * max_length, heads, head_size))?
                .index_select(&indices, 0)
        };
        let packed_q = pack(&q)?;
        let packed_k = pack(&k)?;
        let packed_v = pack(&v)?;
        let packed = match implementation {
            AttentionImplementation::FlashAttention2
            | AttentionImplementation::FlashAttentionBlackwell => {
                crate::kernels::cuda::attention::flash_attn_varlen_windowed(
                    &packed_q,
                    &packed_k,
                    &packed_v,
                    &cumulative,
                    &cumulative,
                    max_length,
                    max_length,
                    scale,
                    window,
                    window,
                )?
            }
            _ => candle_core::bail!("{} support is not compiled in", implementation.cli_name()),
        };
        Tensor::zeros(
            (batch * max_length, heads, head_size),
            packed.dtype(),
            packed.device(),
        )?
        .index_add(&indices, &packed, 0)?
        .reshape((batch, max_length, heads, head_size))?
    };
    attention.transpose(1, 2)
}

pub(crate) const PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/cuda.ptx"));

pub(crate) struct StableF16Projection;

impl CustomOp2 for StableF16Projection {
    fn name(&self) -> &'static str {
        "modernbert-stable-f16-projection"
    }

    fn cpu_fwd(
        &self,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("Stable FP16 projection requires CUDA")
    }

    fn cuda_fwd(
        &self,
        x: &CudaStorage,
        xl: &Layout,
        w: &CudaStorage,
        wl: &Layout,
    ) -> Result<(CudaStorage, Shape)> {
        let mut shape = xl.shape().dims().to_vec();
        let Some(&inner) = shape.last() else {
            candle_core::bail!("Projection requires a feature dimension")
        };
        let (columns, weight_inner) = wl.shape().dims2()?;
        if inner == 0 || columns == 0 || weight_inner != inner || xl.shape().elem_count() == 0 {
            candle_core::bail!("Invalid stable FP16 projection dimensions")
        }
        let rows = xl.shape().elem_count() / inner;
        let count = rows
            .checked_mul(columns)
            .ok_or_else(|| candle_core::Error::Msg("Projection output overflow".into()))?;
        if xl.shape().elem_count() > i32::MAX as usize
            || wl.shape().elem_count() > i32::MAX as usize
            || count > i32::MAX as usize
            || columns.div_ceil(8) > 65535
        {
            candle_core::bail!("Stable FP16 projection exceeds kernel index/grid limits")
        }
        let (xs, xe) = xl
            .contiguous_offsets()
            .ok_or_else(|| candle_core::Error::Msg("Projection input must be contiguous".into()))?;
        let (ws, we) = wl.contiguous_offsets().ok_or_else(|| {
            candle_core::Error::Msg("Projection weight must be contiguous".into())
        })?;
        let input = x.as_cuda_slice::<f16>()?.slice(xs..xe);
        let weight = w.as_cuda_slice::<f16>()?.slice(ws..we);
        let device = x.device().clone();
        let output = unsafe { device.alloc::<f16>(count)? };
        let function =
            device.get_or_load_custom_func("modernbert_projection_f16", "sys1_modernbert", PTX)?;
        let mut launch = function.builder();
        let inner = inner as i32;
        let columns_arg = columns as i32;
        launch
            .arg(&input)
            .arg(&weight)
            .arg(&output)
            .arg(&inner)
            .arg(&columns_arg);
        unsafe {
            launch.launch(LaunchConfig {
                grid_dim: (rows as u32, columns.div_ceil(8) as u32, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .w()?;
        *shape.last_mut().unwrap() = columns;
        Ok((
            CudaStorage::wrap_cuda_slice(output, device),
            Shape::from(shape),
        ))
    }
}

pub(crate) struct Geglu;

impl CustomOp1 for Geglu {
    fn name(&self) -> &'static str {
        "modernbert-geglu"
    }

    fn cpu_fwd(&self, _: &CpuStorage, _: &Layout) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("ModernBERT GeGLU kernel requires CUDA")
    }

    fn cuda_fwd(&self, storage: &CudaStorage, layout: &Layout) -> Result<(CudaStorage, Shape)> {
        let mut dims = layout.shape().dims().to_vec();
        let Some(last) = dims.last_mut() else {
            candle_core::bail!("ModernBERT GeGLU input has no feature dimension")
        };
        if *last == 0 || *last % 2 != 0 {
            candle_core::bail!("ModernBERT GeGLU feature dimension must be positive and even")
        }
        *last /= 2;
        let inner = i32::try_from(*last).map_err(|_| {
            candle_core::Error::Msg("ModernBERT GeGLU feature dimension is too large".into())
        })?;
        let count = layout.shape().elem_count() / 2;
        let count_arg = i64::try_from(count)
            .map_err(|_| candle_core::Error::Msg("ModernBERT GeGLU input is too large".into()))?;
        let blocks = u32::try_from(count.div_ceil(256)).map_err(|_| {
            candle_core::Error::Msg("ModernBERT GeGLU launch grid is too large".into())
        })?;
        let (start, end) = layout.contiguous_offsets().ok_or_else(|| {
            candle_core::Error::Msg("ModernBERT GeGLU input must be contiguous".into())
        })?;
        let device = storage.device().clone();
        let config = LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let output = match storage.dtype() {
            DType::F32 => {
                let input = storage.as_cuda_slice::<f32>()?.slice(start..end);
                let output = unsafe { device.alloc::<f32>(count)? };
                let function = device.get_or_load_custom_func(
                    "modernbert_geglu_f32",
                    "sys1_modernbert",
                    PTX,
                )?;
                let mut launch = function.builder();
                launch.arg(&input).arg(&output).arg(&count_arg).arg(&inner);
                unsafe { launch.launch(config) }.w()?;
                CudaStorage::wrap_cuda_slice(output, device)
            }
            DType::F16 => {
                let input = storage.as_cuda_slice::<f16>()?.slice(start..end);
                let output = unsafe { device.alloc::<f16>(count)? };
                let function = device.get_or_load_custom_func(
                    "modernbert_geglu_f16",
                    "sys1_modernbert",
                    PTX,
                )?;
                let mut launch = function.builder();
                launch.arg(&input).arg(&output).arg(&count_arg).arg(&inner);
                unsafe { launch.launch(config) }.w()?;
                CudaStorage::wrap_cuda_slice(output, device)
            }
            DType::BF16 => {
                let input = storage.as_cuda_slice::<bf16>()?.slice(start..end);
                let output = unsafe { device.alloc::<bf16>(count)? };
                let function = device.get_or_load_custom_func(
                    "modernbert_geglu_bf16",
                    "sys1_modernbert",
                    PTX,
                )?;
                let mut launch = function.builder();
                launch.arg(&input).arg(&output).arg(&count_arg).arg(&inner);
                unsafe { launch.launch(config) }.w()?;
                CudaStorage::wrap_cuda_slice(output, device)
            }
            dtype => candle_core::bail!("unsupported ModernBERT GeGLU dtype {dtype:?}"),
        };
        Ok((output, Shape::from(dims)))
    }
}

pub(crate) fn geglu(input: &Tensor) -> Result<Tensor> {
    input.contiguous()?.apply_op1_no_bwd(&Geglu)
}

pub(crate) struct RopeQkv;

impl CustomOp3 for RopeQkv {
    fn name(&self) -> &'static str {
        "modernbert-rope-qkv"
    }

    fn cpu_fwd(
        &self,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("ModernBERT fused RoPE requires CUDA")
    }

    fn cuda_fwd(
        &self,
        input_storage: &CudaStorage,
        input_layout: &Layout,
        cos_storage: &CudaStorage,
        cos_layout: &Layout,
        sin_storage: &CudaStorage,
        sin_layout: &Layout,
    ) -> Result<(CudaStorage, Shape)> {
        let &[batch, length, 3, heads, dim] = input_layout.shape().dims() else {
            candle_core::bail!("ModernBERT fused RoPE expects [batch, length, 3, heads, dim]")
        };
        if dim == 0 || dim % 2 != 0 || heads == 0 || length == 0 {
            candle_core::bail!("ModernBERT fused RoPE has invalid dimensions")
        }
        if cos_storage.dtype() != DType::F32
            || sin_storage.dtype() != DType::F32
            || cos_layout.shape().dims() != sin_layout.shape().dims()
            || cos_layout.shape().dims().len() != 2
            || cos_layout.shape().dims()[0] < length
            || cos_layout.shape().dims()[1] != dim / 2
        {
            candle_core::bail!("ModernBERT fused RoPE has invalid cosine/sine tables")
        }
        let pairs = batch
            .checked_mul(length)
            .and_then(|value| value.checked_mul(heads))
            .and_then(|value| value.checked_mul(dim / 2))
            .ok_or_else(|| {
                candle_core::Error::Msg("ModernBERT fused RoPE shape is too large".into())
            })?;
        let output_count = input_layout.shape().elem_count();
        let pairs_arg = i64::try_from(pairs).map_err(|_| {
            candle_core::Error::Msg("ModernBERT fused RoPE shape is too large".into())
        })?;
        let length_arg = i32::try_from(length)
            .map_err(|_| candle_core::Error::Msg("ModernBERT sequence is too long".into()))?;
        let heads_arg = i32::try_from(heads)
            .map_err(|_| candle_core::Error::Msg("ModernBERT has too many heads".into()))?;
        let dim_arg = i32::try_from(dim).map_err(|_| {
            candle_core::Error::Msg("ModernBERT head dimension is too large".into())
        })?;
        let blocks = u32::try_from(pairs.div_ceil(256)).map_err(|_| {
            candle_core::Error::Msg("ModernBERT fused RoPE grid is too large".into())
        })?;
        let (input_start, input_end) = input_layout.contiguous_offsets().ok_or_else(|| {
            candle_core::Error::Msg("ModernBERT fused RoPE input must be contiguous".into())
        })?;
        let (cos_start, cos_end) = cos_layout.contiguous_offsets().ok_or_else(|| {
            candle_core::Error::Msg("ModernBERT RoPE cosine table must be contiguous".into())
        })?;
        let (sin_start, sin_end) = sin_layout.contiguous_offsets().ok_or_else(|| {
            candle_core::Error::Msg("ModernBERT RoPE sine table must be contiguous".into())
        })?;
        let device = input_storage.device().clone();
        let cos = cos_storage
            .as_cuda_slice::<f32>()?
            .slice(cos_start..cos_end);
        let sin = sin_storage
            .as_cuda_slice::<f32>()?
            .slice(sin_start..sin_end);
        let config = LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        macro_rules! launch_rope {
            ($type:ty, $kernel:literal) => {{
                let input = input_storage
                    .as_cuda_slice::<$type>()?
                    .slice(input_start..input_end);
                let output = unsafe { device.alloc::<$type>(output_count)? };
                let function = device.get_or_load_custom_func($kernel, "sys1_modernbert", PTX)?;
                let mut launch = function.builder();
                launch
                    .arg(&input)
                    .arg(&cos)
                    .arg(&sin)
                    .arg(&output)
                    .arg(&pairs_arg)
                    .arg(&length_arg)
                    .arg(&heads_arg)
                    .arg(&dim_arg);
                unsafe { launch.launch(config) }.w()?;
                CudaStorage::wrap_cuda_slice(output, device)
            }};
        }
        let output = match input_storage.dtype() {
            DType::F16 => launch_rope!(f16, "modernbert_rope_qkv_f16"),
            DType::BF16 => launch_rope!(bf16, "modernbert_rope_qkv_bf16"),
            dtype => candle_core::bail!("unsupported ModernBERT fused RoPE dtype {dtype:?}"),
        };
        Ok((output, Shape::from((3, batch, length, heads, dim))))
    }
}

impl RotaryEmbedding {
    pub(crate) fn apply_qkv(&self, qkv: &Tensor) -> Result<(Tensor, Tensor, Tensor)> {
        let output = qkv.apply_op3_no_bwd(&self.cos, &self.sin, &RopeQkv)?;
        Ok((
            output.get(0)?.transpose(1, 2)?,
            output.get(1)?.transpose(1, 2)?,
            output.get(2)?.transpose(1, 2)?,
        ))
    }
}

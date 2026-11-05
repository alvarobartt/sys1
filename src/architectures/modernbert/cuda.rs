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

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone, Copy)]
    struct Case {
        name: &'static str,
        batch: usize,
        length: usize,
        lengths: &'static [usize],
        window: Option<usize>,
    }

    fn deterministic_tensor(
        shape: (usize, usize, usize, usize),
        seed: u64,
        device: &Device,
    ) -> Result<Tensor> {
        let count = shape.0 * shape.1 * shape.2 * shape.3;
        let mut state = seed;
        let values = (0..count)
            .map(|_| {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                let unit = ((state >> 40) as u32) as f32 / ((1u32 << 24) as f32);
                (unit - 0.5) * 2.0
            })
            .collect::<Vec<_>>();
        Tensor::from_vec(values, shape, device)?.to_dtype(DType::F16)
    }

    fn attention_mask(
        lengths: &[usize],
        length: usize,
        window: Option<usize>,
        dtype: DType,
        device: &Device,
    ) -> Result<Tensor> {
        let padding = lengths
            .iter()
            .flat_map(|&valid| (0..length).map(move |column| (column < valid) as u32 as f32))
            .collect::<Vec<_>>();
        let padding = Tensor::from_vec(padding, (lengths.len(), length), device)?;
        let global = global_attention_mask(&padding, length, dtype)?;
        match window {
            Some(window) => {
                global.broadcast_add(&local_attention_mask(length, window, dtype, device)?)
            }
            None => Ok(global),
        }
    }

    fn valid_error(actual: &Tensor, reference: &Tensor, lengths: &[usize]) -> Result<(f32, f64)> {
        let (batch_size, heads, length, width) = actual.dims4()?;
        let actual = actual
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let reference = reference.flatten_all()?.to_vec1::<f32>()?;
        let mut maximum = 0f32;
        let mut total = 0f64;
        let mut count = 0usize;
        assert_eq!(lengths.len(), batch_size);
        for (batch, &valid_length) in lengths.iter().enumerate() {
            for head in 0..heads {
                for token in 0..valid_length {
                    for column in 0..width {
                        let index = ((batch * heads + head) * length + token) * width + column;
                        let delta = (actual[index] - reference[index]).abs();
                        maximum = maximum.max(delta);
                        total += delta as f64;
                        count += 1;
                    }
                }
            }
        }
        Ok((maximum, total / count as f64))
    }

    fn run(case: Case, device: &Device) -> Result<()> {
        const HEADS: usize = 12;
        const HEAD_SIZE: usize = 64;
        assert_eq!(case.batch, case.lengths.len());
        let shape = (case.batch, HEADS, case.length, HEAD_SIZE);
        let q = deterministic_tensor(shape, 0x6a09_e667_f3bc_c909, device)?;
        let k = deterministic_tensor(shape, 0xbb67_ae85_84ca_a73b, device)?;
        let v = deterministic_tensor(shape, 0x3c6e_f372_fe94_f82b, device)?;
        let scale = (HEAD_SIZE as f64).powf(-0.5);

        let q32 = q.to_dtype(DType::F32)?;
        let k32 = k.to_dtype(DType::F32)?;
        let v32 = v.to_dtype(DType::F32)?;
        let mask32 = attention_mask(case.lengths, case.length, case.window, DType::F32, device)?;
        let reference = eager_attention_full(&q32, &k32, &v32, scale, Some(&mask32))?;

        let mask16 = attention_mask(case.lengths, case.length, case.window, DType::F16, device)?;
        let eager = eager_attention_full(&q, &k, &v, scale, Some(&mask16))?;
        let flash = scaled_dot_product_attention(
            &q,
            &k,
            &v,
            scale,
            AttentionOptions {
                mask: None,
                implementation: crate::models::attention_for_device(DType::F16, device)
                    .map_err(|error| candle_core::Error::Msg(error.to_string()))?,
                lengths: case.lengths,
                window: case.window,
            },
        )?;

        let (eager_max, eager_mean) = valid_error(&eager, &reference, case.lengths)?;
        let (flash_max, flash_mean) = valid_error(&flash, &reference, case.lengths)?;
        let allowed = 2.0 * eager_max;
        println!(
            "MODERNBERT_FA2_REFERENCE case={} batch={} length={} lengths={:?} window={:?} eager_max={:.9} flash_max={:.9} allowed={:.9} eager_mean={:.9} flash_mean={:.9} pass={}",
            case.name,
            case.batch,
            case.length,
            case.lengths,
            case.window,
            eager_max,
            flash_max,
            allowed,
            eager_mean,
            flash_mean,
            flash_max <= allowed,
        );
        assert!(
            flash_max <= allowed,
            "{}: FA2 max error {flash_max} exceeds 2x eager F16 error {allowed}",
            case.name,
        );
        Ok(())
    }

    #[test]
    fn fa2_forward_error_is_bounded_by_twice_f16_eager_error() -> Result<()> {
        let device =
            crate::device::load().map_err(|error| candle_core::Error::Msg(error.to_string()))?;
        if !crate::kernels::cuda::attention::supported_test_device(&device)? {
            println!("MODERNBERT_FA_REFERENCE_SKIPPED unsupported CUDA architecture");
            return Ok(());
        }
        for case in [
            Case {
                name: "dense_128",
                batch: 1,
                length: 128,
                lengths: &[128],
                window: None,
            },
            Case {
                name: "local_257",
                batch: 1,
                length: 257,
                lengths: &[257],
                window: Some(64),
            },
            Case {
                name: "varlen_dense_129",
                batch: 3,
                length: 129,
                lengths: &[129, 97, 33],
                window: None,
            },
            Case {
                name: "varlen_local_513",
                batch: 3,
                length: 513,
                lengths: &[513, 383, 127],
                window: Some(128),
            },
        ] {
            run(case, &device)?;
        }
        Ok(())
    }
}

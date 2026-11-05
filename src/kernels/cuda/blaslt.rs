#![allow(dead_code)]

use candle_core::cuda_backend::{
    CudaStorage,
    cudarc::{
        cublaslt::{result, sys},
        driver::{CudaSlice, CudaStream, DevicePtr, DevicePtrMut},
    },
};
use candle_core::{DType, Result, Storage, Tensor, backend::BackendStorage, op::BackpropOp};
use std::{cell::RefCell, collections::HashMap, sync::Arc};

fn error<E: std::fmt::Display>(e: E) -> candle_core::Error {
    candle_core::Error::Msg(format!("cuBLASLt projection: {e}"))
}

struct Layout(sys::cublasLtMatrixLayout_t);
impl Drop for Layout {
    fn drop(&mut self) {
        let _ = unsafe { result::destroy_matrix_layout(self.0) };
    }
}
struct Desc(sys::cublasLtMatmulDesc_t);
impl Drop for Desc {
    fn drop(&mut self) {
        let _ = unsafe { result::destroy_matmul_desc(self.0) };
    }
}
struct Pref(sys::cublasLtMatmulPreference_t);
impl Drop for Pref {
    fn drop(&mut self) {
        let _ = unsafe { result::destroy_matmul_pref(self.0) };
    }
}
struct Plan {
    a: Layout,
    b: Layout,
    out: Layout,
    desc: Desc,
    algo: sys::cublasLtMatmulAlgo_t,
    workspace_bytes: usize,
}
type Key = (
    usize,
    usize,
    usize,
    u32,
    u32,
    u32,
    Option<usize>,
    bool,
    bool,
);
struct Context {
    stream: Arc<CudaStream>,
    handle: sys::cublasLtHandle_t,
    workspace: CudaSlice<u8>,

    planning_workspace_bytes: usize,
    plans: HashMap<Key, Plan>,
}
impl Context {
    fn new(stream: Arc<CudaStream>) -> Result<Self> {
        stream.context().bind_to_thread().map_err(error)?;

        let major = stream.context().compute_capability().map_err(error)?.0;
        let bytes = if major >= 9 { 32 << 20 } else { 4 << 20 };
        let workspace = unsafe { stream.alloc::<u8>(bytes) }.map_err(error)?;
        let handle = result::create_handle().map_err(error)?;
        Ok(Self {
            stream,
            handle,
            workspace,
            planning_workspace_bytes: bytes,
            plans: HashMap::new(),
        })
    }
    fn plan(&self, key: Key) -> Result<Plan> {
        let (
            rows,
            inner,
            columns,
            align_a,
            align_b,
            align_out,
            reference_rows,
            transposed_weight,
            cooperative,
        ) = key;
        let ld_k = i64::try_from(inner).map_err(error)?;
        let ld_n = i64::try_from(columns).map_err(error)?;
        let bf16 = sys::cudaDataType::CUDA_R_16BF;

        let a = Layout(
            result::create_matrix_layout(
                bf16,
                if transposed_weight { columns } else { inner } as u64,
                if transposed_weight { inner } else { columns } as u64,
                if transposed_weight { ld_n } else { ld_k },
            )
            .map_err(error)?,
        );
        let b = Layout(
            result::create_matrix_layout(bf16, inner as u64, rows as u64, ld_k).map_err(error)?,
        );
        let out = Layout(
            result::create_matrix_layout(bf16, columns as u64, rows as u64, ld_n).map_err(error)?,
        );
        let desc = Desc(
            result::create_matmul_desc(
                sys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
                sys::cudaDataType::CUDA_R_32F,
            )
            .map_err(error)?,
        );

        let transpose = if transposed_weight {
            candle_core::cuda_backend::cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_N
        } else {
            candle_core::cuda_backend::cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_T
        } as i32;
        let pref = Pref(result::create_matmul_pref().map_err(error)?);
        let bytes = self.planning_workspace_bytes as u64;

        let reduction = sys::cublasLtReductionScheme_t::CUBLASLT_REDUCTION_SCHEME_NONE as u32;
        unsafe {
            result::set_matmul_desc_attribute(
                desc.0,
                sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_TRANSA,
                (&transpose as *const i32).cast(),
                std::mem::size_of_val(&transpose),
            )
            .map_err(error)?;
            for (attribute, value, size) in [
                (sys::cublasLtMatmulPreferenceAttributes_t::CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, (&bytes as *const u64).cast(), std::mem::size_of_val(&bytes)),
                (sys::cublasLtMatmulPreferenceAttributes_t::CUBLASLT_MATMUL_PREF_REDUCTION_SCHEME_MASK, (&reduction as *const u32).cast(), std::mem::size_of_val(&reduction)),
                (sys::cublasLtMatmulPreferenceAttributes_t::CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_A_BYTES, (&align_a as *const u32).cast(), std::mem::size_of_val(&align_a)),
                (sys::cublasLtMatmulPreferenceAttributes_t::CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_B_BYTES, (&align_b as *const u32).cast(), std::mem::size_of_val(&align_b)),
                (sys::cublasLtMatmulPreferenceAttributes_t::CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_C_BYTES, (&align_out as *const u32).cast(), std::mem::size_of_val(&align_out)),
                (sys::cublasLtMatmulPreferenceAttributes_t::CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_D_BYTES, (&align_out as *const u32).cast(), std::mem::size_of_val(&align_out)),
            ] {
                result::set_matmul_pref_attribute(pref.0, attribute, value, size).map_err(error)?;
            }

            let reference_b = reference_rows
                .map(|m| {
                    result::create_matrix_layout(bf16, inner as u64, m as u64, ld_k)
                        .map(Layout)
                        .map_err(error)
                })
                .transpose()?;
            let reference_out = reference_rows
                .map(|m| {
                    result::create_matrix_layout(bf16, columns as u64, m as u64, ld_n)
                        .map(Layout)
                        .map_err(error)
                })
                .transpose()?;
            let mut heuristic = result::get_matmul_algo_heuristic(
                self.handle,
                desc.0,
                a.0,
                reference_b.as_ref().unwrap_or(&b).0,
                reference_out.as_ref().unwrap_or(&out).0,
                reference_out.as_ref().unwrap_or(&out).0,
                pref.0,
            )
            .map_err(error)?;
            let hopper_enabled = {
                #[cfg(test)]
                {
                    EXPANSION_ENABLED.with(std::cell::Cell::get)
                }
                #[cfg(not(test))]
                {
                    true
                }
            };
            if hopper_enabled
                && option_env!("SYS1_CUDA_COMPUTE_CAP") == Some("90")
                && expansion_geometry(rows, inner, columns, reference_rows, transposed_weight)
                && let Some(plan) = checked_expansion(
                    self.handle,
                    desc.0,
                    a.0,
                    b.0,
                    out.0,
                    self.planning_workspace_bytes,
                    columns == 8192 && rows >= 49152,
                )
            {
                heuristic = plan;
                #[cfg(test)]
                EXPANSION_CALLS.with(|slot| slot.set(slot.get() + 1));
            }

            let mid_enabled = {
                #[cfg(test)]
                {
                    MID_DOWN_PLAN_ENABLED.with(std::cell::Cell::get)
                }
                #[cfg(not(test))]
                {
                    true
                }
            };
            if mid_enabled
                && option_env!("SYS1_CUDA_COMPUTE_CAP") == Some("120")
                && (3201..=3712).contains(&rows)
                && inner == 12288
                && columns == 4096
                && !transposed_weight
                && reference_rows.is_none()
            {
                let mut choices: [std::mem::MaybeUninit<sys::cublasLtMatmulHeuristicResult_t>; 16] =
                    [const { std::mem::MaybeUninit::uninit() }; 16];
                let mut returned = 0i32;
                let status = sys::cublasLtMatmulAlgoGetHeuristic(
                    self.handle,
                    desc.0,
                    a.0,
                    b.0,
                    out.0,
                    out.0,
                    pref.0,
                    16,
                    choices.as_mut_ptr().cast(),
                    &mut returned,
                )
                .result();
                if status.is_ok() && (1..=16).contains(&returned) {
                    for slot in &choices[..returned as usize] {
                        let candidate = slot.assume_init_ref();
                        if candidate.state.result().is_err()
                            || candidate.workspaceSize > self.planning_workspace_bytes
                        {
                            continue;
                        }
                        let attribute = |name| -> Option<i32> {
                            let mut value = 0i32;
                            let mut written = 0usize;
                            let status = sys::cublasLtMatmulAlgoConfigGetAttribute(
                                &candidate.algo,
                                name,
                                (&mut value as *mut i32).cast(),
                                std::mem::size_of::<i32>(),
                                &mut written,
                            )
                            .result();
                            (status.is_ok() && written == std::mem::size_of::<i32>())
                                .then_some(value)
                        };
                        use sys::cublasLtMatmulAlgoConfigAttributes_t::*;
                        if attribute(CUBLASLT_ALGO_CONFIG_ID) != Some(21)
                            || attribute(CUBLASLT_ALGO_CONFIG_TILE_ID) != Some(20)
                            || attribute(CUBLASLT_ALGO_CONFIG_STAGES_ID) != Some(10)
                            || attribute(CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING) != Some(0)
                            || attribute(CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION) != Some(0)
                            || !matches!(attribute(CUBLASLT_ALGO_CONFIG_SPLITK_NUM), Some(0 | 1))
                            || attribute(CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME) != Some(0)
                        {
                            continue;
                        }
                        let mut checked =
                            std::mem::MaybeUninit::<sys::cublasLtMatmulHeuristicResult_t>::uninit();
                        if sys::cublasLtMatmulAlgoCheck(
                            self.handle,
                            desc.0,
                            a.0,
                            b.0,
                            out.0,
                            out.0,
                            &candidate.algo,
                            checked.as_mut_ptr(),
                        )
                        .result()
                        .is_err()
                        {
                            continue;
                        }
                        let checked = checked.assume_init();
                        if checked.state.result().is_err()
                            || checked.workspaceSize > self.planning_workspace_bytes
                        {
                            continue;
                        }
                        heuristic.algo = candidate.algo;
                        heuristic.workspaceSize = checked.workspaceSize;
                        #[cfg(test)]
                        MID_DOWN_PLAN_CALLS.with(|count| count.set(count.get() + 1));
                        break;
                    }
                }
            }
            if reference_rows.is_some() {
                let algo = heuristic.algo;
                sys::cublasLtMatmulAlgoCheck(
                    self.handle,
                    desc.0,
                    a.0,
                    b.0,
                    out.0,
                    out.0,
                    &algo,
                    &mut heuristic,
                )
                .result()
                .map_err(error)?;
                heuristic.state.result().map_err(error)?;
                heuristic.algo = algo;
            }
            if reference_rows.is_none() && heuristic.workspaceSize > self.planning_workspace_bytes {
                candle_core::bail!("cuBLASLt heuristic exceeded workspace")
            }
            if cooperative
                && let Ok(plan) = checked_hopper_cooperative_plan(
                    self.handle,
                    desc.0,
                    a.0,
                    b.0,
                    out.0,
                    self.planning_workspace_bytes,
                    (inner, columns),
                )
            {
                heuristic = plan;
                #[cfg(test)]
                HOPPER_DOWN_COOPERATIVE_SELECTIONS.with(|count| count.set(count.get() + 1));
            }
            let mut actual_reduction = 0u32;
            let mut written = 0usize;
            sys::cublasLtMatmulAlgoConfigGetAttribute(
                &heuristic.algo,
                sys::cublasLtMatmulAlgoConfigAttributes_t::CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME,
                (&mut actual_reduction as *mut u32).cast(),
                std::mem::size_of::<u32>(),
                &mut written,
            )
            .result()
            .map_err(error)?;
            if written != std::mem::size_of::<u32>() || actual_reduction != 0 {
                candle_core::bail!(
                    "cuBLASLt selected unsupported reduction scheme {actual_reduction}"
                )
            }
            Ok(Plan {
                a,
                b,
                out,
                desc,
                algo: heuristic.algo,
                workspace_bytes: heuristic.workspaceSize,
            })
        }
    }
}
impl Drop for Context {
    fn drop(&mut self) {
        if self.stream.context().bind_to_thread().is_ok() {
            self.plans.clear();
            let _ = unsafe { result::destroy_handle(self.handle) };
        }
    }
}
thread_local! {

    static CONTEXT: RefCell<Option<Context>> = const { RefCell::new(None) };
}
fn alignment(pointer: u64) -> u32 {
    1u32 << pointer.trailing_zeros().min(8)
}

pub(crate) fn project_unblocked(input: &Tensor, linear: &candle_nn::Linear) -> Result<Tensor> {
    if linear.bias().is_some() {
        candle_core::bail!("Unblocked cuBLASLt only supports unbiased projections")
    }
    let (batch, length, inner) = input.dims3()?;
    let rows = batch
        .checked_mul(length)
        .ok_or_else(|| error("projection row count overflow"))?;
    matmul_bf16(
        &input.contiguous()?.reshape((rows, inner))?,
        linear.weight(),
    )?
    .reshape((batch, length, linear.weight().dim(0)?))
}

pub(crate) fn matmul_bf16(input: &Tensor, weight: &Tensor) -> Result<Tensor> {
    matmul_bf16_impl(input, weight, None, None, false)
}

pub(crate) fn project_fixed512(input: &Tensor, linear: &candle_nn::Linear) -> Result<Tensor> {
    if linear.bias().is_some() {
        candle_core::bail!("Fixed512 cuBLASLt requires unbiased projections")
    }
    let (batch, length, inner) = input.dims3()?;
    let rows = batch
        .checked_mul(length)
        .ok_or_else(|| error("row count overflow"))?;
    if rows == 0 {
        return input.apply(linear);
    }
    let padded_rows = rows
        .checked_add(511)
        .ok_or_else(|| error("padding overflow"))?
        / 512
        * 512;
    let matrix = input.contiguous()?.reshape((rows, inner))?;
    let padded = if padded_rows == rows {
        matrix
    } else {
        let zeros = Tensor::zeros((padded_rows - rows, inner), DType::BF16, input.device())?;
        Tensor::cat(&[&matrix, &zeros], 0)?
    };
    matmul_bf16_impl(&padded, linear.weight(), Some(512), None, false)?
        .narrow(0, 0, rows)?
        .reshape((batch, length, linear.weight().dim(0)?))
}

pub(crate) fn project_pinned512(input: &Tensor, linear: &candle_nn::Linear) -> Result<Tensor> {
    if linear.bias().is_some() {
        candle_core::bail!("Pinned512 cuBLASLt requires unbiased projections")
    }
    let (batch, length, inner) = input.dims3()?;
    let rows = batch
        .checked_mul(length)
        .ok_or_else(|| error("row count overflow"))?;
    if rows == 0 {
        return input.apply(linear);
    }
    matmul_bf16_impl(
        &input.contiguous()?.reshape((rows, inner))?,
        linear.weight(),
        None,
        Some(512),
        false,
    )?
    .reshape((batch, length, linear.weight().dim(0)?))
}

pub(crate) fn project_pinned2048(input: &Tensor, linear: &candle_nn::Linear) -> Result<Tensor> {
    if linear.bias().is_some() {
        candle_core::bail!("Pinned2048 cuBLASLt requires unbiased projections")
    }
    let (batch, length, inner) = input.dims3()?;
    let rows = batch
        .checked_mul(length)
        .ok_or_else(|| error("row count overflow"))?;
    if rows == 0 {
        return input.apply(linear);
    }
    matmul_bf16_impl(
        &input.contiguous()?.reshape((rows, inner))?,
        linear.weight(),
        None,
        Some(2048),
        false,
    )?
    .reshape((batch, length, linear.weight().dim(0)?))
}

fn matmul_bf16_impl(
    input: &Tensor,
    weight: &Tensor,
    block_rows: Option<usize>,
    reference_rows: Option<usize>,
    transposed_weight: bool,
) -> Result<Tensor> {
    matmul_bf16_impl_policy(
        input,
        weight,
        block_rows,
        reference_rows,
        transposed_weight,
        false,
    )
}

fn matmul_bf16_impl_policy(
    input: &Tensor,
    weight: &Tensor,
    block_rows: Option<usize>,
    reference_rows: Option<usize>,
    transposed_weight: bool,
    cooperative: bool,
) -> Result<Tensor> {
    let (rows, inner) = input.dims2()?;
    let (first, second) = weight.dims2()?;
    let (columns, weight_inner) = if transposed_weight {
        (second, first)
    } else {
        (first, second)
    };
    if rows == 0
        || inner == 0
        || columns == 0
        || inner != weight_inner
        || input.dtype() != DType::BF16
        || weight.dtype() != DType::BF16
        || !input.device().is_cuda()
        || !input.device().same_device(weight.device())
    {
        candle_core::bail!(
            "cuBLASLt requires nonempty same-device BF16 matrices with matching inner dimensions"
        )
    }
    let launch_rows = block_rows.unwrap_or(rows);
    if launch_rows == 0 || rows % launch_rows != 0 {
        candle_core::bail!("cuBLASLt fixed geometry requires whole nonempty row blocks")
    }
    let input = input.contiguous()?;
    let weight = weight.contiguous()?;
    let (xs, xl) = input.storage_and_layout();
    let (ws, wl) = weight.storage_and_layout();
    let (Storage::Cuda(xs), Storage::Cuda(ws)) = (&*xs, &*ws) else {
        candle_core::bail!("cuBLASLt requires CUDA storage")
    };
    let (xstart, xend) = xl
        .contiguous_offsets()
        .ok_or_else(|| error("noncontiguous input"))?;
    let (wstart, wend) = wl
        .contiguous_offsets()
        .ok_or_else(|| error("noncontiguous weight"))?;
    let xstorage = xs.as_cuda_slice::<half::bf16>()?;
    let wstorage = ws.as_cuda_slice::<half::bf16>()?;
    if xend > xstorage.len() || wend > wstorage.len() {
        candle_core::bail!("cuBLASLt storage bounds differ")
    }
    let device = xs.device().clone();
    let stream = device.cuda_stream();
    stream.context().bind_to_thread().map_err(error)?;
    let count = rows
        .checked_mul(columns)
        .ok_or_else(|| error("output size overflow"))?;
    let mut output = unsafe { device.alloc::<half::bf16>(count)? };
    let x = xstorage.slice(xstart..xend);
    let w = wstorage.slice(wstart..wend);
    let (xptr, _xguard) = x.device_ptr(&stream);
    let (wptr, _wguard) = w.device_ptr(&stream);
    let (outptr, _outguard) = output.device_ptr_mut(&stream);
    CONTEXT.with(|cell| -> Result<()> {
        let mut context = cell.try_borrow_mut().map_err(error)?;
        if context
            .as_ref()
            .is_none_or(|c| !Arc::ptr_eq(&c.stream, &stream))
        {
            *context = Some(Context::new(stream.clone())?);
        }
        let context = context.as_mut().unwrap();

        for start in (0..rows).step_by(launch_rows) {
            let input_offset = start
                .checked_mul(inner)
                .and_then(|n| n.checked_mul(2))
                .ok_or_else(|| error("input pointer offset overflow"))?;
            let output_offset = start
                .checked_mul(columns)
                .and_then(|n| n.checked_mul(2))
                .ok_or_else(|| error("output pointer offset overflow"))?;
            let block_xptr = xptr
                .checked_add(u64::try_from(input_offset).map_err(error)?)
                .ok_or_else(|| error("input pointer overflow"))?;
            let block_outptr = outptr
                .checked_add(u64::try_from(output_offset).map_err(error)?)
                .ok_or_else(|| error("output pointer overflow"))?;
            let key = (
                launch_rows,
                inner,
                columns,
                alignment(wptr),
                alignment(block_xptr),
                alignment(block_outptr),
                reference_rows,
                transposed_weight,
                cooperative,
            );
            if !context.plans.contains_key(&key) {
                if context.plans.len() >= 128 {
                    context.plans.clear();
                }
                let plan = context.plan(key)?;
                context.plans.insert(key, plan);
            }
            let required_workspace = context.plans[&key].workspace_bytes;
            if required_workspace > context.workspace.len() {
                context.workspace =
                    unsafe { stream.alloc::<u8>(required_workspace) }.map_err(error)?;
            }
            let workspace_bytes = context.workspace.len();
            let plan = &context.plans[&key];
            let (workptr, _workguard) = context.workspace.device_ptr_mut(&stream);
            let alpha = 1f32;
            let beta = 0f32;

            unsafe {
                result::matmul(
                    context.handle,
                    plan.desc.0,
                    (&alpha as *const f32).cast(),
                    (&beta as *const f32).cast(),
                    wptr as *const _,
                    plan.a.0,
                    block_xptr as *const _,
                    plan.b.0,
                    block_outptr as *const _,
                    plan.out.0,
                    block_outptr as *mut _,
                    plan.out.0,
                    &plan.algo,
                    workptr as *mut _,
                    workspace_bytes,
                    stream.cu_stream().cast(),
                )
            }
            .map_err(error)?;
        }
        Ok(())
    })?;
    drop(_outguard);
    Ok(Tensor::from_storage(
        Storage::Cuda(CudaStorage::wrap_cuda_slice(output, device)),
        (rows, columns),
        BackpropOp::none(),
        false,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn blaslt_pinned512_variable_rows_smoke() -> Result<()> {
        let device = crate::device::load().map_err(error)?;
        let candle_core::Device::Cuda(cuda) = &device else {
            return Ok(());
        };
        if cuda
            .cuda_stream()
            .context()
            .compute_capability()
            .map_err(error)?
            .0
            < 8
        {
            println!("BLASLT_PINNED512_SKIPPED BF16 requires SM80");
            return Ok(());
        }

        for rows in [3, 512, 513, 1025] {
            let inner = 128;
            let columns = 256;
            let x: Vec<f32> = (0..rows * inner)
                .map(|i| ((i * 17 % 41) as f32 - 20.) / 16.)
                .collect();
            let w: Vec<f32> = (0..columns * inner)
                .map(|i| ((i * 13 % 37) as f32 - 18.) / 16.)
                .collect();
            let x = Tensor::from_vec(x, (1, rows, inner), &device)?.to_dtype(DType::BF16)?;
            let w = Tensor::from_vec(w, (columns, inner), &device)?.to_dtype(DType::BF16)?;
            let linear = candle_nn::Linear::new(w, None);
            let actual = project_pinned512(&x, &linear)?
                .to_dtype(DType::F32)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            let reference = project_fixed512(&x, &linear)?
                .to_dtype(DType::F32)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            assert_eq!(actual, reference, "pinned algorithm dyadic smoke M={rows}");
        }
        Ok(())
    }
    #[test]
    fn blaslt_fixed512_matches_independent_block_launches() -> Result<()> {
        let device = crate::device::load().map_err(error)?;
        let candle_core::Device::Cuda(cuda) = &device else {
            return Ok(());
        };
        if cuda
            .cuda_stream()
            .context()
            .compute_capability()
            .map_err(error)?
            .0
            < 8
        {
            println!("BLASLT_FIXED512_SKIPPED BF16 requires SM80");
            return Ok(());
        }
        for rows in [3, 512, 513, 1025] {
            let inner = 128;
            let columns = 256;
            let x: Vec<f32> = (0..rows * inner)
                .map(|i| ((i * 17 % 41) as f32 - 20.) / 16.)
                .collect();
            let w: Vec<f32> = (0..columns * inner)
                .map(|i| ((i * 13 % 37) as f32 - 18.) / 32.)
                .collect();
            let input = Tensor::from_vec(x, (1, rows, inner), &device)?.to_dtype(DType::BF16)?;
            let weight = Tensor::from_vec(w, (columns, inner), &device)?.to_dtype(DType::BF16)?;
            let linear = candle_nn::Linear::new(weight, None);
            let actual = project_fixed512(&input, &linear)?
                .flatten_all()?
                .to_dtype(DType::F32)?
                .to_vec1::<f32>()?;
            let matrix = input.reshape((rows, inner))?;
            let mut expected = Vec::new();
            for start in (0..rows).step_by(512) {
                let count = (rows - start).min(512);
                let piece = matrix.narrow(0, start, count)?;
                let padded = if count == 512 {
                    piece
                } else {
                    Tensor::cat(
                        &[
                            &piece,
                            &Tensor::zeros((512 - count, inner), DType::BF16, &device)?,
                        ],
                        0,
                    )?
                };
                expected.extend(
                    matmul_bf16(&padded, linear.weight())?
                        .narrow(0, 0, count)?
                        .flatten_all()?
                        .to_dtype(DType::F32)?
                        .to_vec1::<f32>()?,
                );
            }
            assert_eq!(actual, expected, "fixed512 allocation changed rows={rows}");
        }
        Ok(())
    }
    #[test]
    fn blaslt_bf16_error_against_exact_dyadic_reference() -> Result<()> {
        let device = crate::device::load().map_err(error)?;
        let candle_core::Device::Cuda(cuda) = &device else {
            return Ok(());
        };
        if cuda
            .cuda_stream()
            .context()
            .compute_capability()
            .map_err(error)?
            .0
            < 8
        {
            println!("BLASLT_BF16_SKIPPED compute capability below SM80; no BF16 validation");
            return Ok(());
        }
        for (rows, inner, columns) in [(3, 7, 11), (16, 32, 48), (512, 128, 256)] {
            let x: Vec<f32> = (0..rows * inner)
                .map(|i| ((i * 17 % 41) as f32 - 20.) / 16.)
                .collect();
            let w: Vec<f32> = (0..columns * inner)
                .map(|i| ((i * 13 % 37) as f32 - 18.) / 32.)
                .collect();
            let input =
                Tensor::from_vec(x.clone(), (rows, inner), &device)?.to_dtype(DType::BF16)?;
            let weight =
                Tensor::from_vec(w.clone(), (columns, inner), &device)?.to_dtype(DType::BF16)?;
            let actual = matmul_bf16(&input, &weight)?
                .to_dtype(DType::F32)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            let eager = input
                .matmul(&weight.t()?)?
                .to_dtype(DType::F32)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            let mut lt_max = 0f64;
            let mut eager_max = 0f64;
            for row in 0..rows {
                for col in 0..columns {
                    let reference: f64 = (0..inner)
                        .map(|k| x[row * inner + k] as f64 * w[col * inner + k] as f64)
                        .sum();
                    let index = row * columns + col;
                    lt_max = lt_max.max((actual[index] as f64 - reference).abs());
                    eager_max = eager_max.max((eager[index] as f64 - reference).abs());
                }
            }

            assert!(lt_max <= 2. * eager_max, "Lt {lt_max} vs eager {eager_max}");
            println!(
                "BLASLT_BF16_REFERENCE rows={rows} inner={inner} columns={columns} lt_max={lt_max} eager_max={eager_max}"
            );

            assert_eq!(
                actual,
                matmul_bf16(&input, &weight)?
                    .to_dtype(DType::F32)?
                    .flatten_all()?
                    .to_vec1::<f32>()?
            );
        }
        Ok(())
    }
}

pub(crate) fn project_transposed_weight(
    input: &Tensor,
    weight: &Tensor,
    pinned: bool,
) -> Result<Tensor> {
    let (batch, length, inner) = input.dims3()?;
    matmul_bf16_impl(
        &input.contiguous()?.reshape((batch * length, inner))?,
        weight,
        None,
        if pinned { Some(512) } else { None },
        true,
    )?
    .reshape((batch, length, weight.dim(1)?))
}

#[cfg(test)]
thread_local! {
    static MID_DOWN_PLAN_ENABLED: std::cell::Cell<bool> = const { std::cell::Cell::new(true) };
    static MID_DOWN_PLAN_CALLS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

#[cfg(test)]
use super::sm90::matmul::{EXPANSION_CALLS, EXPANSION_ENABLED};
use super::sm90::matmul::{checked_expansion, expansion_geometry};

fn checked_hopper_cooperative_plan(
    handle: sys::cublasLtHandle_t,
    desc: sys::cublasLtMatmulDesc_t,
    a: sys::cublasLtMatrixLayout_t,
    b: sys::cublasLtMatrixLayout_t,
    y: sys::cublasLtMatrixLayout_t,
    workspace: usize,
    geometry: (usize, usize),
) -> Result<sys::cublasLtMatmulHeuristicResult_t> {
    let (inner, columns) = geometry;
    if option_env!("SYS1_CUDA_COMPUTE_CAP") != Some("90") {
        candle_core::bail!("Cooperative plan requires compiled SM90");
    }
    let tile = if inner == 4096 && matches!(columns, 12288 | 24576) {
        470i32
    } else if columns == 4096 && matches!(inner, 4096 | 12288) {
        502i32
    } else {
        candle_core::bail!("Unproven cooperative geometry K={inner}/F={columns}");
    };
    unsafe {
        let mut algo = std::mem::MaybeUninit::<sys::cublasLtMatmulAlgo_t>::uninit();
        let bf16 = sys::cudaDataType::CUDA_R_16BF;
        sys::cublasLtMatmulAlgoInit(
            handle,
            sys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
            sys::cudaDataType::CUDA_R_32F,
            bf16,
            bf16,
            bf16,
            bf16,
            66,
            algo.as_mut_ptr(),
        )
        .result()
        .map_err(error)?;
        let mut algo = algo.assume_init();
        use sys::cublasLtMatmulAlgoConfigAttributes_t::*;
        for (attribute, value) in [
            (CUBLASLT_ALGO_CONFIG_TILE_ID, tile),
            (CUBLASLT_ALGO_CONFIG_STAGES_ID, 35i32),
            (CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION, 7i32),
            (CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING, 0i32),
            (CUBLASLT_ALGO_CONFIG_SPLITK_NUM, 1i32),
            (CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, 0i32),
        ] {
            sys::cublasLtMatmulAlgoConfigSetAttribute(
                &mut algo,
                attribute,
                (&value as *const i32).cast(),
                std::mem::size_of::<i32>(),
            )
            .result()
            .map_err(error)?;
            let mut actual = 0i32;
            let mut written = 0usize;
            sys::cublasLtMatmulAlgoConfigGetAttribute(
                &algo,
                attribute,
                (&mut actual as *mut i32).cast(),
                std::mem::size_of::<i32>(),
                &mut written,
            )
            .result()
            .map_err(error)?;
            if written != std::mem::size_of::<i32>() || actual != value {
                candle_core::bail!("Cooperative algorithm attribute readback differs");
            }
        }
        let cluster = 5u16;
        sys::cublasLtMatmulAlgoConfigSetAttribute(
            &mut algo,
            CUBLASLT_ALGO_CONFIG_CLUSTER_SHAPE_ID,
            (&cluster as *const u16).cast(),
            std::mem::size_of::<u16>(),
        )
        .result()
        .map_err(error)?;
        let mut actual = 0u16;
        let mut written = 0usize;
        sys::cublasLtMatmulAlgoConfigGetAttribute(
            &algo,
            CUBLASLT_ALGO_CONFIG_CLUSTER_SHAPE_ID,
            (&mut actual as *mut u16).cast(),
            std::mem::size_of::<u16>(),
            &mut written,
        )
        .result()
        .map_err(error)?;
        if actual != cluster || written != std::mem::size_of::<u16>() {
            candle_core::bail!("Cooperative cluster readback differs");
        }
        let mut checked = std::mem::MaybeUninit::<sys::cublasLtMatmulHeuristicResult_t>::uninit();
        sys::cublasLtMatmulAlgoCheck(handle, desc, a, b, y, y, &algo, checked.as_mut_ptr())
            .result()
            .map_err(error)?;
        let mut checked = checked.assume_init();
        checked.state.result().map_err(error)?;
        if checked.workspaceSize > workspace {
            candle_core::bail!("Cooperative probe exceeds unchanged {workspace} byte workspace");
        }
        checked.algo = algo;
        Ok(checked)
    }
}

pub(crate) fn project_hopper_down_cooperative(
    input: &Tensor,
    linear: &candle_nn::Linear,
) -> Result<Tensor> {
    let (batch, length, inner) = input.dims3()?;
    let rows = batch
        .checked_mul(length)
        .ok_or_else(|| error("row overflow"))?;
    if option_env!("SYS1_CUDA_COMPUTE_CAP") != Some("90")
        || input.dtype() != DType::BF16
        || linear.bias().is_some()
        || inner != 12288
        || linear.weight().dim(0)? != 4096
        || !(32768..=65536).contains(&rows)
    {
        candle_core::bail!("Unqualified cooperative down geometry")
    }
    matmul_bf16_impl_policy(
        &input.contiguous()?.reshape((rows, inner))?,
        linear.weight(),
        None,
        Some(512),
        false,
        true,
    )?
    .reshape((batch, length, 4096))
}
#[cfg(test)]
thread_local! {static HOPPER_DOWN_COOPERATIVE_SELECTIONS:std::cell::Cell<usize>=const{std::cell::Cell::new(0)};}

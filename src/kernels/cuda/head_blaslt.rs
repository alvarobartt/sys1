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
    candle_core::Error::Msg(format!("head cuBLASLt FP32: {e}"))
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
}
type Key = (usize, usize, usize, u32, u32, u32, bool);

type AlgorithmKey = (usize, usize, u32, u32, u32, u32, bool);
#[cfg(test)]
static REFERENCE_QUERIES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
struct Context {
    stream: Arc<CudaStream>,
    handle: sys::cublasLtHandle_t,
    workspace: CudaSlice<u8>,
    plans: HashMap<Key, Plan>,
    reference_algorithms: HashMap<AlgorithmKey, sys::cublasLtMatmulAlgo_t>,
}
impl Context {
    fn new(stream: Arc<CudaStream>) -> Result<Self> {
        stream.context().bind_to_thread().map_err(error)?;
        let workspace = unsafe { stream.alloc::<u8>(32 << 20) }.map_err(error)?;
        let handle = result::create_handle().map_err(error)?;
        Ok(Self {
            stream,
            handle,
            workspace,
            plans: HashMap::new(),
            reference_algorithms: HashMap::new(),
        })
    }

    fn plan(&mut self, key: Key, bias_pointer: Option<u64>) -> Result<Plan> {
        let (rows, inner, columns, align_a, align_b, align_out, has_bias) = key;
        let k = i64::try_from(inner).map_err(error)?;
        let n = i64::try_from(columns).map_err(error)?;
        let bf16 = sys::cudaDataType::CUDA_R_16BF;
        let f32 = sys::cudaDataType::CUDA_R_32F;
        let a = Layout(
            result::create_matrix_layout(bf16, inner as u64, columns as u64, k).map_err(error)?,
        );
        let b = Layout(
            result::create_matrix_layout(bf16, inner as u64, rows as u64, k).map_err(error)?,
        );
        let out = Layout(
            result::create_matrix_layout(bf16, columns as u64, rows as u64, n).map_err(error)?,
        );
        #[allow(unused_mut)]
        let mut reference_rows = if inner == 4096 && columns == 64 && !has_bias {
            16384u64
        } else {
            512u64
        };
        #[cfg(test)]
        if inner == 4096 && columns == 64 && !has_bias {
            reference_rows = GDN_PROBE_REFERENCE_ROWS.with(|v| v.get()) as u64;
        }
        let ref_b = Layout(
            result::create_matrix_layout(bf16, inner as u64, reference_rows, k).map_err(error)?,
        );
        let ref_out = Layout(
            result::create_matrix_layout(bf16, columns as u64, reference_rows, n).map_err(error)?,
        );
        let desc = Desc(
            result::create_matmul_desc(sys::cublasComputeType_t::CUBLAS_COMPUTE_32F, f32)
                .map_err(error)?,
        );
        if has_bias {
            let pointer = bias_pointer.ok_or_else(|| error("missing fused bias pointer"))?
                as *const std::ffi::c_void;
            unsafe {
                result::set_matmul_desc_attribute(
                    desc.0,
                    sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_BIAS_POINTER,
                    (&pointer as *const *const std::ffi::c_void).cast(),
                    std::mem::size_of_val(&pointer),
                )
            }
            .map_err(error)?;
            let epilogue = sys::cublasLtEpilogue_t::CUBLASLT_EPILOGUE_BIAS as u32;
            unsafe {
                result::set_matmul_desc_attribute(
                    desc.0,
                    sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_EPILOGUE,
                    (&epilogue as *const u32).cast(),
                    std::mem::size_of_val(&epilogue),
                )
            }
            .map_err(error)?;
        }
        let pref = Pref(result::create_matmul_pref().map_err(error)?);
        let transpose =
            candle_core::cuda_backend::cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_T as i32;
        let bytes = self.workspace.len() as u64;
        let reduction = sys::cublasLtReductionScheme_t::CUBLASLT_REDUCTION_SCHEME_NONE as u32;
        unsafe {
            result::set_matmul_desc_attribute(
                desc.0,
                sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_TRANSA,
                (&transpose as *const i32).cast(),
                std::mem::size_of_val(&transpose),
            )
            .map_err(error)?;
            use sys::cublasLtMatmulPreferenceAttributes_t::*;
            for (attribute, pointer, size) in [
                (
                    CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                    (&bytes as *const u64).cast(),
                    std::mem::size_of_val(&bytes),
                ),
                (
                    CUBLASLT_MATMUL_PREF_REDUCTION_SCHEME_MASK,
                    (&reduction as *const u32).cast(),
                    std::mem::size_of_val(&reduction),
                ),
                (
                    CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_A_BYTES,
                    (&align_a as *const u32).cast(),
                    std::mem::size_of_val(&align_a),
                ),
                (
                    CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_B_BYTES,
                    (&align_b as *const u32).cast(),
                    std::mem::size_of_val(&align_b),
                ),
                (
                    CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_C_BYTES,
                    (&align_out as *const u32).cast(),
                    std::mem::size_of_val(&align_out),
                ),
                (
                    CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_D_BYTES,
                    (&align_out as *const u32).cast(),
                    std::mem::size_of_val(&align_out),
                ),
            ] {
                result::set_matmul_pref_attribute(pref.0, attribute, pointer, size)
                    .map_err(error)?;
            }
            let algorithm_key = (
                inner,
                columns,
                align_a,
                align_b,
                align_out,
                bias_pointer.map(alignment).unwrap_or(0),
                has_bias,
            );
            let cached = self.reference_algorithms.get(&algorithm_key).copied();
            let algo = if let Some(algo) = cached {
                algo
            } else {
                #[cfg(test)]
                REFERENCE_QUERIES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                result::get_matmul_algo_heuristic(
                    self.handle,
                    desc.0,
                    a.0,
                    ref_b.0,
                    ref_out.0,
                    ref_out.0,
                    pref.0,
                )
                .map_err(error)?
                .algo
            };
            let mut checked =
                std::mem::MaybeUninit::<sys::cublasLtMatmulHeuristicResult_t>::uninit();
            sys::cublasLtMatmulAlgoCheck(
                self.handle,
                desc.0,
                a.0,
                b.0,
                out.0,
                out.0,
                &algo,
                checked.as_mut_ptr(),
            )
            .result()
            .map_err(error)?;
            let checked = checked.assume_init();
            checked.state.result().map_err(error)?;
            if checked.workspaceSize > self.workspace.len() {
                candle_core::bail!("head plan exceeds bounded workspace")
            }
            use sys::cublasLtMatmulAlgoConfigAttributes_t::*;
            for (attribute, allowed) in [
                (CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, 0i32),
                (CUBLASLT_ALGO_CONFIG_SPLITK_NUM, 1),
            ] {
                let mut value = 0i32;
                let mut written = 0usize;
                sys::cublasLtMatmulAlgoConfigGetAttribute(
                    &algo,
                    attribute,
                    (&mut value as *mut i32).cast(),
                    std::mem::size_of::<i32>(),
                    &mut written,
                )
                .result()
                .map_err(error)?;
                if written != std::mem::size_of::<i32>() || value < 0 || value > allowed {
                    candle_core::bail!("head plan uses an unsupported split-K/reduction")
                }
            }
            #[cfg(test)]
            if cached.is_none() && inner == 4096 && columns == 64 && !has_bias {
                let mut attributes = Vec::new();
                for attribute in [
                    CUBLASLT_ALGO_CONFIG_ID,
                    CUBLASLT_ALGO_CONFIG_TILE_ID,
                    CUBLASLT_ALGO_CONFIG_STAGES_ID,
                ] {
                    let mut value = 0i32;
                    let mut written = 0usize;
                    sys::cublasLtMatmulAlgoConfigGetAttribute(
                        &algo,
                        attribute,
                        (&mut value as *mut i32).cast(),
                        std::mem::size_of::<i32>(),
                        &mut written,
                    )
                    .result()
                    .map_err(error)?;
                    assert_eq!(written, std::mem::size_of::<i32>());
                    attributes.push(value);
                }
                println!(
                    "GDN_REFERENCE_ALGORITHM {}",
                    serde_json::json!({
                        "reference_rows":reference_rows,"actual_rows":rows,"K":inner,"N":columns,
                        "alignments":[align_a,align_b,align_out],"algorithm_id_tile_stages":attributes,
                        "workspace_bytes":checked.workspaceSize
                    })
                );
            }
            if cached.is_none() {
                if self.reference_algorithms.len() >= 128 {
                    self.reference_algorithms.clear();
                }
                self.reference_algorithms.insert(algorithm_key, algo);
            }
            Ok(Plan {
                a,
                b,
                out,
                desc,
                algo,
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

fn dot(input: &Tensor, weight: &Tensor, bias: Option<&Tensor>) -> Result<Tensor> {
    let (rows, inner) = input.dims2()?;
    let (columns, wi) = weight.dims2()?;
    if rows < 512
        || inner != wi
        || input.dtype() != DType::BF16
        || weight.dtype() != DType::BF16
        || !input.device().same_device(weight.device())
        || !input.device().is_cuda()
    {
        candle_core::bail!("unsupported BF16/F32 head matrices")
    }
    let count = rows
        .checked_mul(columns)
        .ok_or_else(|| error("output size overflow"))?;
    if [input.elem_count(), weight.elem_count(), count]
        .iter()
        .any(|&n| n > i32::MAX as usize)
    {
        candle_core::bail!("head projection index bound")
    }
    let input = input.contiguous()?;
    let weight = weight.contiguous()?;
    let (xs, xl) = input.storage_and_layout();
    let (ws, wl) = weight.storage_and_layout();
    let (Storage::Cuda(xs), Storage::Cuda(ws)) = (&*xs, &*ws) else {
        candle_core::bail!("head projection needs CUDA")
    };
    let (xb, xe) = xl
        .contiguous_offsets()
        .ok_or_else(|| error("input layout"))?;
    let (wb, we) = wl
        .contiguous_offsets()
        .ok_or_else(|| error("weight layout"))?;
    let x = xs.as_cuda_slice::<half::bf16>()?.slice(xb..xe);
    let w = ws.as_cuda_slice::<half::bf16>()?.slice(wb..we);
    let device = xs.device().clone();
    let stream = device.cuda_stream();
    stream.context().bind_to_thread().map_err(error)?;
    let mut output = unsafe { device.alloc::<half::bf16>(count)? };
    let (xp, _xg) = x.device_ptr(&stream);
    let (wp, _wg) = w.device_ptr(&stream);
    let (yp, yg) = output.device_ptr_mut(&stream);

    let bias = bias.map(Tensor::contiguous).transpose()?;
    if let Some(bias) = &bias
        && (bias.dtype() != DType::BF16
            || bias.dims() != [columns]
            || !bias.device().same_device(input.device()))
    {
        candle_core::bail!("invalid fused BF16 head bias")
    }
    let bias_storage = bias.as_ref().map(Tensor::storage_and_layout);
    let bias_values = if let Some((storage, layout)) = &bias_storage {
        let Storage::Cuda(storage) = &**storage else {
            candle_core::bail!("bias needs CUDA")
        };
        let (start, end) = layout
            .contiguous_offsets()
            .ok_or_else(|| error("bias layout"))?;
        Some(storage.as_cuda_slice::<half::bf16>()?.slice(start..end))
    } else {
        None
    };
    let bias_pointer_guard = bias_values.as_ref().map(|v| v.device_ptr(&stream));
    let bias_pointer = bias_pointer_guard.as_ref().map(|(pointer, _)| *pointer);

    CONTEXT.with(|cell| -> Result<()> {
        let mut slot = cell.try_borrow_mut().map_err(error)?;
        if slot
            .as_ref()
            .is_none_or(|c| !Arc::ptr_eq(&c.stream, &stream))
        {
            *slot = Some(Context::new(stream.clone())?);
        }
        let context = slot.as_mut().ok_or_else(|| error("missing context"))?;
        let key = (
            rows,
            inner,
            columns,
            alignment(wp),
            alignment(xp),
            alignment(yp),
            bias_pointer.is_some(),
        );
        if !context.plans.contains_key(&key) {
            if context.plans.len() >= 128 {
                context.plans.clear();
            }
            let plan = context.plan(key, bias_pointer)?;
            context.plans.insert(key, plan);
        }
        let workspace_bytes = context.workspace.len();
        let plan = &context.plans[&key];
        if let Some(pointer) = bias_pointer {
            let pointer = pointer as *const std::ffi::c_void;
            unsafe {
                result::set_matmul_desc_attribute(
                    plan.desc.0,
                    sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_BIAS_POINTER,
                    (&pointer as *const *const std::ffi::c_void).cast(),
                    std::mem::size_of_val(&pointer),
                )
            }
            .map_err(error)?;
        }

        let (work, _work_guard) = context.workspace.device_ptr_mut(&stream);
        let alpha = 1f32;
        let beta = 0f32;
        unsafe {
            result::matmul(
                context.handle,
                plan.desc.0,
                (&alpha as *const f32).cast(),
                (&beta as *const f32).cast(),
                wp as *const _,
                plan.a.0,
                xp as *const _,
                plan.b.0,
                yp as *const _,
                plan.out.0,
                yp as *mut _,
                plan.out.0,
                &plan.algo,
                work as *mut _,
                workspace_bytes,
                stream.cu_stream().cast(),
            )
        }
        .map_err(error)
    })?;
    drop(yg);
    Ok(Tensor::from_storage(
        Storage::Cuda(CudaStorage::wrap_cuda_slice(output, device)),
        (rows, columns),
        BackpropOp::none(),
        false,
    ))
}

pub(crate) fn eligible(input: &Tensor, linear: &candle_nn::Linear) -> bool {
    let Some(&inner) = input.dims().last() else {
        return false;
    };
    let Ok((columns, wi)) = linear.weight().dims2() else {
        return false;
    };
    option_env!("SYS1_CUDA_COMPUTE_CAP") == Some("90")
        && input.device().is_cuda()
        && input.dtype() == DType::BF16
        && linear.weight().dtype() == DType::BF16
        && input.device().same_device(linear.weight().device())
        && linear
            .bias()
            .is_some_and(|b| b.dtype() == DType::BF16 && b.device().same_device(input.device()))
        && wi == inner
        && input.elem_count() > 0
        && inner > 0
        && input.elem_count() / inner <= 65536
        && ((inner == 1024 && matches!(columns, 1024 | 2048 | 3072 | 4096))
            || (inner == 4096 && columns == 1024))
}

pub(crate) fn project(input: &Tensor, linear: &candle_nn::Linear) -> Result<Tensor> {
    let inner = *input
        .dims()
        .last()
        .ok_or_else(|| error("missing inner dimension"))?;
    let rows = input.elem_count() / inner;
    let columns = linear.weight().dim(0)?;
    let matrix = input.contiguous()?.reshape((rows, inner))?;
    let padded = if rows < 512 {
        let zeros = Tensor::zeros((512 - rows, inner), DType::BF16, input.device())?;
        Tensor::cat(&[&matrix, &zeros], 0)?
    } else {
        matrix
    };

    let output = dot(&padded, linear.weight(), linear.bias())?.narrow(0, 0, rows)?;
    let mut shape = input.dims().to_vec();
    *shape
        .last_mut()
        .ok_or_else(|| error("missing output dimension"))? = columns;
    output.reshape(shape)
}

pub(crate) fn project_gdn_ba(input: &Tensor, linear: &candle_nn::Linear) -> Result<Tensor> {
    let inner = input.dim(candle_core::D::Minus1)?;
    let columns = linear.weight().dim(0)?;
    let rows = input.elem_count() / inner.max(1);
    if !matches!(option_env!("SYS1_CUDA_COMPUTE_CAP"), Some("80" | "90"))
        || !input.device().is_cuda()
        || input.dtype() != DType::BF16
        || linear.weight().dtype() != DType::BF16
        || !input.device().same_device(linear.weight().device())
        || linear.bias().is_some()
        || inner != 4096
        || !matches!(columns, 32 | 64)
        || linear.weight().dim(1)? != inner
        || !(1..=65536).contains(&rows)
    {
        candle_core::bail!("unsupported GDN BF16/F32 matrices")
    }
    let matrix = input.contiguous()?.reshape((rows, inner))?;
    let padded = if rows < 512 {
        let zeros = Tensor::zeros((512 - rows, inner), DType::BF16, input.device())?;
        Tensor::cat(&[&matrix, &zeros], 0)?
    } else {
        matrix
    };

    let weight = if columns == 32 {
        let zeros = Tensor::zeros((32, inner), DType::BF16, input.device())?;
        Tensor::cat(&[linear.weight(), &zeros], 0)?
    } else {
        linear.weight().clone()
    };
    let output = dot(&padded, &weight, None)?
        .narrow(0, 0, rows)?
        .narrow(1, 0, columns)?;
    let mut shape = input.dims().to_vec();
    *shape
        .last_mut()
        .ok_or_else(|| error("missing GDN output dimension"))? = columns;
    output.reshape(shape)
}

#[cfg(test)]
thread_local! {
    static GDN_PROBE_REFERENCE_ROWS: std::cell::Cell<usize> = const { std::cell::Cell::new(16384) };
}

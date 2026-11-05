use candle_core::cuda_backend::cudarc::cublaslt::sys;

pub(crate) fn expansion_geometry(
    rows: usize,
    inner: usize,
    columns: usize,
    reference: Option<usize>,
    nn: bool,
) -> bool {
    (16384..=65536).contains(&rows)
        && inner == 4096
        && matches!(columns, 8192 | 12288)
        && reference.is_none()
        && !nn
}

#[cfg(test)]
thread_local! {
    pub(crate) static EXPANSION_ENABLED:std::cell::Cell<bool>=const{std::cell::Cell::new(true)};
    pub(crate) static EXPANSION_CALLS:std::cell::Cell<usize>=const{std::cell::Cell::new(0)};
}

pub(crate) unsafe fn checked_expansion(
    handle: sys::cublasLtHandle_t,
    desc: sys::cublasLtMatmulDesc_t,
    a: sys::cublasLtMatrixLayout_t,
    b: sys::cublasLtMatrixLayout_t,
    y: sys::cublasLtMatrixLayout_t,
    workspace: usize,
    large_qkv: bool,
) -> Option<sys::cublasLtMatmulHeuristicResult_t> {
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
        .ok()?;
        let mut algo = algo.assume_init();
        use sys::cublasLtMatmulAlgoConfigAttributes_t::*;
        for (attribute, value) in [
            (
                CUBLASLT_ALGO_CONFIG_TILE_ID,
                if large_qkv { 197i32 } else { 29i32 },
            ),
            (CUBLASLT_ALGO_CONFIG_STAGES_ID, 35),
            (
                CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION,
                if large_qkv { 2 } else { 1 },
            ),
            (CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING, 0),
            (CUBLASLT_ALGO_CONFIG_SPLITK_NUM, 1),
            (CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, 0),
        ] {
            sys::cublasLtMatmulAlgoConfigSetAttribute(
                &mut algo,
                attribute,
                (&value as *const i32).cast(),
                std::mem::size_of::<i32>(),
            )
            .result()
            .ok()?;
        }
        let cluster = if large_qkv { 5u16 } else { 3u16 };
        sys::cublasLtMatmulAlgoConfigSetAttribute(
            &mut algo,
            CUBLASLT_ALGO_CONFIG_CLUSTER_SHAPE_ID,
            (&cluster as *const u16).cast(),
            std::mem::size_of::<u16>(),
        )
        .result()
        .ok()?;
        let mut checked = std::mem::MaybeUninit::<sys::cublasLtMatmulHeuristicResult_t>::uninit();
        sys::cublasLtMatmulAlgoCheck(handle, desc, a, b, y, y, &algo, checked.as_mut_ptr())
            .result()
            .ok()?;
        let mut checked = checked.assume_init();
        checked.state.result().ok()?;
        if checked.workspaceSize > workspace {
            return None;
        }
        checked.algo = algo;
        Some(checked)
    }
}

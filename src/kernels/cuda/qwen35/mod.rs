pub(crate) mod gated_delta;

const CUDA_PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/cuda.ptx"));

#[cfg(feature = "cuda")]
#[path = "src/kernels/cuda/build.rs"]
mod cuda;
#[cfg(any(feature = "cuda", feature = "metal"))]
#[path = "src/kernels/build.rs"]
mod kernel_build;
#[cfg(feature = "metal")]
#[path = "src/kernels/metal/build.rs"]
mod metal;

fn main() {
    for name in [
        "sys1_cuda_fused_ffn",
        "sys1_cuda_fp64_emulation",
        "sys1_cuda_direct_ffn",
        "sys1_cuda_fa4_sm90",
        "sys1_cuda_gdn_sm80",
        "sys1_cuda_gdn_sm90",
    ] {
        println!("cargo:rustc-check-cfg=cfg({name})");
    }
    #[cfg(feature = "cuda")]
    cuda::compile();
    #[cfg(feature = "metal")]
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("macos") {
        metal::compile();
    }
}

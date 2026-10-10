#[cfg(not(any(feature = "cpu", feature = "cuda", feature = "metal")))]
compile_error!("enable one backend feature: cpu, cuda, or metal");
#[cfg(any(
    all(feature = "cpu", feature = "cuda"),
    all(feature = "cpu", feature = "metal"),
    all(feature = "cuda", feature = "metal")
))]
compile_error!("backend features are mutually exclusive: choose cpu, cuda, or metal");
#[cfg(all(feature = "metal", not(target_os = "macos")))]
compile_error!("the `metal` feature is only supported when targeting macOS");
#[cfg(all(feature = "cuda", target_os = "macos"))]
compile_error!("the `cuda` feature is not supported when targeting macOS");

#[cfg(feature = "metal")]
pub use device::report_dtype_support;

pub mod api;
pub(crate) mod archs;
pub mod batching;
mod device;
pub mod hub;
#[cfg(feature = "metal")]
pub(crate) mod kernels;
pub(crate) mod media;
mod metrics;
pub mod models;
pub mod schema;
pub mod tokenizer;

pub fn validate_backend() -> anyhow::Result<()> {
    device::validate_available()
}

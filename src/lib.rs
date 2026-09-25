#[cfg(not(any(feature = "cpu", feature = "cuda", feature = "metal")))]
compile_error!("enable one backend feature: cpu, cuda, or metal");
#[cfg(any(
    all(feature = "cpu", feature = "cuda"),
    all(feature = "cpu", feature = "metal"),
    all(feature = "cuda", feature = "metal")
))]
compile_error!("backend features are mutually exclusive: choose cpu, cuda, or metal");
#[cfg(all(feature = "flash-attn-2", feature = "flash-attn-3"))]
compile_error!(
    "Flash Attention features are mutually exclusive: choose flash-attn-2 or flash-attn-3"
);
#[cfg(all(feature = "metal", not(target_os = "macos")))]
compile_error!("the `metal` feature is only supported when targeting macOS");
#[cfg(all(feature = "cuda", target_os = "macos"))]
compile_error!("the `cuda` feature is not supported when targeting macOS");

pub mod api;
pub mod batching;
mod device;
pub mod hub;
pub mod models;
pub mod schema;
pub mod tokenizer;

pub fn validate_backend() -> anyhow::Result<()> {
    device::validate_available()
}

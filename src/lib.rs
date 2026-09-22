#[cfg(not(any(feature = "cpu", feature = "cuda", feature = "metal")))]
compile_error!("enable one backend feature: cpu, cuda, or metal");
#[cfg(any(
    all(feature = "cpu", feature = "cuda"),
    all(feature = "cpu", feature = "metal"),
    all(feature = "cuda", feature = "metal")
))]
compile_error!("backend features are mutually exclusive: choose cpu, cuda, or metal");

pub mod api;
pub mod batching;
mod device;
pub mod hub;
pub mod models;
pub mod schema;
pub mod tokenizer;

pub(crate) mod activation;
pub(crate) mod cpu;
pub(crate) mod linear;
pub(crate) mod normalization;

#[cfg(feature = "cuda")]
pub(crate) mod cuda;
#[cfg(feature = "metal")]
pub(crate) mod metal;

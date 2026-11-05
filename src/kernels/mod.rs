pub(crate) mod activation;
pub(crate) mod linear;

#[cfg(feature = "cuda")]
pub(crate) mod cuda;
#[cfg(feature = "metal")]
pub(crate) mod metal;

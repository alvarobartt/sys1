use candle_core::Device;

pub fn validate_available() -> anyhow::Result<()> {
    #[cfg(feature = "cuda")]
    anyhow::ensure!(
        candle_core::utils::cuda_is_available(),
        "CUDA backend selected, but no CUDA device is available"
    );
    #[cfg(feature = "metal")]
    anyhow::ensure!(
        candle_core::utils::metal_is_available(),
        "Metal backend selected, but no Metal device is available"
    );
    Ok(())
}

#[cfg(all(feature = "cpu", not(any(feature = "cuda", feature = "metal"))))]
pub fn load() -> anyhow::Result<Device> {
    Ok(Device::Cpu)
}

#[cfg(all(feature = "cuda", not(any(feature = "cpu", feature = "metal"))))]
pub fn load() -> anyhow::Result<Device> {
    Ok(Device::new_cuda(0)?)
}

#[cfg(all(feature = "metal", not(any(feature = "cpu", feature = "cuda"))))]
pub fn load() -> anyhow::Result<Device> {
    Ok(Device::new_metal(0)?)
}

#[cfg(not(any(
    all(feature = "cpu", not(any(feature = "cuda", feature = "metal"))),
    all(feature = "cuda", not(any(feature = "cpu", feature = "metal"))),
    all(feature = "metal", not(any(feature = "cpu", feature = "cuda")))
)))]
pub fn load() -> anyhow::Result<Device> {
    unreachable!()
}

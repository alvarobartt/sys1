use candle_core::Device;

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

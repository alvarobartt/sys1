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

#[cfg(feature = "metal")]
pub fn report_dtype_support(dtype: candle_core::DType) {
    use objc2_metal::{MTLDevice, MTLGPUFamily};

    if dtype != candle_core::DType::BF16 {
        return;
    }
    let Ok(device) = Device::new_metal(0) else {
        return;
    };
    let Ok(metal) = device.as_metal_device() else {
        return;
    };
    let handle = metal.metal_device();
    let raw = handle.as_ref();
    if raw.supportsFamily(MTLGPUFamily::Apple9) {
        return;
    }
    tracing::warn!(
        gpu = %raw.name(),
        "BF16 matmuls run at F32 speed as Apple families before Apple9 i.e., M-series before \
         M3, have no widening for BF16, so in this machine F16 runs about 25-30% faster on \
         the same shapes."
    );
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

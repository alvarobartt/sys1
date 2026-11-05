use candle_core::{
    Device, Result,
    cuda_backend::cudarc::driver::{CudaEvent, sys},
};
use std::{cell::RefCell, collections::BTreeMap};

struct Checkpoint {
    layer: usize,
    stage: &'static str,
    event: CudaEvent,
}

thread_local! {
    static CHECKPOINTS: RefCell<Option<Vec<Checkpoint>>> = const { RefCell::new(None) };
}

pub(crate) fn begin(device: &Device) -> Result<()> {
    let enabled =
        std::env::var_os("SYS1_CUDA_EVENT_PROFILE").as_deref() == Some(std::ffi::OsStr::new("1"));
    let enabled = enabled && device.is_cuda();
    CHECKPOINTS.with(|checkpoints| {
        *checkpoints.borrow_mut() = enabled.then(Vec::new);
    });
    Ok(())
}

pub(crate) fn mark(device: &Device, layer: usize, stage: &'static str) -> Result<()> {
    let active = CHECKPOINTS.with(|checkpoints| checkpoints.borrow().is_some());
    if !active {
        return Ok(());
    }
    let Device::Cuda(cuda) = device else {
        candle_core::bail!("Qwen3.5 CUDA event profile requires CUDA")
    };
    let event = cuda
        .cuda_stream()
        .record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))
        .map_err(|error| candle_core::Error::Msg(format!("CUDA event record: {error}")))?;
    CHECKPOINTS.with(|checkpoints| {
        checkpoints.borrow_mut().as_mut().unwrap().push(Checkpoint {
            layer,
            stage,
            event,
        });
    });
    Ok(())
}

pub(crate) fn finish(batch: usize, length: usize) -> Result<()> {
    let Some(checkpoints) = CHECKPOINTS.with(|checkpoints| checkpoints.borrow_mut().take()) else {
        return Ok(());
    };
    let mut totals = BTreeMap::<&'static str, f64>::new();
    let mut counts = BTreeMap::<&'static str, usize>::new();
    for pair in checkpoints.windows(2) {
        let [previous, current] = pair else {
            unreachable!()
        };
        if previous.layer != current.layer || current.stage == "start" {
            continue;
        }
        let elapsed = previous.event.elapsed_ms(&current.event).map_err(|error| {
            candle_core::Error::Msg(format!("CUDA event elapsed time: {error}"))
        })?;
        if !elapsed.is_finite() || elapsed < 0. {
            candle_core::bail!("Invalid Qwen3.5 CUDA event duration")
        }
        *totals.entry(current.stage).or_default() += f64::from(elapsed);
        *counts.entry(current.stage).or_default() += 1;
    }
    tracing::info!(
        batch,
        length,
        checkpoints = checkpoints.len(),
        ?totals,
        ?counts,
        "Qwen3.5 CUDA stream event profile (diagnostic, not request latency)"
    );
    Ok(())
}

use crate::kernel_build::kernel_sources;
use std::{env, path::PathBuf, process::Command};

pub(crate) fn compile() {
    let sources = kernel_sources("src/kernels/metal", "metal");
    let output_dir = PathBuf::from(env::var_os("OUT_DIR").unwrap());
    let library = output_dir.join("kernels.metallib");
    let mut air_files = Vec::with_capacity(sources.len());
    for source in sources {
        let air = output_dir
            .join(source.file_stem().unwrap())
            .with_extension("air");
        let metal = Command::new("xcrun")
            .args(["--toolchain", "Metal", "metal", "-std=metal3.0", "-c"])
            .arg(&source)
            .arg("-o")
            .arg(&air)
            .status()
            .expect("failed to run the Metal compiler");
        assert!(metal.success(), "failed to compile {}", source.display());
        air_files.push(air);
    }
    let link = Command::new("xcrun")
        .args(["--toolchain", "Metal", "metallib"])
        .args(&air_files)
        .arg("-o")
        .arg(&library)
        .status()
        .expect("failed to run metallib");
    assert!(link.success(), "failed to link Metal kernels");
    println!(
        "cargo:rustc-env=SYS1_KERNEL_METALLIB_PATH={}",
        library.display()
    );
}

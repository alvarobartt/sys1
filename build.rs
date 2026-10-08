use std::{env, path::PathBuf, process::Command};

fn main() {
    println!("cargo:rerun-if-env-changed=CUDA_COMPUTE_CAP");
    if env::var_os("CARGO_FEATURE_METAL").is_some()
        && env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("macos")
    {
        println!("cargo:rerun-if-changed=src/kernels/qwen35.metal");
        let output_dir = PathBuf::from(env::var_os("OUT_DIR").unwrap());
        let air = output_dir.join("qwen35.air");
        let library = output_dir.join("qwen35.metallib");
        let metal = Command::new("xcrun")
            .args([
                "--toolchain",
                "Metal",
                "metal",
                "-std=metal3.0",
                "-c",
                "src/kernels/qwen35.metal",
                "-o",
            ])
            .arg(&air)
            .status()
            .expect("Failed to run the Metal compiler for Qwen3.5 kernels");
        assert!(metal.success(), "Failed to compile Qwen3.5 Metal kernels");
        let link = Command::new("xcrun")
            .args(["--toolchain", "Metal", "metallib"])
            .arg(&air)
            .arg("-o")
            .arg(&library)
            .status()
            .expect("Failed to run metallib for Qwen3.5 kernels");
        assert!(link.success(), "Failed to link Qwen3.5 Metal kernels");
        println!(
            "cargo:rustc-env=SYS1_KERNEL_METALLIB_PATH={}",
            library.display()
        );
    }
    if env::var_os("CARGO_FEATURE_CUDA").is_some() {
        println!("cargo:rerun-if-changed=src/kernels/qwen35.cu");
        let nvcc = env::var_os("CUDA_ROOT")
            .map(|root| PathBuf::from(root).join("bin/nvcc"))
            .unwrap_or_else(|| PathBuf::from("nvcc"));
        let output = PathBuf::from(env::var_os("OUT_DIR").unwrap()).join("qwen35.ptx");
        let status = Command::new(nvcc)
            .args(["-ptx", "-arch=compute_75", "src/kernels/qwen35.cu", "-o"])
            .arg(&output)
            .status()
            .expect("Failed to run nvcc for Qwen3.5 CUDA kernels");
        assert!(status.success(), "Failed to compile Qwen3.5 CUDA kernels");
        println!("cargo:rustc-env=SYS1_KERNEL_PTX_PATH={}", output.display());
    }

    let flash_attention_2 = env::var_os("CARGO_FEATURE_FLASH_ATTN_2").is_some();
    let flash_attention_3 = env::var_os("CARGO_FEATURE_FLASH_ATTN_3").is_some();
    if !flash_attention_2 && !flash_attention_3 {
        return;
    }

    let Ok(value) = env::var("CUDA_COMPUTE_CAP") else {
        println!(
            "cargo:warning=CUDA_COMPUTE_CAP is not set; the Flash Attention architecture will be checked at startup"
        );
        return;
    };
    let capability = parse_compute_capability(&value);
    match (flash_attention_2, capability) {
        (true, 80..=99) | (false, 90) => {}
        (true, _) => panic!(
            "`flash-attn-2` supports compute capability 8.x or 9.x; got {:?}",
            value
        ),
        (false, _) => panic!(
            "`flash-attn-3` requires Hopper compute capability 9.0; got {:?}",
            value
        ),
    }
}

fn parse_compute_capability(value: &str) -> u32 {
    let normalized = value.trim().to_ascii_lowercase();
    let normalized = normalized
        .strip_prefix("sm_")
        .unwrap_or(&normalized)
        .trim_end_matches(['a', 'f'])
        .replace('.', "");
    normalized.parse::<u32>().unwrap_or_else(|_| {
        panic!(
            "invalid CUDA_COMPUTE_CAP {:?}; expected values such as 80, 89, or 90",
            value
        )
    })
}

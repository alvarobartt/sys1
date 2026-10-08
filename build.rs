use std::{env, path::PathBuf, process::Command};

fn main() {
    println!("cargo:rerun-if-env-changed=CUDA_COMPUTE_CAP");

    if env::var_os("CARGO_FEATURE_CUDA").is_none() {
        return;
    }

    let capability = compute_capability();
    let flash_attention_2 = env::var_os("CARGO_FEATURE_FLASH_ATTN_2").is_some();
    let flash_attention_3 = env::var_os("CARGO_FEATURE_FLASH_ATTN_3").is_some();
    if flash_attention_2 || flash_attention_3 {
        match (flash_attention_2, capability) {
            (true, 80..=99) | (false, 90) => {}
            (true, _) => {
                panic!("`flash-attn-2` supports compute capability 8.x or 9.x; got {capability}")
            }
            (false, _) => {
                panic!("`flash-attn-3` requires Hopper compute capability 9.0; got {capability}")
            }
        }
    }

    println!("cargo:rerun-if-changed=src/kernels/modernbert.cu");
    let output = PathBuf::from(env::var_os("OUT_DIR").unwrap()).join("modernbert.ptx");
    let status = Command::new("nvcc")
        .args([
            "-ptx",
            &format!("-arch=compute_{capability}"),
            "src/kernels/modernbert.cu",
            "-o",
        ])
        .arg(&output)
        .status()
        .expect("failed to run nvcc for ModernBERT CUDA kernels");
    assert!(
        status.success(),
        "failed to compile ModernBERT CUDA kernels for compute capability {capability}"
    );
}

fn compute_capability() -> u32 {
    match env::var("CUDA_COMPUTE_CAP") {
        Ok(value) => parse_compute_capability(&value),
        Err(env::VarError::NotPresent) => {
            let output = Command::new("nvidia-smi")
                .args(["--query-gpu=compute_cap", "--format=csv,noheader"])
                .output()
                .expect("failed to detect CUDA compute capability with nvidia-smi; set CUDA_COMPUTE_CAP when building without a GPU");
            assert!(
                output.status.success(),
                "nvidia-smi failed to detect CUDA compute capability; set CUDA_COMPUTE_CAP when building without a GPU: {}",
                String::from_utf8_lossy(&output.stderr)
            );
            let value = String::from_utf8(output.stdout)
                .expect("nvidia-smi returned invalid UTF-8 for CUDA compute capability");
            parse_compute_capability(value.lines().next().unwrap_or_default())
        }
        Err(error) => panic!("invalid CUDA_COMPUTE_CAP: {error}"),
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

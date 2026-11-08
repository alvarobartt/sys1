use crate::kernel_build::kernel_sources;
use std::{env, fs, path::PathBuf, process::Command};
#[path = "../../assets.rs"]
mod assets;

pub(crate) fn compile() {
    println!("cargo:rerun-if-env-changed=CUDA_COMPUTE_CAP");
    let capability = compute_capability();
    compile_assets(capability);
    println!("cargo:rustc-env=SYS1_CUDA_COMPUTE_CAP={capability}");
    #[cfg(feature = "cuda")]
    compile_fused_gate_up(capability);
    #[cfg(feature = "cuda")]
    compile_direct_gate_up(capability);
    let sources: Vec<_> = kernel_sources("src/kernels/cuda", "cu")
        .into_iter()
        .filter(|source| {
            let parts: Vec<_> = source.components().map(|part| part.as_os_str()).collect();
            if parts.iter().any(|part| *part == "sm80") && capability < 80 {
                return false;
            }
            if parts.iter().any(|part| *part == "sm90") && capability != 90 {
                return false;
            }
            if source
                .file_name()
                .is_some_and(|name| name == "operand_reuse.cu")
            {
                return capability == 89;
            }
            true
        })
        .collect();
    let output_dir = PathBuf::from(env::var_os("OUT_DIR").unwrap());
    let unit = output_dir.join("kernels.cu");
    let includes = sources
        .iter()
        .map(|path| {
            format!(
                "#include \"{}\"\n",
                path.strip_prefix("src/kernels/cuda")
                    .unwrap()
                    .to_str()
                    .unwrap()
            )
        })
        .collect::<String>();
    fs::write(&unit, includes).expect("failed to generate CUDA translation unit");
    let output = output_dir.join("cuda.ptx");
    let status = Command::new("nvcc")
        .args(["-ptx", &format!("-arch=compute_{capability}"), "-I"])
        .arg("src/kernels/cuda")
        .arg(&unit)
        .arg("-o")
        .arg(&output)
        .status()
        .expect("failed to run nvcc for CUDA kernels");
    assert!(
        status.success(),
        "failed to compile CUDA kernels for compute capability {capability}"
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

#[cfg(feature = "cuda")]
fn compile_fused_gate_up(capability: u32) {
    if !matches!(capability, 80 | 120) {
        return;
    }
    println!("cargo:rerun-if-changed=src/kernels/cuda/linear/fused_qkvz.cuh");
    let source = "src/kernels/cuda/linear/fused_gate_up.cuh";
    println!("cargo:rerun-if-changed={source}");
    let out = PathBuf::from(env::var_os("OUT_DIR").unwrap());
    let includes = cudaforge::DependencyManager::new()
        .with_cutlass(Some("7d49e6c7e2f8896c47f586706e67e1fb215529dc"))
        .fetch_all(&out)
        .expect("failed to fetch pinned CUTLASS headers");
    let status = Command::new("nvcc")
        .args(["-x", "cu", "-ptx", "-std=c++17", "-O3", "-lineinfo"])
        .arg(format!("-arch=compute_{capability}"))
        .args(includes)
        .arg(source)
        .arg("-o")
        .arg(out.join("fused_gate_up.ptx"))
        .status()
        .expect("failed to compile fused MLP PTX");
    assert!(status.success(), "fused MLP PTX compile failed");
    println!("cargo:rustc-cfg=sys1_cuda_fused_ffn");
}

fn compile_assets(capability: u32) {
    println!("cargo:rustc-check-cfg=cfg(sys1_cuda_fa4_sm90)");
    #[cfg(feature = "cuda")]
    {
        let out = PathBuf::from(env::var_os("OUT_DIR").unwrap());
        println!("cargo:rustc-check-cfg=cfg(sys1_cuda_gdn_sm80)");
        println!("cargo:rustc-check-cfg=cfg(sys1_cuda_gdn_sm90)");
        let architecture = match capability {
            80 | 86 | 89 => Some(("qwen35_aot_assets_sm80", "sys1_cuda_gdn_sm80")),
            90 => Some(("qwen35_aot_assets_v91", "sys1_cuda_gdn_sm90")),
            _ => None,
        };
        if let Some((architecture, cfg)) = architecture {
            println!("cargo:rustc-cfg={cfg}");
            fs::create_dir_all(out.join(architecture)).expect("create CUDA asset directory");
            for index in 0..5 {
                let name = format!("kernels/{architecture}/{index}.cubin");
                let source = assets::get(&name).expect("download pinned CUDA asset");
                fs::copy(
                    source,
                    out.join(architecture).join(format!("{index}.cubin")),
                )
                .expect("stage CUDA asset");
            }
        }
        if capability == 90 {
            let source = assets::get("attention_fa4_assets_v112/fa4_sm90_bf16_d256.fatbin")
                .expect("download pinned FA4 image");
            fs::copy(source, out.join("fa4_sm90_bf16_d256.fatbin")).expect("stage FA4 image");
            println!("cargo:rustc-cfg=sys1_cuda_fa4_sm90");
        }
    }
    #[cfg(not(feature = "cuda"))]
    let _ = capability;
}

#[cfg(feature = "cuda")]
fn compile_direct_gate_up(capability: u32) {
    if !matches!(capability, 86 | 89) {
        return;
    }
    let source = "src/kernels/cuda/linear/direct_mlp.cuh";
    println!("cargo:rerun-if-changed={source}");
    let out = PathBuf::from(env::var_os("OUT_DIR").unwrap());
    let includes = cudaforge::DependencyManager::new()
        .with_cutlass(Some("7d49e6c7e2f8896c47f586706e67e1fb215529dc"))
        .fetch_all(&out)
        .expect("pinned CUTLASS headers");
    let status = Command::new("nvcc")
        .args(["-x", "cu", "-ptx", "-std=c++17", "-O3", "-lineinfo"])
        .arg(format!("-arch=compute_{capability}"))
        .args(includes)
        .arg(source)
        .arg("-o")
        .arg(out.join("fused_gate_up_direct.ptx"))
        .status()
        .expect("compile original-weight MLP");
    assert!(status.success(), "direct MLP compilation failed");
    println!("cargo:rustc-cfg=sys1_cuda_direct_ffn");
}

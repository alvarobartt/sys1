use std::env;

fn main() {
    println!("cargo:rerun-if-env-changed=CUDA_COMPUTE_CAP");

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

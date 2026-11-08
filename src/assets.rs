use sha2::{Digest, Sha256};
use std::{fs, path::PathBuf, time::Duration};

const REVISION: &str = "fe16e9b22cd05ebb799f242df9ac68470d3e3d4c";
const FILES: &[(&str, &str)] = &[
    (
        "attention_fa4_assets_v112/fa4_sm90_bf16_d256.fatbin",
        "2586383961d29ef0470dfd31294edb194f9329a18b3888aea8cc2236317c1c75",
    ),
    (
        "kernels/qwen35_aot_assets_sm80/0.cubin",
        "f9c7791e1616c7293f76762fc426211cbe5c5379ac4c40bea97ab03223eb15ca",
    ),
    (
        "kernels/qwen35_aot_assets_sm80/1.cubin",
        "241ac534a0f67f58fb683861e3d4e7a5858d73ea3dd07e908dc518640e1bb9d5",
    ),
    (
        "kernels/qwen35_aot_assets_sm80/2.cubin",
        "8e1e6cfd3f92d3eab06a4bc1327b8931c6861b94afc0624335e7818b5e987f5f",
    ),
    (
        "kernels/qwen35_aot_assets_sm80/3.cubin",
        "4882e21b2d925cd845bd3bcda3d488990063b903817c299ca62a89116ff054f6",
    ),
    (
        "kernels/qwen35_aot_assets_sm80/4.cubin",
        "b15c92f459c7e3bab0f266f55f4e235e67d91cb67b67911f8346b40ce80a073e",
    ),
    (
        "kernels/qwen35_aot_assets_v91/0.cubin",
        "2418dd6bbf431add62c383cc2b5f6b63b0e30d49ea6bab16d972a07288e5cf7b",
    ),
    (
        "kernels/qwen35_aot_assets_v91/1.cubin",
        "bea70ef64e71d48bab0214596afad3484650b1bac6c2f8f5b0ceca5fa67ed798",
    ),
    (
        "kernels/qwen35_aot_assets_v91/2.cubin",
        "6daadd369603bf01467ec44902e7cb50afa9426ef86c8ae86bbc631fa2db25e7",
    ),
    (
        "kernels/qwen35_aot_assets_v91/3.cubin",
        "90e25a17d7fa0f4ca174c97804ccc094ce5098ac4dab8a23987e118c5bf9bc94",
    ),
    (
        "kernels/qwen35_aot_assets_v91/4.cubin",
        "e583b7f858c2a767b6ebe69869214157e40b95721fecb417ba4a292ce8264b0c",
    ),
];

pub(crate) fn get(name: &str) -> Result<PathBuf, Box<dyn std::error::Error + Send + Sync>> {
    let expected = FILES.iter().find(|(file, _)| *file == name);
    let expected = expected
        .ok_or_else(|| format!("unknown sys1 asset: {name}"))?
        .1;
    let home = std::env::var_os("HF_HOME")
        .map(PathBuf::from)
        .or_else(|| {
            std::env::var_os("XDG_CACHE_HOME").map(|p| PathBuf::from(p).join("huggingface"))
        })
        .or_else(|| std::env::var_os("HOME").map(|p| PathBuf::from(p).join(".cache/huggingface")))
        .ok_or("set HF_HOME to a writable asset cache")?;
    let path = home.join("sys1").join(REVISION).join(name);
    let valid = |data: &[u8]| format!("{:x}", Sha256::digest(data)) == expected;
    if let Ok(data) = fs::read(&path)
        && valid(&data)
    {
        return Ok(path);
    }
    let url =
        format!("https://huggingface.co/alvarobartt/sys1-artifacts/resolve/{REVISION}/{name}");
    let data = reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs(120))
        .build()?
        .get(url)
        .send()?
        .error_for_status()?
        .bytes()?;
    if !valid(&data) {
        return Err(format!("SHA256 mismatch for sys1 asset {name}").into());
    }
    fs::create_dir_all(path.parent().ok_or("invalid asset path")?)?;
    let mut temporary = tempfile::NamedTempFile::new_in(path.parent().unwrap())?;
    std::io::Write::write_all(&mut temporary, &data)?;
    temporary.persist(&path)?;
    Ok(path)
}

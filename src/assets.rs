use sha2::{Digest, Sha256};
use std::{fs, path::PathBuf, time::Duration};

const REVISION: &str = "fe16e9b22cd05ebb799f242df9ac68470d3e3d4c";
#[cfg(test)]
const FIXTURES: &[(&str, &str)] = &[
    (
        "architectures/fixtures/activation/gelu-erf-bf16.safetensors",
        "fb88d301975af8beb2aa29ccdb0b8cf785907aaad91cd4a30f24f3269a6295bf",
    ),
    (
        "architectures/fixtures/activation/gelu-erf-f16.safetensors",
        "e5ba818f0d68b3393339a156343b25318f239d3d00bf75d862b60c5319011f80",
    ),
    (
        "architectures/fixtures/activation/gelu-erf-f32.safetensors",
        "3748ff24ebf8077eabb216c1c7fce8d3dd2c42a3c1ebb1f1c4b554af4c0c683b",
    ),
    (
        "architectures/fixtures/linear/b2-q3-k7-w128-bf16.safetensors",
        "fe4abade7c20bf20ee02b1035f3bc921dd4bf2ebd46b1163b44cac6d317c78b8",
    ),
    (
        "architectures/fixtures/linear/b2-q3-k7-w128-f16.safetensors",
        "024de5979053af5252ce43913620b0942cdf6a356edb6c30a3ec0424f4e12af8",
    ),
    (
        "architectures/fixtures/linear/b2-q3-k7-w128-f32.safetensors",
        "1e1d98268e3746c2a884f862be4e2b7b855fcabcf07b693c949e226406e483d1",
    ),
    (
        "architectures/fixtures/qwen35_conv/b1-l1-c128-k4-bf16.safetensors",
        "c277a1f7e97159139cf4e6f456dff031db3a0c13600abe7643311e539abb2f15",
    ),
    (
        "architectures/fixtures/qwen35_conv/b1-l1-c128-k4-f32.safetensors",
        "731d430f0cbdac9699ec9a5b9c8c46102560ba669fa55fea6c0d506c66337150",
    ),
    (
        "architectures/fixtures/qwen35_conv/b1-l257-c255-k4-bf16.safetensors",
        "13349f16baee4f2a5767fa5752596bb62f5a685dcb196794ccb2ae7569f55e51",
    ),
    (
        "architectures/fixtures/qwen35_conv/b1-l257-c255-k4-f32.safetensors",
        "041b06c2bfe21b1105e344aeaf9faf59c136f6a4cca4b838ab5ff2c08ab8a121",
    ),
    (
        "architectures/fixtures/qwen35_conv/b2-l129-c128-k4-bf16.safetensors",
        "fc5a588e56981761f9a2724553984aa3143e06073d2e41374d0aec9f82d295a1",
    ),
    (
        "architectures/fixtures/qwen35_conv/b2-l129-c128-k4-f32.safetensors",
        "37c847c837efdacca1e8da0a383db6d0da962e7e6d7d68b45f6ad1ade945c718",
    ),
    (
        "architectures/fixtures/qwen35_conv/b2-l7-c31-k4-bf16.safetensors",
        "c127e3922eb69667e34b496ff64e8d207a97fa2fbee6c5095fe6ba9a8d8d3439",
    ),
    (
        "architectures/fixtures/qwen35_conv/b2-l7-c31-k4-f32.safetensors",
        "979ff42a505b448e6162a4db8cd9696d63a912d620bf57c1de043df64c7017fe",
    ),
    (
        "architectures/fixtures/qwen35_delta/b2-l65-kh2-h4-d16-bf16.safetensors",
        "31b87216b1cac51a59cd3d0e4280d5ca8ede6f9127367a3fef93ff21262625bb",
    ),
    (
        "architectures/fixtures/qwen35_delta/b2-l65-kh2-h4-d16-f32.safetensors",
        "59d26e8f94ef4623dcb537cd4fbadeb2bb93de369db74600008054e3023c14e6",
    ),
    (
        "architectures/fixtures/qwen35_recurrent/post-finalize-bf16.safetensors",
        "0f954cfd4eadf83e14c722e78120677876c1b5e20344b6aadff173ed090f9317",
    ),
    (
        "architectures/fixtures/qwen35_recurrent/post-finalize-f16.safetensors",
        "aa308bc82182536aa50a61ee03d0f3d57e437ccb5389a775f2b64b11ac881354",
    ),
    (
        "architectures/fixtures/qwen35_recurrent/post-finalize-f32.safetensors",
        "010d0b2f8a9eba41380bbb8c8df1590060378f62240535ae727ca94d00db8726",
    ),
    (
        "architectures/fixtures/sdpa/b2-h3-l11-d64-masked-bf16.safetensors",
        "8916df0fe8605257b774b510898bbce31cc8989b5bd100726167d6f9a0b97b94",
    ),
    (
        "architectures/fixtures/sdpa/b2-h3-l11-d64-masked-f16.safetensors",
        "d8cbed54f7b5fbed99a281836fe7f0d8ca3a600c847f5b88b76819f3573dc287",
    ),
    (
        "architectures/fixtures/sdpa/b2-h3-l11-d64-masked-f32.safetensors",
        "81a40615b1e76b3545b336ae032a4d7af142b6589fc141d7732ef5dcbcc4ff8d",
    ),
    (
        "architectures/fixtures/sdpa/b2-h3-l11-d64-unmasked-bf16.safetensors",
        "13ef8ebff0d86f2256173f520d5b1af3cc9bdc24487fdae7ae32344f4be439ca",
    ),
    (
        "architectures/fixtures/sdpa/b2-h3-l11-d64-unmasked-f16.safetensors",
        "bbe644724dbac75357a4ab448819ae152148b15c42c62039eb4828d7aee9c066",
    ),
    (
        "architectures/fixtures/sdpa/b2-h3-l11-d64-unmasked-f32.safetensors",
        "f9749eaa86206d993024fe3b1374dacfedbab01c3931a9488722363b2308ec25",
    ),
    (
        "fa4_driver_abi_cases.bin",
        "7fabe198519ca57d1f1c06a409a94916a51012009956203a7e2d85615da52b26",
    ),
];

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
    #[cfg(test)]
    let expected = expected.or_else(|| FIXTURES.iter().find(|(file, _)| *file == name));
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

#[cfg(test)]
pub(crate) fn fixtures(directory: &str) -> candle_core::Result<PathBuf> {
    let prefix = format!("architectures/fixtures/{directory}/");
    let mut parent = None;
    for (name, _) in FIXTURES
        .iter()
        .filter(|(name, _)| name.starts_with(&prefix))
    {
        let path = get(name).map_err(|error| candle_core::Error::Msg(error.to_string()))?;
        parent = path.parent().map(PathBuf::from);
    }
    parent.ok_or_else(|| candle_core::Error::Msg(format!("unknown fixture directory {directory}")))
}

use crate::schema::MediaInput;
use anyhow::{Context, bail};
use base64::{Engine, engine::general_purpose::STANDARD};
use std::{
    io::{Read, Write},
    net::{IpAddr, ToSocketAddrs},
    process::{Command, Stdio},
    time::{Duration, Instant},
};

const MAX_MEDIA_BYTES: usize = 32 * 1024 * 1024;
const MAX_DECODED_BYTES: usize = 128 * 1024 * 1024;

fn public_ip(address: IpAddr) -> bool {
    match address {
        IpAddr::V4(address) => {
            let [a, b, ..] = address.octets();
            !address.is_private()
                && !address.is_loopback()
                && !address.is_link_local()
                && !address.is_broadcast()
                && !address.is_documentation()
                && !address.is_multicast()
                && !address.is_unspecified()
                && a != 0
                && !(a == 100 && (64..=127).contains(&b))
                && !(a == 198 && (18..=19).contains(&b))
                && a < 240
        }
        IpAddr::V6(address) => {
            if let Some(mapped) = address.to_ipv4_mapped() {
                return public_ip(IpAddr::V4(mapped));
            }
            !address.is_loopback()
                && !address.is_unspecified()
                && !address.is_multicast()
                && !address.is_unique_local()
                && !address.is_unicast_link_local()
        }
    }
}

fn download_url(value: &str) -> anyhow::Result<Vec<u8>> {
    let url = reqwest::Url::parse(value).context("invalid media URL")?;
    anyhow::ensure!(
        matches!(url.scheme(), "http" | "https"),
        "media URL must use http or https"
    );
    anyhow::ensure!(
        url.username().is_empty() && url.password().is_none(),
        "media URL must not contain credentials"
    );
    let host = url.host_str().context("media URL has no host")?;
    let port = url
        .port_or_known_default()
        .context("media URL has no port")?;
    let addresses: Vec<_> = (host, port)
        .to_socket_addrs()
        .with_context(|| format!("could not resolve media host {host:?}"))?
        .collect();
    anyhow::ensure!(!addresses.is_empty(), "media host has no addresses");
    anyhow::ensure!(
        addresses.iter().all(|address| public_ip(address.ip())),
        "media URL resolves to a private or reserved address"
    );
    let client = reqwest::blocking::Client::builder()
        .no_proxy()
        .redirect(reqwest::redirect::Policy::none())
        .timeout(Duration::from_secs(20))
        .resolve_to_addrs(host, &addresses)
        .build()?;
    let response = client.get(url).send()?.error_for_status()?;
    anyhow::ensure!(
        !response.status().is_redirection(),
        "media URL redirects are not followed; use the final URL"
    );
    let mut bytes = Vec::new();
    response
        .take((MAX_MEDIA_BYTES + 1) as u64)
        .read_to_end(&mut bytes)?;
    anyhow::ensure!(
        bytes.len() <= MAX_MEDIA_BYTES,
        "media URL exceeds 32 MiB; use a smaller file"
    );
    Ok(bytes)
}

pub(crate) fn input_bytes(input: &MediaInput, kind: &str) -> anyhow::Result<Vec<u8>> {
    let bytes = match input {
        MediaInput::Text(value) if value.contains("://") => download_url(value)?,
        MediaInput::Url(value) => download_url(&value.url)?,
        MediaInput::Text(value) => {
            let encoded = if let Some((header, payload)) = value.split_once(',') {
                anyhow::ensure!(
                    header.starts_with(&format!("data:{kind}/")) && header.ends_with(";base64"),
                    "{kind} data URL must be base64 encoded"
                );
                payload
            } else {
                value.as_str()
            };
            STANDARD.decode(encoded).context("invalid media base64")?
        }
        MediaInput::Base64(value) => STANDARD
            .decode(&value.base64)
            .context("invalid media base64")?,
        MediaInput::Bytes(value) => value.clone(),
        MediaInput::ByteObject(value) => value.bytes.clone(),
    };
    anyhow::ensure!(
        bytes.len() <= MAX_MEDIA_BYTES,
        "media exceeds 32 MiB; use a smaller file"
    );
    Ok(bytes)
}

fn run_media_command(
    program: &str,
    args: &[String],
    data: Option<&[u8]>,
    kind: &str,
    timeout: Duration,
    max_output: usize,
) -> anyhow::Result<Vec<u8>> {
    let mut child = Command::new(program)
        .args(args)
        .stdin(if data.is_some() {
            Stdio::piped()
        } else {
            Stdio::null()
        })
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .with_context(|| {
            format!("failed to start {program} for {kind} processing; install ffmpeg and ensure {program} is on PATH")
        })?;
    let stdout = child
        .stdout
        .take()
        .context("media processor stdout is unavailable")?;
    let stderr = child
        .stderr
        .take()
        .context("media processor stderr is unavailable")?;
    let output_reader = std::thread::spawn(move || {
        let mut output = Vec::new();
        stdout
            .take((max_output + 1) as u64)
            .read_to_end(&mut output)
            .map(|_| output)
    });
    let error_reader = std::thread::spawn(move || {
        let mut output = Vec::new();
        stderr.take(65_537).read_to_end(&mut output).map(|_| output)
    });
    let writer = data.map(|data| {
        let mut stdin = child.stdin.take().unwrap();
        let payload = data.to_vec();
        std::thread::spawn(move || stdin.write_all(&payload))
    });
    let started = Instant::now();
    let status = loop {
        if let Some(status) = child.try_wait()? {
            break status;
        }
        if started.elapsed() >= timeout {
            child.kill()?;
            child.wait()?;
            return Err(anyhow::anyhow!(
                "{program} exceeded the {kind} processing limit of {:.2} seconds; use a shorter or smaller file",
                timeout.as_secs_f64()
            ));
        }
        std::thread::sleep(Duration::from_millis(10));
    };
    let output = output_reader
        .join()
        .map_err(|_| anyhow::anyhow!("media processor output thread failed"))??;
    let error = error_reader
        .join()
        .map_err(|_| anyhow::anyhow!("media processor error thread failed"))??;
    anyhow::ensure!(
        output.len() <= max_output,
        "decoded {kind} exceeds {max_output} bytes; use a smaller file"
    );
    let write_result = writer.map(|writer| {
        writer
            .join()
            .map_err(|_| anyhow::anyhow!("media processor input thread failed"))?
            .map_err(anyhow::Error::from)
    });
    if !status.success() {
        bail!(
            "{program} could not decode the supplied {kind}: {}",
            String::from_utf8_lossy(&error)
        );
    }
    if let Some(write_result) = write_result {
        write_result?;
    }
    Ok(output)
}

pub(crate) fn run_media_tool(
    program: &str,
    args: &[String],
    data: &[u8],
) -> anyhow::Result<Vec<u8>> {
    run_media_command(
        program,
        args,
        Some(data),
        "image",
        Duration::from_secs(30),
        MAX_DECODED_BYTES,
    )
}

pub(crate) fn run_file_tool(
    program: &str,
    args: &[String],
    max_output: usize,
) -> anyhow::Result<Vec<u8>> {
    run_media_command(
        program,
        args,
        None,
        "video",
        Duration::from_secs(120),
        max_output,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::{MediaBase64, MediaUrl};

    #[cfg(unix)]
    #[test]
    fn media_tools_have_time_and_output_limits() {
        let error = run_media_command(
            "sleep",
            &["1".into()],
            None,
            "video",
            Duration::from_millis(20),
            1024,
        )
        .unwrap_err();
        assert!(error.to_string().contains("processing limit"));

        let error = run_media_command(
            "sh",
            &["-c".into(), "printf 12345".into()],
            None,
            "image",
            Duration::from_secs(1),
            4,
        )
        .unwrap_err();
        assert!(error.to_string().contains("decoded image exceeds 4 bytes"));
    }

    #[test]
    fn media_inputs_accept_bytes_and_base64_and_reject_private_urls() {
        let data = b"P6\n1 1\n255\n\xff\x00\x00".to_vec();
        assert_eq!(
            input_bytes(&MediaInput::Bytes(data.clone()), "image").unwrap(),
            data
        );
        let base64 = MediaInput::Base64(MediaBase64 {
            base64: STANDARD.encode(&data),
        });
        assert_eq!(input_bytes(&base64, "image").unwrap(), data);
        let url = MediaInput::Url(MediaUrl {
            url: "http://127.0.0.1/image.png".into(),
        });
        assert!(
            input_bytes(&url, "image")
                .unwrap_err()
                .to_string()
                .contains("private or reserved")
        );
        assert!(
            input_bytes(&MediaInput::from("file:///tmp/image.png"), "image")
                .unwrap_err()
                .to_string()
                .contains("http or https")
        );
    }
}

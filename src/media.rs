use crate::schema::MediaInput;
use anyhow::{Context, bail};
use base64::{Engine, engine::general_purpose::STANDARD};
use std::{
    io::{Read, Write},
    net::{IpAddr, ToSocketAddrs},
    process::{Command, Stdio},
    sync::OnceLock,
    time::{Duration, Instant},
};

const MAX_MEDIA_BYTES: usize = 32 * 1024 * 1024;
const MAX_DECODED_BYTES: usize = 128 * 1024 * 1024;
static VIDEO_TOOLS_VALIDATED: OnceLock<()> = OnceLock::new();

fn parse_tool_version(output: &str, program: &str) -> anyhow::Result<(u32, u32, u32)> {
    let version = output
        .lines()
        .next()
        .and_then(|line| line.strip_prefix(&format!("{program} version ")))
        .context("unexpected FFmpeg version output")?
        .trim_start_matches('n');
    let mut parts = version.split('.');
    let major = parts
        .next()
        .context("missing FFmpeg major version")?
        .parse()?;
    let minor = parts
        .next()
        .context("missing FFmpeg minor version")?
        .chars()
        .take_while(char::is_ascii_digit)
        .collect::<String>()
        .parse()?;
    let patch = parts
        .next()
        .unwrap_or("0")
        .chars()
        .take_while(char::is_ascii_digit)
        .collect::<String>()
        .parse()?;
    Ok((major, minor, patch))
}

pub(crate) fn require_video_tools() -> anyhow::Result<()> {
    if VIDEO_TOOLS_VALIDATED.get().is_some() {
        return Ok(());
    }
    for program in ["ffmpeg", "ffprobe"] {
        let output = Command::new(program)
            .arg("-version")
            .output()
            .with_context(|| format!("{program} is required for video processing"))?;
        anyhow::ensure!(output.status.success(), "failed to check {program} version");
        let version = String::from_utf8_lossy(&output.stdout);
        let (major, minor, patch) = parse_tool_version(&version, program)?;
        anyhow::ensure!(
            (major, minor, patch) >= (6, 1, 1),
            "{program} 6.1.1 or newer is required for video processing; found {major}.{minor}.{patch}"
        );
    }
    let _ = VIDEO_TOOLS_VALIDATED.set(());
    Ok(())
}

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

const MAX_MEDIA_REDIRECTS: usize = 10;
const MEDIA_DOWNLOAD_TIMEOUT: Duration = Duration::from_secs(20);

fn fetch_media_url(
    url: &reqwest::Url,
    deadline: Instant,
) -> anyhow::Result<reqwest::blocking::Response> {
    let host = url.host_str().context("media URL has no host")?;
    let port = url
        .port_or_known_default()
        .context("media URL has no port")?;
    let addresses: Vec<_> = if let Ok(address) = host.trim_matches(['[', ']']).parse::<IpAddr>() {
        vec![std::net::SocketAddr::new(address, port)]
    } else {
        (host, port)
            .to_socket_addrs()
            .with_context(|| format!("could not resolve media host {host:?}"))?
            .collect()
    };
    anyhow::ensure!(!addresses.is_empty(), "media host has no addresses");
    anyhow::ensure!(
        addresses.iter().all(|address| public_ip(address.ip())),
        "media URL resolves to a private or reserved address"
    );
    let remaining = deadline.saturating_duration_since(Instant::now());
    anyhow::ensure!(!remaining.is_zero(), "media download timed out");
    // Each hop uses only the addresses validated above, including cross-host redirects.
    let client = reqwest::blocking::Client::builder()
        .no_proxy()
        .redirect(reqwest::redirect::Policy::none())
        .timeout(remaining)
        .resolve_to_addrs(host, &addresses)
        .build()?;
    Ok(client.get(url.clone()).send()?)
}

fn download_url(value: &str) -> anyhow::Result<Vec<u8>> {
    download_url_with(value, fetch_media_url)
}

fn download_url_with(
    value: &str,
    mut fetch: impl FnMut(&reqwest::Url, Instant) -> anyhow::Result<reqwest::blocking::Response>,
) -> anyhow::Result<Vec<u8>> {
    let mut url = reqwest::Url::parse(value).context("invalid media URL")?;
    let deadline = Instant::now() + MEDIA_DOWNLOAD_TIMEOUT;
    for redirects in 0..=MAX_MEDIA_REDIRECTS {
        anyhow::ensure!(
            matches!(url.scheme(), "http" | "https"),
            "media URL must use http or https"
        );
        anyhow::ensure!(
            url.username().is_empty() && url.password().is_none(),
            "media URL must not contain credentials"
        );
        anyhow::ensure!(Instant::now() < deadline, "media download timed out");
        let response = fetch(&url, deadline)?.error_for_status()?;
        if response.status().is_redirection() {
            anyhow::ensure!(
                matches!(response.status().as_u16(), 301 | 302 | 303 | 307 | 308),
                "unsupported media redirect status: {}",
                response.status()
            );
            anyhow::ensure!(
                redirects < MAX_MEDIA_REDIRECTS,
                "media URL exceeds {MAX_MEDIA_REDIRECTS} redirects"
            );
            let location = response
                .headers()
                .get(reqwest::header::LOCATION)
                .context("media redirect has no Location header")?
                .to_str()
                .context("invalid media redirect Location header")?;
            url = url.join(location).context("invalid media redirect URL")?;
            continue;
        }
        let mut bytes = Vec::new();
        response
            .take((MAX_MEDIA_BYTES + 1) as u64)
            .read_to_end(&mut bytes)?;
        anyhow::ensure!(Instant::now() < deadline, "media download timed out");
        anyhow::ensure!(
            bytes.len() <= MAX_MEDIA_BYTES,
            "media URL exceeds 32 MiB; use a smaller file"
        );
        return Ok(bytes);
    }
    unreachable!("redirect limit is checked before following a redirect")
}

pub(crate) fn input_bytes(input: &MediaInput, kind: &str) -> anyhow::Result<Vec<u8>> {
    input
        .validate_content_type(kind)
        .map_err(anyhow::Error::msg)?;
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
        MediaInput::Embedded(value) => STANDARD
            .decode(&value.base64)
            .context("invalid media base64")?,
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

    fn media_response(
        status: u16,
        location: Option<&str>,
        body: Vec<u8>,
    ) -> reqwest::blocking::Response {
        let mut builder = axum::http::Response::builder().status(status);
        if let Some(location) = location {
            builder = builder.header(reqwest::header::LOCATION, location);
        }
        builder.body(body).unwrap().into()
    }

    #[test]
    fn follows_relative_and_cross_host_media_redirects() {
        for status in [301, 302, 303, 307, 308] {
            let mut visited = Vec::new();
            let bytes = download_url_with("https://example.com/start", |url, _| {
                visited.push(url.to_string());
                Ok(match visited.len() {
                    1 => media_response(status, Some("/next"), vec![]),
                    2 => media_response(status, Some("https://cdn.example.com/image.jpg"), vec![]),
                    _ => media_response(200, None, b"image bytes".to_vec()),
                })
            })
            .unwrap();
            assert_eq!(bytes, b"image bytes");
            assert_eq!(
                visited,
                [
                    "https://example.com/start",
                    "https://example.com/next",
                    "https://cdn.example.com/image.jpg"
                ]
            );
        }
    }

    #[test]
    fn rejects_unsafe_media_redirect_destinations() {
        for (location, expected) in [
            ("file:///tmp/image.jpg", "http or https"),
            ("https://user:password@example.com/image.jpg", "credentials"),
            ("http://127.0.0.1/image.jpg", "private or reserved"),
            ("http://10.0.0.1/image.jpg", "private or reserved"),
            ("http://[::1]/image.jpg", "private or reserved"),
        ] {
            let mut calls = 0;
            let error = download_url_with("https://example.com/start", |url, deadline| {
                calls += 1;
                if calls == 1 {
                    Ok(media_response(302, Some(location), vec![]))
                } else {
                    fetch_media_url(url, deadline)
                }
            })
            .unwrap_err();
            assert!(error.to_string().contains(expected), "{location}: {error}");
        }
    }

    #[test]
    fn rejects_redirect_loops_and_missing_locations() {
        let mut calls = 0;
        let error = download_url_with("https://example.com/start", |_, _| {
            calls += 1;
            Ok(media_response(307, Some("/start"), vec![]))
        })
        .unwrap_err();
        assert_eq!(calls, MAX_MEDIA_REDIRECTS + 1);
        assert!(error.to_string().contains("exceeds 10 redirects"));
        let error = download_url_with("https://example.com/start", |_, _| {
            Ok(media_response(302, None, vec![]))
        })
        .unwrap_err();
        assert!(error.to_string().contains("no Location header"));
    }

    #[test]
    #[ignore = "requires public network access"]
    fn downloads_huggingface_image_through_redirects() {
        let bytes = download_url("https://huggingface.co/datasets/hf-internal-testing/fixtures_ocr/resolve/main/SROIE-receipt.jpeg").unwrap();
        assert!(
            bytes.starts_with(&[0xff, 0xd8, 0xff]),
            "expected a JPEG image"
        );
    }

    #[test]
    fn parses_ffmpeg_versions() {
        assert_eq!(
            parse_tool_version("ffmpeg version 6.1.1-3ubuntu5 Copyright", "ffmpeg").unwrap(),
            (6, 1, 1)
        );
        assert_eq!(
            parse_tool_version("ffprobe version n5.1.6 Copyright", "ffprobe").unwrap(),
            (5, 1, 6)
        );
        assert_eq!(
            parse_tool_version("ffmpeg version 8.0 Copyright", "ffmpeg").unwrap(),
            (8, 0, 0)
        );
    }

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
    fn decodes_embedded_images_and_videos() {
        for (kind, content_type) in [("image", "image/png"), ("video", "video/mp4")] {
            let input: MediaInput = serde_json::from_value(serde_json::json!({
                "content_type": content_type, "base64": STANDARD.encode(b"file bytes")
            }))
            .unwrap();
            assert_eq!(input_bytes(&input, kind).unwrap(), b"file bytes");
            let wrong_kind = if kind == "image" { "video" } else { "image" };
            assert!(input_bytes(&input, wrong_kind).is_err());
        }
        let invalid: MediaInput = serde_json::from_value(serde_json::json!({
            "content_type": "video/mp4", "base64": "invalid!"
        }))
        .unwrap();
        assert!(
            input_bytes(&invalid, "video")
                .unwrap_err()
                .to_string()
                .contains("invalid media base64")
        );
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

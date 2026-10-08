use super::qwen35_text::VisionConfig;
use crate::models::AttentionImplementation;
use candle_core::{D, DType, Result, Tensor};
use candle_nn::{LayerNorm, Linear, VarBuilder, layer_norm, ops::softmax_last_dim};

type PositionData = ([Vec<u32>; 4], [Vec<f32>; 4], Vec<f32>, Vec<usize>);

fn dense(input: usize, output: usize, vb: VarBuilder) -> Result<Linear> {
    Ok(Linear::new(
        vb.get((output, input), "weight")?,
        Some(vb.get(output, "bias")?),
    ))
}

struct Block {
    norm1: LayerNorm,
    norm2: LayerNorm,
    qkv: Linear,
    proj: Linear,
    fc1: Linear,
    fc2: Linear,
    heads: usize,
    head_dim: usize,
    attention: AttentionImplementation,
}

impl Block {
    fn load(
        config: &VisionConfig,
        vb: VarBuilder,
        attention: AttentionImplementation,
    ) -> Result<Self> {
        Ok(Self {
            norm1: layer_norm(config.hidden_size, 1e-6, vb.pp("norm1"))?,
            norm2: layer_norm(config.hidden_size, 1e-6, vb.pp("norm2"))?,
            qkv: dense(
                config.hidden_size,
                config.hidden_size * 3,
                vb.pp("attn.qkv"),
            )?,
            proj: dense(config.hidden_size, config.hidden_size, vb.pp("attn.proj"))?,
            fc1: dense(
                config.hidden_size,
                config.intermediate_size,
                vb.pp("mlp.linear_fc1"),
            )?,
            fc2: dense(
                config.intermediate_size,
                config.hidden_size,
                vb.pp("mlp.linear_fc2"),
            )?,
            heads: config.num_heads,
            head_dim: config.hidden_size / config.num_heads,
            attention,
        })
    }

    fn attend(&self, x: &Tensor, cos: &Tensor, sin: &Tensor, lengths: &[usize]) -> Result<Tensor> {
        let tokens = x.dim(0)?;
        let parts = x
            .apply(&self.qkv)?
            .reshape((tokens, 3, self.heads, self.head_dim))?;
        let rotate = |input: Tensor| -> Result<Tensor> {
            let half = self.head_dim / 2;
            let first = input.narrow(D::Minus1, 0, half)?;
            let second = input.narrow(D::Minus1, half, half)?;
            let rotated = Tensor::cat(&[&second.neg()?, &first], D::Minus1)?;
            let original = input.to_dtype(DType::F32)?;
            (original.broadcast_mul(cos)? + rotated.to_dtype(DType::F32)?.broadcast_mul(sin)?)?
                .to_dtype(x.dtype())
        };
        let q = rotate(parts.narrow(1, 0, 1)?.squeeze(1)?)?;
        let k = rotate(parts.narrow(1, 1, 1)?.squeeze(1)?)?;
        let v = parts.narrow(1, 2, 1)?.squeeze(1)?;
        let scale = (self.head_dim as f32).powf(-0.5);
        let attention = match self.attention {
            AttentionImplementation::Eager => {
                let mut offset = 0;
                let mut outputs = Vec::with_capacity(lengths.len());
                for &length in lengths {
                    let qi = q.narrow(0, offset, length)?.transpose(0, 1)?.unsqueeze(0)?;
                    let ki = k.narrow(0, offset, length)?.transpose(0, 1)?.unsqueeze(0)?;
                    let vi = v.narrow(0, offset, length)?.transpose(0, 1)?.unsqueeze(0)?;
                    #[cfg(feature = "metal")]
                    if x.device().is_metal() {
                        outputs.push(
                            candle_nn::ops::sdpa(
                                &qi.contiguous()?,
                                &ki.contiguous()?,
                                &vi.contiguous()?,
                                None,
                                false,
                                scale,
                                1.0,
                            )?
                            .squeeze(0)?
                            .transpose(0, 1)?,
                        );
                        offset += length;
                        continue;
                    }
                    let scores = (qi
                        .contiguous()?
                        .matmul(&ki.transpose(D::Minus2, D::Minus1)?.contiguous()?)?
                        * scale as f64)?;
                    let probs =
                        softmax_last_dim(&scores.to_dtype(DType::F32)?)?.to_dtype(x.dtype())?;
                    outputs.push(
                        probs
                            .contiguous()?
                            .matmul(&vi.contiguous()?)?
                            .squeeze(0)?
                            .transpose(0, 1)?,
                    );
                    offset += length;
                }
                Tensor::cat(&outputs.iter().collect::<Vec<_>>(), 0)?
            }
            #[cfg(any(feature = "flash-attn-2", feature = "flash-attn-3"))]
            implementation => {
                let mut cumulative = Vec::with_capacity(lengths.len() + 1);
                cumulative.push(0u32);
                for &length in lengths {
                    cumulative.push(cumulative.last().copied().unwrap() + length as u32);
                }
                let max_length = lengths.iter().copied().max().unwrap_or(0);
                let cumulative = Tensor::from_vec(cumulative, lengths.len() + 1, x.device())?;
                match implementation {
                    #[cfg(feature = "flash-attn-2")]
                    AttentionImplementation::FlashAttention2 => {
                        candle_flash_attn::flash_attn_varlen(
                            &q.contiguous()?,
                            &k.contiguous()?,
                            &v.contiguous()?,
                            &cumulative,
                            &cumulative,
                            max_length,
                            max_length,
                            scale,
                            false,
                        )?
                    }
                    #[cfg(feature = "flash-attn-3")]
                    AttentionImplementation::FlashAttention3 => {
                        candle_flash_attn_v3::flash_attn_varlen(
                            &q.contiguous()?,
                            &k.contiguous()?,
                            &v.contiguous()?,
                            &cumulative,
                            &cumulative,
                            max_length,
                            max_length,
                            scale,
                            false,
                            false,
                        )?
                    }
                    _ => candle_core::bail!(
                        "{} support is not compiled in",
                        implementation.cli_name()
                    ),
                }
            }
            #[cfg(not(any(feature = "flash-attn-2", feature = "flash-attn-3")))]
            other => candle_core::bail!("{} support is not compiled in", other.cli_name()),
        };
        attention
            .reshape((tokens, self.heads * self.head_dim))?
            .apply(&self.proj)
    }

    fn forward(&self, x: &Tensor, cos: &Tensor, sin: &Tensor, lengths: &[usize]) -> Result<Tensor> {
        let mixed = self.attend(&x.apply(&self.norm1)?, cos, sin, lengths)?;
        let x = (x + mixed)?;
        &x + x
            .apply(&self.norm2)?
            .apply(&self.fc1)?
            .gelu()?
            .apply(&self.fc2)?
    }
}

pub struct VisionTower {
    patch_weight: Tensor,
    patch_bias: Tensor,
    positions: Tensor,
    blocks: Vec<Block>,
    merge_norm: LayerNorm,
    merge_fc1: Linear,
    merge_fc2: Linear,
    config: VisionConfig,
}

impl VisionTower {
    pub fn load(
        config: &VisionConfig,
        vb: VarBuilder,
        attention: AttentionImplementation,
    ) -> Result<Self> {
        let patch_width =
            config.in_channels * config.temporal_patch_size * config.patch_size * config.patch_size;
        let merge_width = config.hidden_size * config.spatial_merge_size.pow(2);
        Ok(Self {
            patch_weight: vb
                .get(
                    (
                        config.hidden_size,
                        config.in_channels,
                        config.temporal_patch_size,
                        config.patch_size,
                        config.patch_size,
                    ),
                    "patch_embed.proj.weight",
                )?
                .reshape((config.hidden_size, patch_width))?,
            patch_bias: vb.get(config.hidden_size, "patch_embed.proj.bias")?,
            positions: vb.get(
                (config.num_position_embeddings, config.hidden_size),
                "pos_embed.weight",
            )?,
            blocks: (0..config.depth)
                .map(|index| Block::load(config, vb.pp(format!("blocks.{index}")), attention))
                .collect::<Result<Vec<_>>>()?,
            merge_norm: layer_norm(config.hidden_size, 1e-6, vb.pp("merger.norm"))?,
            merge_fc1: dense(merge_width, merge_width, vb.pp("merger.linear_fc1"))?,
            merge_fc2: dense(
                merge_width,
                config.out_hidden_size,
                vb.pp("merger.linear_fc2"),
            )?,
            config: config.clone(),
        })
    }

    pub fn forward(&self, patches: &Tensor, grids: &[[usize; 3]]) -> Result<Tensor> {
        let total = patches.dim(0)?;
        let expected: usize = grids.iter().map(|[t, h, w]| t * h * w).sum();
        if total != expected {
            candle_core::bail!("vision patches ({total}) do not match grids ({expected})")
        }
        let width = self.config.hidden_size;
        let patch_width = self.patch_weight.dim(1)?;
        if patches.dim(1)? != patch_width {
            candle_core::bail!("vision patch width mismatch")
        }
        let mut x = patches
            .matmul(&self.patch_weight.t()?)?
            .broadcast_add(&self.patch_bias)?;
        let (position_indices, position_weights, rotary, lengths) = self.positions_for(grids)?;
        let mut position = Tensor::zeros((total, width), x.dtype(), x.device())?;
        for corner in 0..4 {
            let ids = Tensor::from_vec(position_indices[corner].clone(), total, x.device())?;
            let values = self.positions.index_select(&ids, 0)?;
            let weights =
                Tensor::from_vec(position_weights[corner].clone(), (total, 1), x.device())?
                    .to_dtype(values.dtype())?;
            position = (position + values.broadcast_mul(&weights)?)?;
        }
        x = (x + position)?;
        let rotary = Tensor::from_vec(
            rotary,
            (total, self.config.hidden_size / self.config.num_heads),
            x.device(),
        )?;
        let cos = rotary.cos()?.unsqueeze(1)?;
        let sin = rotary.sin()?.unsqueeze(1)?;
        for block in &self.blocks {
            x = block.forward(&x, &cos, &sin, &lengths)?;
        }
        let merged = x.apply(&self.merge_norm)?.reshape((
            total / self.config.spatial_merge_size.pow(2),
            self.config.hidden_size * self.config.spatial_merge_size.pow(2),
        ))?;
        merged
            .apply(&self.merge_fc1)?
            .gelu_erf()?
            .apply(&self.merge_fc2)
    }

    fn positions_for(&self, grids: &[[usize; 3]]) -> Result<PositionData> {
        let side = (self.config.num_position_embeddings as f64).sqrt() as usize;
        if side * side != self.config.num_position_embeddings {
            candle_core::bail!("vision position grid must be square")
        }
        let merge = self.config.spatial_merge_size;
        let dim = self.config.hidden_size / self.config.num_heads;
        let mut indices: [Vec<u32>; 4] = std::array::from_fn(|_| Vec::new());
        let mut weights: [Vec<f32>; 4] = std::array::from_fn(|_| Vec::new());
        let mut rotary = Vec::new();
        let mut lengths = Vec::new();
        for &[t, h, w] in grids {
            if h % merge != 0 || w % merge != 0 || h == 0 || w == 0 || t == 0 {
                candle_core::bail!("invalid Qwen vision grid ({t},{h},{w})")
            }
            for _ in 0..t {
                lengths.push(h * w);
                for block_h in 0..h / merge {
                    for block_w in 0..w / merge {
                        for inner_h in 0..merge {
                            for inner_w in 0..merge {
                                let row = block_h * merge + inner_h;
                                let col = block_w * merge + inner_w;
                                let fy = row as f32 * (side - 1) as f32 / (h - 1).max(1) as f32;
                                let fx = col as f32 * (side - 1) as f32 / (w - 1).max(1) as f32;
                                let y0 = fy.floor() as usize;
                                let x0 = fx.floor() as usize;
                                let y1 = (y0 + 1).min(side - 1);
                                let x1 = (x0 + 1).min(side - 1);
                                let dy = fy - y0 as f32;
                                let dx = fx - x0 as f32;
                                for (corner, (y, x, weight)) in [
                                    (y0, x0, (1. - dy) * (1. - dx)),
                                    (y0, x1, (1. - dy) * dx),
                                    (y1, x0, dy * (1. - dx)),
                                    (y1, x1, dy * dx),
                                ]
                                .into_iter()
                                .enumerate()
                                {
                                    indices[corner].push((y * side + x) as u32);
                                    weights[corner].push(weight);
                                }
                                let quarter = dim / 4;
                                let mut phase = Vec::with_capacity(dim / 2);
                                for axis in [row, col] {
                                    for index in 0..quarter {
                                        phase.push(
                                            axis as f32
                                                / 10000f32
                                                    .powf((2 * index) as f32 / (dim / 2) as f32),
                                        );
                                    }
                                }
                                rotary.extend_from_slice(&phase);
                                rotary.extend_from_slice(&phase);
                            }
                        }
                    }
                }
            }
        }
        Ok((indices, weights, rotary, lengths))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vision_tower_merges_one_image_grid_on_cpu() {
        let config = VisionConfig {
            dtype: None,
            depth: 1,
            hidden_size: 8,
            intermediate_size: 16,
            num_heads: 1,
            in_channels: 3,
            patch_size: 16,
            temporal_patch_size: 2,
            spatial_merge_size: 2,
            out_hidden_size: 8,
            num_position_embeddings: 16,
        };
        let device = candle_core::Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let tower = VisionTower::load(&config, vb, AttentionImplementation::Eager).unwrap();
        let patches = Tensor::zeros((4, 3 * 2 * 16 * 16), DType::F32, &device).unwrap();
        let result = tower.forward(&patches, &[[1, 2, 2]]).unwrap();
        assert_eq!(result.dims(), &[1, 8]);
    }
}

pub mod images {
    use super::VisionConfig;
    use crate::{
        media::{input_bytes, run_media_tool},
        schema::MediaInput,
    };
    use anyhow::Context;
    use serde_json::{Value, json};
    use std::{fs, path::Path};

    pub struct Images {
        pub patches: Vec<f32>,
        pub grids: Vec<[usize; 3]>,
        pub token_count: usize,
    }

    pub fn validate_config(path: &Path, vision: &VisionConfig) -> anyhow::Result<()> {
        let config: Value = serde_json::from_slice(&fs::read(path)?)?;
        let processor = config
            .get("image_processor")
            .and_then(Value::as_object)
            .context("processor_config.json has no image_processor")?;
        let expected = json!({
            "image_processor_type": "Qwen2VLImageProcessor",
            "do_convert_rgb": true,
            "do_normalize": true,
            "do_rescale": true,
            "do_resize": true,
            "image_mean": [0.5, 0.5, 0.5],
            "image_std": [0.5, 0.5, 0.5],
            "merge_size": vision.spatial_merge_size,
            "patch_size": vision.patch_size,
            "temporal_patch_size": vision.temporal_patch_size,
            "resample": 3,
            "rescale_factor": 1.0 / 255.0,
            "size": {"shortest_edge": 65536, "longest_edge": 16777216}
        });
        for (key, value) in expected.as_object().unwrap() {
            anyhow::ensure!(
                processor.get(key) == Some(value),
                "unsupported Qwen image processor setting {key:?}"
            );
        }
        Ok(())
    }

    fn dimensions(data: &[u8]) -> anyhow::Result<(usize, usize)> {
        let args = [
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height",
            "-of",
            "csv=s=x:p=0",
            "-i",
            "pipe:0",
        ]
        .map(str::to_owned);
        let output = run_media_tool("ffprobe", &args, data)?;
        let output = std::str::from_utf8(&output)?.trim();
        let (width, height) = output.split_once('x').context("image has no dimensions")?;
        Ok((width.parse()?, height.parse()?))
    }

    fn resized_size(height: usize, width: usize, factor: usize) -> anyhow::Result<(usize, usize)> {
        anyhow::ensure!(height > 0 && width > 0, "image dimensions must be positive");
        anyhow::ensure!(
            height.max(width) <= 200usize.saturating_mul(height.min(width)),
            "image aspect ratio exceeds 200"
        );
        let round = |value: usize| {
            ((value as f64 / factor as f64).round() as usize)
                .max(1)
                .saturating_mul(factor)
        };
        let mut h = round(height);
        let mut w = round(width);
        let pixels = h.saturating_mul(w);
        if pixels > 16_777_216 {
            let beta = (height as f64 * width as f64 / 16_777_216f64).sqrt();
            h = ((height as f64 / beta / factor as f64).floor() as usize).max(1) * factor;
            w = ((width as f64 / beta / factor as f64).floor() as usize).max(1) * factor;
        } else if pixels < 65_536 {
            let beta = (65_536f64 / (height as f64 * width as f64)).sqrt();
            h = (height as f64 * beta / factor as f64).ceil() as usize * factor;
            w = (width as f64 * beta / factor as f64).ceil() as usize * factor;
        }
        Ok((h, w))
    }

    pub fn decode_images(
        images: &[MediaInput],
        config: &VisionConfig,
        max_tokens: usize,
    ) -> anyhow::Result<Images> {
        let mut patches = Vec::new();
        let mut grids = Vec::with_capacity(images.len());
        let mut token_count = 0usize;
        let size = config.patch_size;
        let merge = config.spatial_merge_size;
        anyhow::ensure!(
            config.in_channels == 3 && config.temporal_patch_size == 2,
            "unsupported Qwen vision patch layout"
        );
        for image in images {
            let data = input_bytes(image, "image")?;
            let (width, height) = dimensions(&data)?;
            let (new_h, new_w) = resized_size(height, width, size * merge)?;
            let image_tokens = (new_h / (size * merge)).saturating_mul(new_w / (size * merge));
            token_count = token_count.saturating_add(image_tokens);
            anyhow::ensure!(
                token_count <= max_tokens,
                "images require at least {token_count} visual tokens, but only {max_tokens} remain; reduce image resolution or count, or increase --max-model-len"
            );
            let args = vec![
                "-v".into(),
                "error".into(),
                "-i".into(),
                "pipe:0".into(),
                "-vf".into(),
                format!("scale={new_w}:{new_h}:flags=bicubic"),
                "-frames:v".into(),
                "1".into(),
                "-f".into(),
                "rawvideo".into(),
                "-pix_fmt".into(),
                "rgb24".into(),
                "pipe:1".into(),
            ];
            let pixels = run_media_tool("ffmpeg", &args, &data)?;
            anyhow::ensure!(
                pixels.len() == new_h * new_w * 3,
                "decoded image has an unexpected size"
            );
            let grid_h = new_h / size;
            let grid_w = new_w / size;
            grids.push([1, grid_h, grid_w]);
            for block_h in 0..grid_h / merge {
                for block_w in 0..grid_w / merge {
                    for inner_h in 0..merge {
                        for inner_w in 0..merge {
                            let patch_h = (block_h * merge + inner_h) * size;
                            let patch_w = (block_w * merge + inner_w) * size;
                            for channel in 0..3 {
                                for _frame in 0..2 {
                                    for y in 0..size {
                                        for x in 0..size {
                                            let pixel =
                                                pixels[((patch_h + y) * new_w + patch_w + x) * 3
                                                    + channel];
                                            patches.push((pixel as f32 / 255. - 0.5) / 0.5);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        Ok(Images {
            patches,
            grids,
            token_count,
        })
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use base64::{Engine, engine::general_purpose::STANDARD};

        #[test]
        fn oversized_image_dimensions_resize_without_overflow() {
            let (height, width) = resized_size(usize::MAX, usize::MAX, 32).unwrap();
            assert!(height > 0 && width > 0);
            assert!(height * width <= 16_777_216);
        }

        #[test]
        fn uniform_image_has_qwen_patch_layout_and_normalization() {
            let mut ppm = b"P6\n2 2\n255\n".to_vec();
            for _ in 0..4 {
                ppm.extend_from_slice(&[255, 0, 0]);
            }
            let image = format!(
                "data:image/x-portable-pixmap;base64,{}",
                STANDARD.encode(ppm)
            );
            let config = VisionConfig {
                dtype: None,
                depth: 1,
                hidden_size: 72,
                intermediate_size: 128,
                num_heads: 1,
                in_channels: 3,
                patch_size: 16,
                temporal_patch_size: 2,
                spatial_merge_size: 2,
                out_hidden_size: 72,
                num_position_embeddings: 2304,
            };
            let error = decode_images(&[image.clone().into()], &config, 63)
                .err()
                .unwrap();
            assert!(error.to_string().contains("--max-model-len"));
            let result = decode_images(&[image.into()], &config, 64).unwrap();
            assert_eq!(result.grids, vec![[1, 16, 16]]);
            assert_eq!(result.token_count, 64);
            assert_eq!(result.patches.len(), 256 * 3 * 2 * 16 * 16);
            assert!(
                result.patches[..512]
                    .iter()
                    .all(|value| (*value - 1.).abs() < 1e-5)
            );
            assert!(
                result.patches[512..1024]
                    .iter()
                    .all(|value| (*value + 1.).abs() < 1e-5)
            );
        }

        #[test]
        fn multiple_images_keep_separate_grids_and_patch_ranges() {
            let make_image = |rgb: [u8; 3]| {
                let mut ppm = b"P6\n2 2\n255\n".to_vec();
                for _ in 0..4 {
                    ppm.extend_from_slice(&rgb);
                }
                format!(
                    "data:image/x-portable-pixmap;base64,{}",
                    STANDARD.encode(ppm)
                )
            };
            let config = VisionConfig {
                dtype: None,
                depth: 1,
                hidden_size: 72,
                intermediate_size: 128,
                num_heads: 1,
                in_channels: 3,
                patch_size: 16,
                temporal_patch_size: 2,
                spatial_merge_size: 2,
                out_hidden_size: 72,
                num_position_embeddings: 2304,
            };
            let result = decode_images(
                &[
                    make_image([255, 0, 0]).into(),
                    make_image([0, 0, 255]).into(),
                ],
                &config,
                usize::MAX,
            )
            .unwrap();
            let image_width = 256 * 3 * 2 * 16 * 16;
            assert_eq!(result.grids, vec![[1, 16, 16], [1, 16, 16]]);
            assert_eq!(result.patches.len(), image_width * 2);
            assert!((result.patches[0] - 1.).abs() < 1e-5);
            assert!((result.patches[image_width] + 1.).abs() < 1e-5);
            assert!((result.patches[image_width + 1024] - 1.).abs() < 1e-5);
        }

        #[test]
        fn nonuniform_patch_order_matches_reference_processor() {
            let mut ppm = b"P6\n256 256\n255\n".to_vec();
            for y in 0..256u16 {
                for x in 0..256u16 {
                    ppm.extend_from_slice(&[x as u8, y as u8, (x + y) as u8]);
                }
            }
            let image = format!(
                "data:image/x-portable-pixmap;base64,{}",
                STANDARD.encode(ppm)
            );
            let config = VisionConfig {
                dtype: None,
                depth: 1,
                hidden_size: 72,
                intermediate_size: 128,
                num_heads: 1,
                in_channels: 3,
                patch_size: 16,
                temporal_patch_size: 2,
                spatial_merge_size: 2,
                out_hidden_size: 72,
                num_position_embeddings: 2304,
            };
            let result = decode_images(&[image.into()], &config, usize::MAX).unwrap();
            for (patch, offset, expected) in [
                (0, 0, -1.0),
                (0, 15, -0.882353),
                (0, 512, -1.0),
                (0, 1039, -0.882353),
                (1, 0, -0.87451),
                (4, 0, -0.74902),
                (16, 0, 0.003922),
            ] {
                let actual = result.patches[patch * 1536 + offset];
                assert!(
                    (actual - expected).abs() < 1e-5,
                    "patch={patch} offset={offset}"
                );
            }
        }
    }
}

pub mod video {
    use super::VisionConfig;
    use crate::media::{input_bytes, run_file_tool};
    use crate::schema::MediaInput;
    use anyhow::Context;
    use serde::Deserialize;
    use serde_json::{Value, json};
    use std::{fs, io::Write, path::Path};

    pub struct Videos {
        pub patches: Vec<f32>,
        pub grids: Vec<[usize; 3]>,
        pub timestamps: Vec<Vec<f64>>,
    }

    #[derive(Deserialize)]
    struct Probe {
        streams: Vec<Stream>,
    }

    #[derive(Deserialize)]
    struct Stream {
        width: usize,
        height: usize,
        nb_read_frames: Option<String>,
        avg_frame_rate: Option<String>,
        r_frame_rate: Option<String>,
    }

    fn frame_rate(value: &str) -> Option<f64> {
        let (numerator, denominator) = value.split_once('/')?;
        let numerator: f64 = numerator.parse().ok()?;
        let denominator: f64 = denominator.parse().ok()?;
        let fps = numerator / denominator;
        (fps.is_finite()
            && denominator.is_finite()
            && numerator.is_finite()
            && denominator > 0.
            && numerator > 0.)
            .then_some(fps)
    }

    fn resized_size(
        frames: usize,
        height: usize,
        width: usize,
        factor: usize,
    ) -> anyhow::Result<(usize, usize)> {
        anyhow::ensure!(
            height >= factor && width >= factor,
            "video height and width must each be at least {factor} pixels"
        );
        anyhow::ensure!(
            height.max(width) <= 200usize.saturating_mul(height.min(width)),
            "video aspect ratio exceeds 200"
        );
        let round =
            |value: usize| ((value as f64 / factor as f64).round() as usize).saturating_mul(factor);
        let mut h = round(height);
        let mut w = round(width);
        let total = frames
            .div_ceil(2)
            .saturating_mul(2)
            .saturating_mul(h)
            .saturating_mul(w);
        if total > 25_165_824 {
            let beta = (frames as f64 * height as f64 * width as f64 / 25_165_824.).sqrt();
            h = ((height as f64 / beta / factor as f64).floor() as usize).max(1) * factor;
            w = ((width as f64 / beta / factor as f64).floor() as usize).max(1) * factor;
        } else if total < 4096 {
            let beta = (4096. / (frames as f64 * height as f64 * width as f64)).sqrt();
            h = (height as f64 * beta / factor as f64).ceil() as usize * factor;
            w = (width as f64 * beta / factor as f64).ceil() as usize * factor;
        }
        Ok((h, w))
    }

    pub fn validate_config(path: &Path, vision: &VisionConfig) -> anyhow::Result<()> {
        let config: Value = serde_json::from_slice(&fs::read(path)?)?;
        let processor = config
            .get("video_processor")
            .and_then(Value::as_object)
            .context("processor_config.json has no video_processor")?;
        let expected = json!({
            "video_processor_type": "Qwen3VLVideoProcessor",
            "do_convert_rgb": true,
            "do_normalize": true,
            "do_rescale": true,
            "do_resize": true,
            "do_sample_frames": true,
            "image_mean": [0.5, 0.5, 0.5],
            "image_std": [0.5, 0.5, 0.5],
            "merge_size": vision.spatial_merge_size,
            "patch_size": vision.patch_size,
            "temporal_patch_size": vision.temporal_patch_size,
            "resample": 3,
            "rescale_factor": 1.0 / 255.0,
            "fps": 2,
            "min_frames": 4,
            "max_frames": 768,
            "size": {"shortest_edge": 4096, "longest_edge": 25165824}
        });
        for (key, value) in expected.as_object().unwrap() {
            anyhow::ensure!(
                processor.get(key) == Some(value),
                "unsupported Qwen video processor setting {key:?}"
            );
        }
        Ok(())
    }

    pub fn decode_videos(
        inputs: &[MediaInput],
        config: &VisionConfig,
        max_tokens: usize,
    ) -> anyhow::Result<Videos> {
        crate::media::require_video_tools()?;
        let mut patches = Vec::new();
        let mut grids = Vec::with_capacity(inputs.len());
        let mut timestamps = Vec::with_capacity(inputs.len());
        let mut token_count = 0usize;
        let size = config.patch_size;
        let merge = config.spatial_merge_size;
        anyhow::ensure!(
            config.in_channels == 3 && config.temporal_patch_size == 2,
            "unsupported Qwen video patch layout"
        );
        for input in inputs {
            let data = input_bytes(input, "video")?;
            let mut file = tempfile::NamedTempFile::new()?;
            file.write_all(&data)?;
            file.flush()?;
            let path = file.path().to_string_lossy().into_owned();
            let probe_args = [
                "-v".into(),
                "error".into(),
                "-count_frames".into(),
                "-select_streams".into(),
                "v:0".into(),
                "-show_entries".into(),
                "stream=width,height,nb_read_frames,avg_frame_rate,r_frame_rate".into(),
                "-of".into(),
                "json".into(),
                path.clone(),
            ];
            let probe: Probe =
                serde_json::from_slice(&run_file_tool("ffprobe", &probe_args, 1 << 20)?)?;
            let stream = probe
                .streams
                .first()
                .context("video has no visual stream")?;
            let total_frames: usize = stream
                .nb_read_frames
                .as_ref()
                .context("video frame count is unavailable")?
                .parse()
                .context("invalid video frame count")?;
            anyhow::ensure!(total_frames > 0, "video has no frames");
            let fps = stream
                .avg_frame_rate
                .as_deref()
                .and_then(frame_rate)
                .or_else(|| stream.r_frame_rate.as_deref().and_then(frame_rate))
                .context("video frame rate is unavailable")?;
            let sample_count = ((total_frames as f64 / fps * 2.) as usize)
                .clamp(4, 768)
                .min(total_frames);
            let indices: Vec<_> = (0..sample_count)
                .map(|index| {
                    if sample_count == 1 {
                        0
                    } else {
                        ((total_frames - 1) as f64 * index as f64 / (sample_count - 1) as f64)
                            .round_ties_even() as usize
                    }
                })
                .collect();
            let (height, width) =
                resized_size(sample_count, stream.height, stream.width, size * merge)?;
            let video_tokens = sample_count
                .div_ceil(2)
                .saturating_mul(height / (size * merge))
                .saturating_mul(width / (size * merge));
            token_count = token_count.saturating_add(video_tokens);
            anyhow::ensure!(
                token_count <= max_tokens,
                "videos require at least {token_count} visual tokens, but only {max_tokens} remain; use shorter or lower-resolution clips, or increase --max-model-len"
            );
            let selection = indices
                .iter()
                .map(|index| format!("eq(n\\,{index})"))
                .collect::<Vec<_>>()
                .join("+");
            let filter = format!("select={selection},scale={width}:{height}:flags=bicubic");
            let decode_args = vec![
                "-v".into(),
                "error".into(),
                "-i".into(),
                path,
                "-vf".into(),
                filter,
                "-fps_mode".into(),
                "passthrough".into(),
                "-frames:v".into(),
                sample_count.to_string(),
                "-f".into(),
                "rawvideo".into(),
                "-pix_fmt".into(),
                "rgb24".into(),
                "pipe:1".into(),
            ];
            let pixels = run_file_tool("ffmpeg", &decode_args, 128 << 20)?;
            let frame_width = height
                .checked_mul(width)
                .and_then(|pixels| pixels.checked_mul(3))
                .context("decoded video dimensions exceed addressable memory")?;
            let expected_bytes = sample_count
                .checked_mul(frame_width)
                .context("decoded video dimensions exceed addressable memory")?;
            anyhow::ensure!(
                pixels.len() == expected_bytes,
                "decoded video has {} frames, expected {sample_count}",
                pixels.len() / frame_width
            );
            let grid_t = sample_count.div_ceil(2);
            let grid_h = height / size;
            let grid_w = width / size;
            grids.push([grid_t, grid_h, grid_w]);
            let mut video_timestamps = Vec::with_capacity(grid_t);
            for frame in 0..grid_t {
                let first = frame * 2;
                let second = (first + 1).min(sample_count - 1);
                video_timestamps.push((indices[first] + indices[second]) as f64 / (2. * fps));
                for block_h in 0..grid_h / merge {
                    for block_w in 0..grid_w / merge {
                        for inner_h in 0..merge {
                            for inner_w in 0..merge {
                                let patch_h = (block_h * merge + inner_h) * size;
                                let patch_w = (block_w * merge + inner_w) * size;
                                for channel in 0..3 {
                                    for frame_index in [first, second] {
                                        for y in 0..size {
                                            for x in 0..size {
                                                let pixel = pixels[frame_index * frame_width
                                                    + ((patch_h + y) * width + patch_w + x) * 3
                                                    + channel];
                                                patches.push((pixel as f32 / 255. - 0.5) / 0.5);
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            timestamps.push(video_timestamps);
        }
        Ok(Videos {
            patches,
            grids,
            timestamps,
        })
    }

    #[cfg(test)]
    pub(crate) fn synthetic_test_video() -> Vec<u8> {
        let mut frames = Vec::new();
        for rgb in [[255, 0, 0], [255, 0, 0], [0, 0, 255], [0, 0, 255]] {
            for _ in 0..64 * 64 {
                frames.extend_from_slice(&rgb);
            }
        }
        let args = [
            "-v".into(),
            "error".into(),
            "-f".into(),
            "rawvideo".into(),
            "-pix_fmt".into(),
            "rgb24".into(),
            "-s".into(),
            "64x64".into(),
            "-r".into(),
            "2".into(),
            "-i".into(),
            "pipe:0".into(),
            "-c:v".into(),
            "ffv1".into(),
            "-f".into(),
            "matroska".into(),
            "pipe:1".into(),
        ];
        crate::media::run_media_tool("ffmpeg", &args, &frames).unwrap()
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn oversized_video_dimensions_resize_without_overflow() {
            let (height, width) = resized_size(768, usize::MAX, usize::MAX, 32).unwrap();
            assert!(height > 0 && width > 0);
            assert!(768 * height * width <= 25_165_824);
        }

        #[test]
        fn four_frame_video_has_two_temporal_patch_groups() {
            let video = synthetic_test_video();
            let config = VisionConfig {
                dtype: None,
                depth: 1,
                hidden_size: 72,
                intermediate_size: 128,
                num_heads: 1,
                in_channels: 3,
                patch_size: 16,
                temporal_patch_size: 2,
                spatial_merge_size: 2,
                out_hidden_size: 72,
                num_position_embeddings: 2304,
            };
            let error = decode_videos(&[MediaInput::Bytes(video.clone())], &config, 7)
                .err()
                .unwrap();
            assert!(error.to_string().contains("--max-model-len"));
            let decoded = decode_videos(&[MediaInput::Bytes(video)], &config, 8).unwrap();
            assert_eq!(decoded.grids, vec![[2, 4, 4]]);
            assert_eq!(decoded.timestamps, vec![vec![0.25, 1.25]]);
            assert_eq!(decoded.patches.len(), 2 * 4 * 4 * 1536);
            assert!(decoded.patches[0] > 0.9);
            assert!(decoded.patches[16 * 1536] < -0.9);
            assert!(decoded.patches[16 * 1536 + 1024] > 0.9);
        }
    }
}

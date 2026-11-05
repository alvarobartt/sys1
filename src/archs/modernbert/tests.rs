use super::*;
#[cfg(feature = "metal")]
use crate::archs::metal::ModernBertEncoder as Encoder;

#[test]
fn natural_geometry_eager_attention_preserves_valid_tokens() -> Result<()> {
    let device = Device::Cpu;
    let lengths = [3usize, 5];
    let q = Tensor::arange(0f32, 40f32, &device)?.reshape((2, 1, 5, 4))?;
    let q = (&q / 31.)?;
    let k = (&q * 0.7)?;
    let v = (&q + 0.2)?;
    let keep = Tensor::new(&[[1f32, 1., 1., 0., 0.], [1., 1., 1., 1., 1.]], &device)?;
    let global = global_attention_mask(&keep, 5, DType::F32)?;
    let local = local_attention_mask(5, 2, DType::F32, &device)?;
    for mask in [&global, &global.broadcast_add(&local)?] {
        let actual =
            padding_independent_eager_attention(&q, &k, &v, 0.5, Some(mask), &lengths, None)?;
        for (slot, &length) in lengths.iter().enumerate() {
            let valid = |tensor: &Tensor| {
                tensor
                    .narrow(0, slot, 1)?
                    .narrow(2, 0, length)?
                    .contiguous()
            };
            let valid_mask = mask
                .broadcast_as((2, 1, 5, 5))?
                .narrow(0, slot, 1)?
                .narrow(2, 0, length)?
                .narrow(3, 0, length)?
                .contiguous()?;
            let expected = eager_attention(
                &valid(&q)?,
                &valid(&k)?,
                &valid(&v)?,
                0.5,
                Some(&valid_mask),
            )?;
            assert_eq!(
                valid(&actual)?.flatten_all()?.to_vec1::<f32>()?,
                expected.flatten_all()?.to_vec1::<f32>()?
            );
        }
    }
    assert!(padding_independent_eager_attention(&q, &k, &v, 0.5, None, &[0, 5], None).is_err());
    Ok(())
}
use candle_core::IndexOp;

#[test]
fn local_attention_mask_only_exposes_nearby_tokens() -> Result<()> {
    let mask = local_attention_mask(4, 1, DType::F32, &Device::Cpu)?.to_vec2::<f32>()?;

    assert_eq!(
        mask,
        vec![
            vec![0.0, 0.0, ATTENTION_MASK_VALUE, ATTENTION_MASK_VALUE],
            vec![0.0, 0.0, 0.0, ATTENTION_MASK_VALUE],
            vec![ATTENTION_MASK_VALUE, 0.0, 0.0, 0.0],
            vec![ATTENTION_MASK_VALUE, ATTENTION_MASK_VALUE, 0.0, 0.0],
        ]
    );
    Ok(())
}

#[test]
fn global_attention_mask_hides_padding_tokens() -> Result<()> {
    let mask = Tensor::new(&[[1u32, 1, 0]], &Device::Cpu)?;
    let mask = global_attention_mask(&mask, 2, DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;

    assert_eq!(
        mask,
        vec![
            0.0,
            0.0,
            ATTENTION_MASK_VALUE,
            0.0,
            0.0,
            ATTENTION_MASK_VALUE
        ]
    );
    Ok(())
}

#[tokio::test]
async fn fp32_hidden_states() -> anyhow::Result<()> {
    let path = crate::hub::download(
        "convaiinnovations/laya",
        "aa8c91ca088ec597df95a0d1c76b3063cb2ae5e8",
    )
    .await?;

    let device = crate::device::load()?;
    let config = Config::load(&path.join("encoder/config.json"))?;
    let weights = path.join("model.safetensors");
    // SAFETY: All file-backed mmap constructors are marked `unsafe` because of the potential
    // for undefined behavior using the map if the underlying file is modified or out of
    // process.
    //
    // More information at https://github.com/RazrFalcon/memmap2-rs/blob/a02e2a/src/lib.rs#L135-L165
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[weights], DType::F32, &device)? };
    let vb = vb.rename_f(|name| {
        name.strip_prefix("model.")
            .map(|name| format!("encoder.{name}"))
            .unwrap_or_else(|| name.to_owned())
    });
    let encoder = Encoder::load(vb, &config, DType::F32, AttentionImplementation::Eager)?;

    let tokenizer_path = path.join("tokenizer/tokenizer.json");
    let tokenizer = crate::tokenizer::from_json(&fs::read(tokenizer_path)?)?;
    let cls_id = tokenizer
        .token_to_id("[CLS]")
        .ok_or_else(|| anyhow::anyhow!("tokenizer is missing [CLS]"))?;
    let sep_id = tokenizer
        .token_to_id("[SEP]")
        .ok_or_else(|| anyhow::anyhow!("tokenizer is missing [SEP]"))?;
    let pad_id = tokenizer
        .token_to_id("[PAD]")
        .ok_or_else(|| anyhow::anyhow!("tokenizer is missing [PAD]"))?;

    let mut ids = vec![cls_id];
    ids.extend(tokenizer.encode("What is Deep Learning?")?);
    ids.push(sep_id);
    let valid_length = ids.len();

    let mut mask = vec![1f32; ids.len()];
    ids.resize(32, pad_id);
    mask.resize(32, 0.0);

    let length = ids.len();
    let ids = Tensor::from_vec(ids, (1, length), &device)?;
    let mask = Tensor::from_vec(mask, (1, length), &device)?;
    let hidden = encoder.forward(&ids, &mask, &[valid_length], true)?;

    insta::assert_yaml_snapshot!(
        "modernbert_fp32_hidden_states",
        hidden.i((0, 0, 0..128))?.to_vec1::<f32>()?,
        {
            "[]" => insta::rounded_redaction(3),
        }
    );

    Ok(())
}

#[test]
fn cpu_windowed_attention_matches_dense_attention() -> Result<()> {
    let device = Device::Cpu;
    let values = Tensor::arange(0u32, 48, &device)?
        .to_dtype(DType::F32)?
        .reshape((1, 2, 6, 4))?;
    let q = (&values / 37.0)?;
    let k = (&values / 29.0)?;
    let v = (&values / 19.0)?;
    let local = local_attention_mask(6, 2, DType::F32, &device)?;
    let padding = Tensor::new(&[[1f32, 1.0, 1.0, 1.0, 0.0, 0.0]], &device)?;
    let global = global_attention_mask(&padding, 6, DType::F32)?;

    for mask in [&local, &global.broadcast_add(&local)?] {
        let dense = eager_attention(&q, &k, &v, 0.5, Some(mask))?;
        let windowed = cpu_windowed_attention(&q, &k, &v, 0.5, Some(mask), 2)?;
        let dense = dense.flatten_all()?.to_vec1::<f32>()?;
        let windowed = windowed.flatten_all()?.to_vec1::<f32>()?;
        let difference = dense
            .iter()
            .zip(windowed)
            .map(|(left, right)| (left - right).abs())
            .fold(0f32, f32::max);
        assert!(difference < 1e-5, "maximum difference was {difference}");
    }

    Ok(())
}

#[test]
fn eager_bf16_attention_handles_padding_masks_without_nans() {
    let device = Device::Cpu;
    let q = Tensor::zeros((1, 1, 1, 1), DType::BF16, &device).unwrap();
    let k = Tensor::zeros((1, 1, 2, 1), DType::BF16, &device).unwrap();
    let v = Tensor::new(&[2f32, 3.], &device)
        .unwrap()
        .reshape((1, 1, 2, 1))
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap();
    let mask = Tensor::new(&[0f32, f32::NEG_INFINITY], &device)
        .unwrap()
        .reshape((1, 1, 1, 2))
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap();
    let output = eager_attention(&q, &k, &v, 1., Some(&mask))
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();

    assert!(output.iter().all(|value| value.is_finite()));
    assert_eq!(output, vec![2.0]);
}

#[test]
fn bf16_attention_masks_remain_finite() {
    let device = Device::Cpu;
    let padding = Tensor::new(&[[1f32, 0.0]], &device).unwrap();
    let global = global_attention_mask(&padding, 2, DType::BF16)
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    let local = local_attention_mask(4, 1, DType::BF16, &device)
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();

    assert!(global.iter().chain(&local).all(|value| value.is_finite()));
    assert!(global.iter().any(|value| *value < -1_000.0));
    assert!(local.iter().any(|value| *value < -1_000.0));
}

use anyhow::{Context, bail};
use serde_json::Value;
use std::collections::HashMap;
use tokenizers::{
    canonicalize_value, from_json as pipeline_from_json,
    pipeline::{EncodeOptions, PipelineTokenizer},
};

const MISSING_ATOMS: [char; 13] = [
    'À', 'Á', 'õ', 'ö', '÷', 'ø', 'ù', 'ú', 'û', 'ü', 'ý', 'þ', 'ÿ',
];

struct AddedToken {
    content: String,
    id: u32,
}

pub struct Tokenizer {
    inner: PipelineTokenizer,
    added: Vec<AddedToken>,
    ids: HashMap<String, u32>,
}

impl Tokenizer {
    pub fn token_to_id(&self, token: &str) -> Option<u32> {
        self.ids.get(token).copied()
    }

    pub fn encode(&self, text: &str) -> anyhow::Result<Vec<u32>> {
        let mut output = Vec::new();
        let mut offset = 0;
        while offset < text.len() {
            let next = self
                .added
                .iter()
                .filter_map(|token| {
                    text[offset..]
                        .find(&token.content)
                        .map(|index| (offset + index, token))
                })
                .min_by(|(left_index, left), (right_index, right)| {
                    left_index
                        .cmp(right_index)
                        .then_with(|| right.content.len().cmp(&left.content.len()))
                });
            let Some((index, token)) = next else {
                output.extend(self.encode_base(&text[offset..])?);
                break;
            };
            output.extend(self.encode_base(&text[offset..index])?);
            output.push(token.id);
            offset = index + token.content.len();
        }
        Ok(output)
    }

    fn encode_base(&self, text: &str) -> anyhow::Result<Vec<u32>> {
        if text.is_empty() {
            return Ok(Vec::new());
        }
        let encoding = self
            .inner
            .encode(text, &EncodeOptions::no_specials())
            .wait()
            .map_err(|error| anyhow::anyhow!(error.to_string()))?
            .remove(0);
        Ok(encoding.ids().iter().map(|token| token.id()).collect())
    }
}

pub fn from_json(json: &[u8]) -> anyhow::Result<Tokenizer> {
    let mut value: Value = serde_json::from_slice(json)?;
    migrate(&mut value)?;
    let added = read_added(&value)?;
    let ids = added
        .iter()
        .map(|token| (token.content.clone(), token.id))
        .collect();
    value["added_tokens"] = Value::Array(Vec::new());
    let inner = pipeline_from_json(&serde_json::to_string(&value)?)
        .map_err(|error| anyhow::anyhow!(error.to_string()))?;
    Ok(Tokenizer { inner, added, ids })
}

fn migrate(value: &mut Value) -> anyhow::Result<()> {
    if value.get("laya_migration").and_then(Value::as_u64) == Some(1) {
        return Ok(());
    }
    if value.get("version").and_then(Value::as_str) != Some("1.0") {
        bail!("expected a Laya Tokenizers 1.0 file");
    }
    let added = read_added(value)?;
    let removable: Vec<_> = added
        .iter()
        .filter(|token| (50254..=50279).contains(&token.id))
        .map(|token| token.content.clone())
        .collect();
    if !matches!(removable.len(), 0 | 26) {
        bail!("unexpected Laya added-token layout");
    }
    if !removable.is_empty() {
        let vocab = value
            .get_mut("model")
            .and_then(|model| model.get_mut("vocab"))
            .and_then(Value::as_object_mut)
            .context("tokenizer has no model vocab")?;
        for token in removable {
            vocab.remove(&token);
        }
        for (offset, atom) in MISSING_ATOMS.into_iter().enumerate() {
            vocab.insert(atom.to_string(), Value::from(50254 + offset as u32));
        }
    }
    let model = value
        .get_mut("model")
        .and_then(Value::as_object_mut)
        .context("tokenizer has no model")?;
    migrate_byte_fallback_tabs(model)?;
    canonicalize_value(value).map_err(|error| anyhow::anyhow!(error.to_string()))?;
    value
        .as_object_mut()
        .unwrap()
        .insert("laya_migration".into(), Value::from(1));
    Ok(())
}

fn migrate_byte_fallback_tabs(model: &mut serde_json::Map<String, Value>) -> anyhow::Result<()> {
    if model.get("byte_fallback").and_then(Value::as_bool) == Some(true) {
        let vocab = model
            .get_mut("vocab")
            .and_then(Value::as_object_mut)
            .context("tokenizer has no model vocab")?;
        if !vocab.contains_key("<0x09>") {
            let tab_tokens: Vec<_> = vocab
                .keys()
                .filter(|token| token.contains('\t'))
                .cloned()
                .collect();
            if tab_tokens.is_empty() {
                bail!("byte-fallback tokenizer is missing <0x09>");
            }
            for token in tab_tokens {
                let id = vocab.remove(&token).unwrap();
                vocab.insert(token.replace('\t', "<0x09>"), id);
            }
            let merges = model
                .get_mut("merges")
                .and_then(Value::as_array_mut)
                .context("byte-fallback tokenizer has no BPE merges")?;
            for merge in merges {
                let parts = merge
                    .as_array_mut()
                    .context("byte-fallback tokenizer has an invalid BPE merge")?;
                for part in parts {
                    if let Some(token) = part.as_str()
                        && token.contains('\t')
                    {
                        *part = Value::String(token.replace('\t', "<0x09>"));
                    }
                }
            }
        }
    }
    Ok(())
}

fn read_added(value: &Value) -> anyhow::Result<Vec<AddedToken>> {
    value
        .get("added_tokens")
        .and_then(Value::as_array)
        .context("tokenizer has no added_tokens")?
        .iter()
        .map(|token| {
            Ok(AddedToken {
                content: token
                    .get("content")
                    .and_then(Value::as_str)
                    .context("added token has no content")?
                    .to_owned(),
                id: token
                    .get("id")
                    .and_then(Value::as_u64)
                    .context("added token has no id")? as u32,
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn migrates_tab_tokens_without_duplicate_vocabulary_ids() {
        let mut model = json!({
            "byte_fallback": true,
            "vocab": {"\t": 9, "\t\t": 10, "word": 11},
            "merges": [["\t", "\t"], ["\t\t", "\t"]]
        });

        migrate_byte_fallback_tabs(model.as_object_mut().unwrap()).unwrap();

        assert_eq!(
            model["vocab"],
            json!({"<0x09>": 9, "<0x09><0x09>": 10, "word": 11})
        );
        assert_eq!(
            model["merges"],
            json!([["<0x09>", "<0x09>"], ["<0x09><0x09>", "<0x09>"]])
        );
    }
}

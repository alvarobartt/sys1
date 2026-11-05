use super::{
    Encoded, Question, confidence,
    schema_cache::{CompiledSchema, SchemaCache},
};
use crate::{
    architectures::qwen35_vision as vision,
    schema::{ApiError, DecisionRequest, DecisionResponse, Usage},
    tokenizer,
};
use serde_json::{Map, Value, json};

const PREFIX: &str = "<|im_start|>system\n";
const USER_PREFIX: &str = "<|im_end|>\n<|im_start|>user\nSTATE:\n";
const SUFFIX: &str =
    "\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nJOINT SCHEMA DECISIONS:";

const SYSTEM_PROMPT: &str = "Read the complete state and schema. Decide every field jointly. Each answer must be exactly one of that field's allowed options.";
fn winning_option(probabilities: &[f32]) -> usize {
    probabilities
        .iter()
        .enumerate()
        .max_by(|left, right| left.1.total_cmp(right.1).then_with(|| right.0.cmp(&left.0)))
        .expect("validated questions have at least one option")
        .0
}

#[derive(Default)]
pub(crate) struct JointProtocol {
    schema_cache: SchemaCache,
}

fn tokens(tokenizer: &tokenizer::Tokenizer, value: &str) -> Result<Vec<u32>, ApiError> {
    tokenizer
        .encode(value)
        .map_err(|error| ApiError::new(error.to_string()))
}

impl JointProtocol {
    pub(crate) fn render(value: &Value) -> String {
        if let Some(value) = value.as_str() {
            return value.to_owned();
        }

        fn write(value: &Value, output: &mut String) {
            match value {
                Value::Object(object) => {
                    output.push('{');
                    let mut keys: Vec<_> = object.keys().collect();
                    keys.sort();
                    for (index, key) in keys.into_iter().enumerate() {
                        if index != 0 {
                            output.push(',');
                        }
                        output.push_str(&serde_json::to_string(key).unwrap());
                        output.push(':');
                        write(&object[key], output);
                    }
                    output.push('}');
                }
                Value::Array(array) => {
                    output.push('[');
                    for (index, child) in array.iter().enumerate() {
                        if index != 0 {
                            output.push(',');
                        }
                        write(child, output);
                    }
                    output.push(']');
                }
                Value::Number(number) if number.is_f64() => {
                    let shortest = number.to_string();
                    let negative = shortest.starts_with('-');
                    let unsigned = shortest.trim_start_matches('-');
                    let (mantissa, exponent) = unsigned
                        .split_once('e')
                        .map(|(m, e)| (m, e.parse::<i32>().unwrap()))
                        .unwrap_or((unsigned, 0));
                    let decimal = mantissa.find('.').unwrap_or(mantissa.len()) as i32;
                    let mut digits = mantissa.replace('.', "");
                    let leading = digits.bytes().take_while(|&digit| digit == b'0').count();
                    let zero = leading == digits.len();
                    let mut position = decimal + exponent;
                    if zero {
                        digits = "0".into();
                        position = 1;
                    } else {
                        digits.drain(..leading);
                        position -= leading as i32;
                        while digits.len() > 1 && digits.ends_with('0') {
                            digits.pop();
                        }
                    }
                    if negative {
                        output.push('-');
                    }
                    if !zero && !(-3..=16).contains(&position) {
                        output.push_str(&digits[..1]);
                        if digits.len() > 1 {
                            output.push('.');
                            output.push_str(&digits[1..]);
                        }
                        output.push_str(&format!("e{:+03}", position - 1));
                    } else if position <= 0 {
                        output.push_str("0.");
                        output.extend(std::iter::repeat_n('0', (-position) as usize));
                        output.push_str(&digits);
                    } else if position as usize >= digits.len() {
                        output.push_str(&digits);
                        output.extend(std::iter::repeat_n('0', position as usize - digits.len()));
                        output.push_str(".0");
                    } else {
                        output.push_str(&digits[..position as usize]);
                        output.push('.');
                        output.push_str(&digits[position as usize..]);
                    }
                }
                _ => output.push_str(&value.to_string()),
            }
        }
        let mut output = String::new();
        write(value, &mut output);
        output
    }

    fn instruction(object: &Map<String, Value>, id: &str) -> String {
        match object.get("instructions") {
            None | Some(Value::Null) => id.to_owned(),
            Some(Value::String(value)) if value.is_empty() => id.to_owned(),
            Some(value) => Self::render(value),
        }
    }

    fn options(kind: &str, criteria: &Value, id: &str) -> Result<Vec<(String, Value)>, ApiError> {
        match kind {
            "noul" => {
                let mut options = vec![
                    (
                        "true".to_owned(),
                        json!("The proposition is true or the answer is yes."),
                    ),
                    (
                        "false".to_owned(),
                        json!("The proposition is false or the answer is no."),
                    ),
                ];
                if !criteria.is_null() {
                    let supplied = criteria.as_object().ok_or_else(|| {
                        ApiError::new(format!("question {id:?} noul criteria must be an object"))
                    })?;
                    for (key, value) in &mut options {
                        if let Some(replacement) = supplied.get(key) {
                            *value = replacement.clone();
                        }
                    }
                }
                Ok(options)
            }
            "choice" => {
                let supplied = criteria.as_object().ok_or_else(|| {
                    ApiError::new(format!("question {id:?} choice criteria must be an object"))
                })?;
                if supplied.is_empty() {
                    return Err(ApiError::new(format!(
                        "question {id:?} criteria must not be empty"
                    )));
                }
                let mut options: Vec<_> = supplied
                    .iter()
                    .map(|(key, value)| (key.clone(), value.clone()))
                    .collect();
                options.sort_by(|left, right| left.0.cmp(&right.0));
                Ok(options)
            }
            "score" => {
                let supplied = criteria.as_array().ok_or_else(|| {
                    ApiError::new(format!("question {id:?} score criteria must be an array"))
                })?;
                if supplied.is_empty() {
                    return Err(ApiError::new(format!(
                        "question {id:?} criteria must not be empty"
                    )));
                }
                Ok(supplied
                    .iter()
                    .enumerate()
                    .map(|(index, value)| (index.to_string(), value.clone()))
                    .collect())
            }
            _ => Err(ApiError::new(format!(
                "question {id:?} type must be noul, choice, or score"
            ))),
        }
    }

    fn compile_schema(
        tokenizer: &tokenizer::Tokenizer,
        questions_input: &Map<String, Value>,
    ) -> Result<CompiledSchema, ApiError> {
        let tokens = |value: &str| {
            tokenizer
                .encode(value)
                .map_err(|error| ApiError::new(error.to_string()))
        };
        if questions_input.is_empty() {
            return Err(ApiError::new("questions must not be empty"));
        }
        let mut schema = tokens("\n\nSCHEMA FIELDS:\n")?;
        let mut questions = Vec::with_capacity(questions_input.len());
        for (index, (id, value)) in questions_input.iter().enumerate() {
            let object = value
                .as_object()
                .ok_or_else(|| ApiError::new(format!("question {id:?} must be an object")))?;
            let kind_name = object
                .get("type")
                .and_then(Value::as_str)
                .ok_or_else(|| ApiError::new(format!("question {id:?} requires type")))?;
            let kind = match kind_name {
                "noul" => 0,
                "choice" => 1,
                "score" => 2,
                _ => return Err(ApiError::new(format!("question {id:?} has invalid type"))),
            };
            let criterion = object.get("criteria").cloned().unwrap_or(Value::Null);
            let options = Self::options(kind_name, &criterion, id)?;
            schema.extend(tokens(&format!(
                "\nFIELD {}\nID: {id}\nTYPE: {kind_name}\nINSTRUCTION: ",
                index + 1
            ))?);
            let question_start = schema.len();
            let instruction = Self::instruction(object, id);
            schema.extend(tokens(&instruction)?);
            let question_span = (question_start, schema.len());
            schema.extend(tokens("\nALLOWED OPTIONS:\n")?);
            let mut option_spans = Vec::with_capacity(options.len());
            let mut option_ids = Vec::with_capacity(options.len());
            for (option_index, (option_id, description)) in options.into_iter().enumerate() {
                schema.extend(tokens(&format!("OPTION {}: ", option_index + 1))?);
                let start = schema.len();
                let mut semantics = Map::new();
                semantics.insert("option_id".into(), Value::String(option_id.clone()));
                if !description.is_null() {
                    semantics.insert("description".into(), description);
                }
                schema.extend(tokens(&Self::render(&Value::Object(semantics)))?);
                option_spans.push((start, schema.len()));
                option_ids.push(option_id);
                schema.extend(tokens("\n")?);
            }
            schema.extend(tokens("END FIELD\n")?);
            questions.push(Question {
                id: id.clone(),
                kind,
                question_span,
                option_spans,
                option_ids,
            });
        }
        Ok(CompiledSchema {
            ids: schema,
            questions,
        })
    }

    pub(crate) fn encode(
        &self,
        context: &super::Context,
        request: &DecisionRequest,
    ) -> Result<Encoded, ApiError> {
        let compiled = self.schema_cache.get_or_build(&request.questions, || {
            Self::compile_schema(&context.tokenizer, &request.questions)
        })?;
        let schema = &compiled.ids;
        let mut questions = compiled.questions.clone();
        let criteria = request
            .questions
            .values()
            .map(|value| value.get("criteria").cloned().unwrap_or(Value::Null))
            .collect();
        let mut prefix = tokens(
            &context.tokenizer,
            &format!("{PREFIX}{SYSTEM_PROMPT}{USER_PREFIX}"),
        )?;
        let suffix = tokens(&context.tokenizer, SUFFIX)?;
        let baseline = prefix.len() + schema.len() + suffix.len();
        if baseline > context.max_length {
            return Err(ApiError::new(format!(
                "schema requires {baseline} tokens before state; maximum is {}. Reduce schema size or increase --max-model-len",
                context.max_length
            )));
        }
        let media_budget = context.max_length - baseline;
        let patch_width =
            context.vision_config.patch_size * context.vision_config.spatial_merge_size;
        let min_image_tokens = (65_536 / patch_width.pow(2)).max(1);
        if request.images.len() > media_budget / min_image_tokens {
            return Err(ApiError::new(format!(
                "image count exceeds the {media_budget}-token media budget; use fewer images or increase --max-model-len"
            )));
        }
        let images = if request.images.is_empty() {
            None
        } else {
            Some(
                vision::images::decode_images(
                    &request.images,
                    &context.vision_config,
                    media_budget,
                )
                .map_err(|error| ApiError::new(error.to_string()))?,
            )
        };
        let video_budget = media_budget - images.as_ref().map_or(0, |images| images.token_count);
        if request.videos.len() > video_budget {
            return Err(ApiError::new(format!(
                "video count exceeds the {video_budget}-token media budget; use fewer videos or increase --max-model-len"
            )));
        }
        let videos = if request.videos.is_empty() {
            None
        } else {
            Some(
                vision::video::decode_videos(&request.videos, &context.vision_config, video_budget)
                    .map_err(|error| ApiError::new(error.to_string()))?,
            )
        };
        let mut media_text = String::new();
        if let Some(images) = &images {
            for &[t, h, w] in &images.grids {
                media_text.push_str("<|vision_start|>");
                for _ in 0..t * h * w / context.vision_config.spatial_merge_size.pow(2) {
                    media_text.push_str("<|image_pad|>");
                }
                media_text.push_str("<|vision_end|>");
            }
        }
        if let Some(videos) = &videos {
            for (grid, timestamps) in videos.grids.iter().zip(&videos.timestamps) {
                let &[t, h, w] = grid;
                let count = h * w / context.vision_config.spatial_merge_size.pow(2);
                for timestamp in &timestamps[..t] {
                    media_text.push_str(&format!("<{timestamp:.1} seconds>"));
                    media_text.push_str("<|vision_start|>");
                    for _ in 0..count {
                        media_text.push_str("<|video_pad|>");
                    }
                    media_text.push_str("<|vision_end|>");
                }
            }
        }
        if !media_text.is_empty() {
            media_text.push('\n');
            prefix.extend(tokens(&context.tokenizer, &media_text)?);
        }
        let mut state = tokens(&context.tokenizer, &Self::render(&request.state))?;
        let fixed = prefix.len() + schema.len() + suffix.len();
        if fixed > context.max_length {
            return Err(ApiError::new(format!(
                "schema and media require {fixed} tokens before state; maximum is {}. Reduce media resolution or schema size, or increase --max-model-len",
                context.max_length
            )));
        }
        state.truncate(context.max_length - fixed);
        let offset = prefix.len() + state.len();
        for question in &mut questions {
            question.question_span.0 += offset;
            question.question_span.1 += offset;
            for span in &mut question.option_spans {
                span.0 += offset;
                span.1 += offset;
            }
        }
        prefix.extend(state);
        prefix.extend(schema);
        prefix.extend(suffix);
        context.prepare(prefix, questions, criteria, images, videos)
    }

    pub(crate) fn response_logits(
        &self,
        encoded: &Encoded,
        logits: Vec<Vec<f32>>,
    ) -> Result<DecisionResponse, ApiError> {
        crate::numerics::json("logits", &logits)
            .map_err(|error| ApiError::internal(error.to_string()))?;
        let mut answers = Map::new();
        for ((question, criterion), scores) in
            encoded.questions.iter().zip(&encoded.criteria).zip(logits)
        {
            if scores.iter().any(|score| !score.is_finite()) {
                return Err(ApiError::internal(format!(
                    "non-finite logits for question {:?}",
                    question.id
                )));
            }
            let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut probs: Vec<_> = scores.iter().map(|value| (value - max).exp()).collect();
            let sum: f32 = probs.iter().sum();
            for probability in &mut probs {
                *probability /= sum;
            }
            let rounded = |value: f32| (value as f64 * 10000.).round() / 10000.;
            let answer = match question.kind {
                0 => json!({"type":"noul", "noul": rounded(probs[0])}),
                1 => {
                    let winner = winning_option(&probs);
                    let values = question
                        .option_ids
                        .iter()
                        .cloned()
                        .zip(probs.iter().map(|value| json!(rounded(*value))))
                        .collect::<Map<_, _>>();
                    json!({"type":"choice", "choice":question.option_ids[winner], "confidence":rounded(confidence::choice_confidence(&probs)), "probabilities":values})
                }
                _ => {
                    let values = question
                        .option_ids
                        .iter()
                        .cloned()
                        .zip(probs.iter().map(|value| json!(rounded(*value))))
                        .collect::<Map<_, _>>();
                    let legend = criterion
                        .as_array()
                        .unwrap()
                        .iter()
                        .enumerate()
                        .map(|(index, value)| (index.to_string(), value.clone()))
                        .collect::<Map<_, _>>();
                    let score: f32 = probs
                        .iter()
                        .enumerate()
                        .map(|(index, probability)| index as f32 * probability)
                        .sum();
                    let confidence = confidence::score_confidence(&probs);
                    json!({"type":"score", "score":rounded(score), "confidence":rounded(confidence), "legend":legend, "probabilities":values})
                }
            };
            answers.insert(question.id.clone(), answer);
        }
        Ok(DecisionResponse {
            model: String::new(),
            answers,
            usage: Usage {
                input_tokens: encoded.ids.len(),
                output_tokens: 0,
            },
        })
    }

    pub(crate) fn answer_needs_singleton_confirmation(answer: &Value) -> bool {
        const BF16_RTOL: f64 = 1.6e-2;
        const BF16_ATOL: f64 = 1e-5;
        let object = match answer.as_object() {
            Some(object) => object,
            None => return false,
        };
        let mut probabilities: Vec<f64> =
            if let Some(probabilities) = object.get("probabilities").and_then(Value::as_object) {
                probabilities.values().filter_map(Value::as_f64).collect()
            } else if object.get("type").and_then(Value::as_str) == Some("noul") {
                match object.get("noul").and_then(Value::as_f64) {
                    Some(probability) => vec![probability, 1. - probability],
                    None => return false,
                }
            } else {
                return false;
            };
        if probabilities.len() < 2 {
            return false;
        }
        probabilities.sort_by(|left, right| right.total_cmp(left));
        let first = probabilities[0];
        let second = probabilities[1];

        let combined_close_bound = 2. * BF16_ATOL + BF16_RTOL * (first.abs() + second.abs());
        first - second <= combined_close_bound
    }

    pub(crate) fn response_needs_singleton_confirmation(response: &DecisionResponse) -> bool {
        response
            .answers
            .values()
            .any(Self::answer_needs_singleton_confirmation)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn choice_argmax_keeps_the_first_option_on_exact_ties() {
        assert_eq!(winning_option(&[0.5, 0.5]), 0);
        assert_eq!(winning_option(&[0.2, 0.4, 0.4]), 1);
        assert_eq!(winning_option(&[0.4, 0.2, 0.4]), 0);
        assert_eq!(winning_option(&[1.0]), 0);
        assert_eq!(winning_option(&[0.2, 0.4, 0.40001]), 2);
    }

    #[test]
    fn numeric_json_rendering_matches_provider_spelling() {
        let values: Value =
            serde_json::from_str("[0.00007249627201966407,0.0001,1e-5,1e15,1e16,-0.0]").unwrap();
        assert_eq!(
            JointProtocol::render(&values),
            "[7.249627201966407e-05,0.0001,1e-05,1000000000000000.0,1e+16,-0.0]"
        );
        assert_eq!(
            JointProtocol::render(&json!({"float": 1.0, "integer": 1})),
            r#"{"float":1.0,"integer":1}"#
        );
        let halfway: Value =
            serde_json::from_str("[753906732828695.2,-753906732828695.2,5e-324]").unwrap();
        assert_eq!(
            JointProtocol::render(&halfway),
            "[753906732828695.2,-753906732828695.2,5e-324]"
        );
    }

    #[test]
    fn instructions_render_all_json_values_like_provider() {
        for (instructions, expected) in [
            (
                json!({"z": ["é", {"b": 2, "a": 1}], "a": true}),
                r#"{"a":true,"z":["é",{"a":1,"b":2}]}"#,
            ),
            (json!(["first", "second"]), r#"["first","second"]"#),
            (json!(false), "false"),
            (json!(12), "12"),
            (json!("plain text"), "plain text"),
            (Value::Null, "field_id"),
            (json!(""), "field_id"),
        ] {
            let question = json!({"instructions": instructions});
            assert_eq!(
                JointProtocol::instruction(question.as_object().unwrap(), "field_id"),
                expected
            );
        }
        assert_eq!(
            JointProtocol::instruction(&Map::new(), "field_id"),
            "field_id"
        );
    }

    #[test]
    fn singleton_confirmation_uses_the_bf16_combined_close_bound() {
        assert!(JointProtocol::answer_needs_singleton_confirmation(&json!({
            "type": "choice",
            "choice": "option_2",
            "probabilities": {"option_0": 0.4285, "option_1": 0.0431, "option_2": 0.4348, "option_3": 0.0937}
        })));
        assert!(!JointProtocol::answer_needs_singleton_confirmation(
            &json!({
                "type": "choice",
                "choice": "a",
                "probabilities": {"a": 0.6, "b": 0.4}
            })
        ));
        assert!(JointProtocol::answer_needs_singleton_confirmation(
            &json!({"type": "noul", "noul": 0.503})
        ));
        assert!(!JointProtocol::answer_needs_singleton_confirmation(
            &json!({"type": "noul", "noul": 0.9})
        ));
    }
}

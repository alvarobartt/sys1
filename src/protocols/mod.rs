pub(crate) mod confidence;
pub(crate) mod joint;
mod schema_cache;

use crate::architectures::qwen35_vision as vision;
use serde_json::Value;
#[derive(Clone, Debug)]
pub struct Question {
    pub id: String,
    pub kind: usize,
    pub question_span: (usize, usize),
    pub option_spans: Vec<(usize, usize)>,
    pub option_ids: Vec<String>,
}

pub(crate) struct Encoded {
    pub(crate) ids: Vec<u32>,
    pub(crate) positions: Vec<[u32; 3]>,
    pub(crate) images: Option<vision::images::Images>,
    pub(crate) videos: Option<vision::video::Videos>,
    pub(crate) media_spans: Vec<(usize, usize, bool)>,
    pub(crate) questions: Vec<Question>,
    pub(crate) criteria: Vec<Value>,
}
pub(crate) mod inline;
pub(crate) mod ordinal;

pub(crate) struct Context {
    pub(crate) tokenizer: crate::tokenizer::Tokenizer,
    pub(crate) vision_config: crate::architectures::qwen35_text::VisionConfig,
    pub(crate) image_token_id: u32,
    pub(crate) video_token_id: u32,
    pub(crate) max_length: usize,
}
impl Context {
    pub(crate) fn prepare(
        &self,
        ids: Vec<u32>,
        questions: Vec<crate::protocols::Question>,
        criteria: Vec<serde_json::Value>,
        images: Option<vision::images::Images>,
        videos: Option<vision::video::Videos>,
    ) -> Result<crate::protocols::Encoded, crate::schema::ApiError> {
        let mut positions = Vec::with_capacity(ids.len());
        let mut media_spans = Vec::new();
        let mut current_position = 0u32;
        let mut cursor = 0usize;
        let mut image_index = 0usize;
        let mut video_index = 0usize;
        let video_frames: Vec<_> = videos
            .as_ref()
            .map(|videos| {
                videos
                    .grids
                    .iter()
                    .flat_map(|&[t, h, w]| (0..t).map(move |_| [1, h, w]))
                    .collect()
            })
            .unwrap_or_default();
        while cursor < ids.len() {
            if ids[cursor] == self.image_token_id || ids[cursor] == self.video_token_id {
                let is_video = ids[cursor] == self.video_token_id;
                let &[t, h, w] = if is_video {
                    let grid = video_frames
                        .get(video_index)
                        .ok_or_else(|| crate::schema::ApiError::new("too many video tokens"))?;
                    video_index += 1;
                    grid
                } else {
                    let grid = images
                        .as_ref()
                        .and_then(|images| images.grids.get(image_index))
                        .ok_or_else(|| crate::schema::ApiError::new("too many image tokens"))?;
                    image_index += 1;
                    grid
                };
                let merge = self.vision_config.spatial_merge_size;
                let count = t * h * w / merge.pow(2);
                let token_id = if is_video {
                    self.video_token_id
                } else {
                    self.image_token_id
                };
                if ids
                    .get(cursor..cursor + count)
                    .is_none_or(|ids| ids.iter().any(|id| *id != token_id))
                {
                    return Err(crate::schema::ApiError::new(
                        "media token count does not match visual grid",
                    ));
                }
                media_spans.push((cursor, count, is_video));
                for frame in 0..t {
                    for row in 0..h / merge {
                        for col in 0..w / merge {
                            let _ = frame;
                            positions.push([
                                current_position,
                                current_position + row as u32,
                                current_position + col as u32,
                            ]);
                        }
                    }
                }
                current_position += (h.max(w) / merge) as u32;
                cursor += count;
            } else {
                positions.push([current_position; 3]);
                current_position += 1;
                cursor += 1;
            }
        }
        Ok(crate::protocols::Encoded {
            ids,
            positions,
            images,
            videos,
            media_spans,
            questions,
            criteria,
        })
    }
}

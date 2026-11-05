use crate::protocols::Encoded;
use candle_core::{DType, Result, Tensor};
use candle_nn::{Linear, VarBuilder};

pub(crate) struct Readout(Linear);

impl Readout {
    pub(crate) fn load(input: usize, options: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self(Linear::new(vb.get((options, input), "weight")?, None)))
    }

    pub(crate) fn forward(
        &self,
        encoded: &[&Encoded],
        hidden: &Tensor,
    ) -> Result<Vec<Vec<Vec<f32>>>> {
        let length: usize = encoded.iter().map(|item| item.ids.len()).sum();
        let mut offset = 0;
        let mut markers = Vec::new();
        for item in encoded {
            for question in &item.questions {
                if question.question_span.0 >= item.ids.len() {
                    candle_core::bail!("readout marker exceeds its sequence")
                }
                markers.push((offset + question.question_span.0) as u32);
            }
            offset += item.ids.len();
        }
        let count = markers.len();
        let hidden = hidden.reshape((length, ()))?;
        let markers = Tensor::from_vec(markers, count, hidden.device())?;
        let selected = hidden.index_select(&markers, 0)?;
        let logits = crate::kernels::linear::forward(&selected, &self.0)?
            .to_dtype(DType::F32)?
            .to_vec2::<f32>()?;
        let mut rows = logits.into_iter();
        encoded
            .iter()
            .map(|item| {
                item.questions
                    .iter()
                    .map(|question| {
                        let mut row = rows.next().ok_or_else(|| {
                            candle_core::Error::Msg("missing readout output".into())
                        })?;
                        if question.option_ids.len() > row.len() {
                            candle_core::bail!("decision options exceed readout width")
                        }
                        row.truncate(question.option_ids.len());
                        Ok(row)
                    })
                    .collect()
            })
            .collect()
    }
}

use candle_core::{Layout, Result, Storage, Tensor};
use rayon::prelude::*;

pub(crate) fn delta_rule(
    query: &Tensor,
    key: &Tensor,
    value: &Tensor,
    gate: &Tensor,
    beta: &Tensor,
) -> Result<Tensor> {
    let (batch, heads, length, key_dim) = query.dims4()?;
    let (value_batch, value_heads, value_length, value_dim) = value.dims4()?;
    if key_dim == 0
        || value_dim == 0
        || heads == 0
        || key.dims() != query.dims()
        || (batch, heads, length) != (value_batch, value_heads, value_length)
        || gate.dims() != [batch, heads, length]
        || beta.dims() != gate.dims()
    {
        candle_core::bail!("Invalid CPU delta-rule shape")
    }
    key_dim
        .checked_mul(value_dim)
        .ok_or_else(|| candle_core::Error::Msg("CPU delta-rule state size overflow".into()))?;
    let key_rows = length
        .checked_mul(key_dim)
        .ok_or_else(|| candle_core::Error::Msg("CPU delta-rule key size overflow".into()))?;
    let value_rows = length
        .checked_mul(value_dim)
        .ok_or_else(|| candle_core::Error::Msg("CPU delta-rule value size overflow".into()))?;
    let query = query.contiguous()?;
    let key = key.contiguous()?;
    let value = value.contiguous()?;
    let gate = gate.contiguous()?;
    let beta = beta.contiguous()?;
    let (query_storage, query_layout) = query.storage_and_layout();
    let (key_storage, key_layout) = key.storage_and_layout();
    let (value_storage, value_layout) = value.storage_and_layout();
    let (gate_storage, gate_layout) = gate.storage_and_layout();
    let (beta_storage, beta_layout) = beta.storage_and_layout();
    let query_values = f32_slice(&query_storage, query_layout)?;
    let key_values = f32_slice(&key_storage, key_layout)?;
    let value_values = f32_slice(&value_storage, value_layout)?;
    let gate_values = f32_slice(&gate_storage, gate_layout)?;
    let beta_values = f32_slice(&beta_storage, beta_layout)?;
    let mut output = vec![0f32; value_values.len()];
    if length != 0 {
        output
            .par_chunks_mut(value_rows)
            .enumerate()
            .for_each(|(head, output)| {
                let keys = head * key_rows;
                let values = head * value_rows;
                let gates = head * length;
                delta_rule_head(
                    &query_values[keys..keys + key_rows],
                    &key_values[keys..keys + key_rows],
                    &value_values[values..values + value_rows],
                    &gate_values[gates..gates + length],
                    &beta_values[gates..gates + length],
                    output,
                );
            });
    }
    Tensor::from_vec(output, (batch, heads, length, value_dim), query.device())?.transpose(1, 2)
}

pub(super) fn f32_slice<'a>(storage: &'a Storage, layout: &Layout) -> Result<&'a [f32]> {
    let Storage::Cpu(storage) = storage else {
        candle_core::bail!("Expected CPU tensor storage")
    };
    let (start, end) = layout
        .contiguous_offsets()
        .ok_or_else(|| candle_core::Error::Msg("Expected contiguous CPU tensor".into()))?;
    Ok(&storage.as_slice::<f32>()?[start..end])
}

fn delta_rule_head(
    query: &[f32],
    key: &[f32],
    value: &[f32],
    gate: &[f32],
    beta: &[f32],
    output: &mut [f32],
) {
    let key_dim = query.len() / gate.len();
    let value_dim = value.len() / gate.len();
    let mut state = vec![0f64; key_dim * value_dim];
    let mut delta = vec![0f64; value_dim];
    let mut result = vec![0f64; value_dim];
    for (token, output) in output.chunks_exact_mut(value_dim).enumerate() {
        let query = &query[token * key_dim..(token + 1) * key_dim];
        let key = &key[token * key_dim..(token + 1) * key_dim];
        let value = &value[token * value_dim..(token + 1) * value_dim];
        let decay = f64::from(gate[token]).exp();
        delta.fill(0.);
        for (row, key) in state.chunks_exact_mut(value_dim).zip(key) {
            let key = f64::from(*key);
            for (state, predicted) in row.iter_mut().zip(&mut delta) {
                *state *= decay;
                *predicted += key * *state;
            }
        }
        for (delta, value) in delta.iter_mut().zip(value) {
            *delta = (f64::from(*value) - *delta) * f64::from(beta[token]);
        }
        result.fill(0.);
        for ((row, key), query) in state.chunks_exact_mut(value_dim).zip(key).zip(query) {
            let key = f64::from(*key);
            let query = f64::from(*query);
            for ((state, delta), result) in row.iter_mut().zip(&delta).zip(&mut result) {
                *state += key * delta;
                *result += query * *state;
            }
        }
        for (output, result) in output.iter_mut().zip(&result) {
            *output = *result as f32;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};

    #[test]
    fn cpu_recurrence_checks_shapes_types_and_empty_sequences() -> Result<()> {
        let device = Device::Cpu;
        let q = Tensor::zeros((2, 3, 0, 8), DType::F32, &device)?;
        let v = Tensor::zeros((2, 3, 0, 5), DType::F32, &device)?;
        let gate = Tensor::zeros((2, 3, 0), DType::F32, &device)?;
        assert_eq!(delta_rule(&q, &q, &v, &gate, &gate)?.dims(), [2, 0, 3, 5]);
        let invalid = Tensor::zeros((2, 2, 0), DType::F32, &device)?;
        assert!(delta_rule(&q, &q, &v, &invalid, &gate).is_err());
        assert!(delta_rule(&q.to_dtype(DType::F64)?, &q, &v, &gate, &gate).is_err());
        Ok(())
    }
}

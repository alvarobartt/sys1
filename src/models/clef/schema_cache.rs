use super::clef_head::Question;
use crate::schema::ApiError;
use serde_json::{Map, Value};
use std::{
    collections::{HashMap, VecDeque},
    mem::size_of,
    sync::{Arc, Mutex},
};

pub(super) struct CompiledSchema {
    pub ids: Vec<u32>,
    pub questions: Vec<Question>,
}

impl CompiledSchema {
    fn accounted_bytes(&self, key_bytes: usize) -> usize {
        let mut bytes = key_bytes
            .saturating_add(size_of::<Self>())
            .saturating_add(self.ids.capacity().saturating_mul(size_of::<u32>()))
            .saturating_add(
                self.questions
                    .capacity()
                    .saturating_mul(size_of::<Question>()),
            );
        for question in &self.questions {
            bytes = bytes
                .saturating_add(question.id.capacity())
                .saturating_add(question.option_spans.capacity().saturating_mul(size_of::<(
                    usize,
                    usize,
                )>(
                )))
                .saturating_add(
                    question
                        .option_ids
                        .capacity()
                        .saturating_mul(size_of::<String>()),
                );
            for id in &question.option_ids {
                bytes = bytes.saturating_add(id.capacity());
            }
        }
        bytes
    }
}

struct Entry {
    value: Arc<CompiledSchema>,
    bytes: usize,
}

#[derive(Default)]
struct State {
    entries: HashMap<Arc<str>, Entry>,
    order: VecDeque<Arc<str>>,
    bytes: usize,
}

pub(super) struct SchemaCache {
    state: Mutex<State>,
    maximum_entries: usize,
    maximum_bytes: usize,
}

impl Default for SchemaCache {
    fn default() -> Self {
        Self::with_limits(1024, 16 << 20)
    }
}

impl SchemaCache {
    fn with_limits(maximum_entries: usize, maximum_bytes: usize) -> Self {
        Self {
            state: Mutex::new(State::default()),
            maximum_entries,
            maximum_bytes,
        }
    }

    pub fn get_or_build(
        &self,
        questions: &Map<String, Value>,
        build: impl FnOnce() -> Result<CompiledSchema, ApiError>,
    ) -> Result<Arc<CompiledSchema>, ApiError> {
        let key =
            serde_json::to_string(questions).map_err(|error| ApiError::new(error.to_string()))?;
        if let Ok(state) = self.state.lock()
            && let Some(entry) = state.entries.get(key.as_str())
        {
            return Ok(entry.value.clone());
        }

        let value = Arc::new(build()?);
        let bytes = value.accounted_bytes(key.len());
        if self.maximum_entries == 0 || bytes > self.maximum_bytes {
            return Ok(value);
        }
        if let Ok(mut state) = self.state.lock() {
            if let Some(entry) = state.entries.get(key.as_str()) {
                return Ok(entry.value.clone());
            }
            while state.entries.len() >= self.maximum_entries
                || state.bytes.saturating_add(bytes) > self.maximum_bytes
            {
                let Some(oldest) = state.order.pop_front() else {
                    break;
                };
                if let Some(entry) = state.entries.remove(oldest.as_ref()) {
                    state.bytes = state.bytes.saturating_sub(entry.bytes);
                }
            }
            let key: Arc<str> = Arc::from(key);
            state.bytes = state.bytes.saturating_add(bytes);
            state.order.push_back(key.clone());
            state.entries.insert(
                key,
                Entry {
                    value: value.clone(),
                    bytes,
                },
            );
        }
        Ok(value)
    }

    #[cfg(test)]
    pub fn resident(&self) -> (usize, usize) {
        self.state
            .lock()
            .map(|state| (state.entries.len(), state.bytes))
            .unwrap_or_default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn questions(id: &str) -> Map<String, Value> {
        json!({id: {"type":"noul", "instructions":id}})
            .as_object()
            .unwrap()
            .clone()
    }

    fn schema(id: u32) -> Result<CompiledSchema, ApiError> {
        Ok(CompiledSchema {
            ids: vec![id],
            questions: vec![],
        })
    }

    #[test]
    fn exact_keys_hits_and_fifo_eviction_preserve_outstanding_values() {
        let cache = SchemaCache::with_limits(2, 4096);
        let original = cache.get_or_build(&questions("one"), || schema(1)).unwrap();
        cache.get_or_build(&questions("two"), || schema(2)).unwrap();
        let hit = cache
            .get_or_build(&questions("one"), || panic!("cache hit rebuilt schema"))
            .unwrap();
        assert!(Arc::ptr_eq(&original, &hit));
        cache
            .get_or_build(&questions("three"), || schema(3))
            .unwrap();
        let replacement = cache
            .get_or_build(&questions("one"), || schema(11))
            .unwrap();
        assert_eq!(original.ids, [1]);
        assert_eq!(replacement.ids, [11]);
        assert_eq!(cache.resident().0, 2);
    }

    #[test]
    fn byte_budget_and_failed_compilation_do_not_leave_entries() {
        let cache = SchemaCache::with_limits(4, 1);
        cache
            .get_or_build(&questions("large"), || schema(7))
            .unwrap();
        assert_eq!(cache.resident(), (0, 0));
        assert!(
            cache
                .get_or_build(&questions("bad"), || Err(ApiError::new("invalid schema")))
                .is_err()
        );
        assert_eq!(cache.resident(), (0, 0));
    }

    #[test]
    fn different_values_and_question_order_have_distinct_keys() {
        let cache = SchemaCache::default();
        let first = json!({"a":{"type":"noul"},"b":{"type":"noul"}})
            .as_object()
            .unwrap()
            .clone();
        let second = json!({"b":{"type":"noul"},"a":{"type":"noul"}})
            .as_object()
            .unwrap()
            .clone();
        let third = json!({"a":{"type":"noul","instructions":"changed"},"b":{"type":"noul"}})
            .as_object()
            .unwrap()
            .clone();
        for (index, key) in [first, second, third].into_iter().enumerate() {
            assert_eq!(
                cache
                    .get_or_build(&key, || schema(index as u32))
                    .unwrap()
                    .ids,
                [index as u32]
            );
        }
        assert_eq!(cache.resident().0, 3);
    }
}

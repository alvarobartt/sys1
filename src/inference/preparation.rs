pub(super) fn workers(requests: usize, available: usize) -> usize {
    available.clamp(1, 8).min(requests.div_ceil(32).max(1))
}

pub(super) fn map<T: Sync, R: Send, F: Fn(&T) -> R + Sync>(
    input: &[T],
    worker_limit: usize,
    encode: F,
) -> Vec<R> {
    if input.is_empty() {
        return Vec::new();
    }
    let count = worker_limit.clamp(1, 8).min(input.len());
    if count == 1 {
        return input.iter().map(encode).collect();
    }
    std::thread::scope(|scope| {
        let encode = &encode;
        let handles = input
            .chunks(input.len().div_ceil(count))
            .map(|chunk| scope.spawn(move || chunk.iter().map(encode).collect::<Vec<_>>()))
            .collect::<Vec<_>>();
        let mut result = Vec::with_capacity(input.len());
        for handle in handles {
            match handle.join() {
                Ok(mut part) => result.append(&mut part),
                Err(panic) => std::panic::resume_unwind(panic),
            }
        }
        result
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn preserves_order_and_individual_errors() {
        let input = (0..513).collect::<Vec<_>>();
        let encode = |value: &usize| {
            if value.is_multiple_of(19) {
                Err(*value)
            } else {
                Ok(value * 7)
            }
        };
        for count in [0, 1, 2, 3, 8, 1000] {
            assert_eq!(
                map(&input, count, encode),
                input.iter().map(encode).collect::<Vec<_>>()
            );
        }
    }

    #[test]
    fn bounds_workers_without_empty_input_threads() {
        for requests in 0..1025 {
            for available in [0, 1, 2, 26, 256] {
                let count = workers(requests, available);
                assert!((1..=8).contains(&count));
                assert!(count <= available.max(1));
                assert!(count <= requests.div_ceil(32).max(1));
            }
        }
        assert_eq!(
            map::<usize, usize, _>(&[], 1000, |_| panic!("empty callback")),
            Vec::<usize>::new()
        );
    }

    #[test]
    fn never_creates_more_than_eight_threads() {
        let ids = std::sync::Mutex::new(std::collections::HashSet::new());
        let input = (0..1024).collect::<Vec<_>>();
        let result = map(&input, 1000, |value| {
            ids.lock().unwrap().insert(std::thread::current().id());
            *value
        });
        assert_eq!(result, input);
        assert!(ids.into_inner().unwrap().len() <= 8);
    }

    #[test]
    fn propagates_worker_panic_instead_of_partial_results() {
        let result = std::panic::catch_unwind(|| {
            map(&(0..129).collect::<Vec<_>>(), 8, |value| {
                if *value == 65 {
                    panic!("encoder failure payload");
                }
                *value
            })
        });
        let panic = result.expect_err("a worker panic must not produce partial success");
        assert_eq!(
            panic.downcast_ref::<&str>(),
            Some(&"encoder failure payload")
        );
    }
}

---
title: Batching
---

# Batching

`sys1` runs one inference worker per process. The HTTP router validates a
request and admits it to a bounded queue. The worker takes one request, adds
eligible queued requests to a batch, runs one model call, and returns each
response. A full queue returns HTTP 503.

Batch selection is bounded by **request count** and **question count**, not
estimated token count. Tokenization, deduplication, padding, and attention
masking happen after the batch forms.

## Default behavior

The optional collection window is disabled by default:

```bash
sys1 --model-id convaiinnovations/laya --batch-wait-ms 0
```

With `--batch-wait-ms 0`, the worker does not wait for new arrivals. It can
still batch requests that are already queued when it becomes free, especially
under concurrent load. The default limits are 32 requests per batch and 128
questions across the batch.

Setting `--batch-wait-ms` above zero starts a collection window when the worker
takes the first request. The worker waits until that deadline, unless the batch
reaches its request limit or the next request would exceed its question limit.
This is one window per batch, not one wait added for every request.

## Latency and throughput

A collection window can form larger batches and amortize model-call overhead
when requests arrive close together. Whether it improves throughput depends on
the model, backend, input sizes, and concurrency. Measure your workload before
raising it.

At low concurrency, a positive window often adds nearly its full duration to
each request's latency because there is no other request to collect. Requests
can also spend time waiting behind a running batch in the queue. Start with
the default `0 ms` when response latency matters more than batch size.

`--request-timeout-ms` is a separate deadline. It starts after a request is
admitted and covers queue time, collection time, and inference. Its default is
30,000 ms; `0` disables it. Raising this timeout does **not** make batches
form faster or reduce latency. A request may time out while its batch is still
being processed.

## Controls

| Flag | Default | Effect |
| --- | ---: | --- |
| `--batch-wait-ms` | `0` | Maximum time to collect more requests for a batch. |
| `--max-batch-size` | `32` | Maximum requests in one batch. |
| `--max-batch-questions` | `128` | Maximum total questions in one batch. |
| `--max-questions-per-request` | `64` | Admission limit for a single request; cannot exceed the batch question limit. |
| `--max-queue-size` | `256` | Pending request capacity before HTTP 503. |
| `--request-timeout-ms` | `30000` | Per-request response deadline before HTTP 504; `0` disables it. |

For a throughput experiment, set a small collection window and a bounded
batch. These values are examples, not recommended production defaults:

```bash
sys1 --model-id convaiinnovations/laya \
  --batch-wait-ms 2 \
  --max-batch-size 16 \
  --max-batch-questions 64
```

Compare this with the defaults under the same concurrency and input mix. Use
[`/metrics`](../api.md#metrics) to inspect `sys1_batch_size`,
`sys1_batch_questions`, `sys1_requests_queued`,
`sys1_requests_timed_out_total`, and `sys1_inference_duration_seconds`.
Increase batch limits only if memory headroom and observed throughput justify
it. Size the request timeout for the full queue-plus-inference path, not just
the collection window. See the [CLI](../cli.md#batching-and-limits) for every
flag and [API](../api.md#metrics) for the full metric list.

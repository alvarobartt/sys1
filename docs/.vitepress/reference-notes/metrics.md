Counters grow until the process restarts, gauges report current values, and
histograms record batch size, question count, and inference duration. Metrics
are reset after model warmup.

```bash
curl http://localhost:3000/metrics
```

```text
# HELP sys1_requests_accepted_total Inference requests accepted into the processing queue.
# TYPE sys1_requests_accepted_total counter
sys1_requests_accepted_total 12
# HELP sys1_requests_queued Inference requests currently waiting in the queue.
# TYPE sys1_requests_queued gauge
sys1_requests_queued 0
```

The metric names and descriptions come from the Prometheus response. Each
histogram exposes `_bucket`, `_sum`, and `_count` series. A timeout can occur
after admission, so timed-out requests are not a subset of rejected requests.
The endpoint is unauthenticated; restrict access at the reverse proxy if
needed. See [Batching](advanced/batching.md) for tuning and the
[CLI](cli.md#batching-and-limits) for limits.

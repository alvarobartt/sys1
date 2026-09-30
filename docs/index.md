---
title: sys1 documentation
description: Blazing fast, self-hosted structured decisions for open-weight models with a TypeSafe AI compatible API, written in Rust.
---

<img
  class="overview-banner"
  src="https://github.com/user-attachments/assets/8b2774a7-6439-422e-bbd3-3fdd85117c5d"
  alt="sys1"
/>

# Structured decisions. Your infrastructure.

`sys1` is a Rust inference server for low-latency, high-throughput structured
decisions with open-weight models. It exposes a TypeSafe AI compatible
HTTP API for Choice, Score, and Noul decisions.

It loads one supported checkpoint per process from Hugging Face or a local
path and runs on CPU, Metal, or CUDA. The same API works on every backend.

## Built for production

`sys1` is built for production:

- Dynamic batching groups queued requests within request and question limits.
  See [Advanced batching](advanced/batching.md) for the collection window and
  [CLI](cli.md#batching-and-limits) for the controls.
- Model warmup completes before the server accepts traffic.
- Queue capacity, batch size, request size, and response timeout are bounded.
- `/health` reports worker readiness; [`/metrics`](api.md#metrics) exposes
  Prometheus counters, gauges, and histograms for queue depth, requests, batches,
  and inference duration.
- Request logs and graceful shutdown support deployment operations.

Install the default CPU build:

```bash
cargo install sys1
```

Start [Laya](models/laya.md) with F32 precision:

```bash
sys1 --model-id convaiinnovations/laya --dtype f32
```

Send a structured decision request:

```bash
curl http://localhost:3000/v1/systemone \
  -H 'content-type: application/json' \
  -d '{
    "state": {"message": "I was charged twice. Please refund me today."},
    "questions": {
      "route": {
        "type": "choice",
        "instructions": "Where should this ticket go?",
        "criteria": {"billing": "payments and refunds", "support": "product help"}
      }
    }
  }'
```

See [Get started](getting-started/index.md) for readiness checks and backend
installation, [Advanced batching](advanced/batching.md), [CLI](cli.md) for
options and defaults, and [Models](models/index.md) for supported checkpoints.

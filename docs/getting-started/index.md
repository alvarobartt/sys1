---
title: Getting started
---

# Getting started

Install the default CPU build:

```bash
cargo install sys1
```

Start [Laya](../models/laya.md):

```bash
sys1 --model-id convaiinnovations/laya --dtype auto
```

Wait for [readiness](../api.md#health), then send a
[System One request](../api.md#system-one):

```bash
curl --fail http://localhost:3000/health

curl http://localhost:3000/v1/systemone \
  -H 'content-type: application/json' \
  -d '{
    "state": {"message": "Refund invoice 4411."},
    "questions": {
      "route": {
        "type": "choice",
        "instructions": "Route the request.",
        "criteria": {
          "billing": "payments and refunds",
          "support": "product help"
        }
      }
    }
  }'
```

Choose a deployment path: [CPU](cpu.md), [Metal](metal.md),
[CUDA](cuda.md), or [Docker](docker.md). The API is identical across backends.
See the [CLI](../cli.md) for flags and [Advanced batching](../advanced/batching.md)
for queue and latency tuning.

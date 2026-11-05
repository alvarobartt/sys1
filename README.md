<div align="center">
  <img
    src="https://github.com/user-attachments/assets/8b2774a7-6439-422e-bbd3-3fdd85117c5d"
    alt="sys1"
    width="1200"
  />
  <br/>
  <em>
    Blazing fast structured decisions for System One open-models self-hosted with a TypeSafe AI compatible API, written in Rust.
  </em>
</div>

## Features

- `tokio`, `axum` and `serde`, the usual suspects
- System One compatible API Spec
- `candle` with [`tokenizers` release candidate](https://huggingface.co/blog/tokenizers-v1)!
- Dynamic, token-based batching
- SDPA on CPU, Metal, and CUDA
- Automatic Flash Attention on supported CUDA devices; `--enforce-eager` selects eager attention
- Blazing fast inference for ModernBERT and Qwen3.5
- Multimodal decisions from text, images, and video when supported by the model

## Get started

Rust and Cargo 1.98.1, mandatory. `ffmpeg` and `ffprobe` 6.1.1 or higher for video inputs.

If you run on CUDA you also need `nvcc`. And if you run on Metal, you also need `xcode` and `xcodebuild -downloadComponent MetalToolchain`.

```bash
cargo install sys1 --features cpu
# cargo install sys1 --no-default-features --features metal
# cargo install sys1 --no-default-features --features cuda
```

Then run it with any of the supported models (more coming soon!).

- [`convaiinnovations/laya`](https://huggingface.co/convaiinnovations/laya) for English text, guardrails, email triage
- [`convaiinnovations/laya-multilingual`](https://huggingface.co/convaiinnovations/laya-multilingual) for 100+ languages, ~2.2x faster
- [`convaiinnovations/laya-typed-decisions`](https://huggingface.co/convaiinnovations/laya-typed-decisions) for typed-decisions workflows
- [`Cloudflare/clef`](https://huggingface.co/Cloudflare/clef) for text, image, and video decisions
- [`Cloudflare/clef-flash`](https://huggingface.co/Cloudflare/clef-flash) for text, image, and video decisions (smaller and faster than the 27B variant)

```bash
sys1 --model-id convaiinnovations/laya --dtype auto
```

And, just query your Jev-compatible API at `/v1/systemone` (or `/v1/decide`).

```bash
curl http://localhost:3000/v1/systemone \
    -H "Content-Type: application/json" \
    -d '{
      "model": "convaiinnovations/laya",
      "state": {
        "message": "I was charged twice for invoice 4411. Please refund me today."
      },
      "questions": {
        "route": {
          "type": "choice",
          "instructions": "Where should this ticket go?",
          "criteria": {
            "billing": "payments, refunds, invoices",
            "bug": "the product is broken",
            "account": "login or access"
          }
        },
        "urgency": {
          "type": "score",
          "instructions": "How urgent is this message?",
          "criteria": [
            "routine, no rush",
            "today",
            "urgent",
            "critical, about to churn"
          ]
        },
        "escalate": {
          "type": "noul",
          "instructions": "Escalate to a human immediately?"
        }
      }
    }'
```

Or, install the [TypeSafe AI SDK for Python](https://github.com/typesafe-ai/typesafe-sdk-python), or the [TypeSafe AI SDK for Javascript / Typescript](https://github.com/typesafe-ai/typesafe-sdk-js).

```python
from typesafe_sdk import Choice, TypeSafeClient

client = TypeSafeClient(base_url="http://localhost:3000", api_key="-")

response = client.system_one(
    model="convaiinnovations/laya",
    state={"message": "I was charged twice for invoice 4411."},
    questions={
        "route": Choice(
            instructions="Where should this ticket go?",
            criteria={"billing": None, "bug": None, "account": None},
        ),
    },
)
print(response.choices["route"].choice)
```

## References

- [TypeSafe AI API](https://api.typesafe.ai)
- [TypeSafe AI Python SDK](https://github.com/typesafe-ai/typesafe-sdk-python)
- [Laya: Multilingual, non-autoregressive System 1 decision engine](https://github.com/NandhaKishorM/laya)
- [Introducing Clef: our open-source decision models, and new RL fine-tuning platform](https://blog.cloudflare.com/clef-decision-models/)

### CUDA build assets and benchmarks

CUDA builds fetch checksum-verified device images from the pinned [sys1-artifacts](https://huggingface.co/alvarobartt/sys1-artifacts/tree/fe16e9b22cd05ebb799f242df9ac68470d3e3d4c) revision and embed them in the executable. Inference needs no Python or asset download. External reference-fixture tests are ignored by default; serving and the end-to-end benchmarks do not need them. Third-party notices remain in `licenses/`.

The [benchmark dataset](https://huggingface.co/datasets/alvarobartt/sys1-benchmarks/tree/a3c3616418ef2d20d2b39bfe8d0c2c905305df5d) provides canonical and stress workloads without storing data in Git:

```sh
uv run benchmark.py throughput --dataset-profile clef-flash --concurrency 1 32 128 512 --api-url http://127.0.0.1:3000 --output results.json
uv run --with huggingface-hub==1.33.0 leaderboard.py --base-url http://127.0.0.1:3000 --out results/c1
uv run --with huggingface-hub==1.33.0 leaderboard.py --base-url http://127.0.0.1:3000 --concurrency 512 --stress-continuous --stress-case canonical full-context-64-questions full-context-mixed-types --stress-windows 1 --stress-seconds 300 --stress-warmup-requests 1024 --out results/c512
uv run --with httpx==0.28.1 demo.py --host 127.0.0.1 --port 3000
```

Use the same checkpoint revision, precision, workload, client and hardware for each engine. C512 records queueing latency separately from throughput and drains outstanding requests. The dataset retains the historical client's source and exact input hashes. These cohorts reconstruct supported public requests; they are not the unpublished Decision Index sample.

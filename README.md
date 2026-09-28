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

- [TypeSafe AI compatible API](https://docs.typesafe.ai/api)
- `candle` with [`tokenizers` release candidate](https://huggingface.co/blog/tokenizers-v1)!
- Token-based, dynamic batching
- SDPA on CPU, Metal, and CUDA
- Flash Attention on Ampere, Ada Lovelace, and Hopper
- Blazing fast inference for ModernBERT and Qwen3.5
- Multimodal decisions from text, images, and video when supported by the model

## Get started

Rust and Cargo 1.98.1, mandatory. `ffmpeg` and `ffprobe` 6.1.1 or higher for video inputs.

If you run on CUDA you also need `nvcc`. And if you run on Metal, you also need `xcode` and `xcodebuild -downloadComponent MetalToolchain`.

```bash
cargo install sys1 --features cpu
# cargo install sys1 --no-default-features --features metal
# cargo install sys1 --no-default-features --features cuda
# cargo install sys1 --no-default-features --features cuda,flash-attn-2 # Ampere, Ada Lovelace, or Hopper
# cargo install sys1 --no-default-features --features cuda,flash-attn-3 # Hopper
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

And, just query your TypeSafe AI compatible API at `/v1/systemone` (or `/v1/decide`).

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
- [TypeSafe AI System One](https://docs.typesafe.ai/concepts/system-one)
- [TypeSafe AI Python SDK](https://github.com/typesafe-ai/typesafe-sdk-python)
- [Laya: Multilingual, non-autoregressive System 1 decision engine](https://github.com/NandhaKishorM/laya)
- [Introducing Clef: our open-source decision models, and new RL fine-tuning platform](https://blog.cloudflare.com/clef-decision-models/)

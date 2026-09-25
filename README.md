<div align="center">
  <img
    src="https://github.com/user-attachments/assets/8b2774a7-6439-422e-bbd3-3fdd85117c5d"
    alt="sys1"
    width="1200"
  />
  <br/>
  <em>
    System One compatible API for open decision models, written in Rust.
  </em>
</div>

## Features

- `tokio`, `axum` and `serde`, the usual suspects
- System One compatible API Spec
- `candle` with [`tokenizers` release candidate](https://huggingface.co/blog/tokenizers-v1)!
- Dynamic, token-based batching
- SDPA on CPU, Metal, and CUDA
- Flash Attention on Ampere, Ada Lovelace, and Hopper

## Get started

Install it with support for CPU, Metal or CUDA.

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

## References

- [TypeSafe AI API](https://api.typesafe.ai)
- [TypeSafe AI Python SDK](https://github.com/typesafe-ai/typesafe-sdk-python)
- [Laya: Multilingual, non-autoregressive System 1 decision engine](https://github.com/NandhaKishorM/laya)

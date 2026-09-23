<div align="center">
  <img
    src="https://github.com/user-attachments/assets/8b2774a7-6439-422e-bbd3-3fdd85117c5d"
    alt="System One"
    width="1200"
  />
  <br/>
  <em>
    System One compatible API for open decision models, e.g.
    <a href="https://huggingface.co/convaiinnovations/laya">Laya</a>,
    written in Rust.
  </em>
</div>

## Features

- `tokio`, `axum` and `serde`, the usual suspects
- System One compatible API Spec
- `candle` with [`tokenizers` release candidate](https://huggingface.co/blog/tokenizers-v1)!
- Dynamic, token-based batching
- Support for ModernBert with Laya custom decision heads
- CPU, CUDA and Metal (MPS) supported
- Up to ~14ms per query on NVIDIA RTX Pro 6000

## Get started

Install it with support for CPU, Metal or CUDA.

```bash
cargo install sys1 --features cpu
# cargo install sys1 --no-default-features --features metal
# cargo install sys1 --no-default-features --features cuda
```

Then run it with [`convaiinnovations/laya`](https://huggingface.co/convaiinnovations/laya) (more models coming soon!).

```bash
sys1 --model-id convaiinnovations/laya --dtype auto
```

And, just query your Jev-compatible API at `/v1/systemone` (or `/v1/decide`).

```bash
curl http://localhost:3000/v1/systemone \
    -H "Content-Type: application/json" \
    -d '{
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

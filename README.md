# sys1

System One compatible API for open decision models e.g., [Laya](https://huggingface.co/convaiinnovations/laya), written in Rust.

- `tokio`, `axum` and `serde`, the usual suspects
- `candle` with [`tokenizers` release candidate](https://huggingface.co/blog/tokenizers-v1)!
- Dynamic, token-based batching
- Support for ModernBert with Laya custom decision heads
- CPU, CUDA and Metal (MPS) supported

## Get started!

Install it for CPU with the `--features cpu`, or Metal with `--features metal`, or CUDA with `--features cuda`.

```bash
cargo install sys1 --features cpu
sys1 --model-id convaiinnovations/laya
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

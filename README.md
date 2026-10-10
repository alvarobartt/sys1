<div align="center">
  <img
    src="https://github.com/user-attachments/assets/8b2774a7-6439-422e-bbd3-3fdd85117c5d"
    alt="sys1"
    width="1200"
  />
  <br/>
  <em>
    Blazing fast, self-hosted structured decisions for open models with a TypeSafe AI compatible API, written in Rust.
  </em>
</div>

## Features

- [TypeSafe AI compatible API](https://docs.typesafe.ai/api)
- Backed by [`candle`](https://github.com/huggingface/candle) and [`tokenizers`](https://github.com/huggingface/tokenizers)
- Dynamic, token-based batching bounded by requests
- Compatible with CPU
- Compatible with Metal (macOS M-series)
- Compatible with CUDA (NVIDIA Turing, Ampere, Ada, Hopper, and Blackwell)
- Support for F32, F16, and BF16
- Optimized ModernBERT and Qwen3.5 architectures

## Get started

- Rust and Cargo 1.98.1
- NVIDIA CUDA Compiler (`nvcc`) on CUDA only i.e., `--features cuda`
- XCode and Metal Toolchain on Metal only i.e., `--features metal`
- (Optional) `ffmpeg` and `ffprobe` 6.1.1 or higher for video inputs

```bash
cargo install sys1 --features cpu
# cargo install sys1 --no-default-features --features metal
# cargo install sys1 --no-default-features --features cuda
# cargo install sys1 --no-default-features --features cuda,flash-attn-2 # Ampere, Ada, Hopper, or Blackwell
```

Then run it with any of the supported models (more coming soon!).

- [`convaiinnovations/laya`](https://huggingface.co/convaiinnovations/laya) for English text, guardrails, email triage
- [`convaiinnovations/laya-multilingual`](https://huggingface.co/convaiinnovations/laya-multilingual) for 100+ languages, ~2.2x faster
- [`convaiinnovations/laya-typed-decisions`](https://huggingface.co/convaiinnovations/laya-typed-decisions) for typed-decisions workflows
- [NEW!] [`Cloudflare/clef`](https://huggingface.co/Cloudflare/clef) for text, image, and video decisions
- [NEW!] [`Cloudflare/clef-flash`](https://huggingface.co/Cloudflare/clef-flash) for text, image, and video decisions (smaller and faster than the 27B variant)

```bash
sys1 --model-id convaiinnovations/laya --dtype auto
```

Or, alternatively run with Docker instead.

```bash
docker run -p 3000:3000 ghcr.io/alvarobartt/sys1:0.1.0-cpu --model-id convaiinnovations/laya --dtype auto
# docker run --gpus all -p 3000:3000 ghcr.io/alvarobartt/sys1:0.1.0-turing --model-id convaiinnovations/laya --dtype f16
# docker run --gpus all -p 3000:3000 ghcr.io/alvarobartt/sys1:0.1.0-ampere --model-id convaiinnovations/laya --dtype f16
# docker run --gpus all -p 3000:3000 ghcr.io/alvarobartt/sys1:0.1.0-ada-lovelace --model-id convaiinnovations/laya --dtype f16
# docker run --gpus all -p 3000:3000 ghcr.io/alvarobartt/sys1:0.1.0-hopper --model-id convaiinnovations/laya --dtype f16
# docker run --gpus all -p 3000:3000 ghcr.io/alvarobartt/sys1:0.1.0-blackwell --model-id convaiinnovations/laya --dtype f16
```

Then query the System One compatible endpoint at `/v1/systemone` (or `/v1/decide`).

```bash
curl http://localhost:3000/v1/systemone \
    -H "Content-Type: application/json" \
    -d '{
      "model": "convaiinnovations/laya",
      "state": "I was charged twice for invoice 4411. Please refund me today.",
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
    state="I was charged twice for invoice 4411.",
    questions={
        "route": Choice(
            instructions="Where should this ticket go?",
            criteria={"billing": None, "bug": None, "account": None},
        ),
    },
)
print(response.choices["route"].choice)
```

---

Alternatively, if you were to deploy [`Cloudflare/clef-flash`](https://huggingface.co/Cloudflare/clef-flash) or any other model with vision capabilities as:

```bash
sys1 --model-id Cloudflare/clef-flash --dtype auto
```

Then you can send requests to `/v1/systemone` with `images` or `videos` (it accepts public HTTP(s) URLs, raw base64 strings, base64 data URLs, or embedded objects).

```bash
curl http://localhost:3000/v1/systemone \
    -H "Content-Type: application/json" \
    -d '{
      "model": "Cloudflare/clef-flash",
      "state": "Classify the attached document.",
      "images": [
        "https://huggingface.co/datasets/hf-internal-testing/fixtures_ocr/resolve/main/SROIE-receipt.jpeg"
      ],
      "questions": {
        "document_type": {
          "type": "choice",
          "instructions": "What kind of document is this?",
          "criteria": {
            "receipt": "proof of a completed purchase",
            "invoice": "a request for payment",
            "other": "neither a receipt nor an invoice"
          }
        }
      }
    }'
```

And, note `images` or `videos` won't be accepted via the TypeSafe AI SDKs as it's not officially supported in the TypeSafe AI API Spec ([yet](https://x.com/CompleteSkeptic/status/2108629012474175685)).

## References

- [TypeSafe AI API](https://docs.typesafe.ai/api)
- [TypeSafe AI System One](https://docs.typesafe.ai/concepts/system-one)
- [TypeSafe AI Python SDK](https://github.com/typesafe-ai/typesafe-sdk-python)
- [Laya: Multilingual, non-autoregressive System 1 decision engine](https://github.com/NandhaKishorM/laya)
- [Introducing Clef: our open-source decision models, and new RL fine-tuning platform](https://blog.cloudflare.com/clef-decision-models/)

---
title: Models
---

# Models

Supported checkpoints use the same HTTP API and differ by encoder, language,
and specialization.

| Model | Encoder | Parameters | Languages | Status |
| --- | --- | ---: | --- | --- |
| [Laya](laya.md) | ModernBERT-large | 421M | English | Stable |
| [Laya Multilingual](laya-multilingual.md) | mmBERT-base | 322M | 100+ | Stable |
| [Laya Typed Decisions](laya-typed-decisions.md) | ModernBERT-large | 421M | English | Stable |

Start a model by passing its Hugging Face identifier:

```bash
sys1 --model-id convaiinnovations/laya
```

One server process hosts one checkpoint. Use `--served-model-name` to expose a
deployment-specific alias in responses and [`/v1/models`](../api.md#list-models).
The [Get started guides](../getting-started/index.md) cover installation.
See [Advanced batching](../advanced/batching.md) for queue and batch limits.

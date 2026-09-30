---
title: Laya
---

# Laya

<ModelPublisher model-id="convaiinnovations/laya" />

`convaiinnovations/laya` is the English ModernBERT checkpoint and the default
model loaded by `sys1`.

| Property | Value |
| --- | --- |
| Hugging Face ID | `convaiinnovations/laya` |
| Encoder | ModernBERT-large |
| Parameters | Approximately 421M |
| Primary use | English routing, guardrails, email triage |

```bash
sys1 --model-id convaiinnovations/laya --dtype auto
```

Use this checkpoint when inputs are primarily English and the domain matches
the model's evaluation. Measure decision quality on application data before
placing confidence thresholds in production.

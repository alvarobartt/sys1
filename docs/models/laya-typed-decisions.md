---
title: Laya Typed Decisions
---

# Laya Typed Decisions

<ModelPublisher model-id="convaiinnovations/laya-typed-decisions" />

`convaiinnovations/laya-typed-decisions` is specialized for the public typed
decision workflows.

| Property | Value |
| --- | --- |
| Hugging Face ID | `convaiinnovations/laya-typed-decisions` |
| Encoder | ModernBERT-large |
| Parameters | Approximately 421M |
| Primary use | Workflows resembling its typed-decisions training set |

```bash
sys1 --model-id convaiinnovations/laya-typed-decisions --dtype auto
```

Prefer this model when its specialization matches the decisions being served.
Treat performance gains as workload-specific and evaluate against the base model
on held-out application data.

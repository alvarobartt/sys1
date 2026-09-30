---
title: Laya Multilingual
---

# Laya Multilingual

<ModelPublisher model-id="convaiinnovations/laya-multilingual" />

`convaiinnovations/laya-multilingual` targets more than 100 languages and uses
the smaller mmBERT encoder.

| Property | Value |
| --- | --- |
| Hugging Face ID | `convaiinnovations/laya-multilingual` |
| Encoder | mmBERT-base |
| Parameters | Approximately 322M |
| Primary use | Multilingual routing and classification |

```bash
sys1 --model-id convaiinnovations/laya-multilingual --dtype auto
```

The checkpoint is typically faster than the English model. Validate each
language and question type used by the application; multilingual performance
is not uniform across tasks.

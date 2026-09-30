---
title: Metal
---

# Metal

## Cargo

```bash
cargo install sys1 --no-default-features --features metal
sys1 --model-id convaiinnovations/laya --dtype f16
```

Use `--dtype f32` for the reproducibility baseline. Validate F16 numeric drift
against application thresholds.

Metal is native-only. Docker Desktop does not expose the Apple GPU to Linux
containers. Use [Get started](index.md) to send a first request.

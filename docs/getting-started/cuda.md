---
title: CUDA
---

# CUDA

## Cargo

```bash
cargo install sys1 --no-default-features --features cuda
sys1 --model-id convaiinnovations/laya --dtype f16
```

Optional attention kernels:

```bash
# Ampere, Ada Lovelace, or Hopper
cargo install sys1 --no-default-features --features cuda,flash-attn-2

# Hopper
cargo install sys1 --no-default-features --features cuda,flash-attn-3
```

Select a compiled kernel with `--attention flash-attn-2` or
`--attention flash-attn-3`.

For GPU images, use the [Docker guide](docker.md). See [Batching](../advanced/batching.md)
before changing queue or batch limits.

---
title: CPU
---

# CPU

## Cargo

```bash
cargo install sys1 --features cpu
sys1 --model-id convaiinnovations/laya --dtype f32
```

For a source build with host CPU instructions:

```bash
RUSTFLAGS='-C target-cpu=native' \
  cargo build --release --locked --no-default-features --features cpu
```

For a container, use the [Docker guide](docker.md). The default CPU build is
also used in [Get started](index.md). See [Batching](../advanced/batching.md)
before changing queue or batch limits.

---
title: Docker
---

# Docker

## CPU

The CPU image supports Linux AMD64 and ARM64:

```bash
docker run --rm -p 3000:3000 \
  -v sys1-models:/home/sys1/.cache/huggingface \
  ghcr.io/alvarobartt/sys1:latest \
  --model-id convaiinnovations/laya --dtype f32
```

## CUDA

Choose the image for your GPU architecture. The NVIDIA Container Toolkit must
make the GPU available to Docker.

| GPU | Image |
| --- | --- |
| Turing | `latest-turing` |
| Ampere | `latest-ampere` |
| Ada Lovelace | `latest-ada-lovelace` |
| Hopper | `latest-hopper` |
| Blackwell | `latest-blackwell` |

For example, on Ampere:

```bash
docker run --rm --gpus all -p 3000:3000 \
  -v sys1-models:/home/sys1/.cache/huggingface \
  ghcr.io/alvarobartt/sys1:latest-ampere \
  --model-id convaiinnovations/laya \
  --dtype f16 \
  --attention flash-attn-2
```

Metal is native-only. Docker Desktop does not expose the Apple GPU to Linux
containers. Use the [Metal guide](metal.md) instead.

Once the container is ready, follow the [first request](index.md) and check
the [CLI](../cli.md) for configuration flags.

---
title: CLI
---

# CLI

`sys1` serves one model per process. The options, defaults, and environment
variables below are generated from the crate's Clap help on each docs build.
Run `sys1 --help` to inspect your installed version, or
[view the exported help text](/cli-help.txt) for this documentation version.
CLI arguments take precedence over environment variables.

<!--@include: ./.vitepress/generated/cli.md-->

The default `--batch-wait-ms 0` disables the extra collection window, but
requests already queued can still be batched. The scheduler limits requests
and questions, not estimated token counts. See
[Advanced batching](advanced/batching.md) for the latency and throughput
tradeoff, and [Get started](getting-started/index.md) for backend setup.

## Examples

Serve a local checkpoint under a stable API name:

```bash
sys1 --model-path /models/laya --served-model-name laya --dtype f32
```

Limit local access and cap the queue:

```bash
sys1 --host 127.0.0.1 --max-queue-size 64
```

---
title: Changelog
---

# Changelog

User-facing changes only. Dates use UTC.

## Unreleased

- Changed [`GET /metrics`](api.md#metrics) to Prometheus text format with
  request, batch, and inference-duration metrics.
- Added environment variable support, short aliases, and grouped help for
  [CLI options](cli.md).
- Added windowed eager attention on CPU.

## 0.0.3, 2026-09-25

- Added [Laya Multilingual](models/laya-multilingual.md) and
  [Laya Typed Decisions](models/laya-typed-decisions.md).
- Added `--max-model-len`.
- Added CUDA FlashAttention 2 and FlashAttention 3 build features. See the
  [CUDA guide](getting-started/cuda.md).
- Added memory-map safety documentation.
- Updated dependencies.

## 0.0.2, 2026-09-23

- Added [`GET /metrics`](api.md#metrics).
- Added explicit F32 and F16 selection with `--dtype`.
- Added CPU, Metal, and CUDA inference paths.
- Added OpenAPI JSON and interactive Swagger UI.
- Added architecture-specific CUDA container builds. See the
  [CUDA image tags](getting-started/docker.md#cuda).
- Enabled native CPU optimizations on Linux.

## 0.0.1, 2026-09-22

- Published the initial Rust inference server.
- Added the System One compatible decision endpoint.
- Added CPU and container release workflows.

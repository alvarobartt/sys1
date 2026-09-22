#!/usr/bin/env bash

set -euo pipefail

exec /usr/local/bin/sccache /usr/local/cuda/bin/nvcc "$@"

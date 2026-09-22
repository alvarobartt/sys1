#!/usr/bin/env bash

set -euo pipefail

if [[ -d /usr/local/nvidia/bin ]]; then
    export PATH="${PATH}:/usr/local/nvidia/bin"
fi
if [[ -d /usr/local/nvidia/lib64 ]]; then
    export LD_LIBRARY_PATH="/usr/local/nvidia/lib64:${LD_LIBRARY_PATH:-}"
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "error: nvidia-smi was not provided by the NVIDIA Container Toolkit" >&2
    exit 1
fi

driver_cuda="$({
    nvidia-smi 2>/dev/null |
        grep -oE 'CUDA[[:space:]]+([A-Za-z]+[[:space:]])?Version:[[:space:]]*[0-9]+(\.[0-9]+)+' |
        grep -oE '[0-9]+(\.[0-9]+)+' |
        head -n1
} || true)"

version_number() {
    local version="$1" major minor patch
    IFS=. read -r major minor patch <<<"${version}"
    minor="${minor:-0}"
    patch="${patch:-0}"
    printf '%d' "$((10#${major} * 10000 + 10#${minor} * 100 + 10#${patch}))"
}

if [[ -d /usr/local/cuda/compat && -n "${driver_cuda}" ]]; then
    if (($(version_number "${driver_cuda}") < $(version_number "${CUDA_VERSION}"))); then
        export LD_LIBRARY_PATH="/usr/local/cuda/compat:${LD_LIBRARY_PATH:-}"
    fi
fi

compute_cap="$({
    nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null |
        head -n1 |
        tr -d '.[:space:]'
} || true)"

case "${compute_cap}" in
75)
    binary=/usr/local/bin/sys1-75
    ;;
8[0-9])
    binary=/usr/local/bin/sys1-80
    ;;
90)
    binary=/usr/local/bin/sys1-90
    ;;
100)
    binary=/usr/local/bin/sys1-100
    ;;
120)
    binary=/usr/local/bin/sys1-120
    ;;
*)
    echo "error: CUDA compute capability '${compute_cap:-unknown}' is not supported by this image" >&2
    exit 1
    ;;
esac

exec /opt/nvidia/nvidia_entrypoint.sh "${binary}" "$@"

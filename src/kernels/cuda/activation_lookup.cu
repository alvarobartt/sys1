#include <cuda_bf16.h>
#include <math.h>

extern "C" __global__ void build_activation_lookup_bf16(__nv_bfloat16 *table) {
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= 131072u)
        return;
    const float x = __bfloat162float(__ushort_as_bfloat16(static_cast<unsigned short>(i)));
    const float denominator = __fadd_rn(1.f, expf(-x));
    const float value = i >= 65536u ? __fdiv_rn(1.f, denominator) : __fdiv_rn(x, denominator);
    table[i] = __float2bfloat16_rn(value);
}

#include <cuda_bf16.h>
#include <cstdint>

template <int V>
__device__ __forceinline__ void activation_vector_impl(const __nv_bfloat16 *input,
    const __nv_bfloat16 *up, const __nv_bfloat16 *table, __nv_bfloat16 *output, unsigned int count,
    int mode) {
    const size_t first = (size_t(blockIdx.x) * blockDim.x + threadIdx.x) * V;
    if (first >= count)
        return;
    constexpr uintptr_t Mask = V * 2 - 1;
    const bool aligned = ((reinterpret_cast<uintptr_t>(input) | reinterpret_cast<uintptr_t>(up) |
                              reinterpret_cast<uintptr_t>(output)) &
                             Mask) == 0;
    if (aligned && first + V <= count) {
        alignas(16) unsigned short xv[V], uv[V], yv[V];
        if constexpr (V == 8) {
            *reinterpret_cast<uint4 *>(xv) = *reinterpret_cast<const uint4 *>(input + first);
            if (mode == 2)
                *reinterpret_cast<uint4 *>(uv) = *reinterpret_cast<const uint4 *>(up + first);
        } else {
            *reinterpret_cast<uint2 *>(xv) = *reinterpret_cast<const uint2 *>(input + first);
            if (mode == 2)
                *reinterpret_cast<uint2 *>(uv) = *reinterpret_cast<const uint2 *>(up + first);
        }
#pragma unroll
        for (int p = 0; p < V; ++p) {
            const __nv_bfloat16 activated = table[unsigned(xv[p]) + (mode == 1 ? 65536u : 0u)];
            yv[p] = __bfloat16_as_ushort(mode == 2
                    ? __float2bfloat16_rn(__fmul_rn(__bfloat162float(activated),
                          __bfloat162float(__ushort_as_bfloat16(uv[p]))))
                    : activated);
        }
        if constexpr (V == 8)
            *reinterpret_cast<uint4 *>(output + first) = *reinterpret_cast<uint4 *>(yv);
        else
            *reinterpret_cast<uint2 *>(output + first) = *reinterpret_cast<uint2 *>(yv);
    } else {
#pragma unroll
        for (int p = 0; p < V; ++p) {
            const size_t i = first + p;
            if (i < count) {
                const unsigned index = __bfloat16_as_ushort(input[i]) + (mode == 1 ? 65536u : 0u);
                const __nv_bfloat16 activated = table[index];
                output[i] = mode == 2 ? __float2bfloat16_rn(__fmul_rn(
                                            __bfloat162float(activated), __bfloat162float(up[i])))
                                      : activated;
            }
        }
    }
}

extern "C" __global__ void activation_lookup_bf16(const __nv_bfloat16 *x, const __nv_bfloat16 *u,
    const __nv_bfloat16 *table, __nv_bfloat16 *y, unsigned n, int mode) {
    activation_vector_impl<8>(x, u, table, y, n, mode);
}

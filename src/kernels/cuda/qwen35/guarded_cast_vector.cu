

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <math.h>

template <bool Value>
__device__ __forceinline__ void qwen35_vector_cast134(const float *input, const float *g,
                                                    const float *beta, __half *output,
                                                    float *aligned_g, float *aligned_beta,
                                                    uint64_t count, float limit, unsigned *bad) {
    bool invalid = false;
    const bool vectorized = (reinterpret_cast<uintptr_t>(input) & 15u) == 0 &&
                            (reinterpret_cast<uintptr_t>(output) & 7u) == 0;
    for (uint64_t i = (uint64_t(blockIdx.x) * blockDim.x + threadIdx.x) * 4; i < count;
         i += uint64_t(gridDim.x) * blockDim.x * 4) {
        if (vectorized && i + 3 < count) {
            const float4 v = *reinterpret_cast<const float4 *>(input + i);
            invalid |= !(fabsf(v.x) <= limit) || !(fabsf(v.y) <= limit) || !(fabsf(v.z) <= limit) ||
                       !(fabsf(v.w) <= limit);
            const __half a = __float2half_rn(v.x), b = __float2half_rn(v.y),
                         c = __float2half_rn(v.z), d = __float2half_rn(v.w);
            const uint2 packed =
                make_uint2(unsigned(__half_as_ushort(a)) | (unsigned(__half_as_ushort(b)) << 16),
                           unsigned(__half_as_ushort(c)) | (unsigned(__half_as_ushort(d)) << 16));
            *reinterpret_cast<uint2 *>(output + i) = packed;
        } else {
#pragma unroll
            for (int j = 0; j < 4; ++j)
                if (i + j < count) {
                    const float v = input[i + j];
                    invalid |= !(fabsf(v) <= limit);
                    output[i + j] = __float2half_rn(v);
                }
        }
        if constexpr (Value) {
            if ((i & 127) == 0) {
                const uint64_t head = i / 128;
                const float gate = g[head], b = beta[head];
                aligned_g[head] = gate;
                aligned_beta[head] = b;
                invalid |= !(isfinite(gate) && gate <= 0.f && b >= 0.f && b <= 1.f);
            }
        }
    }
    if (invalid)
        atomicOr(bad, Value ? 4u : 1u);
}

extern "C" __global__ void qwen35_vector_cast_q134(const float *input, __half *output, uint64_t count,
                                                 unsigned *bad) {
    qwen35_vector_cast134<false>(input, nullptr, nullptr, output, nullptr, nullptr, count, 1.f, bad);
}

extern "C" __global__ void qwen35_vector_cast(const float *input, const float *g, const float *beta,
                                            __half *output, float *aligned_g, float *aligned_beta,
                                            uint64_t count, float limit, unsigned *bad) {
    qwen35_vector_cast134<true>(input, g, beta, output, aligned_g, aligned_beta, count, limit, bad);
}

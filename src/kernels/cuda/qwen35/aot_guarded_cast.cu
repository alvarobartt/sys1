#include <cuda_fp16.h>
#include <math.h>
#include <stdint.h>

extern "C" __global__ void qwen35_aot91_cast_q(
    const float *input, __half *output, uint64_t count, unsigned *bad) {
    bool invalid = false;
    for (uint64_t i = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < count;
         i += uint64_t(gridDim.x) * blockDim.x) {
        const float value = input[i];
        invalid |= !(fabsf(value) <= 1.f);
        output[i] = __float2half_rn(value);
    }
    if (invalid)
        atomicOr(bad, 1u);
}

extern "C" __global__ void qwen35_aot91_cast_k(
    const float *input, __half *output, uint64_t rows, unsigned *bad) {
    const int lane = threadIdx.x & 31;
    for (uint64_t row = uint64_t(blockIdx.x) * 4 + threadIdx.x / 32; row < rows;
         row += uint64_t(gridDim.x) * 4) {
        float norm = 0.f;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const uint64_t index = row * 128 + lane + j * 32;
            const __half value = __float2half_rn(input[index]);
            output[index] = value;
            const float rounded = __half2float(value);
            norm += rounded * rounded;
        }
#pragma unroll
        for (int offset = 16; offset; offset /= 2)
            norm += __shfl_down_sync(0xffffffff, norm, offset);
        if (lane == 0 && !(norm <= 1.25f))
            atomicOr(bad, 2u);
    }
}

extern "C" __global__ void qwen35_aot91_cast_v(const float *input, const float *g,
    const float *beta, __half *output, float *aligned_g, float *aligned_beta, uint64_t count,
    float limit, unsigned *bad) {
    bool invalid = false;
    for (uint64_t i = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < count;
         i += uint64_t(gridDim.x) * blockDim.x) {
        const float value = input[i];
        invalid |= !(fabsf(value) <= limit);
        output[i] = __float2half_rn(value);
        if ((i & 127) == 0) {
            const uint64_t head = i / 128;
            const float gate = g[head], b = beta[head];
            aligned_g[head] = gate;
            aligned_beta[head] = b;
            invalid |= !(isfinite(gate) && gate <= 0.f && b >= 0.f && b <= 1.f);
        }
    }
    if (invalid)
        atomicOr(bad, 4u);
}

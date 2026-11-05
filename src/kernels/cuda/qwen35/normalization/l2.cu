#include <cuda_bf16.h>
#include <float.h>
#include <math.h>
#include <mma.h>

extern "C" __global__ void qwen35_l2_128_f32(const float *input, float *output, unsigned int rows,
                                             float scale) {
    const unsigned int row = blockIdx.x * 4 + threadIdx.x / 32;
    const unsigned int lane = threadIdx.x & 31;
    if (row >= rows)
        return;
    const unsigned int offset = row * 128 + lane * 4;
    float values[4];
#pragma unroll
    for (int j = 0; j < 4; ++j)
        values[j] = input[offset + j];
    float sum = __fmul_rn(values[0], values[0]);
#pragma unroll
    for (int j = 1; j < 4; ++j)
        sum = __fadd_rn(sum, __fmul_rn(values[j], values[j]));
#pragma unroll
    for (int step = 16; step > 0; step >>= 1)
        sum = __fadd_rn(sum, __shfl_xor_sync(0xffffffff, sum, step));
    sum = __shfl_sync(0xffffffff, sum, 0);
    const float denom = __fsqrt_rn(__fadd_rn(sum, 1e-6f));
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const float normalized = __fdiv_rn(values[j], denom);
        output[offset + j] = scale == 1.f ? normalized : __fmaf_rn(normalized, scale, 0.f);
    }
}

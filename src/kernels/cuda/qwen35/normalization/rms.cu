#include <cuda_bf16.h>
#include <float.h>
#include <math.h>
#include <mma.h>

__device__ __forceinline__ float qwen35_warp_sum(float value) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_xor_sync(0xffffffff, value, offset, 32);
    }
    return value;
}

extern "C" __global__ void qwen35_rms_norm_delta_bf16(const __nv_bfloat16 *input,
    const __nv_bfloat16 *delta_weight, __nv_bfloat16 *output, int columns, float epsilon) {
    const int row = blockIdx.x;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    float sum = 0.f;
    for (int column = threadIdx.x; column < columns; column += blockDim.x) {
        const float value = float(input[row * columns + column]);
        sum += value * value;
    }
    sum = qwen35_warp_sum(sum);
    __shared__ float warp_sums[32];
    if (lane == 0)
        warp_sums[warp] = sum;
    __syncthreads();
    sum = lane < (blockDim.x >> 5) ? warp_sums[lane] : 0.f;
    sum = qwen35_warp_sum(sum);
    const float scale = rsqrtf(sum / float(columns) + epsilon);
    for (int column = threadIdx.x; column < columns; column += blockDim.x) {
        const int index = row * columns + column;
        output[index] =
            __nv_bfloat16(scale * float(input[index]) * (1.f + float(delta_weight[column])));
    }
}

extern "C" __global__ void qwen35_residual_rms_norm_delta_bf16(const __nv_bfloat16 *input,
    const __nv_bfloat16 *mixed, const __nv_bfloat16 *delta_weight, __nv_bfloat16 *residual,
    __nv_bfloat16 *normalized, int columns, float epsilon) {
    const int row = blockIdx.x;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    float sum = 0.f;
    for (int column = threadIdx.x; column < columns; column += blockDim.x) {
        const int index = row * columns + column;
        const __nv_bfloat16 value = __hadd(input[index], mixed[index]);
        residual[index] = value;
        const float rounded = float(value);
        sum += rounded * rounded;
    }
    sum = qwen35_warp_sum(sum);
    __shared__ float warp_sums[32];
    if (lane == 0)
        warp_sums[warp] = sum;
    __syncthreads();
    sum = lane < (blockDim.x >> 5) ? warp_sums[lane] : 0.f;
    sum = qwen35_warp_sum(sum);
    const float scale = rsqrtf(sum / float(columns) + epsilon);
    for (int column = threadIdx.x; column < columns; column += blockDim.x) {
        const int index = row * columns + column;
        normalized[index] =
            __nv_bfloat16(scale * float(residual[index]) * (1.f + float(delta_weight[column])));
    }
}

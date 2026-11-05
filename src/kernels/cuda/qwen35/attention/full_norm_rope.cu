#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <math.h>

__device__ __forceinline__ float qwen35_fullnr_warp_sum(float value) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        value += __shfl_xor_sync(0xffffffff, value, offset, 32);
    return value;
}

extern "C" __global__ void
qwen35_full_norm_rope_bf16(const __nv_bfloat16 *input, const __nv_bfloat16 *delta_weight,
                         const __nv_bfloat16 *cosine, const __nv_bfloat16 *sine,
                         __nv_bfloat16 *output, int width, int rotary, int heads, int length,
                         int row_stride, float epsilon) {
    const int row = blockIdx.x, lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int position = (row / heads) % length;
    float sum = 0.f;
    for (int column = threadIdx.x; column < width; column += blockDim.x) {
        const float value = float(input[size_t(row) * row_stride + column]);
        sum += value * value;
    }
    sum = qwen35_fullnr_warp_sum(sum);
    __shared__ float warp_sums[32];
    __shared__ __nv_bfloat16 normalized[1024];
    if (lane == 0)
        warp_sums[warp] = sum;
    __syncthreads();
    sum = lane < (blockDim.x >> 5) ? warp_sums[lane] : 0.f;
    sum = qwen35_fullnr_warp_sum(sum);
    const float scale = rsqrtf(sum / float(width) + epsilon);
    for (int column = threadIdx.x; column < width; column += blockDim.x) {

        normalized[column] = __nv_bfloat16(scale * float(input[size_t(row) * row_stride + column]) *
                                           (1.f + float(delta_weight[column])));
    }
    __syncthreads();
    for (int column = threadIdx.x; column < width; column += blockDim.x) {
        __nv_bfloat16 value = normalized[column];
        if (column < rotary) {
            const int partner = column < rotary / 2 ? column + rotary / 2 : column - rotary / 2;
            const __nv_bfloat16 rotated =
                column < rotary / 2 ? __hneg(normalized[partner]) : normalized[partner];

            value = __hadd_rn(__hmul_rn(value, cosine[position * rotary + column]),
                              __hmul_rn(rotated, sine[position * rotary + column]));
        }
        output[size_t(row) * width + column] = value;
    }
}

#include <cuda_bf16.h>
__device__ __forceinline__ float post_strided_warp_sum(float value) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_xor_sync(0xffffffff, value, offset, 32);
    }
    return value;
}
extern "C" __global__ void qwen35_post_finalize_strided(const float *input,
    const __nv_bfloat16 *gate, const __nv_bfloat16 *norm, __nv_bfloat16 *output, int columns,
    float epsilon, unsigned long long length, unsigned long long heads, unsigned long long stride_b,
    unsigned long long stride_t, unsigned long long stride_h) {
    const int row = blockIdx.x;
    const unsigned long long h = row % heads;
    const unsigned long long token = (row / heads) % length;
    const unsigned long long batch = row / (heads * length);
    const unsigned long long source_row = batch * stride_b + token * stride_t + h * stride_h;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    float sum = 0.f;
    for (int column = threadIdx.x; column < columns; column += blockDim.x) {
        const float value = float(__nv_bfloat16(input[source_row + column]));
        sum += value * value;
    }
    sum = post_strided_warp_sum(sum);
    __shared__ float warp_sums[32];
    if (lane == 0)
        warp_sums[warp] = sum;
    __syncthreads();
    sum = lane < (blockDim.x >> 5) ? warp_sums[lane] : 0.f;
    sum = post_strided_warp_sum(sum);
    const float scale = rsqrtf(sum / float(columns) + epsilon);
    for (int column = threadIdx.x; column < columns; column += blockDim.x) {
        const int index = row * columns + column;
        const float gate_value = float(gate[index]);
        const float gate_silu = gate_value / (1.f + expf(-gate_value));
        const float recurrent = float(__nv_bfloat16(input[source_row + column]));
        const float normalized = float(__nv_bfloat16(__fmul_rn(recurrent, scale)));
        const float weighted = float(__nv_bfloat16(__fmul_rn(normalized, float(norm[column]))));
        output[index] = __nv_bfloat16(__fmul_rn(weighted, gate_silu));
    }
}

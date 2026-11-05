#include <cuda_bf16.h>
#include <float.h>
#include <math.h>
#include <mma.h>

extern "C" __global__ void qwen35_post_finalize_bf16(const float *input, const __nv_bfloat16 *gate,
    const __nv_bfloat16 *norm, __nv_bfloat16 *output, int columns, float epsilon) {
    const int row = blockIdx.x;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    float sum = 0.f;
    for (int column = threadIdx.x; column < columns; column += blockDim.x) {
        const float value = float(__nv_bfloat16(input[row * columns + column]));
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
        const float gate_value = float(gate[index]);
        const float gate_silu = gate_value / (1.f + expf(-gate_value));
        const float recurrent = float(__nv_bfloat16(input[index]));
        const float normalized = float(__nv_bfloat16(__fmul_rn(recurrent, scale)));
        const float weighted = float(__nv_bfloat16(__fmul_rn(normalized, float(norm[column]))));
        output[index] = __nv_bfloat16(__fmul_rn(weighted, gate_silu));
    }
}

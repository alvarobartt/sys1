#include <cuda_bf16.h>

extern "C" __global__ void weighted_summary_bf16(const __nv_bfloat16 *values,
                                                      const __nv_bfloat16 *weights,
                                                      __nv_bfloat16 *output, unsigned int rows,
                                                      unsigned int columns) {
    const unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= columns)
        return;
    float sum = 0.f;
    for (unsigned int row = 0; row < rows; ++row) {
        const float product = __fmul_rn(__bfloat162float(values[row * columns + col]),
                                        __bfloat162float(weights[row]));
        const __nv_bfloat16 rounded = __float2bfloat16_rn(product);
        sum = __fadd_rn(sum, __bfloat162float(rounded));
    }
    output[col] = __float2bfloat16_rn(sum);
}

#include <cuda_bf16.h>
#include <math.h>

extern "C" __global__ void activation_bf16(const __nv_bfloat16 *input, const __nv_bfloat16 *up,
                                                __nv_bfloat16 *output, unsigned int count,
                                                int mode) {
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count)
        return;
    const float x = __bfloat162float(input[i]);
    const float denominator = __fadd_rn(1.f, expf(-x));
    const float value = mode == 1 ? __fdiv_rn(1.f, denominator) : __fdiv_rn(x, denominator);
    const __nv_bfloat16 activated = __float2bfloat16_rn(value);
    output[i] =
        mode == 2
            ? __float2bfloat16_rn(__fmul_rn(__bfloat162float(activated), __bfloat162float(up[i])))
            : activated;
}

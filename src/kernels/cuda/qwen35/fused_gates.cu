#include <cuda_bf16.h>
#include <math.h>


#include <cuda_bf16.h>
#include <math.h>

extern "C" __global__ void qwen35_packed_ba_nonlinearity_bf16(const __nv_bfloat16 *a,
                                                            const __nv_bfloat16 *b,
                                                            const float *bias, const float *a_log,
                                                            float *gate, float *beta, int count,
                                                            int heads) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count)
        return;
    const float bv = __bfloat162float(b[(i / heads) * (2 * heads) + i % heads]);
    const float denominator = __fadd_rn(1.f, expf(-bv));

    const float probability = static_cast<float>(1.0 / static_cast<double>(denominator));
    beta[i] = __bfloat162float(__float2bfloat16_rn(probability));
    const int head = i % heads;
    const float input =
        __fadd_rn(__bfloat162float(a[(i / heads) * (2 * heads) + heads + i % heads]), bias[head]);
    const float correction = logf(__fadd_rn(expf(-fabsf(input)), 1.f));
    gate[i] = __fmul_rn(__fadd_rn(fmaxf(input, 0.f), correction), a_log[head]);
}

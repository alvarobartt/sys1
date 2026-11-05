#include <cuda_bf16.h>
#include <math.h>

extern "C" __global__ void span_means_bf16(const __nv_bfloat16 *source, const unsigned int *ids,
    const unsigned int *spans, __nv_bfloat16 *output, unsigned int source_rows,
    unsigned int columns, int lexical) {
    __shared__ float partial[1024];
    const unsigned int start = spans[blockIdx.y * 3];
    const unsigned int length = spans[blockIdx.y * 3 + 1];
    const unsigned int destination = spans[blockIdx.y * 3 + 2];
    const unsigned int column = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int slot = threadIdx.y * blockDim.x + threadIdx.x;
    float sum = 0.f;
    if (column < columns) {
        for (unsigned long long row = threadIdx.y; row < length; row += blockDim.y) {
            const unsigned long long token = start + row;
            const unsigned long long source_row = lexical ? ids[token] : token;

            const float value = source_row < source_rows
                ? __bfloat162float(source[source_row * columns + column])
                : NAN;
            sum = __fadd_rn(sum, value);
        }
    }
    partial[slot] = sum;
    for (unsigned int step = blockDim.y / 2; step > 0; step >>= 1) {
        __syncthreads();
        if (threadIdx.y < step)
            partial[slot] = __fadd_rn(partial[slot], partial[slot + step * blockDim.x]);
    }
    __syncthreads();
    if (threadIdx.y == 0 && column < columns) {
        const float reciprocal = __fdiv_rn(1.f, float(length));
        output[static_cast<unsigned long long>(destination) * columns + column] =
            __float2bfloat16_rn(__fmaf_rn(partial[slot], reciprocal, 0.f));
    }
}

#include <cuda_runtime.h>
#include <math.h>

template <bool Packed>
__device__ __forceinline__ void
qwen35_delta_reuse(const float *query, const float *key, const float *value,
                 const float *gate, const float *beta, float *output,
                 const unsigned int *starts, int extent, int heads,
                 int key_heads, int value_dim) {
    constexpr int Columns = 2;
    const int head = blockIdx.x % heads, sequence = blockIdx.x / heads;
    const int begin = Packed ? int(starts[sequence]) : sequence * extent;
    const int length = Packed ? int(starts[sequence + 1]) - begin : extent;
    const int lane = threadIdx.x & 15;
    const int first = (blockIdx.y * 8 + (threadIdx.x >> 4)) * Columns;
    if (first >= value_dim)
        return;
    const unsigned mask = __activemask();
    float state[Columns][2][4] = {};
    for (int token = 0; token < length; ++token) {
        const int token_head = (begin + token) * heads + head;
        const int qk_head = (begin + token) * key_heads + head / (heads / key_heads);
        const float decay = Packed ? expf(gate[token_head]) : gate[token_head];
        const float weight = beta[token_head];
        float keys[2][4], queries[2][4];
#pragma unroll
        for (int half = 0; half < 2; ++half) {
#pragma unroll
            for (int part = 0; part < 4; ++part) {
                const int offset = qk_head * 128 + part * 32 + lane + half * 16;
                keys[half][part] = key[offset];
                queries[half][part] = query[offset];
            }
        }
#pragma unroll
        for (int col = 0; col < Columns; ++col) {
            float remembered[2] = {};
#pragma unroll
            for (int half = 0; half < 2; ++half) {
#pragma unroll
                for (int part = 0; part < 4; ++part) {
                    state[col][half][part] = __fmul_rn(state[col][half][part], decay);
                    remembered[half] = __fmaf_rn(keys[half][part], state[col][half][part], remembered[half]);
                }
            }
            float total = __fadd_rn(remembered[0], remembered[1]);
#pragma unroll
            for (int step = 8; step > 0; step >>= 1)
                total = __fadd_rn(total, __shfl_down_sync(mask, total, step, 16));
            const float v = first + col < value_dim ? value[token_head * value_dim + first + col] : 0.f;
            const float delta = __fmul_rn(__fsub_rn(v, __shfl_sync(mask, total, 0, 16)), weight);
            float answer[2] = {};
#pragma unroll
            for (int half = 0; half < 2; ++half) {
#pragma unroll
                for (int part = 0; part < 4; ++part) {
                    state[col][half][part] = __fmaf_rn(keys[half][part], delta, state[col][half][part]);
                    answer[half] = __fmaf_rn(queries[half][part], state[col][half][part], answer[half]);
                }
            }
            total = __fadd_rn(answer[0], answer[1]);
#pragma unroll
            for (int step = 8; step > 0; step >>= 1)
                total = __fadd_rn(total, __shfl_down_sync(mask, total, step, 16));
            if (lane == 0 && first + col < value_dim)
                output[token_head * value_dim + first + col] = total;
        }
    }
}

extern "C" __global__ void qwen35_uniform_delta_reuse_f32(
    const float *q, const float *k, const float *v, const float *decay,
    const float *beta, float *output, int length, int heads, int key_heads, int width) {
    qwen35_delta_reuse<false>(q, k, v, decay, beta, output, nullptr, length, heads, key_heads, width);
}

extern "C" __global__ void qwen35_packed_delta_reuse_f32(
    const float *q, const float *k, const float *v, const float *gate,
    const float *beta, float *output, const unsigned int *starts,
    int extent, int heads, int key_heads, int width) {
    qwen35_delta_reuse<true>(q, k, v, gate, beta, output, starts, extent, heads, key_heads, width);
}

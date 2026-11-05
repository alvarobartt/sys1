

#include <cuda_runtime.h>

template <int Columns>
__device__ __forceinline__ void
packed_columns(const float *query, const float *key, const float *value, const float *decay_values,
               const float *beta, float *output, const unsigned int *starts, int extent, int heads,
               int key_heads, int value_dim) {
    const int head = blockIdx.x % heads, sequence = blockIdx.x / heads;
    const int begin = int(starts[sequence]);
    const int length = int(starts[sequence + 1]) - begin;
    const int lane = threadIdx.x & 31;
    const int first = blockIdx.y * (4 * Columns) + (threadIdx.x >> 5) * Columns;
    if (first >= value_dim)
        return;
    float state[Columns][4] = {};
    for (int token = 0; token < length; ++token) {
        const int token_head = (begin + token) * heads + head;
        const int qk_head = (begin + token) * key_heads + head / (heads / key_heads);
        const float decay = decay_values[token_head], weight = beta[token_head];
        const float *token_query = query + qk_head * 128;
        const float *token_key = key + qk_head * 128;
        const float *token_value = value + token_head * value_dim;
        float keys[4], queries[4], remembered[Columns] = {};
#pragma unroll
        for (int part = 0; part < 4; ++part) {
            keys[part] = token_key[part * 32 + lane];
            queries[part] = token_query[part * 32 + lane];
#pragma unroll
            for (int col = 0; col < Columns; ++col) {
                state[col][part] *= decay;
                remembered[col] += keys[part] * state[col][part];
            }
        }
#pragma unroll
        for (int step = 16; step > 0; step >>= 1) {
#pragma unroll
            for (int col = 0; col < Columns; ++col)
                remembered[col] += __shfl_down_sync(0xffffffff, remembered[col], step);
        }
        float delta[Columns], answer[Columns] = {};
#pragma unroll
        for (int col = 0; col < Columns; ++col) {
            const float v = first + col < value_dim ? token_value[first + col] : 0.f;
            delta[col] = (v - __shfl_sync(0xffffffff, remembered[col], 0)) * weight;
        }
#pragma unroll
        for (int part = 0; part < 4; ++part) {
#pragma unroll
            for (int col = 0; col < Columns; ++col) {
                state[col][part] += keys[part] * delta[col];
                answer[col] += queries[part] * state[col][part];
            }
        }
#pragma unroll
        for (int step = 16; step > 0; step >>= 1) {
#pragma unroll
            for (int col = 0; col < Columns; ++col)
                answer[col] += __shfl_down_sync(0xffffffff, answer[col], step);
        }
        if (lane == 0) {
#pragma unroll
            for (int col = 0; col < Columns; ++col)
                if (first + col < value_dim)
                    output[token_head * value_dim + first + col] = answer[col];
        }
    }
}

extern "C" __global__ void qwen35_packed_delta_warp32_cols2_preexp_f32(
    const float *q, const float *k, const float *v, const float *decay, const float *beta,
    float *output, const unsigned int *starts, int extent, int heads, int key_heads, int width) {
    packed_columns<2>(q, k, v, decay, beta, output, starts, extent, heads, key_heads, width);
}

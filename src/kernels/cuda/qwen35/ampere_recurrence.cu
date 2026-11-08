#include <cuda_runtime.h>

template <bool Packed>
__device__ __forceinline__ void qwen35_ampere_warp32_delta(const float *query, const float *key,
    const float *value, const float *decay_values, const float *beta, float *output,
    const unsigned int *starts, int extent, int heads, int key_heads, int value_dim) {
    const int batch_head = blockIdx.x;
    const int head = batch_head % heads, sequence = batch_head / heads;
    const int begin = Packed ? int(starts[sequence]) : sequence * extent;
    const int length = Packed ? int(starts[sequence + 1]) - begin : extent;
    const int lane = threadIdx.x & 31;
    const int column = blockIdx.y * 4 + (threadIdx.x >> 5);
    if (column >= value_dim)
        return;
    float state[4] = {};
    for (int token = 0; token < length; ++token) {
        const int token_head = (begin + token) * heads + head;
        const int qk_head = (begin + token) * key_heads + head / (heads / key_heads);
        const int output_offset = Packed ? token_head : batch_head * length + token;
        const float decay = decay_values[token_head], weight = beta[token_head];
        const float *token_query = query + qk_head * 128;
        const float *token_key = key + qk_head * 128;
        const float *token_value = value + token_head * value_dim;
        float keys[4], remembered = 0.f;
#pragma unroll
        for (int part = 0; part < 4; ++part) {
            keys[part] = token_key[part * 32 + lane];
            state[part] *= decay;
            remembered += keys[part] * state[part];
        }
#pragma unroll
        for (int step = 16; step > 0; step >>= 1)
            remembered += __shfl_down_sync(0xffffffff, remembered, step);
        const float delta = (token_value[column] - __shfl_sync(0xffffffff, remembered, 0)) * weight;
        float answer = 0.f;
#pragma unroll
        for (int part = 0; part < 4; ++part) {
            state[part] += keys[part] * delta;
            answer += token_query[part * 32 + lane] * state[part];
        }
#pragma unroll
        for (int step = 16; step > 0; step >>= 1)
            answer += __shfl_down_sync(0xffffffff, answer, step);
        if (lane == 0)
            output[output_offset * value_dim + column] = answer;
    }
}

extern "C" __global__ void qwen35_delta_warp32_preexp_f32(const float *q, const float *k,
    const float *v, const float *decay, const float *beta, float *output, int length, int heads,
    int key_heads, int value_dim) {
    qwen35_ampere_warp32_delta<false>(
        q, k, v, decay, beta, output, nullptr, length, heads, key_heads, value_dim);
}

extern "C" __global__ void qwen35_packed_delta_warp32_preexp_f32(const float *q, const float *k,
    const float *v, const float *decay, const float *beta, float *output,
    const unsigned int *starts, int total_length, int heads, int key_heads, int value_dim) {
    qwen35_ampere_warp32_delta<true>(
        q, k, v, decay, beta, output, starts, total_length, heads, key_heads, value_dim);
}

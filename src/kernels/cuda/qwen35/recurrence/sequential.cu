#include <cuda_bf16.h>
#include <float.h>
#include <math.h>
#include <mma.h>

extern "C" __global__ void qwen35_delta_f32(const float *query, const float *key,
                                            const float *value, const float *gate,
                                            const float *beta, float *output, int length, int heads,
                                            int key_heads, int value_dim) {
    const int batch_head = blockIdx.x;
    const int head = batch_head % heads;
    const int batch = batch_head / heads;
    const int lane = threadIdx.x & 31;
    const int column = blockIdx.y * 4 + (threadIdx.x >> 5);
    if (column >= value_dim)
        return;

    float state[4] = {0.f, 0.f, 0.f, 0.f};
    for (int token = 0; token < length; ++token) {
        const int token_head = (batch * length + token) * heads + head;
        const int qk_head = (batch * length + token) * key_heads + head / (heads / key_heads);
        const int output_offset = batch_head * length + token;
        const float decay = expf(gate[token_head]);
        const float weight = beta[token_head];
        const float *token_query = query + qk_head * 128;
        const float *token_key = key + qk_head * 128;
        const float *token_value = value + token_head * value_dim;
        float keys[4];
        float remembered = 0.f;
#pragma unroll
        for (int part = 0; part < 4; ++part) {
            keys[part] = token_key[part * 32 + lane];
            state[part] *= decay;
            remembered += keys[part] * state[part];
        }
#pragma unroll
        for (int step = 16; step > 0; step >>= 1) {
            remembered += __shfl_down_sync(0xffffffff, remembered, step);
        }
        const float delta = (token_value[column] - __shfl_sync(0xffffffff, remembered, 0)) * weight;
        float answer = 0.f;
#pragma unroll
        for (int part = 0; part < 4; ++part) {
            state[part] += keys[part] * delta;
            answer += token_query[part * 32 + lane] * state[part];
        }
#pragma unroll
        for (int step = 16; step > 0; step >>= 1) {
            answer += __shfl_down_sync(0xffffffff, answer, step);
        }
        if (lane == 0) {
            output[output_offset * value_dim + column] = answer;
        }
    }
}

extern "C" __global__ void qwen35_delta_subwarp16_f32(const float *query, const float *key,
                                                      const float *value, const float *gate,
                                                      const float *beta, float *output, int length,
                                                      int heads, int key_heads, int value_dim) {
    const int batch_head = blockIdx.x;
    const int head = batch_head % heads;
    const int batch = batch_head / heads;
    const int lane = threadIdx.x & 15;
    const int column = blockIdx.y * 8 + (threadIdx.x >> 4);
    if (column >= value_dim)
        return;
    const unsigned mask = __activemask();
    float state[2][4] = {{0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}};
    for (int token = 0; token < length; ++token) {
        const int token_head = (batch * length + token) * heads + head;
        const int qk_head = (batch * length + token) * key_heads + head / (heads / key_heads);
        const int output_offset = batch_head * length + token;
        const float decay = expf(gate[token_head]);
        const float weight = beta[token_head];
        const float *token_query = query + qk_head * 128;
        const float *token_key = key + qk_head * 128;
        const float *token_value = value + token_head * value_dim;
        float keys[2][4];
        float remembered[2] = {0.f, 0.f};
#pragma unroll
        for (int half = 0; half < 2; ++half) {
#pragma unroll
            for (int part = 0; part < 4; ++part) {
                keys[half][part] = token_key[part * 32 + lane + half * 16];
                state[half][part] *= decay;
                remembered[half] += keys[half][part] * state[half][part];
            }
        }

        float total = remembered[0] + remembered[1];
#pragma unroll
        for (int step = 8; step > 0; step >>= 1) {
            total += __shfl_down_sync(mask, total, step, 16);
        }
        const float delta = (token_value[column] - __shfl_sync(mask, total, 0, 16)) * weight;
        float answer[2] = {0.f, 0.f};
#pragma unroll
        for (int half = 0; half < 2; ++half) {
#pragma unroll
            for (int part = 0; part < 4; ++part) {
                state[half][part] += keys[half][part] * delta;
                answer[half] += token_query[part * 32 + lane + half * 16] * state[half][part];
            }
        }
        total = answer[0] + answer[1];
#pragma unroll
        for (int step = 8; step > 0; step >>= 1) {
            total += __shfl_down_sync(mask, total, step, 16);
        }
        if (lane == 0)
            output[output_offset * value_dim + column] = total;
    }
}

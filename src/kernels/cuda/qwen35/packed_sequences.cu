#include <cuda_bf16.h>
#include <math.h>
template <typename T>
__device__ void qwen35_packed_conv(const T *input, const T *weights, T *output,
                                 const unsigned int *starts, int count, int length, int channels,
                                 int kernel) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count)
        return;
    const int channel = index % channels;
    const int token = index / channels;
    const int begin = starts[token];

    float value = 0.f;
    for (int tap = 0; tap < kernel; ++tap) {
        const int source_token = token + tap - kernel + 1;
        if (source_token >= begin) {
            const float product = __fmul_rn(float(input[source_token * channels + channel]),
                                            float(weights[channel * kernel + tap]));
            value = __fadd_rn(value, product);
        }
    }

    const float x = float(T(value));
    output[index] = T(x / (1.f + expf(-x)));
}
extern "C" __global__ void qwen35_packed_conv_bf16(const __nv_bfloat16 *x, const __nv_bfloat16 *w,
                                                 __nv_bfloat16 *out, const unsigned int *starts,
                                                 int count, int length, int channels, int kernel) {
    qwen35_packed_conv(x, w, out, starts, count, length, channels, kernel);
}
extern "C" __global__ void qwen35_packed_conv_f32(const float *x, const float *w, float *out,
                                                const unsigned int *starts, int count, int length,
                                                int channels, int kernel) {
    qwen35_packed_conv(x, w, out, starts, count, length, channels, kernel);
}
extern "C" __global__ void qwen35_packed_delta_subwarp16_f32(const float *query, const float *key,
                                                           const float *value, const float *gate,
                                                           const float *beta, float *output,
                                                           const unsigned int *cu_seqlens,
                                                           int total_length, int heads,
                                                           int key_heads, int value_dim) {
    const int batch_head = blockIdx.x;
    const int head = batch_head % heads;
    const int batch = batch_head / heads;
    const int begin = cu_seqlens[batch];
    const int length = cu_seqlens[batch + 1] - begin;
    const int lane = threadIdx.x & 15;
    const int column = blockIdx.y * 8 + (threadIdx.x >> 4);
    if (column >= value_dim)
        return;
    const unsigned mask = __activemask();
    float state[2][4] = {{0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}};
    for (int token = 0; token < length; ++token) {
        const int token_head = (begin + token) * heads + head;
        const int qk_head = (begin + token) * key_heads + head / (heads / key_heads);
        const int output_offset = token_head;
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

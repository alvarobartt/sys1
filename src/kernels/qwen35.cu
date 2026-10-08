#include <cuda_bf16.h>

template <typename T>
__device__ void qwen35_conv(
    const T* input,
    const T* weights,
    T* output,
    int count,
    int length,
    int channels,
    int kernel) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    const int channel = index % channels;
    const int token = index / channels % length;
    const int batch = index / (channels * length);
    T value = T(0.f);
    for (int tap = 0; tap < kernel; ++tap) {
        const int source_token = token + tap - kernel + 1;
        if (source_token >= 0) {
            const float product = __fmul_rn(
                float(input[(batch * length + source_token) * channels + channel]),
                float(weights[channel * kernel + tap]));
            value = T(__fadd_rn(float(value), float(T(product))));
        }
    }
    const float x = float(value);
    output[index] = T(x / (1.f + expf(-x)));
}

extern "C" __global__ void qwen35_conv_bf16(
    const __nv_bfloat16* input,
    const __nv_bfloat16* weights,
    __nv_bfloat16* output,
    int count,
    int length,
    int channels,
    int kernel) {
    qwen35_conv(input, weights, output, count, length, channels, kernel);
}

extern "C" __global__ void qwen35_conv_f32(
    const float* input,
    const float* weights,
    float* output,
    int count,
    int length,
    int channels,
    int kernel) {
    qwen35_conv(input, weights, output, count, length, channels, kernel);
}

extern "C" __global__ void qwen35_delta_f32(
    const float* input,
    float* output,
    int length,
    int value_dim) {
    const int head = blockIdx.x;
    const int lane = threadIdx.x & 31;
    const int column = blockIdx.y * 4 + (threadIdx.x >> 5);
    if (column >= value_dim) return;

    float state[4] = {0.f, 0.f, 0.f, 0.f};
    for (int token = 0; token < length; ++token) {
        const int offset = (head * length + token);
        const float* token_input = input + offset * (258 + value_dim);
        const float decay = expf(token_input[256 + value_dim]);
        const float weight = token_input[257 + value_dim];
        float keys[4];
        float remembered = 0.f;
#pragma unroll
        for (int part = 0; part < 4; ++part) {
            keys[part] = token_input[128 + part * 32 + lane];
            state[part] *= decay;
            remembered += keys[part] * state[part];
        }
#pragma unroll
        for (int step = 16; step > 0; step >>= 1) {
            remembered += __shfl_down_sync(0xffffffff, remembered, step);
        }
        const float delta =
            (token_input[256 + column] - __shfl_sync(0xffffffff, remembered, 0)) * weight;
        float answer = 0.f;
#pragma unroll
        for (int part = 0; part < 4; ++part) {
            state[part] += keys[part] * delta;
            answer += token_input[part * 32 + lane] * state[part];
        }
#pragma unroll
        for (int step = 16; step > 0; step >>= 1) {
            answer += __shfl_down_sync(0xffffffff, answer, step);
        }
        if (lane == 0) {
            output[offset * value_dim + column] = answer;
        }
    }
}

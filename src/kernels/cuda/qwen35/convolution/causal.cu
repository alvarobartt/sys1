#include <cuda_bf16.h>
#include <float.h>
#include <math.h>
#include <mma.h>

template <typename T>
__device__ void qwen35_conv(
    const T *input, const T *weights, T *output, int count, int length, int channels, int kernel) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count)
        return;
    const int channel = index % channels;
    const int token = index / channels % length;
    const int batch = index / (channels * length);

    float value = 0.f;
    for (int tap = 0; tap < kernel; ++tap) {
        const int source_token = token + tap - kernel + 1;
        if (source_token >= 0) {
            const float product =
                __fmul_rn(float(input[(batch * length + source_token) * channels + channel]),
                    float(weights[channel * kernel + tap]));
            value = __fadd_rn(value, product);
        }
    }

    const float x = float(T(value));
    output[index] = T(x / (1.f + expf(-x)));
}

extern "C" __global__ void qwen35_conv_bf16(const __nv_bfloat16 *input,
    const __nv_bfloat16 *weights, __nv_bfloat16 *output, int count, int length, int channels,
    int kernel) {
    qwen35_conv(input, weights, output, count, length, channels, kernel);
}

extern "C" __global__ void qwen35_conv_f32(const float *input, const float *weights, float *output,
    int count, int length, int channels, int kernel) {
    qwen35_conv(input, weights, output, count, length, channels, kernel);
}

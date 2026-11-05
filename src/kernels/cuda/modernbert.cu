#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <math.h>

extern "C" __global__ void modernbert_projection_f16(
    const __half *input, const __half *weight, __half *output, int inner, int columns) {
    const int row = blockIdx.x;
    const int lane = threadIdx.x & 31;
    const int column = blockIdx.y * 8 + (threadIdx.x >> 5);
    if (column >= columns)
        return;
    float running = 0.f, correction = 0.f;
    bool exceptional = false;
    for (int k = lane; k < inner; k += 32) {
        const float product =
            __fmul_rn(float(input[row * inner + k]), float(weight[column * inner + k]));
        exceptional |= !isfinite(product);
        const float updated = __fadd_rn(running, product);
        const float virtual_product = __fsub_rn(updated, running);
        const float error = __fadd_rn(__fsub_rn(running, __fsub_rn(updated, virtual_product)),
            __fsub_rn(product, virtual_product));
        correction = __fadd_rn(correction, error);
        running = updated;
    }
    double sum = __dadd_rn(double(running), double(correction));
    if (__any_sync(0xffffffff, exceptional)) {
        sum = 0.;
        for (int k = lane; k < inner; k += 32) {
            sum = __fma_rn(double(float(input[row * inner + k])),
                double(float(weight[column * inner + k])), sum);
        }
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum = __dadd_rn(sum, __shfl_down_sync(0xffffffff, sum, offset));
    }
    if (lane == 0)
        output[row * columns + column] = __float2half_rn(__double2float_rn(sum));
}

template <typename T> __device__ void geglu(const T *input, T *output, long long count, int inner) {
    const long long index = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count)
        return;
    const long long row = index / inner;
    const int column = index % inner;
    const long long offset = row * (2LL * inner) + column;
    const float value = float(input[offset]);
    const T activated = T(value * normcdff(value));
    output[index] = T(float(activated) * float(input[offset + inner]));
}

extern "C" __global__ void modernbert_geglu_f32(
    const float *input, float *output, long long count, int inner) {
    geglu(input, output, count, inner);
}

extern "C" __global__ void modernbert_geglu_f16(
    const __half *input, __half *output, long long count, int inner) {
    geglu(input, output, count, inner);
}

extern "C" __global__ void modernbert_geglu_bf16(
    const __nv_bfloat16 *input, __nv_bfloat16 *output, long long count, int inner) {
    geglu(input, output, count, inner);
}

template <typename T>
__device__ void rope_qkv(const T *input, const float *cos, const float *sin, T *output,
    long long pairs, int length, int heads, int dim) {
    const long long index = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= pairs)
        return;
    const int half = dim / 2;
    const int column = index % half;
    const int head = (index / half) % heads;
    const long long token = index / ((long long)heads * half);
    const int position = token % length;
    const long long input_base = token * (3LL * heads * dim) + head * dim + column;
    const long long output_base = token * ((long long)heads * dim) + head * dim + column;
    const long long plane = pairs * 2;
    const float c = cos[position * half + column];
    const float s = sin[position * half + column];

    const float q0 = float(input[input_base]);
    const float q1 = float(input[input_base + half]);
    output[output_base] = T(q0 * c - q1 * s);
    output[output_base + half] = T(q0 * s + q1 * c);

    const long long key_base = input_base + (long long)heads * dim;
    const float k0 = float(input[key_base]);
    const float k1 = float(input[key_base + half]);
    output[plane + output_base] = T(k0 * c - k1 * s);
    output[plane + output_base + half] = T(k0 * s + k1 * c);

    const long long value_base = key_base + (long long)heads * dim;
    output[2 * plane + output_base] = input[value_base];
    output[2 * plane + output_base + half] = input[value_base + half];
}

extern "C" __global__ void modernbert_rope_qkv_f16(const __half *input, const float *cos,
    const float *sin, __half *output, long long pairs, int length, int heads, int dim) {
    rope_qkv(input, cos, sin, output, pairs, length, heads, dim);
}

extern "C" __global__ void modernbert_rope_qkv_bf16(const __nv_bfloat16 *input, const float *cos,
    const float *sin, __nv_bfloat16 *output, long long pairs, int length, int heads, int dim) {
    rope_qkv(input, cos, sin, output, pairs, length, heads, dim);
}

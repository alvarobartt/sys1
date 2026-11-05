#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <math.h>

template <bool Merged>
__device__ __forceinline__ void qwen35_full_qkv_finish_impl(
    const __nv_bfloat16 *qg, const __nv_bfloat16 *keys, const __nv_bfloat16 *values,
    const __nv_bfloat16 *qw, const __nv_bfloat16 *kw, const __nv_bfloat16 *cosine,
    const __nv_bfloat16 *sine, const __nv_bfloat16 *table, __nv_bfloat16 *output, int width,
    int rotary, int heads, int kv_heads, int length, float epsilon) {
    const int head = blockIdx.x % (heads + kv_heads), token = blockIdx.x / (heads + kv_heads);
    const bool query = head < heads;
    const int h = query ? head : head - heads;
    const size_t qspan = size_t(heads) * width, kspan = size_t(kv_heads) * width;
    const size_t stride = Merged ? 2 * qspan + 2 * kspan : 2 * qspan;
    const __nv_bfloat16 *input = query
                                     ? qg + token * stride + size_t(h) * 2 * width
                                     : (Merged ? qg + token * stride + 2 * qspan + size_t(h) * width
                                               : keys + token * kspan + size_t(h) * width);
    const __nv_bfloat16 *weight = query ? qw : kw;
    const size_t total_tokens = size_t(gridDim.x) / (heads + kv_heads);
    const size_t qcount = total_tokens * qspan, kcount = total_tokens * kspan;
    __nv_bfloat16 *destination = query ? output + token * qspan + size_t(h) * width
                                       : output + qcount + token * kspan + size_t(h) * width;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, position = token % length;
    float sum = 0.f;
    for (int column = threadIdx.x; column < width; column += blockDim.x) {
        const float value = float(input[column]);
        sum += value * value;
    }
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_xor_sync(0xffffffff, sum, offset, 32);
    __shared__ float warp_sums[32];
    __shared__ __nv_bfloat16 normalized[1024];
    if (lane == 0)
        warp_sums[warp] = sum;
    __syncthreads();
    sum = lane < (blockDim.x >> 5) ? warp_sums[lane] : 0.f;
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_xor_sync(0xffffffff, sum, offset, 32);
    const float scale = rsqrtf(sum / float(width) + epsilon);
    for (int column = threadIdx.x; column < width; column += blockDim.x)
        normalized[column] =
            __nv_bfloat16(scale * float(input[column]) * (1.f + float(weight[column])));
    __syncthreads();
    for (int column = threadIdx.x; column < width; column += blockDim.x) {
        __nv_bfloat16 value = normalized[column];
        if (column < rotary) {
            const int partner = column < rotary / 2 ? column + rotary / 2 : column - rotary / 2;
            const __nv_bfloat16 rotated =
                column < rotary / 2 ? __hneg(normalized[partner]) : normalized[partner];
            value = __hadd_rn(__hmul_rn(value, cosine[position * rotary + column]),
                              __hmul_rn(rotated, sine[position * rotary + column]));
        }
        destination[column] = value;
        if (query) {
            output[qcount + kcount + token * qspan + size_t(h) * width + column] =
                table[65536u + unsigned(__bfloat16_as_ushort(input[width + column]))];
        } else if constexpr (Merged) {
            output[2 * qcount + kcount + token * kspan + size_t(h) * width + column] =
                qg[token * stride + 2 * qspan + kspan + size_t(h) * width + column];
        }
    }
}

#define SYS1_FINISH_ARGS                                                                           \
    const __nv_bfloat16 *qg, const __nv_bfloat16 *k, const __nv_bfloat16 *v,                       \
        const __nv_bfloat16 *qw, const __nv_bfloat16 *kw, const __nv_bfloat16 *c,                  \
        const __nv_bfloat16 *s, const __nv_bfloat16 *t, __nv_bfloat16 *o, int d, int r, int h,     \
        int kh, int l, float eps
extern "C" __global__ void qwen35_full_qkv_finish_merged_bf16(SYS1_FINISH_ARGS) {
    qwen35_full_qkv_finish_impl<true>(qg, k, v, qw, kw, c, s, t, o, d, r, h, kh, l, eps);
}
extern "C" __global__ void qwen35_full_qkv_finish_separate_bf16(SYS1_FINISH_ARGS) {
    qwen35_full_qkv_finish_impl<false>(qg, k, v, qw, kw, c, s, t, o, d, r, h, kh, l, eps);
}
#undef SYS1_FINISH_ARGS

extern "C" __global__ void qwen35_full_finish_warp127(
    const __nv_bfloat16 *qg, const __nv_bfloat16 *keys, const __nv_bfloat16 *values,
    const __nv_bfloat16 *qw, const __nv_bfloat16 *kw, const __nv_bfloat16 *cosine,
    const __nv_bfloat16 *sine, const __nv_bfloat16 *table, __nv_bfloat16 *output, int width,
    int rotary, int heads, int kv_heads, int length, float epsilon) {
    const int lane = threadIdx.x & 31;
    const int head = blockIdx.y * 4 + (threadIdx.x >> 5), token = blockIdx.x;
    if (head >= heads + kv_heads)
        return;
    const bool query = head < heads;
    const int h = query ? head : head - heads;
    const size_t qspan = size_t(heads) * 256, kspan = size_t(kv_heads) * 256;
    const size_t qcount = size_t(gridDim.x) * qspan, kcount = size_t(gridDim.x) * kspan;
    const auto *input = query ? qg + size_t(token) * 2 * qspan + size_t(h) * 512
                              : keys + size_t(token) * kspan + size_t(h) * 256;
    const auto *weight = query ? qw : kw;
    auto *destination = query ? output + size_t(token) * qspan + size_t(h) * 256
                              : output + qcount + size_t(token) * kspan + size_t(h) * 256;
    float sums[8];
    __nv_bfloat16 normalized[8];
#pragma unroll
    for (int group = 0; group < 8; ++group) {
        const float value = float(input[group * 32 + lane]);
        float sum = 0.f;
        sum += value * value;
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            sum += __shfl_xor_sync(0xffffffff, sum, offset, 32);

        sums[group] = __shfl_sync(0xffffffff, sum, 0);
    }

    float sum = lane < 8 ? sums[lane] : 0.f;
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_xor_sync(0xffffffff, sum, offset, 32);
    const float scale = rsqrtf(sum / 256.f + epsilon);
#pragma unroll
    for (int group = 0; group < 8; ++group) {
        const int column = group * 32 + lane;
        normalized[group] =
            __nv_bfloat16(scale * float(input[column]) * (1.f + float(weight[column])));
    }
#pragma unroll
    for (int group = 0; group < 8; ++group) {
        const int column = group * 32 + lane;
        auto value = normalized[group];
        if (group < 2) {
            const auto rotated = group == 0 ? __hneg(normalized[1]) : normalized[0];
            const size_t position = size_t(token % length) * 64 + column;
            value =
                __hadd_rn(__hmul_rn(value, cosine[position]), __hmul_rn(rotated, sine[position]));
        }
        destination[column] = value;
        if (query)
            output[qcount + kcount + size_t(token) * qspan + size_t(h) * 256 + column] =
                table[65536u + unsigned(__bfloat16_as_ushort(input[256 + column]))];
    }
}

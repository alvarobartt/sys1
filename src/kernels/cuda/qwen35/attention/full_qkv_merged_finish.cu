#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <math.h>

extern "C" __global__ void qwen35_full_finish_merged_warp128(const __nv_bfloat16 *qg,
    const __nv_bfloat16 *keys, const __nv_bfloat16 *values, const __nv_bfloat16 *qw,
    const __nv_bfloat16 *kw, const __nv_bfloat16 *cosine, const __nv_bfloat16 *sine,
    const __nv_bfloat16 *table, __nv_bfloat16 *output, int width, int rotary, int heads,
    int kv_heads, int length, float epsilon) {
    const int lane = threadIdx.x & 31;
    const int head = blockIdx.y * 4 + (threadIdx.x >> 5), token = blockIdx.x;
    if (head >= heads + kv_heads)
        return;
    const bool query = head < heads;
    const int h = query ? head : head - heads;
    const size_t qspan = size_t(heads) * 256, kspan = size_t(kv_heads) * 256;
    const size_t qcount = size_t(gridDim.x) * qspan, kcount = size_t(gridDim.x) * kspan;
    const size_t stride = 2 * qspan + 2 * kspan;
    const auto *input =
        qg + size_t(token) * stride + (query ? size_t(h) * 512 : 2 * qspan + size_t(h) * 256);
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
    if (!query) {
#pragma unroll
        for (int group = 0; group < 8; ++group) {
            const int column = group * 32 + lane;
            output[2 * qcount + kcount + size_t(token) * kspan + size_t(h) * 256 + column] =
                qg[size_t(token) * stride + 2 * qspan + kspan + size_t(h) * 256 + column];
        }
    }
}

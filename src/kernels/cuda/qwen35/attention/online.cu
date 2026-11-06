#include <cuda_bf16.h>
#include <float.h>
#include <math.h>
#include <mma.h>

extern "C" __global__ void qwen35_online_attention_bf16(const __nv_bfloat16 *q,
    const __nv_bfloat16 *k, const __nv_bfloat16 *v, __nv_bfloat16 *output, int query_heads,
    int queries, int keys, int heads, int kv_heads, int width, float scale, int causal) {
    const int lane = threadIdx.x & 31;
    const int item = blockIdx.x * 4 + threadIdx.x / 32;
    if (item >= query_heads)
        return;
    const int head = item % heads;
    const int row = (item / heads) % queries;
    const int batch = item / (heads * queries);
    const int kv_head = head / (heads / kv_heads);
    const int channels = width / 32;
    float query[8], acc[8];
#pragma unroll
    for (int channel = 0; channel < 8; ++channel) {
        query[channel] = channel < channels ? float(q[item * width + lane + channel * 32]) : 0.f;
        acc[channel] = 0.f;
    }
    float maximum = -INFINITY, denominator = 0.f;
    const int count = causal ? row + 1 : keys;
    for (int key = 0; key < count; ++key) {
        const int offset = ((batch * keys + key) * kv_heads + kv_head) * width;
        float score = 0.f;
#pragma unroll
        for (int channel = 0; channel < 8; ++channel) {
            if (channel < channels)
                score = __fmaf_rn(query[channel], float(k[offset + lane + channel * 32]), score);
        }
#pragma unroll
        for (int step = 16; step > 0; step >>= 1)
            score = __fadd_rn(score, __shfl_down_sync(0xffffffff, score, step));
        score = __fmul_rn(__shfl_sync(0xffffffff, score, 0), scale);
        float alpha = 0.f, probability = 0.f;
        if (lane == 0) {
            const float next = fmaxf(maximum, score);
            alpha = maximum == -INFINITY ? 0.f : expf(__fsub_rn(maximum, next));
            probability = expf(__fsub_rn(score, next));
            denominator = __fmaf_rn(denominator, alpha, probability);
            maximum = next;
        }
        alpha = __shfl_sync(0xffffffff, alpha, 0);
        probability = __shfl_sync(0xffffffff, probability, 0);
#pragma unroll
        for (int channel = 0; channel < 8; ++channel) {
            if (channel < channels)
                acc[channel] = __fmaf_rn(probability, float(v[offset + lane + channel * 32]),
                    __fmul_rn(acc[channel], alpha));
        }
    }
    const float inverse = __shfl_sync(0xffffffff, lane == 0 ? __fdiv_rn(1.f, denominator) : 0.f, 0);
#pragma unroll
    for (int channel = 0; channel < 8; ++channel) {
        if (channel < channels)
            output[item * width + lane + channel * 32] =
                __float2bfloat16_rn(__fmul_rn(acc[channel], inverse));
    }
}

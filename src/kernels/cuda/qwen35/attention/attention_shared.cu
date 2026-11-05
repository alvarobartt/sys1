#include <cuda_bf16.h>
#include <math.h>

extern "C" __global__ void qwen35_online_attention_shared_bf16(const __nv_bfloat16 *q,
    const __nv_bfloat16 *k, const __nv_bfloat16 *v, __nv_bfloat16 *output, int query_heads,
    int queries, int keys, int heads, int kv_heads, int width, float scale, int causal) {
    const int lane = threadIdx.x & 31;
    const int item = blockIdx.x * 4 + threadIdx.x / 32;

    if (blockIdx.x * 4 >= query_heads)
        return;
    const int head = item % heads;
    const int row = (item / heads) % queries;
    const int batch = item / (heads * queries);
    const int kv_head = head / (heads / kv_heads);
    const int channels = width / 32;
    __shared__ __nv_bfloat16 shared_k[32 * 256];
    __shared__ __nv_bfloat16 shared_v[32 * 256];
    float query[8], acc[8];
#pragma unroll
    for (int channel = 0; channel < 8; ++channel) {
        query[channel] = channel < channels ? float(q[item * width + lane + channel * 32]) : 0.f;
        acc[channel] = 0.f;
    }
    float maximum = -INFINITY, denominator = 0.f;
    const int count = causal ? row + 1 : keys;
    for (int first_key = 0; first_key < count; first_key += 32) {
        const int tile_keys = min(32, count - first_key);
        for (int index = threadIdx.x; index < tile_keys * width; index += blockDim.x) {
            const int key = first_key + index / width;
            const int offset = ((batch * keys + key) * kv_heads + kv_head) * width + index % width;
            shared_k[index] = k[offset];
            shared_v[index] = v[offset];
        }
        __syncthreads();
        for (int tile_key = 0; tile_key < tile_keys; ++tile_key) {
            const int offset = tile_key * width;
            float score = 0.f;
#pragma unroll
            for (int channel = 0; channel < 8; ++channel) {
                if (channel < channels)
                    score = __fmaf_rn(
                        query[channel], float(shared_k[offset + lane + channel * 32]), score);
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
                    acc[channel] =
                        __fmaf_rn(probability, float(shared_v[offset + lane + channel * 32]),
                            __fmul_rn(acc[channel], alpha));
            }
        }
        __syncthreads();
    }
    const float inverse = __shfl_sync(0xffffffff, lane == 0 ? __fdiv_rn(1.f, denominator) : 0.f, 0);
#pragma unroll
    for (int channel = 0; channel < 8; ++channel) {
        if (channel < channels)
            output[item * width + lane + channel * 32] =
                __float2bfloat16_rn(__fmul_rn(acc[channel], inverse));
    }
}

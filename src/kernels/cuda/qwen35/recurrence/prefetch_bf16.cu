#include <cuda_runtime.h>
#include <cuda_bf16.h>

#if __CUDA_ARCH__ >= 800
template <int Bytes>
__device__ __forceinline__ void qwen35_bfstaged_copy(void *dst, const void *src, bool valid) {
    const unsigned address = static_cast<unsigned>(__cvta_generic_to_shared(dst));
    const int available = valid ? Bytes : 0;
    asm volatile("cp.async.ca.shared.global [%0], [%1], %2, %3;" ::"r"(address), "l"(src),
                 "n"(Bytes), "r"(available)
                 : "memory");
}
template <int Columns> struct __align__(16) Sys1BfStagedSlot {
    float query[8][128], key[8][128], value[8][4 * Columns], decay[8], beta[8];
};
template <int Columns>
__device__ __forceinline__ void qwen35_bfstaged_stage(Sys1BfStagedSlot<Columns> &slot,
    const float *query, const float *key, const float *value, const float *decay, const float *beta,
    int begin, int first, int length, int head, int key_head, int heads, int key_heads,
    int first_column) {
    for (int i = threadIdx.x; i < 8 * 32; i += 128) {
        const int t = i / 32, k = (i % 32) * 4;
        const bool valid = first + t < length;
        const int token = begin + (valid ? first + t : 0);
        const int offset = (token * key_heads + key_head) * 128 + k;
        qwen35_bfstaged_copy<16>(&slot.query[t][k], query + offset, valid);
        qwen35_bfstaged_copy<16>(&slot.key[t][k], key + offset, valid);
    }
    for (int i = threadIdx.x; i < 8 * Columns; i += 128) {
        const int t = i / Columns, col = (i % Columns) * 4;
        const bool valid = first + t < length;
        const int token = begin + (valid ? first + t : 0);
        qwen35_bfstaged_copy<16>(
            &slot.value[t][col], value + (token * heads + head) * 128 + first_column + col, valid);
    }
    if (threadIdx.x < 8) {
        const int t = threadIdx.x;
        const bool valid = first + t < length;
        const int offset = (begin + (valid ? first + t : 0)) * heads + head;
        qwen35_bfstaged_copy<4>(&slot.decay[t], decay + offset, valid);
        qwen35_bfstaged_copy<4>(&slot.beta[t], beta + offset, valid);
    }
    asm volatile("cp.async.commit_group;" ::: "memory");
}
template <int Columns>
__device__ __forceinline__ void qwen35_bfstaged_body(Sys1BfStagedSlot<Columns> *slots,
    const float *query, const float *key, const float *value, const float *decay, const float *beta,
    __nv_bfloat16 *output, const unsigned int *starts, int heads, int key_heads) {
    const int head = blockIdx.x % heads, sequence = blockIdx.x / heads;
    const int begin = int(starts[sequence]), length = int(starts[sequence + 1]) - begin;
    const int lane = threadIdx.x & 31, first_column = blockIdx.y * (4 * Columns);
    const int column = (threadIdx.x >> 5) * Columns, key_head = head / (heads / key_heads);
    float state[Columns][4] = {};
    qwen35_bfstaged_stage<Columns>(slots[0], query, key, value, decay, beta, begin, 0, length, head,
        key_head, heads, key_heads, first_column);
    asm volatile("cp.async.wait_group 0;" ::: "memory");
    __syncthreads();
    for (int first = 0, bank = 0; first < length; first += 8, bank ^= 1) {
        const bool next = first + 8 < length;
        if (next)
            qwen35_bfstaged_stage<Columns>(slots[bank ^ 1], query, key, value, decay, beta, begin,
                first + 8, length, head, key_head, heads, key_heads, first_column);
        const int count = min(8, length - first);
        for (int t = 0; t < count; ++t) {
            const float d = slots[bank].decay[t], weight = slots[bank].beta[t];
            float keys[4], queries[4], remembered[Columns] = {};
#pragma unroll
            for (int part = 0; part < 4; ++part) {
                keys[part] = slots[bank].key[t][part * 32 + lane];
                queries[part] = slots[bank].query[t][part * 32 + lane];
#pragma unroll
                for (int c = 0; c < Columns; ++c) {
                    state[c][part] *= d;
                    remembered[c] += keys[part] * state[c][part];
                }
            }
#pragma unroll
            for (int step = 16; step > 0; step >>= 1) {
#pragma unroll
                for (int c = 0; c < Columns; ++c)
                    remembered[c] += __shfl_down_sync(0xffffffff, remembered[c], step);
            }
            float delta[Columns], answer[Columns] = {};
#pragma unroll
            for (int c = 0; c < Columns; ++c)
                delta[c] =
                    (slots[bank].value[t][column + c] - __shfl_sync(0xffffffff, remembered[c], 0)) *
                    weight;
#pragma unroll
            for (int part = 0; part < 4; ++part) {
#pragma unroll
                for (int c = 0; c < Columns; ++c) {
                    state[c][part] += keys[part] * delta[c];
                    answer[c] += queries[part] * state[c][part];
                }
            }
#pragma unroll
            for (int step = 16; step > 0; step >>= 1) {
#pragma unroll
                for (int c = 0; c < Columns; ++c)
                    answer[c] += __shfl_down_sync(0xffffffff, answer[c], step);
            }
            if (lane == 0) {
#pragma unroll
                for (int c = 0; c < Columns; ++c)
                    output[((begin + first + t) * heads + head) * 128 + first_column + column + c] =
                        __float2bfloat16_rn(answer[c]);
            }
        }
        if (next) {
            asm volatile("cp.async.wait_group 0;" ::: "memory");
            __syncthreads();
        }
    }
}
#endif

extern "C" __global__ void qwen35_packed_delta_prefetch8_columns4_preexp_bf16(const float *q,
    const float *k, const float *v, const float *decay, const float *beta, __nv_bfloat16 *output,
    const unsigned int *starts, int extent, int heads, int key_heads, int width) {
#if __CUDA_ARCH__ >= 800
    __shared__ Sys1BfStagedSlot<4> slots[2];
    qwen35_bfstaged_body<4>(slots, q, k, v, decay, beta, output, starts, heads, key_heads);
#endif
}

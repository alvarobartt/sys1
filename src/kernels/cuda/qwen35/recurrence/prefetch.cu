

#include <cuda_runtime.h>

template <int Columns>
__device__ __forceinline__ void
qwen35_prefetch_scalar_columns(const float *query, const float *key, const float *value,
                             const float *decay_values, const float *beta, float *output,
                             const unsigned int *starts, int extent, int heads, int key_heads,
                             int value_dim) {
    const int head = blockIdx.x % heads, sequence = blockIdx.x / heads;
    const int begin = int(starts[sequence]);
    const int length = int(starts[sequence + 1]) - begin;
    const int lane = threadIdx.x & 31;
    const int first = blockIdx.y * (4 * Columns) + (threadIdx.x >> 5) * Columns;
    if (first >= value_dim)
        return;
    float state[Columns][4] = {};
    for (int token = 0; token < length; ++token) {
        const int token_head = (begin + token) * heads + head;
        const int qk_head = (begin + token) * key_heads + head / (heads / key_heads);
        const float decay = decay_values[token_head], weight = beta[token_head];
        const float *token_query = query + qk_head * 128;
        const float *token_key = key + qk_head * 128;
        const float *token_value = value + token_head * value_dim;
        float keys[4], queries[4], remembered[Columns] = {};
#pragma unroll
        for (int part = 0; part < 4; ++part) {
            keys[part] = token_key[part * 32 + lane];
            queries[part] = token_query[part * 32 + lane];
#pragma unroll
            for (int col = 0; col < Columns; ++col) {
                state[col][part] *= decay;
                remembered[col] += keys[part] * state[col][part];
            }
        }
#pragma unroll
        for (int step = 16; step > 0; step >>= 1) {
#pragma unroll
            for (int col = 0; col < Columns; ++col)
                remembered[col] += __shfl_down_sync(0xffffffff, remembered[col], step);
        }
        float delta[Columns], answer[Columns] = {};
#pragma unroll
        for (int col = 0; col < Columns; ++col) {
            const float v = first + col < value_dim ? token_value[first + col] : 0.f;
            delta[col] = (v - __shfl_sync(0xffffffff, remembered[col], 0)) * weight;
        }
#pragma unroll
        for (int part = 0; part < 4; ++part) {
#pragma unroll
            for (int col = 0; col < Columns; ++col) {
                state[col][part] += keys[part] * delta[col];
                answer[col] += queries[part] * state[col][part];
            }
        }
#pragma unroll
        for (int step = 16; step > 0; step >>= 1) {
#pragma unroll
            for (int col = 0; col < Columns; ++col)
                answer[col] += __shfl_down_sync(0xffffffff, answer[col], step);
        }
        if (lane == 0) {
#pragma unroll
            for (int col = 0; col < Columns; ++col)
                if (first + col < value_dim)
                    output[token_head * value_dim + first + col] = answer[col];
        }
    }
}

#if __CUDA_ARCH__ >= 800
template <int Bytes>
__device__ __forceinline__ void qwen35_prefetch_copy(void *destination, const void *source,
                                                   bool valid) {
    const unsigned address = static_cast<unsigned>(__cvta_generic_to_shared(destination));
    const int available = valid ? Bytes : 0;
    asm volatile("cp.async.ca.shared.global [%0], [%1], %2, %3;" ::"r"(address), "l"(source),
                 "n"(Bytes), "r"(available)
                 : "memory");
}
template <int Tile> struct __align__(16) Sys1PrefetchSlot {
    float query[Tile][128], key[Tile][128], value[Tile][8], decay[Tile], beta[Tile];
};
template <int Tile>
__device__ __forceinline__ void
qwen35_prefetch_stage(Sys1PrefetchSlot<Tile> &slot, const float *query, const float *key,
                    const float *value, const float *decay, const float *beta, int begin, int first,
                    int length, int head, int key_head, int heads, int key_heads,
                    int first_column) {
    for (int i = threadIdx.x; i < Tile * 32; i += 128) {
        const int t = i / 32, k = (i % 32) * 4;
        const bool valid = first + t < length;
        const int global_token = begin + (valid ? first + t : 0);
        const int offset = (global_token * key_heads + key_head) * 128 + k;
        qwen35_prefetch_copy<16>(&slot.query[t][k], query + offset, valid);
        qwen35_prefetch_copy<16>(&slot.key[t][k], key + offset, valid);
    }
    for (int i = threadIdx.x; i < Tile * 2; i += 128) {
        const int t = i / 2, col = (i % 2) * 4;
        const bool valid = first + t < length;
        const int global_token = begin + (valid ? first + t : 0);
        qwen35_prefetch_copy<16>(&slot.value[t][col],
                               value + (global_token * heads + head) * 128 + first_column + col,
                               valid);
    }
    if (threadIdx.x < Tile) {
        const int t = threadIdx.x;
        const bool valid = first + t < length;
        const int offset = (begin + (valid ? first + t : 0)) * heads + head;
        qwen35_prefetch_copy<4>(&slot.decay[t], decay + offset, valid);
        qwen35_prefetch_copy<4>(&slot.beta[t], beta + offset, valid);
    }
    asm volatile("cp.async.commit_group;" ::: "memory");
}
template <int Tile>
__device__ __forceinline__ void
qwen35_prefetch_body(Sys1PrefetchSlot<Tile> *slots, const float *query, const float *key,
                   const float *value, const float *decay, const float *beta, float *output,
                   const unsigned int *starts, int heads, int key_heads) {
    const int head = blockIdx.x % heads, sequence = blockIdx.x / heads;
    const int begin = int(starts[sequence]), length = int(starts[sequence + 1]) - begin;
    const int lane = threadIdx.x & 31, first_column = blockIdx.y * 8;
    const int column = (threadIdx.x >> 5) * 2;
    const int key_head = head / (heads / key_heads);
    float state[2][4] = {};
    qwen35_prefetch_stage<Tile>(slots[0], query, key, value, decay, beta, begin, 0, length, head,
                              key_head, heads, key_heads, first_column);
    asm volatile("cp.async.wait_group 0;" ::: "memory");
    __syncthreads();
    for (int first = 0, bank = 0; first < length; first += Tile, bank ^= 1) {
        const bool next = first + Tile < length;
        if (next)
            qwen35_prefetch_stage<Tile>(slots[bank ^ 1], query, key, value, decay, beta, begin,
                                      first + Tile, length, head, key_head, heads, key_heads,
                                      first_column);
        const int count = min(Tile, length - first);
        for (int t = 0; t < count; ++t) {
            const float d = slots[bank].decay[t], weight = slots[bank].beta[t];
            float keys[4], queries[4], remembered[2] = {};
#pragma unroll
            for (int part = 0; part < 4; ++part) {
                keys[part] = slots[bank].key[t][part * 32 + lane];
                queries[part] = slots[bank].query[t][part * 32 + lane];
#pragma unroll
                for (int c = 0; c < 2; ++c) {
                    state[c][part] *= d;
                    remembered[c] += keys[part] * state[c][part];
                }
            }
#pragma unroll
            for (int step = 16; step > 0; step >>= 1) {
#pragma unroll
                for (int c = 0; c < 2; ++c)
                    remembered[c] += __shfl_down_sync(0xffffffff, remembered[c], step);
            }
            float delta[2], answer[2] = {};
#pragma unroll
            for (int c = 0; c < 2; ++c)
                delta[c] =
                    (slots[bank].value[t][column + c] - __shfl_sync(0xffffffff, remembered[c], 0)) *
                    weight;
#pragma unroll
            for (int part = 0; part < 4; ++part) {
#pragma unroll
                for (int c = 0; c < 2; ++c) {
                    state[c][part] += keys[part] * delta[c];
                    answer[c] += queries[part] * state[c][part];
                }
            }
#pragma unroll
            for (int step = 16; step > 0; step >>= 1) {
#pragma unroll
                for (int c = 0; c < 2; ++c)
                    answer[c] += __shfl_down_sync(0xffffffff, answer[c], step);
            }
            if (lane == 0) {
#pragma unroll
                for (int c = 0; c < 2; ++c)
                    output[((begin + first + t) * heads + head) * 128 + first_column + column + c] =
                        answer[c];
            }
        }
        if (next) {
            asm volatile("cp.async.wait_group 0;" ::: "memory");
            __syncthreads();
        }
    }
}
#endif
extern "C" __global__ void qwen35_packed_delta_prefetch8_preexp_f32(
    const float *q, const float *k, const float *v, const float *decay, const float *beta,
    float *output, const unsigned int *starts, int extent, int heads, int key_heads, int width) {
#if __CUDA_ARCH__ >= 800
    if (width == 128) {
        __shared__ Sys1PrefetchSlot<8> slots[2];
        qwen35_prefetch_body<8>(slots, q, k, v, decay, beta, output, starts, heads, key_heads);
        return;
    }
#endif
    qwen35_prefetch_scalar_columns<2>(q, k, v, decay, beta, output, starts, extent, heads, key_heads,
                                    width);
}

#include <stdint.h>

__device__ __forceinline__ void qwen35_uniform_prefetch_scalar(const float *query, const float *key,
                                                             const float *value, const float *decay,
                                                             const float *beta, float *output,
                                                             int length, int heads, int key_heads,
                                                             int width) {
    const int bh = blockIdx.x, head = bh % heads, sequence = bh / heads, lane = threadIdx.x & 31;
    const int first = blockIdx.y * 8 + (threadIdx.x >> 5) * 2;
    if (first >= width)
        return;
    float state[2][4] = {};
    for (int t = 0; t < length; ++t) {
        const int th = (sequence * length + t) * heads + head;
        const int qh = (sequence * length + t) * key_heads + head / (heads / key_heads);
        const float d = decay[th], weight = beta[th];
        float keys[4], queries[4], remembered[2] = {};
#pragma unroll
        for (int p = 0; p < 4; ++p) {
            keys[p] = key[qh * 128 + p * 32 + lane];
            queries[p] = query[qh * 128 + p * 32 + lane];
#pragma unroll
            for (int c = 0; c < 2; ++c) {
                state[c][p] *= d;
                remembered[c] += keys[p] * state[c][p];
            }
        }
#pragma unroll
        for (int step = 16; step > 0; step >>= 1) {
#pragma unroll
            for (int c = 0; c < 2; ++c)
                remembered[c] += __shfl_down_sync(0xffffffff, remembered[c], step);
        }
        float delta[2], answer[2] = {};
#pragma unroll
        for (int c = 0; c < 2; ++c)
            delta[c] = ((first + c < width ? value[th * width + first + c] : 0.f) -
                        __shfl_sync(0xffffffff, remembered[c], 0)) *
                       weight;
#pragma unroll
        for (int p = 0; p < 4; ++p) {
#pragma unroll
            for (int c = 0; c < 2; ++c) {
                state[c][p] += keys[p] * delta[c];
                answer[c] += queries[p] * state[c][p];
            }
        }
#pragma unroll
        for (int step = 16; step > 0; step >>= 1) {
#pragma unroll
            for (int c = 0; c < 2; ++c)
                answer[c] += __shfl_down_sync(0xffffffff, answer[c], step);
        }
        if (lane == 0) {
#pragma unroll
            for (int c = 0; c < 2; ++c)
                if (first + c < width)
                    output[(bh * length + t) * width + first + c] = answer[c];
        }
    }
}

#if __CUDA_ARCH__ >= 800
__device__ __forceinline__ void qwen35_uniform_prefetch_body(Sys1PrefetchSlot<8> *slots,
                                                           const float *query, const float *key,
                                                           const float *value, const float *decay,
                                                           const float *beta, float *output,
                                                           int length, int heads, int key_heads) {
    const int bh = blockIdx.x, head = bh % heads, sequence = bh / heads, begin = sequence * length;
    const int lane = threadIdx.x & 31, first_column = blockIdx.y * 8,
              column = (threadIdx.x >> 5) * 2;
    const int key_head = head / (heads / key_heads);
    float state[2][4] = {};
    qwen35_prefetch_stage<8>(slots[0], query, key, value, decay, beta, begin, 0, length, head,
                           key_head, heads, key_heads, first_column);
    asm volatile("cp.async.wait_group 0;" ::: "memory");
    __syncthreads();
    for (int first = 0, bank = 0; first < length; first += 8, bank ^= 1) {
        const bool next = first + 8 < length;
        if (next)
            qwen35_prefetch_stage<8>(slots[bank ^ 1], query, key, value, decay, beta, begin,
                                   first + 8, length, head, key_head, heads, key_heads,
                                   first_column);
        const int count = min(8, length - first);
        for (int t = 0; t < count; ++t) {
            const float d = slots[bank].decay[t], weight = slots[bank].beta[t];
            float keys[4], queries[4], remembered[2] = {};
#pragma unroll
            for (int p = 0; p < 4; ++p) {
                keys[p] = slots[bank].key[t][p * 32 + lane];
                queries[p] = slots[bank].query[t][p * 32 + lane];
#pragma unroll
                for (int c = 0; c < 2; ++c) {
                    state[c][p] *= d;
                    remembered[c] += keys[p] * state[c][p];
                }
            }
#pragma unroll
            for (int step = 16; step > 0; step >>= 1) {
#pragma unroll
                for (int c = 0; c < 2; ++c)
                    remembered[c] += __shfl_down_sync(0xffffffff, remembered[c], step);
            }
            float delta[2], answer[2] = {};
#pragma unroll
            for (int c = 0; c < 2; ++c)
                delta[c] =
                    (slots[bank].value[t][column + c] - __shfl_sync(0xffffffff, remembered[c], 0)) *
                    weight;
#pragma unroll
            for (int p = 0; p < 4; ++p) {
#pragma unroll
                for (int c = 0; c < 2; ++c) {
                    state[c][p] += keys[p] * delta[c];
                    answer[c] += queries[p] * state[c][p];
                }
            }
#pragma unroll
            for (int step = 16; step > 0; step >>= 1) {
#pragma unroll
                for (int c = 0; c < 2; ++c)
                    answer[c] += __shfl_down_sync(0xffffffff, answer[c], step);
            }
            if (lane == 0) {
#pragma unroll
                for (int c = 0; c < 2; ++c)
                    output[((bh * length + first + t) * 128) + first_column + column + c] =
                        answer[c];
            }
        }
        if (next) {
            asm volatile("cp.async.wait_group 0;" ::: "memory");
            __syncthreads();
        }
    }
}
#endif

extern "C" __global__ void
qwen35_uniform_delta_prefetch8_preexp_f32(const float *q, const float *k, const float *v,
                                        const float *decay, const float *beta, float *output,
                                        int length, int heads, int key_heads, int width) {
#if __CUDA_ARCH__ >= 800

    if (width == 128 && ((reinterpret_cast<uintptr_t>(q) | reinterpret_cast<uintptr_t>(k) |
                          reinterpret_cast<uintptr_t>(v)) &
                         15u) == 0) {
        __shared__ Sys1PrefetchSlot<8> slots[2];
        qwen35_uniform_prefetch_body(slots, q, k, v, decay, beta, output, length, heads, key_heads);
        return;
    }
#endif
    qwen35_uniform_prefetch_scalar(q, k, v, decay, beta, output, length, heads, key_heads, width);
}

#if __CUDA_ARCH__ >= 800
template <int Tile> struct __align__(16) Sys1PrefetchColumns4Slot {
    float query[Tile][128], key[Tile][128], value[Tile][16], decay[Tile], beta[Tile];
};
template <int Tile>
__device__ __forceinline__ void
qwen35_prefetch_columns4_stage(Sys1PrefetchColumns4Slot<Tile> &slot, const float *query,
                             const float *key, const float *value, const float *decay,
                             const float *beta, int begin, int first, int length, int head,
                             int key_head, int heads, int key_heads, int first_column) {
    for (int i = threadIdx.x; i < Tile * 32; i += 128) {
        const int t = i / 32, k = (i % 32) * 4;
        const bool valid = first + t < length;
        const int global_token = begin + (valid ? first + t : 0);
        const int offset = (global_token * key_heads + key_head) * 128 + k;
        qwen35_prefetch_copy<16>(&slot.query[t][k], query + offset, valid);
        qwen35_prefetch_copy<16>(&slot.key[t][k], key + offset, valid);
    }
    for (int i = threadIdx.x; i < Tile * 4; i += 128) {
        const int t = i / 4, col = (i % 4) * 4;
        const bool valid = first + t < length;
        const int global_token = begin + (valid ? first + t : 0);
        qwen35_prefetch_copy<16>(&slot.value[t][col],
                               value + (global_token * heads + head) * 128 + first_column + col,
                               valid);
    }
    if (threadIdx.x < Tile) {
        const int t = threadIdx.x;
        const bool valid = first + t < length;
        const int offset = (begin + (valid ? first + t : 0)) * heads + head;
        qwen35_prefetch_copy<4>(&slot.decay[t], decay + offset, valid);
        qwen35_prefetch_copy<4>(&slot.beta[t], beta + offset, valid);
    }
    asm volatile("cp.async.commit_group;" ::: "memory");
}
template <int Tile>
__device__ __forceinline__ void
qwen35_prefetch_columns4_body(Sys1PrefetchColumns4Slot<Tile> *slots, const float *query,
                            const float *key, const float *value, const float *decay,
                            const float *beta, float *output, const unsigned int *starts, int heads,
                            int key_heads) {
    const int head = blockIdx.x % heads, sequence = blockIdx.x / heads;
    const int begin = int(starts[sequence]), length = int(starts[sequence + 1]) - begin;
    const int lane = threadIdx.x & 31, first_column = blockIdx.y * 16;
    const int column = (threadIdx.x >> 5) * 4;
    const int key_head = head / (heads / key_heads);
    float state[4][4] = {};
    qwen35_prefetch_columns4_stage<Tile>(slots[0], query, key, value, decay, beta, begin, 0, length,
                                       head, key_head, heads, key_heads, first_column);
    asm volatile("cp.async.wait_group 0;" ::: "memory");
    __syncthreads();
    for (int first = 0, bank = 0; first < length; first += Tile, bank ^= 1) {
        const bool next = first + Tile < length;
        if (next)
            qwen35_prefetch_columns4_stage<Tile>(slots[bank ^ 1], query, key, value, decay, beta,
                                               begin, first + Tile, length, head, key_head, heads,
                                               key_heads, first_column);
        const int count = min(Tile, length - first);
        for (int t = 0; t < count; ++t) {
            const float d = slots[bank].decay[t], weight = slots[bank].beta[t];
            float keys[4], queries[4], remembered[4] = {};
#pragma unroll
            for (int part = 0; part < 4; ++part) {
                keys[part] = slots[bank].key[t][part * 32 + lane];
                queries[part] = slots[bank].query[t][part * 32 + lane];
#pragma unroll
                for (int c = 0; c < 4; ++c) {
                    state[c][part] *= d;
                    remembered[c] += keys[part] * state[c][part];
                }
            }
#pragma unroll
            for (int step = 16; step > 0; step >>= 1) {
#pragma unroll
                for (int c = 0; c < 4; ++c)
                    remembered[c] += __shfl_down_sync(0xffffffff, remembered[c], step);
            }
            float delta[4], answer[4] = {};
#pragma unroll
            for (int c = 0; c < 4; ++c)
                delta[c] =
                    (slots[bank].value[t][column + c] - __shfl_sync(0xffffffff, remembered[c], 0)) *
                    weight;
#pragma unroll
            for (int part = 0; part < 4; ++part) {
#pragma unroll
                for (int c = 0; c < 4; ++c) {
                    state[c][part] += keys[part] * delta[c];
                    answer[c] += queries[part] * state[c][part];
                }
            }
#pragma unroll
            for (int step = 16; step > 0; step >>= 1) {
#pragma unroll
                for (int c = 0; c < 4; ++c)
                    answer[c] += __shfl_down_sync(0xffffffff, answer[c], step);
            }
            if (lane == 0) {
#pragma unroll
                for (int c = 0; c < 4; ++c)
                    output[((begin + first + t) * heads + head) * 128 + first_column + column + c] =
                        answer[c];
            }
        }
        if (next) {
            asm volatile("cp.async.wait_group 0;" ::: "memory");
            __syncthreads();
        }
    }
}
#endif
extern "C" __global__ void qwen35_packed_delta_prefetch8_columns4_preexp_f32(
    const float *q, const float *k, const float *v, const float *decay, const float *beta,
    float *output, const unsigned int *starts, int extent, int heads, int key_heads, int width) {
#if __CUDA_ARCH__ >= 800
    if (width == 128) {
        __shared__ Sys1PrefetchColumns4Slot<8> slots[2];
        qwen35_prefetch_columns4_body<8>(slots, q, k, v, decay, beta, output, starts, heads,
                                       key_heads);
        return;
    }
#endif
    qwen35_prefetch_scalar_columns<4>(q, k, v, decay, beta, output, starts, extent, heads, key_heads,
                                    width);
}

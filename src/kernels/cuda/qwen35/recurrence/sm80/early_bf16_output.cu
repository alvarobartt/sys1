#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
__device__ __forceinline__ void qwen35_c64bfout_load4(unsigned *out, const __half *pointer) {
    const unsigned address = unsigned(__cvta_generic_to_shared(pointer));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3},[%4];"
                 : "=r"(out[0]), "=r"(out[1]), "=r"(out[2]), "=r"(out[3])
                 : "r"(address)
                 : "memory");
}
__device__ __forceinline__ void qwen35_c64bfout_load2(unsigned *out, const __half *pointer) {
    const unsigned address = unsigned(__cvta_generic_to_shared(pointer));
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1},[%2];"
                 : "=r"(out[0]), "=r"(out[1])
                 : "r"(address)
                 : "memory");
}
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math.h>

__device__ __forceinline__ void qwen35_c64bfout_mma(float *c, const unsigned *a, const unsigned *b) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

template <int Phase>
__device__ __forceinline__ float qwen35_c64bfout_a(int r, int k, const float *q, const float *key,
                                                 const float *w, const float *causal,
                                                 const float *decay, int instance, int begin,
                                                 int length, int qk_base, int qk_stride) {
    if constexpr (Phase == 0)
        return w[(instance * 64 + r) * 128 + k];
    if constexpr (Phase == 1)
        return begin + r < length
                   ? __fmul_rn(decay[instance * 128 + r], q[qk_base + (begin + r) * qk_stride + k])
                   : 0.f;
    if constexpr (Phase == 2)
        return causal[(instance * 64 + r) * 64 + k];
    if constexpr (Phase == 3)
        return begin + k < length ? __fmul_rn(decay[instance * 128 + 64 + k],
                                              key[qk_base + (begin + k) * qk_stride + r])
                                  : 0.f;
    return 0.f;
}

__device__ __forceinline__ int qwen35_c64bfout_swizzle(int row, int col, int stride) {
    return row * stride + (col ^ ((row & 7) << 3));
}

template <int Phase, int N, bool ReuseB = false>
__device__ __forceinline__ void
qwen35_c64bfout_product(const float *q, const float *key, const float *w, const float *causal,
                      const float *decay, float *updates, float (&carry)[4], float (&out)[4],
                      __half *ahalf, __half *alow, __half *bhalf, __half *blow,
                      int *cached_b_unsafe, int instance, int begin, int length, int qk_base,
                      int qk_stride) {
    constexpr int Rows = Phase == 3 ? 128 : 64, Inner = Phase < 2 ? 128 : 64, S = N + 4;
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32, group = lane / 4, tid = lane % 4;
    const int r = (warp / (N / 8)) * 16, c = (warp % (N / 8)) * 8;
    const bool active = warp < (Rows / 16) * (N / 8);
    float high[4] = {}, cross[4] = {}, saved[4] = {};
    if (active) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            if constexpr (Phase == 0)
                saved[i] = updates[(r + group + (i / 2) * 8) * S + c + tid * 2 + i % 2];
            if constexpr (Phase == 2)
                saved[i] = __fmul_rn(out[i], 1.f);
            if constexpr (Phase == 3)
                saved[i] = __fmul_rn(carry[i], decay[instance * 128 + 63]);
        }
    }
    bool unsafe_a = false;
    for (int index = threadIdx.x; index < Rows * Inner; index += blockDim.x) {

        const int row = Phase == 3 ? index % Rows : index / Inner;
        const int col = Phase == 3 ? index / Rows : index % Inner;
        const float x = qwen35_c64bfout_a<Phase>(row, col, q, key, w, causal, decay, instance, begin,
                                               length, qk_base, qk_stride);
        unsafe_a |= !(fabsf(x) <= 32768.f);
        const auto high = __float2half_rn(x);
        const int address = qwen35_c64bfout_swizzle(row, col, Inner);
        ahalf[address] = high;
        alow[address] = __float2half_rn(__fmul_rn(__fsub_rn(x, __half2float(high)), 4096.f));
    }
    unsigned unsafe_b = 0;
    if constexpr (Phase == 0) {

#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int row = r + group + (i / 2) * 8, col = c + tid * 2 + i % 2;
            const float x = carry[i];
            if (!(fabsf(x) <= 32768.f))
                unsafe_b |= 1u << (col / 16);
            const auto high = __float2half_rn(x);
            const int address = qwen35_c64bfout_swizzle(col, row, Inner);
            bhalf[address] = high;
            blow[address] = __float2half_rn(__fmul_rn(__fsub_rn(x, __half2float(high)), 4096.f));
        }
        if (unsafe_b)
            atomicOr(cached_b_unsafe, int(unsafe_b));
    } else if constexpr (ReuseB) {
        unsafe_b = unsigned(*cached_b_unsafe);
    } else {
        for (int index = threadIdx.x; index < Inner * N; index += blockDim.x) {
            const int col = index / Inner, row = index % Inner;
            const float x = updates[row * S + col];
            if (!(fabsf(x) <= 32768.f))
                unsafe_b |= 1u << (col / 16);
            const auto high = __float2half_rn(x);
            const int address = qwen35_c64bfout_swizzle(col, row, Inner);
            bhalf[address] = high;
            blow[address] = __float2half_rn(__fmul_rn(__fsub_rn(x, __half2float(high)), 4096.f));
        }
    }
    if constexpr (Phase == 2) {
        if (unsafe_b)
            atomicOr(cached_b_unsafe, int(unsafe_b));
    }

    const bool any_unsafe = __syncthreads_or(unsafe_a || unsafe_b != 0);
    bool fallback0 = false, fallback1 = false;
    if (any_unsafe) {
        fallback0 = __syncthreads_or(unsafe_a || (unsafe_b & 1u));
        if constexpr (N == 32)
            fallback1 = __syncthreads_or(unsafe_a || (unsafe_b & 2u));
    }
    const bool fallback = c < 16 ? fallback0 : fallback1;
    if (active && !fallback) {
        for (int k = 0; k < Inner; k += 16) {
            unsigned ah[4], al[4], bh[2], bl[2];

            const int ar = r + (lane & 15), ac = k + (lane / 16) * 8;
            const int br = c + (lane & 7), bc = k + ((lane / 8) & 1) * 8;
            qwen35_c64bfout_load4(ah, ahalf + qwen35_c64bfout_swizzle(ar, ac, Inner));
            qwen35_c64bfout_load4(al, alow + qwen35_c64bfout_swizzle(ar, ac, Inner));
            qwen35_c64bfout_load2(bh, bhalf + qwen35_c64bfout_swizzle(br, bc, Inner));
            qwen35_c64bfout_load2(bl, blow + qwen35_c64bfout_swizzle(br, bc, Inner));
            qwen35_c64bfout_mma(high, ah, bh);
            qwen35_c64bfout_mma(cross, ah, bl);
            qwen35_c64bfout_mma(cross, al, bh);
        }
    }

    if constexpr (Phase < 2) {
        if (fallback0 || fallback1) {
            __syncthreads();
            auto *original_state = reinterpret_cast<float *>(ahalf);
#pragma unroll
            for (int i = 0; i < 4; ++i)
                original_state[(r + group + (i / 2) * 8) * S + c + tid * 2 + i % 2] = carry[i];
            __syncthreads();
        }
    }
    if (active) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int row = r + group + (i / 2) * 8, col = c + tid * 2 + i % 2;
            const float initial = (Phase == 2 || Phase == 3) ? saved[i] : 0.f;
            float value;
            if (fallback) {
                value = initial;
                const float *original_b =
                    Phase < 2 ? reinterpret_cast<const float *>(ahalf) : updates;
                for (int k = 0; k < Inner; ++k)
                    value = __fadd_rn(
                        value,
                        __fmul_rn(qwen35_c64bfout_a<Phase>(row, k, q, key, w, causal, decay, instance,
                                                         begin, length, qk_base, qk_stride),
                                  original_b[k * S + col]));
            } else {
                value = __fadd_rn(initial, __fadd_rn(high[i], __fmul_rn(cross[i], 0x1p-12f)));
            }
            if constexpr (Phase == 0)
                updates[row * S + col] = __fsub_rn(saved[i], value);
            if constexpr (Phase == 1 || Phase == 2)
                out[i] = value;
            if constexpr (Phase == 3)
                carry[i] = value;
        }
    }
    if constexpr (Phase == 1) {
        if (threadIdx.x == 0)
            *cached_b_unsafe = 0;
    }
    __syncthreads();
}

template <int N>
__device__ __forceinline__ void
qwen35_c64bfout_state_impl(const float *q, const float *key, const float *w, const float *u,
                         const float *causal, const float *decay, __nv_bfloat16 *output,
                         int physical_start, int length, int head, int heads, int key_heads,
                         int value_dim, int instance_base, int instance_stride) {
    constexpr int S = N + 4;
    extern __shared__ __align__(32) unsigned char qwen35_c64bfout_storage[];
    auto *updates = reinterpret_cast<float *>(qwen35_c64bfout_storage);
    auto *ahalf = reinterpret_cast<__half *>(updates + 64 * S);
    auto *alow = ahalf + 8192;
    auto *bhalf = alow + 8192;
    auto *blow = bhalf + N * 128;
    auto *cached_b_unsafe = reinterpret_cast<int *>(blow + N * 128);
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int r = (warp / (N / 8)) * 16, c = (warp % (N / 8)) * 8, group = lane / 4, tid = lane % 4;
    const int kh = head / (heads / key_heads), first_col = blockIdx.y * N;
    const int qk_base = (physical_start * key_heads + kh) * 128, qk_stride = key_heads * 128;
    float carry[4] = {};
    for (int chunk = 0; chunk < (length + 63) / 64; ++chunk) {
        const int instance = instance_base + chunk * instance_stride, begin = chunk * 64;
        if (threadIdx.x == 0)
            *cached_b_unsafe = 0;
        for (int index = threadIdx.x; index < 64 * N; index += blockDim.x) {
            const int row = index / N, col = first_col + index % N;
            updates[row * S + index % N] =
                col < value_dim ? u[(instance * 64 + row) * value_dim + col] : 0.f;
        }
        __syncthreads();
        float out[4] = {};
        qwen35_c64bfout_product<0, N>(q, key, w, causal, decay, updates, carry, out, ahalf, alow,
                                    bhalf, blow, cached_b_unsafe, instance, begin, length, qk_base,
                                    qk_stride);
        qwen35_c64bfout_product<1, N, true>(q, key, w, causal, decay, updates, carry, out, ahalf,
                                          alow, bhalf, blow, cached_b_unsafe, instance, begin,
                                          length, qk_base, qk_stride);
        qwen35_c64bfout_product<2, N>(q, key, w, causal, decay, updates, carry, out, ahalf, alow,
                                    bhalf, blow, cached_b_unsafe, instance, begin, length, qk_base,
                                    qk_stride);
        if (warp < (64 / 16) * (N / 8)) {
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int row = r + group + (i / 2) * 8, col = first_col + c + tid * 2 + i % 2;
                if (begin + row < length && col < value_dim)
                    output[((physical_start + begin + row) * heads + head) * value_dim + col] =
                        __float2bfloat16_rn(out[i]);
            }
        }
        qwen35_c64bfout_product<3, N, true>(q, key, w, causal, decay, updates, carry, out, ahalf,
                                          alow, bhalf, blow, cached_b_unsafe, instance, begin,
                                          length, qk_base, qk_stride);
    }
}

extern "C" __global__ __launch_bounds__(1024, 1) void qwen35_c64bfout_uniform32_f32(
    const float *q, const float *k, const float *g, const float *w, const float *u, const float *c,
    const float *d, __nv_bfloat16 *out, float *snapshots, int length, int heads, int kh, int vd) {
    const int bh = blockIdx.x, chunks = (length + 63) / 64;
    qwen35_c64bfout_state_impl<32>(q, k, w, u, c, d, out, (bh / heads) * length, length, bh % heads,
                                 heads, kh, vd, bh * chunks, 1);
}
extern "C" __global__ __launch_bounds__(1024, 1) void qwen35_c64bfout_varlen32_f32(
    const float *q, const float *k, const float *g, const float *w, const float *u, const float *c,
    const float *d, __nv_bfloat16 *out, float *snapshots, const unsigned *cumulative,
    const unsigned *chunk_offsets, const unsigned *long_ids, int total_length, int heads, int kh,
    int vd) {
    const int head = blockIdx.x % heads, seq = long_ids[blockIdx.x / heads],
              start = cumulative[seq];
    qwen35_c64bfout_state_impl<32>(q, k, w, u, c, d, out, start, cumulative[seq + 1] - start, head,
                                 heads, kh, vd, int(chunk_offsets[seq]) * heads + head, heads);
}

extern "C" __global__ void qwen35_c64bfout_short_bf16(const float *query, const float *key,
                                                    const float *value, const float *gate,
                                                    const float *beta, __nv_bfloat16 *output,
                                                    const unsigned int *cu_seqlens,
                                                    const unsigned int *short_ids, int total_length,
                                                    int heads, int key_heads, int value_dim) {
    const int batch_head = blockIdx.x;
    const int head = batch_head % heads;
    const int batch = short_ids[batch_head / heads];
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
            output[output_offset * value_dim + column] = __float2bfloat16_rn(total);
    }
}

extern "C" __global__ void qwen35_packed_delta_bfout(const float *query, const float *key,
                                                   const float *value, const float *gate,
                                                   const float *beta, __nv_bfloat16 *output,
                                                   const unsigned int *cu_seqlens, int total_length,
                                                   int heads, int key_heads, int value_dim) {
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
            output[output_offset * value_dim + column] = __float2bfloat16_rn(total);
    }
}

extern "C" __global__ void qwen35_uniform_delta_bfout(const float *query, const float *key,
                                                    const float *value, const float *gate,
                                                    const float *beta, __nv_bfloat16 *output,
                                                    int length, int heads, int key_heads,
                                                    int value_dim) {
    const int batch_head = blockIdx.x;
    const int head = batch_head % heads;
    const int batch = batch_head / heads;
    const int lane = threadIdx.x & 15;
    const int column = blockIdx.y * 8 + (threadIdx.x >> 4);
    if (column >= value_dim)
        return;
    const unsigned mask = __activemask();
    float state[2][4] = {{0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}};
    for (int token = 0; token < length; ++token) {
        const int token_head = (batch * length + token) * heads + head;
        const int qk_head = (batch * length + token) * key_heads + head / (heads / key_heads);
        const int output_offset = batch_head * length + token;
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
            output[output_offset * value_dim + column] = __float2bfloat16_rn(total);
    }
}

#include <cuda_bf16.h>
__device__ __forceinline__ float qwen35_bfinput_warp_sum(float value) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_xor_sync(0xffffffff, value, offset, 32);
    }
    return value;
}
extern "C" __global__ void
qwen35_post_finalize_bf16_input(const __nv_bfloat16 *input, const __nv_bfloat16 *gate,
                              const __nv_bfloat16 *norm, __nv_bfloat16 *output, int columns,
                              float epsilon, unsigned long long length, unsigned long long heads,
                              unsigned long long stride_b, unsigned long long stride_t,
                              unsigned long long stride_h) {
    const int row = blockIdx.x;
    const unsigned long long h = row % heads;
    const unsigned long long token = (row / heads) % length;
    const unsigned long long batch = row / (heads * length);
    const unsigned long long source_row = batch * stride_b + token * stride_t + h * stride_h;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    float sum = 0.f;
    for (int column = threadIdx.x; column < columns; column += blockDim.x) {
        const float value = float(__nv_bfloat16(input[source_row + column]));
        sum += value * value;
    }
    sum = qwen35_bfinput_warp_sum(sum);
    __shared__ float warp_sums[32];
    if (lane == 0)
        warp_sums[warp] = sum;
    __syncthreads();
    sum = lane < (blockDim.x >> 5) ? warp_sums[lane] : 0.f;
    sum = qwen35_bfinput_warp_sum(sum);
    const float scale = rsqrtf(sum / float(columns) + epsilon);
    for (int column = threadIdx.x; column < columns; column += blockDim.x) {
        const int index = row * columns + column;
        const float gate_value = float(gate[index]);
        const float gate_silu = gate_value / (1.f + expf(-gate_value));
        const float recurrent = float(__nv_bfloat16(input[source_row + column]));
        const float normalized = float(__nv_bfloat16(__fmul_rn(recurrent, scale)));
        const float weighted = float(__nv_bfloat16(__fmul_rn(normalized, float(norm[column]))));
        output[index] = __nv_bfloat16(__fmul_rn(weighted, gate_silu));
    }
}

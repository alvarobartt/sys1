__device__ __forceinline__ void qwen35_c64warp_scale_lower(float *, const float *);
__device__ __forceinline__ void qwen35_c64warp_solve(float *, float *, const float *);
template <int Mode>
__device__ __forceinline__ void qwen35_c64v_gram(const float *query, const float *key,
    const float *beta, const float *prefix, bool &key_unsafe, int physical_start, int begin,
    int length, int head, int heads, int kh, int key_heads, __half *ahalf, __half *alow,
    __half *bhalf, __half *blow, float *out) {
    constexpr int C = 64, K = 128, P = 136;
    bool unsafe = Mode == 1 ? key_unsafe : false;
    for (int i = threadIdx.x; i < C * K; i += blockDim.x) {
        const int t = i / K, d = i % K, token = begin + t;
        const int offset = ((physical_start + token) * key_heads + kh) * K + d;
        float lhs = 0.f;
        if constexpr (Mode == 0) {
            const float kval = token < length ? key[offset] : 0.f;
            lhs = token < length ? __fmul_rn(kval, beta[((physical_start + token) * heads + head)])
                                 : 0.f;
            key_unsafe |= !(fabsf(kval) <= 32768.f);
            const auto b = __float2half_rn(kval);
            bhalf[t * P + d] = b;
            blow[t * P + d] = __float2half_rn(__fmul_rn(__fsub_rn(kval, __half2float(b)), 4096.f));
        } else {
            lhs = token < length ? query[offset] : 0.f;
        }
        unsafe |= !(fabsf(lhs) <= 32768.f);
        const auto a = __float2half_rn(lhs);
        ahalf[t * P + d] = a;
        alow[t * P + d] = __float2half_rn(__fmul_rn(__fsub_rn(lhs, __half2float(a)), 4096.f));
    }
    if constexpr (Mode == 0)
        unsafe |= key_unsafe;
    const bool fallback = __syncthreads_or(unsafe);
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int row = (warp / 8) * 16, col = (warp % 8) * 8, group = lane / 4, tid = lane % 4;
    float high[4] = {}, cross[4] = {};
    if (!fallback) {
        for (int k = 0; k < K; k += 16) {
            unsigned ah[4], al[4], bh[2], bl[2];
            const int ar = row + (lane & 15), ac = k + (lane / 16) * 8;
            const int br = col + (lane & 7), bc = k + ((lane / 8) & 1) * 8;
            qwen35_gramld_load4(ah, ahalf + ar * P + ac);
            qwen35_gramld_load4(al, alow + ar * P + ac);
            qwen35_gramld_load2(bh, bhalf + br * P + bc);
            qwen35_gramld_load2(bl, blow + br * P + bc);
            qwen35_c64_half_mma16816(high, ah, bh);
            qwen35_c64_half_mma16816(cross, ah, bl);
            qwen35_c64_half_mma16816(cross, al, bh);
        }
#pragma unroll
        for (int p = 0; p < 4; ++p)
            high[p] = __fadd_rn(high[p], __fmul_rn(cross[p], 0x1p-12f));
    } else {
#pragma unroll
        for (int p = 0; p < 4; ++p) {
            const int t = row + group + (p / 2) * 8, j = col + tid * 2 + p % 2;
            const int token = begin + t, previous = begin + j;
            float dot = 0.f;
            if (token < length && previous < length) {
                for (int d = 0; d < K; ++d) {
                    const int offset = ((physical_start + token) * key_heads + kh) * K + d;
                    const float lhs = Mode == 0
                        ? __fmul_rn(key[offset], beta[(physical_start + token) * heads + head])
                        : query[offset];
                    dot = __fmaf_rn(
                        lhs, key[((physical_start + previous) * key_heads + kh) * K + d], dot);
                }
            }
            high[p] = dot;
        }
    }
#pragma unroll
    for (int p = 0; p < 4; ++p) {
        const int t = row + group + (p / 2) * 8, j = col + tid * 2 + p % 2;
        if constexpr (Mode == 1)
            out[t * C + j] =
                j <= t ? __fmul_rn(high[p], expf(__fsub_rn(prefix[t], prefix[j]))) : 0.f;
        else
            out[t * C + j] = high[p];
    }
    __syncthreads();
}

extern "C" __global__ __launch_bounds__(1024, 1) void qwen35_c64v_prepare_f32(const float *query,
    const float *key, const float *value, const float *gate, const float *beta, float *w, float *u,
    float *causal, float *decays, const unsigned int *cumulative, const unsigned int *chunk_offsets,
    int sequence_count, int total_length, int heads, int key_heads, int value_dim) {
    constexpr int C = 64, K = 128, S = 129, P = 136;
    extern __shared__ __align__(32) unsigned char qwen35_c64v_storage[];
    auto *ahalf = reinterpret_cast<__half *>(qwen35_c64v_storage);
    auto *alow = ahalf + C * P;
    auto *bhalf = alow + C * P;
    auto *blow = bhalf + C * P;
    auto *lower = reinterpret_cast<float *>(blow + C * P);
    auto *prefix = lower + C * C;
    auto *betas = prefix + C;
    auto *tile_w = reinterpret_cast<float *>(qwen35_c64v_storage);
    auto *tile_u = tile_w + C * S;
    const int instance = blockIdx.x, global_chunk = instance / heads, head = instance % heads;
    int lo = 0, hi = sequence_count;
    while (lo < hi) {
        const int mid = (lo + hi + 1) / 2;
        if (chunk_offsets[mid] <= global_chunk)
            lo = mid;
        else
            hi = mid - 1;
    }
    const int sequence = lo, physical_start = cumulative[sequence],
              length = cumulative[sequence + 1] - physical_start;
    const int kh = head / (heads / key_heads),
              begin = (global_chunk - int(chunk_offsets[sequence])) * C;
    if (threadIdx.x < C) {
        const int t = threadIdx.x, token = begin + t, th = (physical_start + token) * heads + head;
        float effective = 0.f;
        if (token < length) {
            const float decay = expf(gate[th]);
            effective = decay > 0.f ? logf(decay) : gate[th];
        }
        prefix[t] = effective;
        betas[t] = token < length ? beta[th] : 0.f;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        float running = 0.f;
        for (int t = 0; t < C; ++t) {
            if (begin + t < length)
                running = __fadd_rn(running, prefix[t]);
            prefix[t] = running;
        }
    }
    __syncthreads();
    if (threadIdx.x < C) {
        const int t = threadIdx.x;
        decays[instance * 128 + t] = expf(prefix[t]);
        decays[instance * 128 + 64 + t] = expf(__fsub_rn(prefix[63], prefix[t]));
    }
    bool key_unsafe = false;
    qwen35_c64v_gram<0>(query, key, beta, prefix, key_unsafe, physical_start, begin, length, head,
        heads, kh, key_heads, ahalf, alow, bhalf, blow, lower);
    qwen35_c64warp_scale_lower(lower, prefix);
    qwen35_c64v_gram<1>(query, key, beta, prefix, key_unsafe, physical_start, begin, length, head,
        heads, kh, key_heads, ahalf, alow, bhalf, blow, causal + instance * C * C);
    for (int i = threadIdx.x; i < C * K; i += blockDim.x) {
        const int t = i / K, d = i % K, token = begin + t;
        const float kval =
            token < length ? key[((physical_start + token) * key_heads + kh) * K + d] : 0.f;
        const float val = token < length && d < value_dim
            ? value[((physical_start + token) * heads + head) * value_dim + d]
            : 0.f;
        tile_w[t * S + d] = __fmul_rn(__fmul_rn(kval, betas[t]), expf(prefix[t]));
        tile_u[t * S + d] = __fmul_rn(val, betas[t]);
    }
    __syncthreads();
    qwen35_c64warp_solve(tile_w, tile_u, lower);
    for (int i = threadIdx.x; i < C * K; i += blockDim.x) {
        const int t = i / K, d = i % K;
        w[(instance * C + t) * K + d] = tile_w[t * S + d];
        if (d < value_dim)
            u[(instance * C + t) * value_dim + d] = tile_u[t * S + d];
    }
    __syncthreads();
}

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math.h>

extern "C" __global__ void qwen35_c64v_short_f32(const float *query, const float *key,
    const float *value, const float *gate, const float *beta, float *output,
    const unsigned int *cu_seqlens, const unsigned int *short_ids, int total_length, int heads,
    int key_heads, int value_dim) {
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
            output[output_offset * value_dim + column] = total;
    }
}

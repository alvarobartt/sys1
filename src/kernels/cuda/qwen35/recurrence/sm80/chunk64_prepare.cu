#include <cuda_runtime.h>
#include <math.h>

__device__ __forceinline__ int qwen35_c64warp_lower_index(int row, int col) {
    return row * 64 + (col ^ (row & 31));
}

__device__ __forceinline__ void qwen35_c64warp_scale_lower(float *lower, const float *prefix) {
    float saved[4];
#pragma unroll
    for (int item = 0; item < 4; ++item) {
        const int index = threadIdx.x + item * 1024, row = index / 64, col = index % 64;
        saved[item] =
            col < row ? __fmul_rn(lower[index], expf(__fsub_rn(prefix[row], prefix[col]))) : 0.f;
    }
    __syncthreads();
#pragma unroll
    for (int item = 0; item < 4; ++item) {
        const int index = threadIdx.x + item * 1024, row = index / 64, col = index % 64;
        lower[qwen35_c64warp_lower_index(row, col)] = saved[item];
    }
}

template <int Quarter>
__device__ __forceinline__ void qwen35_c64warp_subwarp_phase(float (&rhs)[4][4], const float *lower,
                                                           int lane) {
#pragma unroll 1
    for (int j = Quarter * 16; j < (Quarter + 1) * 16; ++j) {
        float coefficient[4];
#pragma unroll
        for (int part = Quarter; part < 4; ++part)
            coefficient[part] = lower[qwen35_c64warp_lower_index(lane + part * 16, j)];
#pragma unroll
        for (int column = 0; column < 4; ++column) {
            const float value = __shfl_sync(0xffffffff, rhs[Quarter][column], j & 15, 16);
#pragma unroll
            for (int part = Quarter; part < 4; ++part) {
                if (part > Quarter || lane + part * 16 > j)
                    rhs[part][column] =
                        __fsub_rn(rhs[part][column], __fmul_rn(coefficient[part], value));
            }
        }
    }
}

__device__ __forceinline__ void qwen35_c64warp_solve(float *tile_w, float *tile_u,
                                                   const float *lower) {
    constexpr int S = 129;
    const int lane = threadIdx.x & 15;
    const int first_column = (threadIdx.x / 32) * 8 + ((threadIdx.x & 31) / 16) * 4;
    float rhs[4][4];
#pragma unroll
    for (int part = 0; part < 4; ++part) {
#pragma unroll
        for (int column = 0; column < 4; ++column) {
            const int c = first_column + column, row = lane + part * 16;
            rhs[part][column] = (c < 128 ? tile_w : tile_u)[row * S + (c & 127)];
        }
    }
    qwen35_c64warp_subwarp_phase<0>(rhs, lower, lane);
    qwen35_c64warp_subwarp_phase<1>(rhs, lower, lane);
    qwen35_c64warp_subwarp_phase<2>(rhs, lower, lane);
    qwen35_c64warp_subwarp_phase<3>(rhs, lower, lane);
#pragma unroll
    for (int part = 0; part < 4; ++part) {
#pragma unroll
        for (int column = 0; column < 4; ++column) {
            const int c = first_column + column, row = lane + part * 16;
            (c < 128 ? tile_w : tile_u)[row * S + (c & 127)] = rhs[part][column];
        }
    }
    __syncthreads();
}

#include <cuda_runtime.h>
#include <cuda_fp16.h>
__device__ __forceinline__ void qwen35_gramld_load4(unsigned *out, const __half *pointer) {
    const unsigned address = unsigned(__cvta_generic_to_shared(pointer));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3},[%4];"
                 : "=r"(out[0]), "=r"(out[1]), "=r"(out[2]), "=r"(out[3])
                 : "r"(address)
                 : "memory");
}
__device__ __forceinline__ void qwen35_gramld_load2(unsigned *out, const __half *pointer) {
    const unsigned address = unsigned(__cvta_generic_to_shared(pointer));
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1},[%2];"
                 : "=r"(out[0]), "=r"(out[1])
                 : "r"(address)
                 : "memory");
}
__device__ __forceinline__ void qwen35_c64warp_scale_lower(float *, const float *);
__device__ __forceinline__ void qwen35_c64warp_solve(float *, float *, const float *);
template <int Mode>
__device__ __forceinline__ void
qwen35_c64mma_gram(const float *query, const float *key, const float *beta, const float *prefix,
                 bool &key_unsafe, int batch, int begin, int length, int head, int heads, int kh,
                 int key_heads, __half *ahalf, __half *alow, __half *bhalf, __half *blow,
                 float *out) {
    constexpr int C = 64, K = 128, P = 136;
    bool unsafe = Mode == 1 ? key_unsafe : false;
    for (int i = threadIdx.x; i < C * K; i += blockDim.x) {
        const int t = i / K, d = i % K, token = begin + t;
        const int offset = ((batch * length + token) * key_heads + kh) * K + d;
        float lhs = 0.f;
        if constexpr (Mode == 0) {
            const float kval = token < length ? key[offset] : 0.f;
            lhs = token < length ? __fmul_rn(kval, beta[((batch * length + token) * heads + head)])
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
                    const int offset = ((batch * length + token) * key_heads + kh) * K + d;
                    const float lhs =
                        Mode == 0
                            ? __fmul_rn(key[offset], beta[(batch * length + token) * heads + head])
                            : query[offset];
                    dot = __fmaf_rn(
                        lhs, key[((batch * length + previous) * key_heads + kh) * K + d], dot);
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

extern "C" __global__ __launch_bounds__(1024, 1) void qwen35_c64mma_prepare_f32(
    const float *query, const float *key, const float *value, const float *gate, const float *beta,
    float *w, float *u, float *causal, float *decays, int length, int heads, int key_heads,
    int value_dim) {
    constexpr int C = 64, K = 128, S = 129, P = 136;
    extern __shared__ __align__(32) unsigned char qwen35_c64mma_storage[];
    auto *ahalf = reinterpret_cast<__half *>(qwen35_c64mma_storage);
    auto *alow = ahalf + C * P;
    auto *bhalf = alow + C * P;
    auto *blow = bhalf + C * P;
    auto *lower = reinterpret_cast<float *>(blow + C * P);
    auto *prefix = lower + C * C;
    auto *betas = prefix + C;
    auto *tile_w = reinterpret_cast<float *>(qwen35_c64mma_storage);
    auto *tile_u = tile_w + C * S;
    const int chunks = (length + C - 1) / C, instance = blockIdx.x, bh = instance / chunks,
              chunk = instance % chunks;
    const int head = bh % heads, batch = bh / heads, kh = head / (heads / key_heads),
              begin = chunk * C;
    if (threadIdx.x < C) {
        const int t = threadIdx.x, token = begin + t, th = (batch * length + token) * heads + head;
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
    qwen35_c64mma_gram<0>(query, key, beta, prefix, key_unsafe, batch, begin, length, head, heads, kh,
                        key_heads, ahalf, alow, bhalf, blow, lower);
    qwen35_c64warp_scale_lower(lower, prefix);
    qwen35_c64mma_gram<1>(query, key, beta, prefix, key_unsafe, batch, begin, length, head, heads, kh,
                        key_heads, ahalf, alow, bhalf, blow, causal + instance * C * C);
    for (int i = threadIdx.x; i < C * K; i += blockDim.x) {
        const int t = i / K, d = i % K, token = begin + t;
        const float kval =
            token < length ? key[((batch * length + token) * key_heads + kh) * K + d] : 0.f;
        const float val = token < length && d < value_dim
                              ? value[((batch * length + token) * heads + head) * value_dim + d]
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

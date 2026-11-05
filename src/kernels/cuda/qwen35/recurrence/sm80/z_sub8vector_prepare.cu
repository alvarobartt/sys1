__device__ __forceinline__ int qwen35_c64sub8vector_coefficient_index(int row, int col) {
    return col * 64 + (row / 32) * 32 + (row & 7) * 4 + ((row / 8) & 3);
}

__device__ __forceinline__ int qwen35_c64sub8vector_half_index(int row, int col) {
    return row * 64 + (col ^ ((row & 7) * 8));
}

template <int Mode>
__device__ __forceinline__ void qwen35_c64sub8vector_gram(const float *query, const float *key,
    const float *beta, const float *prefix, bool &key_unsafe, int physical_start, int begin,
    int length, int head, int heads, int kh, int key_heads, __half *ahalf, __half *alow,
    __half *bhalf, __half *blow, float *out) {
    constexpr int C = 64, K = 128;
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32, group = lane / 4, tid = lane % 4;
    const int col = (warp % 8) * 8;
    float high[2][4] = {}, cross[2][4] = {};
    bool unsafe = Mode == 1 ? key_unsafe : false;
    bool fallback = false;
    for (int slice = 0; slice < K; slice += 64) {
        for (int i = threadIdx.x; i < C * 64; i += blockDim.x) {
            const int t = i / 64, d = i % 64, token = begin + t;
            const int offset = ((physical_start + token) * key_heads + kh) * K + slice + d;
            const float kval = token < length ? key[offset] : 0.f;
            const float lhs = Mode == 0
                ? (token < length ? __fmul_rn(kval, beta[(physical_start + token) * heads + head])
                                  : 0.f)
                : (token < length ? query[offset] : 0.f);
            key_unsafe |= !(fabsf(kval) <= 32768.f);
            unsafe |= !(fabsf(kval) <= 32768.f) || !(fabsf(lhs) <= 32768.f);
            const auto a = __float2half_rn(lhs), b = __float2half_rn(kval);
            const int at = qwen35_c64sub8vector_half_index(t, d);
            ahalf[at] = a;
            alow[at] = __float2half_rn(__fmul_rn(__fsub_rn(lhs, __half2float(a)), 4096.f));
            bhalf[at] = b;
            blow[at] = __float2half_rn(__fmul_rn(__fsub_rn(kval, __half2float(b)), 4096.f));
        }
        fallback = __syncthreads_or(unsafe);
        if (!fallback) {
            for (int k = 0; k < 64; k += 16) {
#pragma unroll
                for (int part = 0; part < 2; ++part) {
                    const int row = (warp / 8) * 16 + part * 32;
                    unsigned ah[4], al[4], bh[2], bl[2];
                    const int ar = row + (lane & 15), ac = k + (lane / 16) * 8;
                    const int br = col + (lane & 7), bc = k + ((lane / 8) & 1) * 8;
                    qwen35_gramld_load4(ah, ahalf + qwen35_c64sub8vector_half_index(ar, ac));
                    qwen35_gramld_load4(al, alow + qwen35_c64sub8vector_half_index(ar, ac));
                    qwen35_gramld_load2(bh, bhalf + qwen35_c64sub8vector_half_index(br, bc));
                    qwen35_gramld_load2(bl, blow + qwen35_c64sub8vector_half_index(br, bc));
                    qwen35_c64_half_mma16816(high[part], ah, bh);
                    qwen35_c64_half_mma16816(cross[part], ah, bl);
                    qwen35_c64_half_mma16816(cross[part], al, bh);
                }
            }
        }

        __syncthreads();
    }
#pragma unroll
    for (int part = 0; part < 2; ++part) {
        const int row = (warp / 8) * 16 + part * 32;
#pragma unroll
        for (int p = 0; p < 4; ++p) {
            const int t = row + group + (p / 2) * 8, j = col + tid * 2 + p % 2;
            float dot;
            if (fallback) {
                dot = 0.f;
                const int token = begin + t, previous = begin + j;
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
            } else
                dot = __fadd_rn(high[part][p], __fmul_rn(cross[part][p], 0x1p-12f));
            if constexpr (Mode == 1)
                out[t * C + j] =
                    j <= t ? __fmul_rn(dot, expf(__fsub_rn(prefix[t], prefix[j]))) : 0.f;
            else
                out[qwen35_c64sub8vector_coefficient_index(t, j)] =
                    j < t ? __fmul_rn(dot, expf(__fsub_rn(prefix[t], prefix[j]))) : 0.f;
        }
    }
    __syncthreads();
}

template <int Eighth>
__device__ __forceinline__ void qwen35_c64sub8vector_rhs_phase(
    float (&rhs)[8][4], const float *lower, int lane) {
#pragma unroll 1
    for (int j = Eighth * 8; j < (Eighth + 1) * 8; ++j) {
        float coefficient[8];
        const float *aligned = lower + qwen35_c64sub8vector_coefficient_index(lane, j);
        const unsigned address = static_cast<unsigned>(__cvta_generic_to_shared(aligned));
        if constexpr (Eighth < 4) {
            asm volatile("ld.shared.v4.f32 {%0,%1,%2,%3}, [%4];"
                         : "=f"(coefficient[0]), "=f"(coefficient[1]), "=f"(coefficient[2]),
                         "=f"(coefficient[3])
                         : "r"(address)
                         : "memory");
        }
        if constexpr (Eighth < 6) {
            asm volatile("ld.shared.v4.f32 {%0,%1,%2,%3}, [%4];"
                         : "=f"(coefficient[4]), "=f"(coefficient[5]), "=f"(coefficient[6]),
                         "=f"(coefficient[7])
                         : "r"(address + 128)
                         : "memory");
        } else if constexpr (Eighth == 6) {
            asm volatile("ld.shared.v2.f32 {%0,%1}, [%2];"
                         : "=f"(coefficient[6]), "=f"(coefficient[7])
                         : "r"(address + 136)
                         : "memory");
        } else
            coefficient[7] = aligned[35];
#pragma unroll
        for (int column = 0; column < 4; ++column) {
            const float value = __shfl_sync(0xffffffff, rhs[Eighth][column], j & 7, 8);
#pragma unroll
            for (int part = Eighth; part < 8; ++part) {
                if (part > Eighth || lane + part * 8 > j)
                    rhs[part][column] =
                        __fsub_rn(rhs[part][column], __fmul_rn(coefficient[part], value));
            }
        }
    }
}

__device__ __forceinline__ void qwen35_c64sub8vector_prepare(const float *query, const float *key,
    const float *value, const float *gate, const float *beta, float *w, float *u, float *causal,
    float *decays, int physical_start, int begin, int length, int head, int heads, int key_heads,
    int value_dim, int instance) {
    constexpr int C = 64, K = 128, S = 129;
    extern __shared__ __align__(32) unsigned char qwen35_c64sub8vector_storage[];
    auto *ahalf = reinterpret_cast<__half *>(qwen35_c64sub8vector_storage);
    auto *alow = ahalf + C * 64;
    auto *bhalf = alow + C * 64;
    auto *blow = bhalf + C * 64;
    auto *lower = reinterpret_cast<float *>(blow + C * 64);
    auto *prefix = lower + C * C;
    auto *betas = prefix + C;
    const int kh = head / (heads / key_heads);
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
    qwen35_c64sub8vector_gram<0>(query, key, beta, prefix, key_unsafe, physical_start, begin,
        length, head, heads, kh, key_heads, ahalf, alow, bhalf, blow, lower);

    qwen35_c64sub8vector_gram<1>(query, key, beta, prefix, key_unsafe, physical_start, begin,
        length, head, heads, kh, key_heads, ahalf, alow, bhalf, blow, causal + instance * C * C);
    const int lane = threadIdx.x & 7,
              first_column = (threadIdx.x / 32) * 16 + ((threadIdx.x & 31) / 8) * 4;
    float rhs[8][4];
#pragma unroll
    for (int part = 0; part < 8; ++part) {
        const int row = lane + part * 8, token = begin + row;
        const float beta_value = betas[row], decay = expf(prefix[row]);
#pragma unroll
        for (int column = 0; column < 4; ++column) {
            const int c = first_column + column, d = c & 127;
            if (c < 128) {
                const float kval =
                    token < length ? key[((physical_start + token) * key_heads + kh) * K + d] : 0.f;
                rhs[part][column] = __fmul_rn(__fmul_rn(kval, beta_value), decay);
            } else {
                const float val = token < length && d < value_dim
                    ? value[((physical_start + token) * heads + head) * value_dim + d]
                    : 0.f;
                rhs[part][column] = __fmul_rn(val, beta_value);
            }
        }
    }
    qwen35_c64sub8vector_rhs_phase<0>(rhs, lower, lane);
    qwen35_c64sub8vector_rhs_phase<1>(rhs, lower, lane);
    qwen35_c64sub8vector_rhs_phase<2>(rhs, lower, lane);
    qwen35_c64sub8vector_rhs_phase<3>(rhs, lower, lane);
    qwen35_c64sub8vector_rhs_phase<4>(rhs, lower, lane);
    qwen35_c64sub8vector_rhs_phase<5>(rhs, lower, lane);
    qwen35_c64sub8vector_rhs_phase<6>(rhs, lower, lane);
    qwen35_c64sub8vector_rhs_phase<7>(rhs, lower, lane);
    __syncthreads();
    auto *tile = reinterpret_cast<float *>(qwen35_c64sub8vector_storage);
#pragma unroll
    for (int field = 0; field < 2; ++field) {
        if ((first_column >= 128) == bool(field)) {
#pragma unroll
            for (int part = 0; part < 8; ++part) {
#pragma unroll
                for (int column = 0; column < 4; ++column)
                    tile[(lane + part * 8) * S + ((first_column + column) & 127)] =
                        rhs[part][column];
            }
        }
        __syncthreads();
        for (int i = threadIdx.x; i < C * K; i += blockDim.x) {
            const int t = i / K, d = i % K;
            if (field == 0)
                w[(instance * C + t) * K + d] = tile[t * S + d];
            else if (d < value_dim)
                u[(instance * C + t) * value_dim + d] = tile[t * S + d];
        }
        __syncthreads();
    }
}

extern "C" __global__ __launch_bounds__(512, 2) void qwen35_c64sub8vector_uniform_prepare_f32(
    const float *q, const float *k, const float *v, const float *g, const float *b, float *w,
    float *u, float *c, float *d, int length, int heads, int key_heads, int vd) {
    const int chunks = (length + 63) / 64, instance = blockIdx.x, bh = instance / chunks,
              chunk = instance % chunks;
    const int batch = bh / heads, head = bh % heads;
    qwen35_c64sub8vector_prepare(q, k, v, g, b, w, u, c, d, batch * length, chunk * 64, length,
        head, heads, key_heads, vd, instance);
}
extern "C" __global__ __launch_bounds__(512, 2) void qwen35_c64sub8vector_varlen_prepare_f32(
    const float *q, const float *k, const float *v, const float *g, const float *b, float *w,
    float *u, float *c, float *d, const unsigned *cumulative, const unsigned *chunks,
    int sequence_count, int total_length, int heads, int key_heads, int vd) {
    const int global_chunk = blockIdx.x / heads, head = blockIdx.x % heads;
    int lo = 0, hi = sequence_count;
    while (lo < hi) {
        const int mid = (lo + hi + 1) / 2;
        if (chunks[mid] <= global_chunk)
            lo = mid;
        else
            hi = mid - 1;
    }
    const int start = cumulative[lo], length = cumulative[lo + 1] - start,
              begin = (global_chunk - int(chunks[lo])) * 64;
    qwen35_c64sub8vector_prepare(q, k, v, g, b, w, u, c, d, start, begin, length, head, heads,
        key_heads, vd, global_chunk * heads + head);
}

#include <stdint.h>
__device__ __forceinline__ int qwen35_c64prepvload2_coefficient_index(int row, int col) {
    return col * 64 + (row / 32) * 32 + (row & 7) * 4 + ((row / 8) & 3);
}

__device__ __forceinline__ int qwen35_c64prepvload2_half_index(int row, int col) {
    return row * 128 + (col ^ ((row & 7) * 8));
}

template <int Mode>
__device__ __forceinline__ void qwen35_c64prepvload2_gram(const float *query, const float *key,
    const float *beta, const float *prefix, bool &key_unsafe, int physical_start, int begin,
    int length, int head, int heads, int kh, int key_heads, __half *ahalf, __half *alow,
    __half *bhalf, __half *blow, float *out) {
    constexpr int C = 64, K = 128;
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32, group = lane / 4, tid = lane % 4;
    const int col = (warp % 8) * 8;
    float high[2][4] = {}, cross[2][4] = {};
    bool unsafe = Mode == 1 ? key_unsafe : false;
    bool fallback = false;
    for (int slice = 0; slice < K; slice += 128) {
        for (int i = threadIdx.x * 4; i < C * K; i += blockDim.x * 4) {
            const int t = i / K, d = i % K, token = begin + t;
            float4 keys = make_float4(0.f, 0.f, 0.f, 0.f);
            float4 queries = keys;
            float beta_value = 0.f;
            if (token < length) {
                const int offset = ((physical_start + token) * key_heads + kh) * K + slice + d;
                const float *kp = key + offset;
                keys = (reinterpret_cast<uintptr_t>(kp) & 15u)
                    ? make_float4(kp[0], kp[1], kp[2], kp[3])
                    : *reinterpret_cast<const float4 *>(kp);
                if constexpr (Mode == 0)
                    beta_value = beta[(physical_start + token) * heads + head];
                else {
                    const float *qp = query + offset;
                    queries = (reinterpret_cast<uintptr_t>(qp) & 15u)
                        ? make_float4(qp[0], qp[1], qp[2], qp[3])
                        : *reinterpret_cast<const float4 *>(qp);
                }
            }
            const float kvals[4] = {keys.x, keys.y, keys.z, keys.w};
            const float qvals[4] = {queries.x, queries.y, queries.z, queries.w};
            __half av[4], alv[4], bv[4], blv[4];
#pragma unroll
            for (int item = 0; item < 4; ++item) {
                const float kval = kvals[item];
                const float lhs =
                    Mode == 0 ? (token < length ? __fmul_rn(kval, beta_value) : 0.f) : qvals[item];
                key_unsafe |= !(fabsf(kval) <= 32768.f);
                unsafe |= !(fabsf(kval) <= 32768.f) || !(fabsf(lhs) <= 32768.f);
                av[item] = __float2half_rn(lhs);
                bv[item] = __float2half_rn(kval);
                alv[item] =
                    __float2half_rn(__fmul_rn(__fsub_rn(lhs, __half2float(av[item])), 4096.f));
                blv[item] =
                    __float2half_rn(__fmul_rn(__fsub_rn(kval, __half2float(bv[item])), 4096.f));
            }
            const int at = qwen35_c64prepvload2_half_index(t, d);
#pragma unroll
            for (int which = 0; which < 4; ++which) {
                if (which == 1 || which == 3)
                    continue;
                __half *destination = which == 0 ? ahalf
                    : which == 1                 ? alow
                    : which == 2                 ? bhalf
                                                 : blow;
                const __half *values = which == 0 ? av : which == 1 ? alv : which == 2 ? bv : blv;
                const __half_raw v0 = values[0], v1 = values[1], v2 = values[2], v3 = values[3];
                *reinterpret_cast<uint2 *>(destination + at) =
                    make_uint2(unsigned(v0.x) | (unsigned(v1.x) << 16),
                        unsigned(v2.x) | (unsigned(v3.x) << 16));
            }
        }
        fallback = __syncthreads_or(unsafe);
        if (!fallback) {
            for (int k = 0; k < K; k += 16) {
#pragma unroll
                for (int part = 0; part < 2; ++part) {
                    const int row = (warp / 8) * 16 + part * 32;
                    unsigned ah[4], al[4], bh[2], bl[2];
                    const int ar = row + (lane & 15), ac = k + (lane / 16) * 8;
                    const int br = col + (lane & 7), bc = k + ((lane / 8) & 1) * 8;
                    qwen35_gramld_load4(ah, ahalf + qwen35_c64prepvload2_half_index(ar, ac));
                    qwen35_gramld_load2(bh, bhalf + qwen35_c64prepvload2_half_index(br, bc));
                    qwen35_c64_half_mma16816(high[part], ah, bh);
                }
            }
        }

        __syncthreads();
    }
#pragma unroll
    for (int part = 0; part < 2; ++part) {
        const int row = (warp / 8) * 16 + part * 32;
#pragma unroll
        for (int p = 0; p < 4; ++p) {
            const int t = row + group + (p / 2) * 8, j = col + tid * 2 + p % 2;
            float dot;
            if (fallback) {
                dot = 0.f;
                const int token = begin + t, previous = begin + j;
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
            } else
                dot = high[part][p];
            if constexpr (Mode == 1)
                out[t * C + j] =
                    j <= t ? __fmul_rn(dot, expf(__fsub_rn(prefix[t], prefix[j]))) : 0.f;
            else
                out[qwen35_c64prepvload2_coefficient_index(t, j)] =
                    j < t ? __fmul_rn(dot, expf(__fsub_rn(prefix[t], prefix[j]))) : 0.f;
        }
    }
    __syncthreads();
}

template <int Eighth>
__device__ __forceinline__ void qwen35_c64prepvload2_rhs_phase(
    float (&rhs)[8][4], const float *lower, int lane) {
#pragma unroll 1
    for (int j = Eighth * 8; j < (Eighth + 1) * 8; ++j) {
        float coefficient[8];
        const float *aligned = lower + qwen35_c64prepvload2_coefficient_index(lane, j);
        const unsigned address = static_cast<unsigned>(__cvta_generic_to_shared(aligned));
        if constexpr (Eighth < 4) {
            asm volatile("ld.shared.v4.f32 {%0,%1,%2,%3}, [%4];"
                         : "=f"(coefficient[0]), "=f"(coefficient[1]), "=f"(coefficient[2]),
                         "=f"(coefficient[3])
                         : "r"(address)
                         : "memory");
        }
        if constexpr (Eighth < 6) {
            asm volatile("ld.shared.v4.f32 {%0,%1,%2,%3}, [%4];"
                         : "=f"(coefficient[4]), "=f"(coefficient[5]), "=f"(coefficient[6]),
                         "=f"(coefficient[7])
                         : "r"(address + 128)
                         : "memory");
        } else if constexpr (Eighth == 6) {
            asm volatile("ld.shared.v2.f32 {%0,%1}, [%2];"
                         : "=f"(coefficient[6]), "=f"(coefficient[7])
                         : "r"(address + 136)
                         : "memory");
        } else
            coefficient[7] = aligned[35];
#pragma unroll
        for (int column = 0; column < 4; ++column) {
            const float value = __shfl_sync(0xffffffff, rhs[Eighth][column], j & 7, 8);
#pragma unroll
            for (int part = Eighth; part < 8; ++part) {
                if (part > Eighth || lane + part * 8 > j)
                    rhs[part][column] =
                        __fsub_rn(rhs[part][column], __fmul_rn(coefficient[part], value));
            }
        }
    }
}

__device__ __forceinline__ void qwen35_c64prepvload2_prepare(const float *query, const float *key,
    const float *value, const float *gate, const float *beta, float *w, float *u, float *causal,
    float *decays, int physical_start, int begin, int length, int head, int heads, int key_heads,
    int value_dim, int instance) {
    constexpr int C = 64, K = 128, S = 129;
    extern __shared__ __align__(32) unsigned char qwen35_c64prepvload2_storage[];
    auto *ahalf = reinterpret_cast<__half *>(qwen35_c64prepvload2_storage);
    auto *alow = ahalf + C * 64;
    auto *bhalf = alow + C * 64;
    auto *blow = bhalf + C * 64;
    auto *lower = reinterpret_cast<float *>(blow + C * 64);
    auto *prefix = lower + C * C;
    auto *betas = prefix + C;
    const int kh = head / (heads / key_heads);
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
    qwen35_c64prepvload2_gram<0>(query, key, beta, prefix, key_unsafe, physical_start, begin,
        length, head, heads, kh, key_heads, ahalf, alow, bhalf, blow, lower);

    qwen35_c64prepvload2_gram<1>(query, key, beta, prefix, key_unsafe, physical_start, begin,
        length, head, heads, kh, key_heads, ahalf, alow, bhalf, blow, causal + instance * C * C);
    const int lane = threadIdx.x & 7,
              first_column = (threadIdx.x / 32) * 16 + ((threadIdx.x & 31) / 8) * 4;

    float rhs[8][4];
    const int warp_lane = threadIdx.x & 31;
    const int load_column = (threadIdx.x / 32) * 16 + (warp_lane & 3) * 4;

    const int source_lane = (warp_lane & 7) * 4 + (warp_lane / 8);
#pragma unroll
    for (int part = 0; part < 8; ++part) {
        const int row = warp_lane / 4 + part * 8, token = begin + row, col = load_column & 127;
        float4 values = make_float4(0.f, 0.f, 0.f, 0.f);
        const bool key_field = load_column < 128;
        if (token < length && (key_field || col < value_dim)) {
            const float *pointer = key_field
                ? key + ((physical_start + token) * key_heads + kh) * K + col
                : value + ((physical_start + token) * heads + head) * value_dim + col;
            if (key_field || col + 3 < value_dim) {
                values = (reinterpret_cast<uintptr_t>(pointer) & 15u)
                    ? make_float4(pointer[0], pointer[1], pointer[2], pointer[3])
                    : *reinterpret_cast<const float4 *>(pointer);
            } else {
                values = make_float4(col < value_dim ? pointer[0] : 0.f,
                    col + 1 < value_dim ? pointer[1] : 0.f, col + 2 < value_dim ? pointer[2] : 0.f,
                    col + 3 < value_dim ? pointer[3] : 0.f);
            }
        }
        const float beta_value = betas[row];
        values.x = __fmul_rn(values.x, beta_value);
        values.y = __fmul_rn(values.y, beta_value);
        values.z = __fmul_rn(values.z, beta_value);
        values.w = __fmul_rn(values.w, beta_value);
        if (key_field) {
            const float decay = decays[instance * 128 + row];
            values.x = __fmul_rn(values.x, decay);
            values.y = __fmul_rn(values.y, decay);
            values.z = __fmul_rn(values.z, decay);
            values.w = __fmul_rn(values.w, decay);
        }
        rhs[part][0] = __shfl_sync(0xffffffff, values.x, source_lane);
        rhs[part][1] = __shfl_sync(0xffffffff, values.y, source_lane);
        rhs[part][2] = __shfl_sync(0xffffffff, values.z, source_lane);
        rhs[part][3] = __shfl_sync(0xffffffff, values.w, source_lane);
    }
    qwen35_c64prepvload2_rhs_phase<0>(rhs, lower, lane);
    qwen35_c64prepvload2_rhs_phase<1>(rhs, lower, lane);
    qwen35_c64prepvload2_rhs_phase<2>(rhs, lower, lane);
    qwen35_c64prepvload2_rhs_phase<3>(rhs, lower, lane);
    qwen35_c64prepvload2_rhs_phase<4>(rhs, lower, lane);
    qwen35_c64prepvload2_rhs_phase<5>(rhs, lower, lane);
    qwen35_c64prepvload2_rhs_phase<6>(rhs, lower, lane);
    qwen35_c64prepvload2_rhs_phase<7>(rhs, lower, lane);
    __syncthreads();
    auto *tile = reinterpret_cast<float *>(qwen35_c64prepvload2_storage);
#pragma unroll
    for (int field = 0; field < 2; ++field) {
        if ((first_column >= 128) == bool(field)) {
#pragma unroll
            for (int part = 0; part < 8; ++part) {
#pragma unroll
                for (int column = 0; column < 4; ++column)
                    tile[(lane + part * 8) * S + ((first_column + column) & 127)] =
                        rhs[part][column];
            }
        }
        __syncthreads();
        for (int i = threadIdx.x; i < C * K; i += blockDim.x) {
            const int t = i / K, d = i % K;
            if (field == 0)
                w[(instance * C + t) * K + d] = tile[t * S + d];
            else if (d < value_dim)
                u[(instance * C + t) * value_dim + d] = tile[t * S + d];
        }
        __syncthreads();
    }
}

extern "C" __global__ __launch_bounds__(512, 2) void qwen35_c64prepvload2_uniform_prepare_f32(
    const float *q, const float *k, const float *v, const float *g, const float *b, float *w,
    float *u, float *c, float *d, int length, int heads, int key_heads, int vd) {
    const int chunks = (length + 63) / 64, instance = blockIdx.x, bh = instance / chunks,
              chunk = instance % chunks;
    const int batch = bh / heads, head = bh % heads;
    qwen35_c64prepvload2_prepare(q, k, v, g, b, w, u, c, d, batch * length, chunk * 64, length,
        head, heads, key_heads, vd, instance);
}
extern "C" __global__ __launch_bounds__(512, 2) void qwen35_c64prepvload2_varlen_prepare_f32(
    const float *q, const float *k, const float *v, const float *g, const float *b, float *w,
    float *u, float *c, float *d, const unsigned *cumulative, const unsigned *chunks,
    int sequence_count, int total_length, int heads, int key_heads, int vd) {
    const int global_chunk = blockIdx.x / heads, head = blockIdx.x % heads;
    int lo = 0, hi = sequence_count;
    while (lo < hi) {
        const int mid = (lo + hi + 1) / 2;
        if (chunks[mid] <= global_chunk)
            lo = mid;
        else
            hi = mid - 1;
    }
    const int start = cumulative[lo], length = cumulative[lo + 1] - start,
              begin = (global_chunk - int(chunks[lo])) * 64;
    qwen35_c64prepvload2_prepare(q, k, v, g, b, w, u, c, d, start, begin, length, head, heads,
        key_heads, vd, global_chunk * heads + head);
}

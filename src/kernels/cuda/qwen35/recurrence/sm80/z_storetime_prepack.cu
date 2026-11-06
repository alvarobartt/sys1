#include <stdint.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math.h>
__device__ __forceinline__ void qwen35_store15_half_mma16816(
    float *c, const unsigned *a, const unsigned *b) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

__device__ __forceinline__ void qwen35_store15_gramld_load4(unsigned *out, const __half *pointer) {
    const unsigned address = unsigned(__cvta_generic_to_shared(pointer));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3},[%4];"
                 : "=r"(out[0]), "=r"(out[1]), "=r"(out[2]), "=r"(out[3])
                 : "r"(address)
                 : "memory");
}
__device__ __forceinline__ void qwen35_store15_gramld_load2(unsigned *out, const __half *pointer) {
    const unsigned address = unsigned(__cvta_generic_to_shared(pointer));
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1},[%2];"
                 : "=r"(out[0]), "=r"(out[1])
                 : "r"(address)
                 : "memory");
}

__device__ __forceinline__ int qwen35_c64storepack15_coefficient_index(int row, int col) {
    return col * 64 + (row / 32) * 32 + (row & 7) * 4 + ((row / 8) & 3);
}

__device__ __forceinline__ int qwen35_c64storepack15_half_index(int row, int col) {
    return row * 128 + (col ^ ((row & 7) * 8));
}

__device__ __forceinline__ int qwen35_store15_swizzle(int row, int col, int stride) {
    return row * stride + (col ^ ((row & 7) * 8));
}

__device__ __forceinline__ void qwen35_c64storepack15_gram(const float *query, const float *key,
    const float *beta, const float *prefix, int physical_start, int begin, int length, int head,
    int heads, int kh, int key_heads, __half *storage, float *lower, float *causal, __half *packed,
    unsigned *flags) {
    constexpr int C = 64, K = 128;
    __half *ak = storage;
    __half *aq = ak + C * K;
    __half *bk = aq + C * K;
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32, group = lane / 4, tid = lane % 4;
    const int col = (warp % 8) * 8;
    bool bad_kk = false, bad_qk = false;
    for (int i = threadIdx.x * 4; i < C * K; i += blockDim.x * 4) {
        const int t = i / K, d = i % K, token = begin + t;
        float4 keys = make_float4(0.f, 0.f, 0.f, 0.f), queries = keys;
        float beta_value = 0.f;
        if (token < length) {
            const int offset = ((physical_start + token) * key_heads + kh) * K + d;
            const float *kp = key + offset;
            const float *qp = query + offset;
            keys = (reinterpret_cast<uintptr_t>(kp) & 15u) ? make_float4(kp[0], kp[1], kp[2], kp[3])
                                                           : *reinterpret_cast<const float4 *>(kp);
            queries = (reinterpret_cast<uintptr_t>(qp) & 15u)
                ? make_float4(qp[0], qp[1], qp[2], qp[3])
                : *reinterpret_cast<const float4 *>(qp);
            beta_value = beta[(physical_start + token) * heads + head];
        }
        const float kvals[4] = {keys.x, keys.y, keys.z, keys.w};
        const float qvals[4] = {queries.x, queries.y, queries.z, queries.w};
        __half kv[4], qv[4], bkv[4];
#pragma unroll
        for (int item = 0; item < 4; ++item) {
            const float kval = kvals[item], qval = qvals[item];
            const float scaled = token < length ? __fmul_rn(kval, beta_value) : 0.f;
            bad_kk |= !(fabsf(kval) <= 32768.f) || !(fabsf(scaled) <= 32768.f);
            bad_qk |= !(fabsf(kval) <= 32768.f) || !(fabsf(qval) <= 32768.f);
            kv[item] = __float2half_rn(kval);
            qv[item] = __float2half_rn(qval);
            bkv[item] = __float2half_rn(scaled);
        }
        const int at = qwen35_c64storepack15_half_index(t, d);
#pragma unroll
        for (int which = 0; which < 3; ++which) {
            __half *destination = which == 0 ? ak : which == 1 ? aq : bk;
            const __half *values = which == 0 ? bkv : which == 1 ? qv : kv;
            const __half_raw v0 = values[0], v1 = values[1], v2 = values[2], v3 = values[3];
            *reinterpret_cast<uint2 *>(destination + at) = make_uint2(
                unsigned(v0.x) | (unsigned(v1.x) << 16), unsigned(v2.x) | (unsigned(v3.x) << 16));
        }
    }
    const bool fallback_kk = __syncthreads_or(bad_kk);
    const bool fallback_qk = __syncthreads_or(bad_qk);
    float high_kk[2][4] = {}, high_qk[2][4] = {};
    for (int k = 0; k < K; k += 16) {
        unsigned bh[2];
        const int br = col + (lane & 7), bc = k + ((lane / 8) & 1) * 8;
        if (!fallback_kk || !fallback_qk)
            qwen35_store15_gramld_load2(bh, bk + qwen35_c64storepack15_half_index(br, bc));
#pragma unroll
        for (int part = 0; part < 2; ++part) {
            const int row = (warp / 8) * 16 + part * 32;
            const int ar = row + (lane & 15), ac = k + (lane / 16) * 8;
            unsigned ah[4];
            if (!fallback_kk) {
                qwen35_store15_gramld_load4(ah, ak + qwen35_c64storepack15_half_index(ar, ac));
                qwen35_store15_half_mma16816(high_kk[part], ah, bh);
            }
            if (!fallback_qk) {
                qwen35_store15_gramld_load4(ah, aq + qwen35_c64storepack15_half_index(ar, ac));
                qwen35_store15_half_mma16816(high_qk[part], ah, bh);
            }
        }
    }
    __syncthreads();
    unsigned pack_unsafe = 0;
#pragma unroll
    for (int part = 0; part < 2; ++part) {
        const int row = (warp / 8) * 16 + part * 32;
#pragma unroll
        for (int p = 0; p < 4; ++p) {
            const int t = row + group + (p / 2) * 8, j = col + tid * 2 + p % 2;
            float kk = high_kk[part][p], qk = high_qk[part][p];
            if (fallback_kk || fallback_qk) {
                const int token = begin + t, previous = begin + j;
                if (fallback_kk)
                    kk = 0.f;
                if (fallback_qk)
                    qk = 0.f;
                if (token < length && previous < length) {
                    for (int d = 0; d < K; ++d) {
                        const int offset = ((physical_start + token) * key_heads + kh) * K + d;
                        const float previous_key =
                            key[((physical_start + previous) * key_heads + kh) * K + d];
                        if (fallback_kk) {
                            const float lhs = __fmul_rn(
                                key[offset], beta[(physical_start + token) * heads + head]);
                            kk = __fmaf_rn(lhs, previous_key, kk);
                        }
                        if (fallback_qk)
                            qk = __fmaf_rn(query[offset], previous_key, qk);
                    }
                }
            }
            const float factor = expf(__fsub_rn(prefix[t], prefix[j]));
            lower[qwen35_c64storepack15_coefficient_index(t, j)] =
                j < t ? __fmul_rn(kk, factor) : 0.f;
            const float x = j <= t ? __fmul_rn(qk, factor) : 0.f;
            causal[t * C + j] = x;
            packed[16384 + qwen35_store15_swizzle(t, j, 64)] = __float2half_rn(x);
            pack_unsafe |= !(fabsf(x) <= 32768.f) ? 4u : 0u;
        }
    }
    if (pack_unsafe)
        atomicOr(flags, pack_unsafe);
    __syncthreads();
}

template <int Eighth>
__device__ __forceinline__ void qwen35_c64storepack15_rhs_phase(
    float (&rhs)[8][4], const float *lower, int lane) {
#pragma unroll 1
    for (int j = Eighth * 8; j < (Eighth + 1) * 8; ++j) {
        float coefficient[8];
        const float *aligned = lower + qwen35_c64storepack15_coefficient_index(lane, j);
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

__device__ __forceinline__ float4 load4_75(const float *p) {
    return (reinterpret_cast<uintptr_t>(p) & 15u) ? make_float4(p[0], p[1], p[2], p[3])
                                                  : *reinterpret_cast<const float4 *>(p);
}
__device__ __forceinline__ void half4_75(__half *p, const float (&v)[4]) {
    unsigned h[4];
#pragma unroll
    for (int j = 0; j < 4; ++j)
        h[j] = __half_as_ushort(__float2half_rn(v[j]));
    *reinterpret_cast<uint2 *>(p) = make_uint2(h[0] | (h[1] << 16), h[2] | (h[3] << 16));
}
__device__ __forceinline__ void qwen35_c64storepack15_prepare(const float *query, const float *key,
    const float *value, const float *gate, const float *beta, float *w, float *u, float *causal,
    float *decays, int physical_start, int begin, int length, int head, int heads, int key_heads,
    int value_dim, int instance, __half *packed, unsigned *flags) {
    constexpr int C = 64, K = 128, S = 129;
    extern __shared__ __align__(32) unsigned char qwen35_c64storepack15_storage[];
    auto *ahalf = reinterpret_cast<__half *>(qwen35_c64storepack15_storage);
    auto *alow = ahalf + C * 64;
    auto *bhalf = alow + C * 64;
    auto *blow = bhalf + C * 64;
    auto *lower = reinterpret_cast<float *>(qwen35_c64storepack15_storage);
    auto *prefix = reinterpret_cast<float *>(qwen35_c64storepack15_storage + 49152);
    auto *betas = prefix + C;
    packed += size_t(instance) * 28672;
    flags += instance;
    if (threadIdx.x == 0)
        *flags = 0;
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
    qwen35_c64storepack15_gram(query, key, beta, prefix, physical_start, begin, length, head, heads,
        kh, key_heads, ahalf, lower, causal + instance * C * C, packed, flags);
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
    qwen35_c64storepack15_rhs_phase<0>(rhs, lower, lane);
    qwen35_c64storepack15_rhs_phase<1>(rhs, lower, lane);
    qwen35_c64storepack15_rhs_phase<2>(rhs, lower, lane);
    qwen35_c64storepack15_rhs_phase<3>(rhs, lower, lane);
    qwen35_c64storepack15_rhs_phase<4>(rhs, lower, lane);
    qwen35_c64storepack15_rhs_phase<5>(rhs, lower, lane);
    qwen35_c64storepack15_rhs_phase<6>(rhs, lower, lane);
    qwen35_c64storepack15_rhs_phase<7>(rhs, lower, lane);
    __syncthreads();
    auto *tile = reinterpret_cast<float *>(qwen35_c64storepack15_storage);
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

        auto *staged_key = reinterpret_cast<__half *>(qwen35_c64storepack15_storage + 33280);
        unsigned pack_unsafe = 0;
        for (int i = threadIdx.x * 4; i < C * K; i += blockDim.x * 4) {
            const int t = i / K, d = i % K;
            float x[4];
#pragma unroll
            for (int j = 0; j < 4; ++j)
                x[j] = tile[t * S + d + j];
            if (field == 0) {
                float *dst = w + (size_t(instance) * C + t) * K + d;
                if (!(reinterpret_cast<uintptr_t>(dst) & 15u))
                    *reinterpret_cast<float4 *>(dst) = make_float4(x[0], x[1], x[2], x[3]);
                else {
#pragma unroll
                    for (int j = 0; j < 4; ++j)
                        dst[j] = x[j];
                }
                const size_t at = (size_t(physical_start + begin + t) * key_heads + kh) * K + d;
                float4 q = make_float4(0.f, 0.f, 0.f, 0.f), k = q;
                if (begin + t < length) {
                    q = load4_75(query + at);
                    k = load4_75(key + at);
                }
                float qs[4] = {q.x, q.y, q.z, q.w}, ks[4] = {k.x, k.y, k.z, k.w};
                const float dq = decays[size_t(instance) * 128 + t],
                            dk = decays[size_t(instance) * 128 + 64 + t];
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    qs[j] = begin + t < length ? __fmul_rn(dq, qs[j]) : 0.f;
                    ks[j] = begin + t < length ? __fmul_rn(dk, ks[j]) : 0.f;
                    pack_unsafe |= (!(fabsf(x[j]) <= 32768.f) ? 1u : 0u) |
                        (!(fabsf(qs[j]) <= 32768.f) ? 2u : 0u) |
                        (!(fabsf(ks[j]) <= 32768.f) ? 8u : 0u);
                }
                const int offset = qwen35_store15_swizzle(t, d, 128);
                half4_75(packed + offset, x);
                half4_75(packed + 8192 + offset, qs);
                half4_75(staged_key + offset, ks);
            } else {
                float *dst = u + (size_t(instance) * C + t) * value_dim + d;
                if (d + 3 < value_dim && !(reinterpret_cast<uintptr_t>(dst) & 15u))
                    *reinterpret_cast<float4 *>(dst) = make_float4(x[0], x[1], x[2], x[3]);
                else {
#pragma unroll
                    for (int j = 0; j < 4; ++j)
                        if (d + j < value_dim)
                            dst[j] = x[j];
                }
            }
        }
        if (pack_unsafe)
            atomicOr(flags, pack_unsafe);
        __syncthreads();
        if (field == 0) {
            for (int i = threadIdx.x * 4; i < 128 * 64; i += blockDim.x * 4) {
                const int d = i / 64, t = i % 64;
                unsigned h[4];
#pragma unroll
                for (int j = 0; j < 4; ++j)
                    h[j] = __half_as_ushort(staged_key[qwen35_store15_swizzle(t + j, d, 128)]);
                *reinterpret_cast<uint2 *>(packed + 20480 + qwen35_store15_swizzle(d, t, 64)) =
                    make_uint2(h[0] | (h[1] << 16), h[2] | (h[3] << 16));
            }
        }
    }
}

extern "C" __global__ __launch_bounds__(512, 2) void qwen35_c64storepack15_uniform_prepare_f32(
    const float *q, const float *k, const float *v, const float *g, const float *b, float *w,
    float *u, float *c, float *d, int length, int heads, int key_heads, int vd, __half *packed,
    unsigned *flags) {
    const int chunks = (length + 63) / 64, instance = blockIdx.x, bh = instance / chunks,
              chunk = instance % chunks;
    const int batch = bh / heads, head = bh % heads;
    qwen35_c64storepack15_prepare(q, k, v, g, b, w, u, c, d, batch * length, chunk * 64, length,
        head, heads, key_heads, vd, instance, packed, flags);
}
extern "C" __global__ __launch_bounds__(512, 2) void qwen35_c64storepack15_varlen_prepare_f32(
    const float *q, const float *k, const float *v, const float *g, const float *b, float *w,
    float *u, float *c, float *d, const unsigned *cumulative, const unsigned *chunks,
    int sequence_count, int total_length, int heads, int key_heads, int vd, __half *packed,
    unsigned *flags) {
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
    qwen35_c64storepack15_prepare(q, k, v, g, b, w, u, c, d, start, begin, length, head, heads,
        key_heads, vd, global_chunk * heads + head, packed, flags);
}

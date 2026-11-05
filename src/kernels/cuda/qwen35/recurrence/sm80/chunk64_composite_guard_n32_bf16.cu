#include <stdint.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
__device__ __forceinline__ void qwen35_c64zguard_n32_load4(unsigned *out, const __half *pointer) {
    const unsigned address = unsigned(__cvta_generic_to_shared(pointer));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3},[%4];"
                 : "=r"(out[0]), "=r"(out[1]), "=r"(out[2]), "=r"(out[3])
                 : "r"(address)
                 : "memory");
}
__device__ __forceinline__ void qwen35_c64zguard_n32_load2(unsigned *out, const __half *pointer) {
    const unsigned address = unsigned(__cvta_generic_to_shared(pointer));
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1},[%2];"
                 : "=r"(out[0]), "=r"(out[1])
                 : "r"(address)
                 : "memory");
}
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math.h>

__device__ __forceinline__ void qwen35_c64zguard_n32_mma(float *c, const unsigned *a,
                                                       const unsigned *b) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

__device__ __forceinline__ void qwen35_c64zguard_n32_load2_trans(unsigned *out,
                                                               const __half *pointer) {
    const unsigned address = unsigned(__cvta_generic_to_shared(pointer));
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1},[%2];"
                 : "=r"(out[0]), "=r"(out[1])
                 : "r"(address)
                 : "memory");
}
__device__ __forceinline__ int qwen35_c64zguard_n32_bindex(int row, int col) {

    return (row * 32 + col) ^ ((row & 7) << 3);
}
template <int Phase>
__device__ __forceinline__ float
qwen35_c64zguard_n32_a(int r, int k, const float *q, const float *key, const float *w,
                     const float *causal, const float *decay, int instance, int begin, int length,
                     int qk_base, int qk_stride) {
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

__device__ __forceinline__ int qwen35_c64zguard_n32_swizzle(int row, int col, int stride) {
    return row * stride + (col ^ ((row & 7) << 3));
}

template <int Phase, int N, bool ReuseB = false>
__device__ __forceinline__ void
qwen35_c64zguard_n32_product(const float *q, const float *key, const float *w, const float *causal,
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
    if constexpr (Phase < 3) {
        for (int index = threadIdx.x * 4; index < Rows * Inner; index += blockDim.x * 4) {
            const int row = index / Inner, col = index % Inner;
            float4 values;
            if constexpr (Phase == 0) {
                values = *reinterpret_cast<const float4 *>(w + (instance * 64 + row) * 128 + col);
            } else if constexpr (Phase == 2) {
                values =
                    *reinterpret_cast<const float4 *>(causal + (instance * 64 + row) * 64 + col);
            } else {
                if (begin + row < length) {
                    const float *pointer = q + qk_base + (begin + row) * qk_stride + col;

                    values = (reinterpret_cast<uintptr_t>(pointer) & 15u)
                                 ? make_float4(pointer[0], pointer[1], pointer[2], pointer[3])
                                 : *reinterpret_cast<const float4 *>(pointer);
                    const float scale = decay[instance * 128 + row];
                    values.x = __fmul_rn(scale, values.x);
                    values.y = __fmul_rn(scale, values.y);
                    values.z = __fmul_rn(scale, values.z);
                    values.w = __fmul_rn(scale, values.w);
                } else
                    values = make_float4(0.f, 0.f, 0.f, 0.f);
            }
            unsafe_a |= !(fabsf(values.x) <= 32768.f) || !(fabsf(values.y) <= 32768.f) ||
                        !(fabsf(values.z) <= 32768.f) || !(fabsf(values.w) <= 32768.f);
            const __half2 hi0 = __floats2half2_rn(values.x, values.y);
            const __half2 hi1 = __floats2half2_rn(values.z, values.w);
            const float2 round0 = __half22float2(hi0), round1 = __half22float2(hi1);
            const __half2 lo0 = __floats2half2_rn(__fmul_rn(__fsub_rn(values.x, round0.x), 4096.f),
                                                  __fmul_rn(__fsub_rn(values.y, round0.y), 4096.f));
            const __half2 lo1 = __floats2half2_rn(__fmul_rn(__fsub_rn(values.z, round1.x), 4096.f),
                                                  __fmul_rn(__fsub_rn(values.w, round1.y), 4096.f));
            const __half2_raw h0 = hi0, h1 = hi1, l0 = lo0, l1 = lo1;
            const uint2 high = make_uint2(unsigned(h0.x) | (unsigned(h0.y) << 16),
                                          unsigned(h1.x) | (unsigned(h1.y) << 16));
            const uint2 low = make_uint2(unsigned(l0.x) | (unsigned(l0.y) << 16),
                                         unsigned(l1.x) | (unsigned(l1.y) << 16));

            const int address = qwen35_c64zguard_n32_swizzle(row, col, Inner);
            *reinterpret_cast<uint2 *>(ahalf + address) = high;
            *reinterpret_cast<uint2 *>(alow + address) = low;
        }
    } else {
        for (int index = threadIdx.x; index < Rows * Inner; index += blockDim.x) {

            const int row = Phase == 3 ? index % Rows : index / Inner;
            const int col = Phase == 3 ? index / Rows : index % Inner;
            const float x = qwen35_c64zguard_n32_a<Phase>(
                row, col, q, key, w, causal, decay, instance, begin, length, qk_base, qk_stride);
            unsafe_a |= !(fabsf(x) <= 32768.f);
            const auto high = __float2half_rn(x);
            const int address = qwen35_c64zguard_n32_swizzle(row, col, Inner);
            ahalf[address] = high;
            alow[address] = __float2half_rn(__fmul_rn(__fsub_rn(x, __half2float(high)), 4096.f));
        }
    }
    unsigned unsafe_b = 0;
    static_assert(N == 32);
    if constexpr (Phase == 0) {
#pragma unroll
        for (int i = 0; i < 4; i += 2) {
            const int row = r + group + (i / 2) * 8, col = c + tid * 2;
            const float x = carry[i], y = carry[i + 1];
            if (!(fabsf(x) <= 32768.f) || !(fabsf(y) <= 32768.f))
                unsafe_b |= 1u << (col / 16);
            const __half2 hi = __floats2half2_rn(x, y);
            const float2 rounded = __half22float2(hi);
            const __half2 lo = __floats2half2_rn(__fmul_rn(__fsub_rn(x, rounded.x), 4096.f),
                                                 __fmul_rn(__fsub_rn(y, rounded.y), 4096.f));
            const int address = qwen35_c64zguard_n32_bindex(row, col);
            *reinterpret_cast<__half2 *>(bhalf + address) = hi;
            *reinterpret_cast<__half2 *>(blow + address) = lo;
        }
        if (unsafe_b)
            atomicOr(cached_b_unsafe, int(unsafe_b));
    } else if constexpr (ReuseB) {
        unsafe_b = unsigned(*cached_b_unsafe);
    } else {
        for (int index = threadIdx.x * 2; index < Inner * N; index += blockDim.x * 2) {
            const int row = index / N, col = index % N;
            const float2 pair = *reinterpret_cast<const float2 *>(updates + row * S + col);
            if (!(fabsf(pair.x) <= 32768.f) || !(fabsf(pair.y) <= 32768.f))
                unsafe_b |= 1u << (col / 16);
            const __half2 hi = __floats2half2_rn(pair.x, pair.y);
            const float2 rounded = __half22float2(hi);
            const __half2 lo = __floats2half2_rn(__fmul_rn(__fsub_rn(pair.x, rounded.x), 4096.f),
                                                 __fmul_rn(__fsub_rn(pair.y, rounded.y), 4096.f));
            const int address = qwen35_c64zguard_n32_bindex(row, col);
            *reinterpret_cast<__half2 *>(bhalf + address) = hi;
            *reinterpret_cast<__half2 *>(blow + address) = lo;
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
            qwen35_c64zguard_n32_load4(ah, ahalf + qwen35_c64zguard_n32_swizzle(ar, ac, Inner));
            qwen35_c64zguard_n32_load4(al, alow + qwen35_c64zguard_n32_swizzle(ar, ac, Inner));
            const int tr = k + (lane & 15), tc = c;
            qwen35_c64zguard_n32_load2_trans(bh, bhalf + qwen35_c64zguard_n32_bindex(tr, tc));
            qwen35_c64zguard_n32_load2_trans(bl, blow + qwen35_c64zguard_n32_bindex(tr, tc));
            qwen35_c64zguard_n32_mma(high, ah, bh);
            qwen35_c64zguard_n32_mma(cross, ah, bl);
            qwen35_c64zguard_n32_mma(cross, al, bh);
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
                    value =
                        __fadd_rn(value, __fmul_rn(qwen35_c64zguard_n32_a<Phase>(
                                                       row, k, q, key, w, causal, decay, instance,
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
qwen35_c64zguard_n32_state_impl(const float *q, const float *key, const float *w, const float *u,
                              const float *causal, const float *decay, const float *first_zero,
                              __nv_bfloat16 *output, int physical_start, int length, int head,
                              int heads, int key_heads, int value_dim, int instance_base,
                              int instance_stride) {
    constexpr int S = N + 4;
    extern __shared__ __align__(32) unsigned char qwen35_c64zguard_n32_storage[];
    auto *updates = reinterpret_cast<float *>(qwen35_c64zguard_n32_storage);
    auto *ahalf = reinterpret_cast<__half *>(updates + 64 * S);
    auto *alow = ahalf + 8192;
    auto *bhalf = alow + 8192;
    auto *blow = bhalf + 4096;
    auto *cached_b_unsafe = reinterpret_cast<int *>(blow + 4096);
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int r = (warp / (N / 8)) * 16, c = (warp % (N / 8)) * 8, group = lane / 4, tid = lane % 4;
    const int kh = head / (heads / key_heads), first_col = blockIdx.y * N;
    const int qk_base = (physical_start * key_heads + kh) * 128, qk_stride = key_heads * 128;
    float carry[4] = {};
    for (int chunk = 0; chunk < (length + 63) / 64; ++chunk) {
        const int instance = instance_base + chunk * instance_stride, begin = chunk * 64;
        const bool skip_zero = chunk == 0 && first_zero[instance] == 1.f;
        if (threadIdx.x == 0)
            *cached_b_unsafe = 0;
        for (int index = threadIdx.x; index < 64 * N; index += blockDim.x) {
            const int row = index / N, col = first_col + index % N;
            const float initial =
                col < value_dim ? u[(instance * 64 + row) * value_dim + col] : 0.f;
            updates[row * S + index % N] = skip_zero ? __fsub_rn(initial, 0.f) : initial;
        }
        __syncthreads();
        float out[4] = {};
        if (!skip_zero) {
            qwen35_c64zguard_n32_product<0, N>(q, key, w, causal, decay, updates, carry, out, ahalf,
                                             alow, bhalf, blow, cached_b_unsafe, instance, begin,
                                             length, qk_base, qk_stride);
            qwen35_c64zguard_n32_product<1, N, true>(q, key, w, causal, decay, updates, carry, out,
                                                   ahalf, alow, bhalf, blow, cached_b_unsafe,
                                                   instance, begin, length, qk_base, qk_stride);
        }
        qwen35_c64zguard_n32_product<2, N>(q, key, w, causal, decay, updates, carry, out, ahalf, alow,
                                         bhalf, blow, cached_b_unsafe, instance, begin, length,
                                         qk_base, qk_stride);
        if (warp < (64 / 16) * (N / 8)) {
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int row = r + group + (i / 2) * 8, col = first_col + c + tid * 2 + i % 2;
                if (begin + row < length && col < value_dim)
                    output[((physical_start + begin + row) * heads + head) * value_dim + col] =
                        __float2bfloat16_rn(out[i]);
            }
        }
        qwen35_c64zguard_n32_product<3, N, true>(q, key, w, causal, decay, updates, carry, out, ahalf,
                                               alow, bhalf, blow, cached_b_unsafe, instance, begin,
                                               length, qk_base, qk_stride);
    }
}

extern "C" __global__ __launch_bounds__(1024, 1) void qwen35_c64zguard_n32_uniform32_f32(
    const float *q, const float *k, const float *g, const float *w, const float *u, const float *c,
    const float *d, __nv_bfloat16 *out, float *snapshots, int length, int heads, int kh, int vd) {
    const int bh = blockIdx.x, chunks = (length + 63) / 64;
    qwen35_c64zguard_n32_state_impl<32>(q, k, w, u, c, d, snapshots, out, (bh / heads) * length,
                                      length, bh % heads, heads, kh, vd, bh * chunks, 1);
}
extern "C" __global__ __launch_bounds__(1024, 1) void qwen35_c64zguard_n32_varlen32_f32(
    const float *q, const float *k, const float *g, const float *w, const float *u, const float *c,
    const float *d, __nv_bfloat16 *out, float *snapshots, const unsigned *cumulative,
    const unsigned *chunk_offsets, const unsigned *long_ids, int total_length, int heads, int kh,
    int vd) {
    const int head = blockIdx.x % heads, seq = long_ids[blockIdx.x / heads],
              start = cumulative[seq];
    qwen35_c64zguard_n32_state_impl<32>(q, k, w, u, c, d, snapshots, out, start,
                                      cumulative[seq + 1] - start, head, heads, kh, vd,
                                      int(chunk_offsets[seq]) * heads + head, heads);
}

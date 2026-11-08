#include <stdint.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math.h>

__device__ __forceinline__ void qwen35_c64uniformvote10_n64_load4(
    unsigned *out, const __half *pointer) {
    const unsigned address = unsigned(__cvta_generic_to_shared(pointer));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3},[%4];"
                 : "=r"(out[0]), "=r"(out[1]), "=r"(out[2]), "=r"(out[3])
                 : "r"(address)
                 : "memory");
}
__device__ __forceinline__ void qwen35_c64uniformvote10_n64_load2(
    unsigned *out, const __half *pointer) {
    const unsigned address = unsigned(__cvta_generic_to_shared(pointer));
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1},[%2];"
                 : "=r"(out[0]), "=r"(out[1])
                 : "r"(address)
                 : "memory");
}
__device__ __forceinline__ void qwen35_c64uniformvote10_n64_mma(
    float *out, const unsigned *a, const unsigned *b) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                 "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
                 : "+f"(out[0]), "+f"(out[1]), "+f"(out[2]), "+f"(out[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
__device__ __forceinline__ int qwen35_c64uniformvote10_n64_swizzle(int row, int col, int stride) {
    return row * stride + (col ^ ((row & 7) << 3));
}
__device__ __forceinline__ void qwen35_c64uniformvote10_n64_load2_trans(
    unsigned *out, const __half *pointer) {
    const unsigned address = unsigned(__cvta_generic_to_shared(pointer));
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1},[%2];"
                 : "=r"(out[0]), "=r"(out[1])
                 : "r"(address)
                 : "memory");
}

__device__ __forceinline__ void qwen35_prepack6_copy(__half *dst, const __half *src, int bytes) {
    for (int i = threadIdx.x * 16; i < bytes; i += blockDim.x * 16) {
        const unsigned address =
            unsigned(__cvta_generic_to_shared(reinterpret_cast<unsigned char *>(dst) + i));
        const auto *source = reinterpret_cast<const unsigned char *>(src) + i;
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(address), "l"(source)
                     : "memory");
    }
    asm volatile("cp.async.commit_group;" ::: "memory");
    asm volatile("cp.async.wait_group 0;" ::: "memory");
}

template <int Phase>
__device__ __forceinline__ float qwen35_c64uniformvote10_n64_a(int row, int col, const float *q,
    const float *key, const float *w, const float *causal, const float *decay, int instance,
    int begin, int length, int qk_base, int qk_stride) {
    if constexpr (Phase == 0)
        return w[(instance * 64 + row) * 128 + col];
    if constexpr (Phase == 1)
        return begin + row < length
            ? __fmul_rn(decay[instance * 128 + row], q[qk_base + (begin + row) * qk_stride + col])
            : 0.f;
    if constexpr (Phase == 2)
        return causal[(instance * 64 + row) * 64 + col];
    if constexpr (Phase == 3)
        return begin + col < length ? __fmul_rn(decay[instance * 128 + 64 + col],
                                          key[qk_base + (begin + col) * qk_stride + row])
                                    : 0.f;
    return 0.f;
}

template <int Phase, bool ReuseB = false>
__device__ __forceinline__ void qwen35_c64uniformvote10_n64_product(const float *q,
    const float *key, const float *w, const float *causal, const float *decay, float *updates,
    float (&carry)[2][4], float (&out)[4], __half *ahalf, __half *alow, __half *bhalf, __half *blow,
    int *cached_b_unsafe, int instance, int begin, int length, int qk_base, int qk_stride) {
    constexpr int N = 64, S = 68, Rows = Phase == 3 ? 128 : 64, Inner = Phase < 2 ? 128 : 64,
                  Parts = Phase == 3 ? 2 : 1;
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32, group = lane / 4, tid = lane % 4;
    const int r = (warp / 8) * 16, c = (warp % 8) * 8;
    float high[Parts][4] = {}, cross[Parts][4] = {}, saved[Parts][4] = {};
#pragma unroll
    for (int part = 0; part < Parts; ++part) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            if constexpr (Phase == 0)
                saved[part][i] = updates[(r + group + (i / 2) * 8) * S + c + tid * 2 + i % 2];
            if constexpr (Phase == 2)
                saved[part][i] = __fmul_rn(out[i], 1.f);
            if constexpr (Phase == 3)
                saved[part][i] = __fmul_rn(carry[part][i], decay[instance * 128 + 63]);
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
            const __half2_raw h0 = hi0, h1 = hi1;
            const uint2 high = make_uint2(
                unsigned(h0.x) | (unsigned(h0.y) << 16), unsigned(h1.x) | (unsigned(h1.y) << 16));

            const int address = qwen35_c64uniformvote10_n64_swizzle(row, col, Inner);
            *reinterpret_cast<uint2 *>(ahalf + address) = high;
        }
    } else {
        for (int index = threadIdx.x; index < Rows * Inner; index += blockDim.x) {
            const int row = Phase == 3 ? index % Rows : index / Inner,
                      col = Phase == 3 ? index / Rows : index % Inner;
            const float x = qwen35_c64uniformvote10_n64_a<Phase>(
                row, col, q, key, w, causal, decay, instance, begin, length, qk_base, qk_stride);
            unsafe_a |= !(fabsf(x) <= 32768.f);
            const auto a = __float2half_rn(x);
            const int address = qwen35_c64uniformvote10_n64_swizzle(row, col, Inner);
            ahalf[address] = a;
        }
    }
    unsigned unsafe_b = 0;
    static_assert(N == 64);
    if constexpr (Phase == 0) {
#pragma unroll
        for (int part = 0; part < 2; ++part) {
#pragma unroll
            for (int i = 0; i < 4; i += 2) {
                const int row = r + part * 64 + group + (i / 2) * 8, col = c + tid * 2;
                const float x = carry[part][i], y = carry[part][i + 1];
                if (!(fabsf(x) <= 32768.f) || !(fabsf(y) <= 32768.f))
                    unsafe_b |= 1u << (col / 16);
                const __half2 hi = __floats2half2_rn(x, y);
                const int address = qwen35_c64uniformvote10_n64_swizzle(row, col, N);
                *reinterpret_cast<__half2 *>(bhalf + address) = hi;
            }
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
            const int address = qwen35_c64uniformvote10_n64_swizzle(row, col, N);
            *reinterpret_cast<__half2 *>(bhalf + address) = hi;
        }
    }
    if constexpr (Phase == 2) {
        if (unsafe_b)
            atomicOr(cached_b_unsafe, int(unsafe_b));
    }
    const bool any_unsafe = __syncthreads_or(unsafe_a || unsafe_b != 0);
    bool fallback_groups[4] = {};
    if (any_unsafe) {
#pragma unroll
        for (int g = 0; g < 4; ++g)
            fallback_groups[g] = __syncthreads_or(unsafe_a || (unsafe_b & (1u << g)));
    }
    const bool fallback = fallback_groups[c / 16];
    if (!fallback) {
#pragma unroll
        for (int part = 0; part < Parts; ++part) {
            for (int k = 0; k < Inner; k += 16) {
                unsigned ah[4], al[4], bh[2], bl[2];
                const int ar = r + part * 64 + (lane & 15), ac = k + (lane / 16) * 8;
                const int br = c + (lane & 7), bc = k + ((lane / 8) & 1) * 8;
                qwen35_c64uniformvote10_n64_load4(
                    ah, ahalf + qwen35_c64uniformvote10_n64_swizzle(ar, ac, Inner));
                const int tr = k + (lane & 15), tc = c;
                qwen35_c64uniformvote10_n64_load2_trans(
                    bh, bhalf + qwen35_c64uniformvote10_n64_swizzle(tr, tc, N));
                qwen35_c64uniformvote10_n64_mma(high[part], ah, bh);
            }
        }
    }
    if constexpr (Phase < 2) {
        if (any_unsafe) {
            __syncthreads();
            auto *original_state = reinterpret_cast<float *>(ahalf);

#pragma unroll
            for (int part = 0; part < 2; ++part) {
#pragma unroll
                for (int i = 0; i < 4; ++i)
                    original_state[(r + part * 64 + group + (i / 2) * 8) * N + c + tid * 2 +
                        i % 2] = carry[part][i];
            }
            __syncthreads();
        }
    }
#pragma unroll
    for (int part = 0; part < Parts; ++part) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int row = r + part * 64 + group + (i / 2) * 8, col = c + tid * 2 + i % 2;
            const float initial = (Phase == 2 || Phase == 3) ? saved[part][i] : 0.f;
            float value;
            if (fallback) {
                value = initial;
                const float *original_b =
                    Phase < 2 ? reinterpret_cast<const float *>(ahalf) : updates;
                constexpr int BStride = Phase < 2 ? N : S;
                for (int k = 0; k < Inner; ++k)
                    value = __fadd_rn(value,
                        __fmul_rn(qwen35_c64uniformvote10_n64_a<Phase>(row, k, q, key, w, causal,
                                      decay, instance, begin, length, qk_base, qk_stride),
                            original_b[k * BStride + col]));
            } else
                value = __fadd_rn(
                    initial, __fadd_rn(high[part][i], __fmul_rn(cross[part][i], 0x1p-12f)));
            if constexpr (Phase == 0)
                updates[row * S + col] = __fsub_rn(saved[part][i], value);
            if constexpr (Phase == 1 || Phase == 2)
                out[i] = value;
            if constexpr (Phase == 3)
                carry[part][i] = value;
        }
    }
    if constexpr (Phase == 1) {
        if (threadIdx.x == 0)
            *cached_b_unsafe = 0;
    }
    __syncthreads();
}

__device__ __forceinline__ void qwen35_c64uniformvote10_n64_product23(const float *q,
    const float *key, const float *w, const float *causal, const float *decay, float *updates,
    float (&carry)[2][4], float (&out)[4], __half *scratch, int *cached_b_unsafe, int instance,
    int begin, int length, int qk_base, int qk_stride, const __half *packed,
    const unsigned *pack_flags) {
    constexpr int N = 64, S = 68, Inner = 64;
    auto *ahalf = scratch;
    auto *alow = ahalf + 192 * Inner;
    auto *bhalf = alow + 192 * Inner;
    auto *blow = bhalf + Inner * N;
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32, group = lane / 4, tid = lane % 4;
    const int r = (warp / 8) * 16, c = (warp % 8) * 8;
    float high[3][4] = {}, cross[3][4] = {};
    const unsigned packed_flags = pack_flags[instance];
    const bool unsafe_l = packed_flags & 4u, unsafe_k = packed_flags & 8u;
    qwen35_prepack6_copy(ahalf, packed + size_t(instance) * 28672 + 16384, 24576);
    unsigned unsafe_b = 0;
    for (int index = threadIdx.x * 2; index < Inner * N; index += blockDim.x * 2) {
        const int row = index / N, col = index % N;
        const float2 pair = *reinterpret_cast<const float2 *>(updates + row * S + col);
        if (!(fabsf(pair.x) <= 32768.f) || !(fabsf(pair.y) <= 32768.f))
            unsafe_b |= 1u << (col / 16);
        const __half2 hi = __floats2half2_rn(pair.x, pair.y);
        const int address = qwen35_c64uniformvote10_n64_swizzle(row, col, N);
        *reinterpret_cast<__half2 *>(bhalf + address) = hi;
    }
    if (unsafe_b)
        atomicOr(cached_b_unsafe, int(unsafe_b));

    __syncthreads();
    const bool bad_l = unsafe_l;
    const bool bad_k = unsafe_k;
    const bool bad_column = (unsigned(*cached_b_unsafe) & (1u << (c / 16))) != 0;
    for (int k = 0; k < Inner; k += 16) {
        unsigned bh[2], bl[2];
        const int tr = k + (lane & 15);
        qwen35_c64uniformvote10_n64_load2_trans(
            bh, bhalf + qwen35_c64uniformvote10_n64_swizzle(tr, c, N));
#pragma unroll
        for (int part = 0; part < 3; ++part) {
            if (!bad_column && !(part == 0 ? bad_l : bad_k)) {
                unsigned ah[4], al[4];
                const int ar = r + part * 64 + (lane & 15), ac = k + (lane / 16) * 8;
                qwen35_c64uniformvote10_n64_load4(
                    ah, ahalf + qwen35_c64uniformvote10_n64_swizzle(ar, ac, Inner));
                qwen35_c64uniformvote10_n64_mma(high[part], ah, bh);
            }
        }
    }
#pragma unroll
    for (int part = 0; part < 3; ++part) {
        const bool fallback = bad_column || (part == 0 ? bad_l : bad_k);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int row = r + (part == 0 ? 0 : (part - 1) * 64) + group + (i / 2) * 8,
                      col = c + tid * 2 + i % 2;

            const float initial = part == 0
                ? __fmul_rn(out[i], 1.f)
                : __fmul_rn(carry[part - 1][i], decay[instance * 128 + 63]);
            float value;
            if (fallback) {
                value = initial;
                for (int k = 0; k < Inner; ++k) {
                    const float a = part == 0
                        ? qwen35_c64uniformvote10_n64_a<2>(row, k, q, key, w, causal, decay,
                              instance, begin, length, qk_base, qk_stride)
                        : qwen35_c64uniformvote10_n64_a<3>(row, k, q, key, w, causal, decay,
                              instance, begin, length, qk_base, qk_stride);
                    value = __fadd_rn(value, __fmul_rn(a, updates[k * S + col]));
                }
            } else
                value = __fadd_rn(
                    initial, __fadd_rn(high[part][i], __fmul_rn(cross[part][i], 0x1p-12f)));
            if (part == 0)
                out[i] = value;
            else
                carry[part - 1][i] = value;
        }
    }

    __syncthreads();
}

__device__ __forceinline__ void qwen35_c64uniformvote10_n64_product01(const float *q,
    const float *key, const float *w, const float *causal, const float *decay, float *updates,
    const float (&original_u)[4], float (&carry)[2][4], float (&out)[4], __half *scratch,
    int *cached_b_unsafe, int instance, int begin, int length, int qk_base, int qk_stride,
    const __half *packed, const unsigned *pack_flags) {
    constexpr int N = 64, S = 68, Inner = 128;
    auto *ahalf = scratch;
    auto *alow = ahalf + 128 * Inner;
    auto *bhalf = alow + 128 * Inner;
    auto *blow = bhalf + Inner * N;
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32, group = lane / 4, tid = lane % 4;
    const int r = (warp / 8) * 16, c = (warp % 8) * 8;
    float high[2][4] = {}, cross[2][4] = {};
    const unsigned packed_flags = pack_flags[instance];
    const bool unsafe_w = packed_flags & 1u, unsafe_q = packed_flags & 2u;
    qwen35_prepack6_copy(ahalf, packed + size_t(instance) * 28672, 32768);
    unsigned unsafe_b = 0;
#pragma unroll
    for (int part = 0; part < 2; ++part) {
#pragma unroll
        for (int i = 0; i < 4; i += 2) {
            const int row = r + part * 64 + group + (i / 2) * 8, col = c + tid * 2;
            const float x = carry[part][i], y = carry[part][i + 1];
            if (!(fabsf(x) <= 32768.f) || !(fabsf(y) <= 32768.f))
                unsafe_b |= 1u << (col / 16);
            const __half2 hi = __floats2half2_rn(x, y);
            const int address = qwen35_c64uniformvote10_n64_swizzle(row, col, N);
            *reinterpret_cast<__half2 *>(bhalf + address) = hi;
        }
    }
    if (unsafe_b)
        atomicOr(cached_b_unsafe, int(unsafe_b));

    __syncthreads();
    const bool bad_w = unsafe_w;
    const bool bad_q = unsafe_q;
    const unsigned bad_b = unsigned(*cached_b_unsafe);
    const bool bad_column = (bad_b & (1u << (c / 16))) != 0;
    for (int k = 0; k < Inner; k += 16) {
        unsigned bh[2], bl[2];
        const int tr = k + (lane & 15);
        qwen35_c64uniformvote10_n64_load2_trans(
            bh, bhalf + qwen35_c64uniformvote10_n64_swizzle(tr, c, N));
#pragma unroll
        for (int part = 0; part < 2; ++part) {
            if (!bad_column && !(part == 0 ? bad_w : bad_q)) {
                unsigned ah[4], al[4];
                const int ar = r + part * 64 + (lane & 15), ac = k + (lane / 16) * 8;
                qwen35_c64uniformvote10_n64_load4(
                    ah, ahalf + qwen35_c64uniformvote10_n64_swizzle(ar, ac, Inner));
                qwen35_c64uniformvote10_n64_mma(high[part], ah, bh);
            }
        }
    }

    __syncthreads();
    if (bad_w || bad_q || bad_b != 0) {
        auto *original_state = reinterpret_cast<float *>(ahalf);
#pragma unroll
        for (int part = 0; part < 2; ++part) {
#pragma unroll
            for (int i = 0; i < 4; ++i)
                original_state[(r + part * 64 + group + (i / 2) * 8) * N + c + tid * 2 + i % 2] =
                    carry[part][i];
        }
        __syncthreads();
    }
#pragma unroll
    for (int part = 0; part < 2; ++part) {
        const bool fallback = bad_column || (part == 0 ? bad_w : bad_q);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int row = r + group + (i / 2) * 8, col = c + tid * 2 + i % 2;
            float value;
            if (fallback) {
                value = 0.f;
                const auto *original_state = reinterpret_cast<const float *>(ahalf);
                for (int k = 0; k < Inner; ++k) {
                    const float a = part == 0
                        ? qwen35_c64uniformvote10_n64_a<0>(row, k, q, key, w, causal, decay,
                              instance, begin, length, qk_base, qk_stride)
                        : qwen35_c64uniformvote10_n64_a<1>(row, k, q, key, w, causal, decay,
                              instance, begin, length, qk_base, qk_stride);
                    value = __fadd_rn(value, __fmul_rn(a, original_state[k * N + col]));
                }
            } else
                value =
                    __fadd_rn(0.f, __fadd_rn(high[part][i], __fmul_rn(cross[part][i], 0x1p-12f)));
            if (part == 0)
                updates[row * S + col] = __fsub_rn(original_u[i], value);
            else
                out[i] = value;
        }
    }
    if (threadIdx.x == 0)
        *cached_b_unsafe = 0;
    __syncthreads();
}

__device__ __forceinline__ void qwen35_c64uniformvote10_n64_state(const float *q, const float *key,
    const float *w, const float *u, const float *causal, const float *decay, __nv_bfloat16 *output,
    int physical_start, int length, int head, int heads, int key_heads, int value_dim,
    int instance_base, int instance_stride, const __half *packed, const unsigned *pack_flags) {
    constexpr int S = 68;
    extern __shared__ __align__(32) unsigned char qwen35_c64uniformvote10_n64_storage[];
    auto *ahalf = reinterpret_cast<__half *>(qwen35_c64uniformvote10_n64_storage);
    auto *updates = reinterpret_cast<float *>(qwen35_c64uniformvote10_n64_storage + 65536);
    auto *cached_b_unsafe = reinterpret_cast<int *>(qwen35_c64uniformvote10_n64_storage + 98304);
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int r = (warp / 8) * 16, c = (warp % 8) * 8, group = lane / 4, tid = lane % 4;
    const int kh = head / (heads / key_heads),
              first_col = (blockIdx.x % ((value_dim + 63) / 64)) * 64;
    const int qk_base = (physical_start * key_heads + kh) * 128, qk_stride = key_heads * 128;
    float carry[2][4] = {};
    for (int chunk = 0; chunk < (length + 63) / 64; ++chunk) {
        const int instance = instance_base + chunk * instance_stride, begin = chunk * 64;
        if (threadIdx.x == 0)
            *cached_b_unsafe = 0;
        float original_u[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int row = r + group + (i / 2) * 8, col = first_col + c + tid * 2 + i % 2;
            original_u[i] = col < value_dim ? u[(instance * 64 + row) * value_dim + col] : 0.f;
        }
        float out[4] = {};
        qwen35_c64uniformvote10_n64_product01(q, key, w, causal, decay, updates, original_u, carry,
            out, ahalf, cached_b_unsafe, instance, begin, length, qk_base, qk_stride, packed,
            pack_flags);
        qwen35_c64uniformvote10_n64_product23(q, key, w, causal, decay, updates, carry, out, ahalf,
            cached_b_unsafe, instance, begin, length, qk_base, qk_stride, packed, pack_flags);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int row = r + group + (i / 2) * 8, col = first_col + c + tid * 2 + i % 2;
            if (begin + row < length && col < value_dim)
                output[((physical_start + begin + row) * heads + head) * value_dim + col] =
                    __float2bfloat16_rn(out[i]);
        }
    }
}
extern "C" __global__ __launch_bounds__(1024, 1) void qwen35_c64uniformvote10_n64_varlen_f32(
    const float *q, const float *k, const float *g, const float *w, const float *u, const float *c,
    const float *d, __nv_bfloat16 *out, float *snapshots, const unsigned *cumulative,
    const unsigned *chunk_offsets, const unsigned *long_ids, int total_length, int heads, int kh,
    int vd, const __half *packed, const unsigned *pack_flags) {
    const int logical = blockIdx.x / ((vd + 63) / 64);
    const int head = logical % heads, seq = long_ids[logical / heads], start = cumulative[seq];
    qwen35_c64uniformvote10_n64_state(q, k, w, u, c, d, out, start, cumulative[seq + 1] - start,
        head, heads, kh, vd, int(chunk_offsets[seq]) * heads + head, heads, packed, pack_flags);
}

extern "C" __global__ __launch_bounds__(1024, 1) void qwen35_c64uniformvote10_n64_uniform_f32(
    const float *q, const float *k, const float *g, const float *w, const float *u, const float *c,
    const float *d, __nv_bfloat16 *out, float *snapshots, int length, int heads, int kh, int vd,
    const __half *packed, const unsigned *pack_flags) {
    const int logical = blockIdx.x / ((vd + 63) / 64);
    const int head = logical % heads, seq = logical / heads;
    const int chunks = (length + 63) / 64;
    qwen35_c64uniformvote10_n64_state(q, k, w, u, c, d, out, seq * length, length, head, heads, kh,
        vd, (seq * heads + head) * chunks, 1, packed, pack_flags);
}

extern "C" __global__ __launch_bounds__(256) void qwen35_c64uniformvote10_prepare_f32(
    const float *q, const float *key, const float *w, const float *causal, const float *decay,
    __half *packed, unsigned *flags, const unsigned *cumulative, const unsigned *offsets,
    int sequences, int heads, int key_heads) {
    extern __shared__ __align__(32) unsigned char prepack_storage[];
    auto *staged_key = reinterpret_cast<__half *>(prepack_storage);
    const int instance = blockIdx.x, head = instance % heads, chunk_index = instance / heads;
    int lower = 0, upper = sequences;
    while (lower + 1 < upper) {
        const int mid = (lower + upper) / 2;
        if (int(offsets[mid]) <= chunk_index)
            lower = mid;
        else
            upper = mid;
    }
    const int start = cumulative[lower], length = cumulative[lower + 1] - start;
    const int begin = (chunk_index - int(offsets[lower])) * 64;
    const int qk_base = (start * key_heads + head / (heads / key_heads)) * 128;
    auto *target = packed + size_t(instance) * 28672;
    if (threadIdx.x == 0)
        flags[instance] = 0;
    __syncthreads();
    unsigned unsafe = 0;
    for (int i = threadIdx.x; i < 64 * 128; i += blockDim.x) {
        const int t = i / 128, d = i % 128;
        const float weight = w[(instance * 64 + t) * 128 + d];
        const float query = begin + t < length
            ? __fmul_rn(decay[instance * 128 + t], q[qk_base + (begin + t) * key_heads * 128 + d])
            : 0.f;
        const float k = begin + t < length ? __fmul_rn(decay[instance * 128 + 64 + t],
                                                 key[qk_base + (begin + t) * key_heads * 128 + d])
                                           : 0.f;
        unsafe |= (!(fabsf(weight) <= 32768.f) ? 1u : 0u) | (!(fabsf(query) <= 32768.f) ? 2u : 0u) |
            (!(fabsf(k) <= 32768.f) ? 8u : 0u);
        const int at = qwen35_c64uniformvote10_n64_swizzle(t, d, 128);
        target[at] = __float2half_rn(weight);
        target[8192 + at] = __float2half_rn(query);
        staged_key[at] = __float2half_rn(k);
    }
    for (int i = threadIdx.x; i < 64 * 64; i += blockDim.x) {
        const int row = i / 64, col = i % 64;
        const float x = causal[(instance * 64 + row) * 64 + col];
        unsafe |= !(fabsf(x) <= 32768.f) ? 4u : 0u;
        target[16384 + qwen35_c64uniformvote10_n64_swizzle(row, col, 64)] = __float2half_rn(x);
    }
    if (unsafe)
        atomicOr(flags + instance, unsafe);
    __syncthreads();
    for (int i = threadIdx.x; i < 128 * 64; i += blockDim.x) {
        const int d = i / 64, t = i % 64;
        target[20480 + qwen35_c64uniformvote10_n64_swizzle(d, t, 64)] =
            staged_key[qwen35_c64uniformvote10_n64_swizzle(t, d, 128)];
    }
}

#include <stdint.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math.h>

__device__ __forceinline__ void qwen35_c64packed32v40_n32_load4(
    unsigned *out, const __half *pointer) {
    const unsigned address = unsigned(__cvta_generic_to_shared(pointer));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3},[%4];"
                 : "=r"(out[0]), "=r"(out[1]), "=r"(out[2]), "=r"(out[3])
                 : "r"(address)
                 : "memory");
}
__device__ __forceinline__ void qwen35_c64packed32v40_n32_load2(
    unsigned *out, const __half *pointer) {
    const unsigned address = unsigned(__cvta_generic_to_shared(pointer));
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1},[%2];"
                 : "=r"(out[0]), "=r"(out[1])
                 : "r"(address)
                 : "memory");
}
__device__ __forceinline__ void qwen35_c64packed32v40_n32_mma(
    float *out, const unsigned *a, const unsigned *b) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                 "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
                 : "+f"(out[0]), "+f"(out[1]), "+f"(out[2]), "+f"(out[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
__device__ __forceinline__ int qwen35_c64packed32v40_n32_swizzle(int row, int col, int stride) {
    return row * stride + (col ^ ((row & ((stride / 8 - 1) < 7 ? (stride / 8 - 1) : 7)) << 3));
}
__device__ __forceinline__ void qwen35_c64packed32v40_n32_load2_trans(
    unsigned *out, const __half *pointer) {
    const unsigned address = unsigned(__cvta_generic_to_shared(pointer));
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1},[%2];"
                 : "=r"(out[0]), "=r"(out[1])
                 : "r"(address)
                 : "memory");
}

__device__ __forceinline__ void qwen35_packed32v40_copy(__half *dst, const __half *src, int bytes) {
    for (int i = threadIdx.x * 16; i < bytes; i += blockDim.x * 16) {
        const unsigned address =
            unsigned(__cvta_generic_to_shared(reinterpret_cast<unsigned char *>(dst) + i));
        const auto *source = reinterpret_cast<const unsigned char *>(src) + i;
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(address), "l"(source)
                     : "memory");
    }
    asm volatile("cp.async.commit_group;" ::: "memory");
    asm volatile("cp.async.wait_group 0;" ::: "memory");
}

template <int Phase>
__device__ __forceinline__ float qwen35_c64packed32v40_n32_a(int row, int col, const float *q,
    const float *key, const float *w, const float *causal, const float *decay, int instance,
    int begin, int length, int qk_base, int qk_stride) {
    if constexpr (Phase == 0)
        return w[(instance * 64 + row) * 128 + col];
    if constexpr (Phase == 1)
        return begin + row < length
            ? __fmul_rn(decay[instance * 128 + row], q[qk_base + (begin + row) * qk_stride + col])
            : 0.f;
    if constexpr (Phase == 2)
        return causal[(instance * 64 + row) * 64 + col];
    if constexpr (Phase == 3)
        return begin + col < length ? __fmul_rn(decay[instance * 128 + 64 + col],
                                          key[qk_base + (begin + col) * qk_stride + row])
                                    : 0.f;
    return 0.f;
}

__device__ __forceinline__ void qwen35_c64packed32v40_n32_product23(const float *q,
    const float *key, const float *w, const float *causal, const float *decay, float *updates,
    float (&carry)[2][4], float (&out)[4], __half *scratch, int *cached_b_unsafe, int instance,
    int begin, int length, int qk_base, int qk_stride, const __half *packed,
    const unsigned *pack_flags) {
    constexpr int N = 32, S = 36, Inner = 64;
    auto *ahalf = scratch;
    auto *alow = ahalf + 192 * Inner;
    auto *bhalf = ahalf + 192 * Inner;
    auto *blow = bhalf + Inner * N;
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32, group = lane / 4, tid = lane % 4;
    const int r = (warp / 4) * 16, c = (warp % 4) * 8;
    float high[3][4] = {}, cross[3][4] = {};
    const unsigned packed_flags = pack_flags[instance];
    const bool unsafe_l = packed_flags & 4u, unsafe_k = packed_flags & 8u;
    qwen35_packed32v40_copy(ahalf, packed + size_t(instance) * 28672 + 16384, 24576);
    unsigned unsafe_b = 0;
    for (int index = threadIdx.x * 2; index < Inner * N; index += blockDim.x * 2) {
        const int row = index / N, col = index % N;
        const float2 pair = *reinterpret_cast<const float2 *>(updates + row * S + col);
        if (!(fabsf(pair.x) <= 32768.f) || !(fabsf(pair.y) <= 32768.f))
            unsafe_b |= 1u << (col / 16);
        const __half2 hi = __floats2half2_rn(pair.x, pair.y);
        const int address = qwen35_c64packed32v40_n32_swizzle(row, col, N);
        *reinterpret_cast<__half2 *>(bhalf + address) = hi;
    }
    if (unsafe_b)
        atomicOr(cached_b_unsafe, int(unsafe_b));

    __syncthreads();
    const bool bad_l = unsafe_l;
    const bool bad_k = unsafe_k;
    const bool bad_column = (unsigned(*cached_b_unsafe) & (1u << (c / 16))) != 0;
    for (int k = 0; k < Inner; k += 16) {
        unsigned bh[2], bl[2];
        const int tr = k + (lane & 15);
        qwen35_c64packed32v40_n32_load2_trans(
            bh, bhalf + qwen35_c64packed32v40_n32_swizzle(tr, c, N));
#pragma unroll
        for (int part = 0; part < 3; ++part) {
            if (!bad_column && !(part == 0 ? bad_l : bad_k)) {
                unsigned ah[4], al[4];
                const int ar = r + part * 64 + (lane & 15), ac = k + (lane / 16) * 8;
                qwen35_c64packed32v40_n32_load4(
                    ah, ahalf + qwen35_c64packed32v40_n32_swizzle(ar, ac, Inner));
                qwen35_c64packed32v40_n32_mma(high[part], ah, bh);
            }
        }
    }
#pragma unroll
    for (int part = 0; part < 3; ++part) {
        const bool fallback = bad_column || (part == 0 ? bad_l : bad_k);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int row = r + (part == 0 ? 0 : (part - 1) * 64) + group + (i / 2) * 8,
                      col = c + tid * 2 + i % 2;

            const float initial = part == 0
                ? __fmul_rn(out[i], 1.f)
                : __fmul_rn(carry[part - 1][i], decay[instance * 128 + 63]);
            float value;
            if (fallback) {
                value = initial;
                for (int k = 0; k < Inner; ++k) {
                    const float a = part == 0
                        ? qwen35_c64packed32v40_n32_a<2>(row, k, q, key, w, causal, decay, instance,
                              begin, length, qk_base, qk_stride)
                        : qwen35_c64packed32v40_n32_a<3>(row, k, q, key, w, causal, decay, instance,
                              begin, length, qk_base, qk_stride);
                    value = __fadd_rn(value, __fmul_rn(a, updates[k * S + col]));
                }
            } else
                value = __fadd_rn(
                    initial, __fadd_rn(high[part][i], __fmul_rn(cross[part][i], 0x1p-12f)));
            if (part == 0)
                out[i] = value;
            else
                carry[part - 1][i] = value;
        }
    }

    __syncthreads();
}

__device__ __forceinline__ void qwen35_c64packed32v40_n32_product01(const float *q,
    const float *key, const float *w, const float *causal, const float *decay, float *updates,
    const float (&original_u)[4], float (&carry)[2][4], float (&out)[4], __half *scratch,
    int *cached_b_unsafe, int instance, int begin, int length, int qk_base, int qk_stride,
    const __half *packed, const unsigned *pack_flags) {
    constexpr int N = 32, S = 36, Inner = 128;
    auto *ahalf = scratch;
    auto *alow = ahalf + 128 * Inner;
    auto *bhalf = ahalf + 128 * Inner;
    auto *blow = bhalf + Inner * N;
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32, group = lane / 4, tid = lane % 4;
    const int r = (warp / 4) * 16, c = (warp % 4) * 8;
    float high[2][4] = {}, cross[2][4] = {};
    const unsigned packed_flags = pack_flags[instance];
    const bool unsafe_w = packed_flags & 1u, unsafe_q = packed_flags & 2u;
    qwen35_packed32v40_copy(ahalf, packed + size_t(instance) * 28672, 32768);
    unsigned unsafe_b = 0;
#pragma unroll
    for (int part = 0; part < 2; ++part) {
#pragma unroll
        for (int i = 0; i < 4; i += 2) {
            const int row = r + part * 64 + group + (i / 2) * 8, col = c + tid * 2;
            const float x = carry[part][i], y = carry[part][i + 1];
            if (!(fabsf(x) <= 32768.f) || !(fabsf(y) <= 32768.f))
                unsafe_b |= 1u << (col / 16);
            const __half2 hi = __floats2half2_rn(x, y);
            const int address = qwen35_c64packed32v40_n32_swizzle(row, col, N);
            *reinterpret_cast<__half2 *>(bhalf + address) = hi;
        }
    }
    if (unsafe_b)
        atomicOr(cached_b_unsafe, int(unsafe_b));

    __syncthreads();
    const bool bad_w = unsafe_w;
    const bool bad_q = unsafe_q;
    const unsigned bad_b = unsigned(*cached_b_unsafe);
    const bool bad_column = (bad_b & (1u << (c / 16))) != 0;
    for (int k = 0; k < Inner; k += 16) {
        unsigned bh[2], bl[2];
        const int tr = k + (lane & 15);
        qwen35_c64packed32v40_n32_load2_trans(
            bh, bhalf + qwen35_c64packed32v40_n32_swizzle(tr, c, N));
#pragma unroll
        for (int part = 0; part < 2; ++part) {
            if (!bad_column && !(part == 0 ? bad_w : bad_q)) {
                unsigned ah[4], al[4];
                const int ar = r + part * 64 + (lane & 15), ac = k + (lane / 16) * 8;
                qwen35_c64packed32v40_n32_load4(
                    ah, ahalf + qwen35_c64packed32v40_n32_swizzle(ar, ac, Inner));
                qwen35_c64packed32v40_n32_mma(high[part], ah, bh);
            }
        }
    }

    __syncthreads();
    if (bad_w || bad_q || bad_b != 0) {
        auto *original_state = reinterpret_cast<float *>(ahalf);
#pragma unroll
        for (int part = 0; part < 2; ++part) {
#pragma unroll
            for (int i = 0; i < 4; ++i)
                original_state[(r + part * 64 + group + (i / 2) * 8) * N + c + tid * 2 + i % 2] =
                    carry[part][i];
        }
        __syncthreads();
    }
#pragma unroll
    for (int part = 0; part < 2; ++part) {
        const bool fallback = bad_column || (part == 0 ? bad_w : bad_q);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int row = r + group + (i / 2) * 8, col = c + tid * 2 + i % 2;
            float value;
            if (fallback) {
                value = 0.f;
                const auto *original_state = reinterpret_cast<const float *>(ahalf);
                for (int k = 0; k < Inner; ++k) {
                    const float a = part == 0
                        ? qwen35_c64packed32v40_n32_a<0>(row, k, q, key, w, causal, decay, instance,
                              begin, length, qk_base, qk_stride)
                        : qwen35_c64packed32v40_n32_a<1>(row, k, q, key, w, causal, decay, instance,
                              begin, length, qk_base, qk_stride);
                    value = __fadd_rn(value, __fmul_rn(a, original_state[k * N + col]));
                }
            } else
                value =
                    __fadd_rn(0.f, __fadd_rn(high[part][i], __fmul_rn(cross[part][i], 0x1p-12f)));
            if (part == 0)
                updates[row * S + col] = __fsub_rn(original_u[i], value);
            else
                out[i] = value;
        }
    }
    if (threadIdx.x == 0)
        *cached_b_unsafe = 0;
    __syncthreads();
}

__device__ __forceinline__ void qwen35_c64packed32v40_n32_state(const float *q, const float *key,
    const float *w, const float *u, const float *causal, const float *decay, __nv_bfloat16 *output,
    int physical_start, int length, int head, int heads, int key_heads, int value_dim,
    int instance_base, int instance_stride, const __half *packed, const unsigned *pack_flags) {
    constexpr int S = 36;
    extern __shared__ __align__(32) unsigned char qwen35_c64packed32v40_n32_storage[];
    auto *ahalf = reinterpret_cast<__half *>(qwen35_c64packed32v40_n32_storage);
    auto *updates = reinterpret_cast<float *>(qwen35_c64packed32v40_n32_storage + 32768);
    auto *cached_b_unsafe = reinterpret_cast<int *>(qwen35_c64packed32v40_n32_storage + 49152);
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int r = (warp / 4) * 16, c = (warp % 4) * 8, group = lane / 4, tid = lane % 4;
    const int kh = head / (heads / key_heads),
              first_col = (blockIdx.x % ((value_dim + 31) / 32)) * 32;
    const int qk_base = (physical_start * key_heads + kh) * 128, qk_stride = key_heads * 128;
    float carry[2][4] = {};
    for (int chunk = 0; chunk < (length + 63) / 64; ++chunk) {
        const int instance = instance_base + chunk * instance_stride, begin = chunk * 64;
        if (threadIdx.x == 0)
            *cached_b_unsafe = 0;
        float original_u[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int row = r + group + (i / 2) * 8, col = first_col + c + tid * 2 + i % 2;
            original_u[i] = col < value_dim ? u[(instance * 64 + row) * value_dim + col] : 0.f;
        }
        float out[4] = {};
        qwen35_c64packed32v40_n32_product01(q, key, w, causal, decay, updates, original_u, carry,
            out, ahalf, cached_b_unsafe, instance, begin, length, qk_base, qk_stride, packed,
            pack_flags);
        qwen35_c64packed32v40_n32_product23(q, key, w, causal, decay, updates, carry, out, ahalf,
            cached_b_unsafe, instance, begin, length, qk_base, qk_stride, packed, pack_flags);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int row = r + group + (i / 2) * 8, col = first_col + c + tid * 2 + i % 2;
            if (begin + row < length && col < value_dim)
                output[((physical_start + begin + row) * heads + head) * value_dim + col] =
                    __float2bfloat16_rn(out[i]);
        }
    }
}
extern "C" __global__ __launch_bounds__(512, 2) void qwen35_c64packed32v40_n32_varlen_f32(
    const float *q, const float *k, const float *g, const float *w, const float *u, const float *c,
    const float *d, __nv_bfloat16 *out, float *snapshots, const unsigned *cumulative,
    const unsigned *chunk_offsets, const unsigned *long_ids, int total_length, int heads, int kh,
    int vd, const __half *packed, const unsigned *pack_flags) {
    const int logical = blockIdx.x / ((vd + 31) / 32);
    const int head = logical % heads, seq = long_ids[logical / heads], start = cumulative[seq];
    qwen35_c64packed32v40_n32_state(q, k, w, u, c, d, out, start, cumulative[seq + 1] - start, head,
        heads, kh, vd, int(chunk_offsets[seq]) * heads + head, heads, packed, pack_flags);
}

extern "C" __global__ __launch_bounds__(512, 2) void qwen35_c64packed32v40_n32_uniform_f32(
    const float *q, const float *k, const float *g, const float *w, const float *u, const float *c,
    const float *d, __nv_bfloat16 *out, float *snapshots, int length, int heads, int kh, int vd,
    const __half *packed, const unsigned *pack_flags) {
    const int logical = blockIdx.x / ((vd + 31) / 32);
    const int head = logical % heads, seq = logical / heads;
    const int chunks = (length + 63) / 64;
    qwen35_c64packed32v40_n32_state(q, k, w, u, c, d, out, seq * length, length, head, heads, kh,
        vd, (seq * heads + head) * chunks, 1, packed, pack_flags);
}

extern "C" __global__ __launch_bounds__(256) void qwen35_c64packed32v40_prepare_f32(const float *q,
    const float *key, const float *w, const float *causal, const float *decay, __half *packed,
    unsigned *flags, const unsigned *cumulative, const unsigned *offsets, int sequences, int heads,
    int key_heads) {
    extern __shared__ __align__(32) unsigned char prepack_storage[];
    auto *staged_key = reinterpret_cast<__half *>(prepack_storage);
    const int instance = blockIdx.x, head = instance % heads, chunk_index = instance / heads;
    int lower = 0, upper = sequences;
    while (lower + 1 < upper) {
        const int mid = (lower + upper) / 2;
        if (int(offsets[mid]) <= chunk_index)
            lower = mid;
        else
            upper = mid;
    }
    const int start = cumulative[lower], length = cumulative[lower + 1] - start;
    const int begin = (chunk_index - int(offsets[lower])) * 64;
    const int qk_base = (start * key_heads + head / (heads / key_heads)) * 128;
    auto *target = packed + size_t(instance) * 28672;
    if (threadIdx.x == 0)
        flags[instance] = 0;
    __syncthreads();
    unsigned unsafe = 0;
    for (int i = threadIdx.x; i < 64 * 128; i += blockDim.x) {
        const int t = i / 128, d = i % 128;
        const float weight = w[(instance * 64 + t) * 128 + d];
        const float query = begin + t < length
            ? __fmul_rn(decay[instance * 128 + t], q[qk_base + (begin + t) * key_heads * 128 + d])
            : 0.f;
        const float k = begin + t < length ? __fmul_rn(decay[instance * 128 + 64 + t],
                                                 key[qk_base + (begin + t) * key_heads * 128 + d])
                                           : 0.f;
        unsafe |= (!(fabsf(weight) <= 32768.f) ? 1u : 0u) | (!(fabsf(query) <= 32768.f) ? 2u : 0u) |
            (!(fabsf(k) <= 32768.f) ? 8u : 0u);
        const int at = qwen35_c64packed32v40_n32_swizzle(t, d, 128);
        target[at] = __float2half_rn(weight);
        target[8192 + at] = __float2half_rn(query);
        staged_key[at] = __float2half_rn(k);
    }
    for (int i = threadIdx.x; i < 64 * 64; i += blockDim.x) {
        const int row = i / 64, col = i % 64;
        const float x = causal[(instance * 64 + row) * 64 + col];
        unsafe |= !(fabsf(x) <= 32768.f) ? 4u : 0u;
        target[16384 + qwen35_c64packed32v40_n32_swizzle(row, col, 64)] = __float2half_rn(x);
    }
    if (unsafe)
        atomicOr(flags + instance, unsafe);
    __syncthreads();
    for (int i = threadIdx.x; i < 128 * 64; i += blockDim.x) {
        const int d = i / 64, t = i % 64;
        target[20480 + qwen35_c64packed32v40_n32_swizzle(d, t, 64)] =
            staged_key[qwen35_c64packed32v40_n32_swizzle(t, d, 128)];
    }
}

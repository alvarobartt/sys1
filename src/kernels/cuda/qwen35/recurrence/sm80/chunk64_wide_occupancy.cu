#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math.h>

__device__ __forceinline__ void qwen35_c64wide_load4(unsigned *out, const __half *pointer) {
    const unsigned address = unsigned(__cvta_generic_to_shared(pointer));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3},[%4];"
                 : "=r"(out[0]), "=r"(out[1]), "=r"(out[2]), "=r"(out[3])
                 : "r"(address)
                 : "memory");
}
__device__ __forceinline__ void qwen35_c64wide_load2(unsigned *out, const __half *pointer) {
    const unsigned address = unsigned(__cvta_generic_to_shared(pointer));
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1},[%2];"
                 : "=r"(out[0]), "=r"(out[1])
                 : "r"(address)
                 : "memory");
}
__device__ __forceinline__ void qwen35_c64wide_mma(
    float *out, const unsigned *a, const unsigned *b) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                 "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
                 : "+f"(out[0]), "+f"(out[1]), "+f"(out[2]), "+f"(out[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
__device__ __forceinline__ int qwen35_c64wide_swizzle(int row, int col, int stride) {
    return row * stride + (col ^ ((row & 7) << 3));
}
template <int Phase>
__device__ __forceinline__ float qwen35_c64wide_a(int row, int col, const float *q,
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
__device__ __forceinline__ void qwen35_c64wide_product(const float *q, const float *key,
    const float *w, const float *causal, const float *decay, float *updates, float (&carry)[2][4],
    float (&out)[4], __half *ahalf, __half *alow, __half *bhalf, __half *blow, int *cached_b_unsafe,
    int instance, int begin, int length, int qk_base, int qk_stride) {
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
    for (int index = threadIdx.x; index < Rows * Inner; index += blockDim.x) {
        const int row = Phase == 3 ? index % Rows : index / Inner,
                  col = Phase == 3 ? index / Rows : index % Inner;
        const float x = qwen35_c64wide_a<Phase>(
            row, col, q, key, w, causal, decay, instance, begin, length, qk_base, qk_stride);
        unsafe_a |= !(fabsf(x) <= 32768.f);
        const auto a = __float2half_rn(x);
        const int address = qwen35_c64wide_swizzle(row, col, Inner);
        ahalf[address] = a;
        alow[address] = __float2half_rn(__fmul_rn(__fsub_rn(x, __half2float(a)), 4096.f));
    }
    unsigned unsafe_b = 0;
    if constexpr (Phase == 0) {
#pragma unroll
        for (int part = 0; part < 2; ++part) {
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int row = r + part * 64 + group + (i / 2) * 8, col = c + tid * 2 + i % 2;
                const float x = carry[part][i];
                if (!(fabsf(x) <= 32768.f))
                    unsafe_b |= 1u << (col / 16);
                const auto b = __float2half_rn(x);
                const int address = qwen35_c64wide_swizzle(col, row, Inner);
                bhalf[address] = b;
                blow[address] = __float2half_rn(__fmul_rn(__fsub_rn(x, __half2float(b)), 4096.f));
            }
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
            const auto b = __float2half_rn(x);
            const int address = qwen35_c64wide_swizzle(col, row, Inner);
            bhalf[address] = b;
            blow[address] = __float2half_rn(__fmul_rn(__fsub_rn(x, __half2float(b)), 4096.f));
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
                qwen35_c64wide_load4(ah, ahalf + qwen35_c64wide_swizzle(ar, ac, Inner));
                qwen35_c64wide_load4(al, alow + qwen35_c64wide_swizzle(ar, ac, Inner));
                qwen35_c64wide_load2(bh, bhalf + qwen35_c64wide_swizzle(br, bc, Inner));
                qwen35_c64wide_load2(bl, blow + qwen35_c64wide_swizzle(br, bc, Inner));
                qwen35_c64wide_mma(high[part], ah, bh);
                qwen35_c64wide_mma(cross[part], ah, bl);
                qwen35_c64wide_mma(cross[part], al, bh);
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
                        __fmul_rn(qwen35_c64wide_a<Phase>(row, k, q, key, w, causal, decay,
                                      instance, begin, length, qk_base, qk_stride),
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

__device__ __forceinline__ void qwen35_c64wide_state(const float *q, const float *key,
    const float *w, const float *u, const float *causal, const float *decay, __nv_bfloat16 *output,
    int physical_start, int length, int head, int heads, int key_heads, int value_dim,
    int instance_base, int instance_stride) {
    constexpr int S = 68;
    extern __shared__ __align__(32) unsigned char qwen35_c64wide_storage[];
    auto *updates = reinterpret_cast<float *>(qwen35_c64wide_storage);
    auto *ahalf = reinterpret_cast<__half *>(updates + 64 * S);
    auto *alow = ahalf + 8192;
    auto *bhalf = alow + 8192;
    auto *blow = bhalf + 64 * 128;
    auto *cached_b_unsafe = reinterpret_cast<int *>(blow + 64 * 128);
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int r = (warp / 8) * 16, c = (warp % 8) * 8, group = lane / 4, tid = lane % 4;
    const int kh = head / (heads / key_heads), first_col = blockIdx.y * 64;
    const int qk_base = (physical_start * key_heads + kh) * 128, qk_stride = key_heads * 128;
    float carry[2][4] = {};
    for (int chunk = 0; chunk < (length + 63) / 64; ++chunk) {
        const int instance = instance_base + chunk * instance_stride, begin = chunk * 64;
        if (threadIdx.x == 0)
            *cached_b_unsafe = 0;
        for (int index = threadIdx.x; index < 64 * 64; index += blockDim.x) {
            const int row = index / 64, col = first_col + index % 64;
            updates[row * S + index % 64] =
                col < value_dim ? u[(instance * 64 + row) * value_dim + col] : 0.f;
        }
        __syncthreads();
        float out[4] = {};
        qwen35_c64wide_product<0>(q, key, w, causal, decay, updates, carry, out, ahalf, alow, bhalf,
            blow, cached_b_unsafe, instance, begin, length, qk_base, qk_stride);
        qwen35_c64wide_product<1, true>(q, key, w, causal, decay, updates, carry, out, ahalf, alow,
            bhalf, blow, cached_b_unsafe, instance, begin, length, qk_base, qk_stride);
        qwen35_c64wide_product<2>(q, key, w, causal, decay, updates, carry, out, ahalf, alow, bhalf,
            blow, cached_b_unsafe, instance, begin, length, qk_base, qk_stride);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int row = r + group + (i / 2) * 8, col = first_col + c + tid * 2 + i % 2;
            if (begin + row < length && col < value_dim)
                output[((physical_start + begin + row) * heads + head) * value_dim + col] =
                    __float2bfloat16_rn(out[i]);
        }
        qwen35_c64wide_product<3, true>(q, key, w, causal, decay, updates, carry, out, ahalf, alow,
            bhalf, blow, cached_b_unsafe, instance, begin, length, qk_base, qk_stride);
    }
}
extern "C" __global__ __launch_bounds__(1024, 1) void qwen35_c64wide_varlen_f32(const float *q,
    const float *k, const float *g, const float *w, const float *u, const float *c, const float *d,
    __nv_bfloat16 *out, float *snapshots, const unsigned *cumulative, const unsigned *chunk_offsets,
    const unsigned *long_ids, int total_length, int heads, int kh, int vd) {
    const int head = blockIdx.x % heads, seq = long_ids[blockIdx.x / heads],
              start = cumulative[seq];
    qwen35_c64wide_state(q, k, w, u, c, d, out, start, cumulative[seq + 1] - start, head, heads, kh,
        vd, int(chunk_offsets[seq]) * heads + head, heads);
}

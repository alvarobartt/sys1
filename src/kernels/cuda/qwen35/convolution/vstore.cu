#include <cuda_bf16.h>
#include <math.h>
#include <cstdint>
#include <cuda_runtime.h>

__device__ __forceinline__ float qwen35_convvstore_decode(unsigned word, int half) {
    return __bfloat162float(__ushort_as_bfloat16(static_cast<unsigned short>(word >> (half * 16))));
}
template <int FixedKernel>
__device__ __forceinline__ void qwen35_convvstore_four(const __nv_bfloat16 *x, const __nv_bfloat16 *w,
                                                     float (&out)[4], int token, int channel,
                                                     int begin, int channels, int kernel) {
    float sums[4] = {};
    unsigned weights[8] = {};
    const bool vector_input = (reinterpret_cast<uintptr_t>(x) & 7u) == 0;
    const bool vector_weights = FixedKernel == 4 && (reinterpret_cast<uintptr_t>(w) & 15u) == 0;
    if constexpr (FixedKernel == 4) {
        if (vector_weights) {
            const uint4 a = *reinterpret_cast<const uint4 *>(w + channel * 4);
            const uint4 b = *reinterpret_cast<const uint4 *>(w + (channel + 2) * 4);
            weights[0] = a.x;
            weights[1] = a.y;
            weights[2] = a.z;
            weights[3] = a.w;
            weights[4] = b.x;
            weights[5] = b.y;
            weights[6] = b.z;
            weights[7] = b.w;
        }
    }
#pragma unroll
    for (int tap = 0; tap < (FixedKernel ? FixedKernel : kernel); ++tap) {
        const int source = token + tap - kernel + 1;
        if (source >= begin) {
            float inputs[4];
            if (vector_input) {
                const uint2 packed =
                    *reinterpret_cast<const uint2 *>(x + source * channels + channel);
                inputs[0] = qwen35_convvstore_decode(packed.x, 0);
                inputs[1] = qwen35_convvstore_decode(packed.x, 1);
                inputs[2] = qwen35_convvstore_decode(packed.y, 0);
                inputs[3] = qwen35_convvstore_decode(packed.y, 1);
            } else {
#pragma unroll
                for (int j = 0; j < 4; ++j)
                    inputs[j] = __bfloat162float(x[source * channels + channel + j]);
            }
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                float weight;
                if constexpr (FixedKernel == 4) {
                    weight = vector_weights
                                 ? qwen35_convvstore_decode(weights[j * 2 + tap / 2], tap & 1)
                                 : __bfloat162float(w[(channel + j) * kernel + tap]);
                } else
                    weight = __bfloat162float(w[(channel + j) * kernel + tap]);
                sums[j] = __fadd_rn(sums[j], __fmul_rn(inputs[j], weight));
            }
        }
    }
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const float rounded = __bfloat162float(__float2bfloat16_rn(sums[j]));
        out[j] = __bfloat162float(__float2bfloat16_rn(rounded / (1.f + expf(-rounded))));
    }
}
__device__ __forceinline__ void qwen35_convvstore_element4(const __nv_bfloat16 *x,
                                                         const __nv_bfloat16 *w, float (&out)[4],
                                                         int token, int channel, int begin,
                                                         int channels, int kernel) {
    if (kernel == 4)
        qwen35_convvstore_four<4>(x, w, out, token, channel, begin, channels, kernel);
    else
        qwen35_convvstore_four<0>(x, w, out, token, channel, begin, channels, kernel);
}
template <bool Packed>
__device__ __forceinline__ void
qwen35_convvstore_impl(const __nv_bfloat16 *mixed, const __nv_bfloat16 *weights, float *query,
                     float *key, __nv_bfloat16 *value, const unsigned *starts, unsigned int rows,
                     int key_heads, int value_heads, float scale, int length, int kernel) {
    const unsigned int row = blockIdx.x * 4 + threadIdx.x / 32;
    const unsigned int lane = threadIdx.x & 31;
    if (row >= rows)
        return;
    const unsigned int token = row / key_heads, head = row % key_heads;
    const unsigned int key_width = key_heads * 128, value_width = value_heads * 128;
    const int channels = 2 * key_width + value_width,
              begin = Packed ? int(starts[token]) : int(token / length) * length;
    const unsigned int source = token * (2 * key_width + value_width) + head * 128 + lane * 4;
    const unsigned int target = row * 128 + lane * 4;
    float q[4], k[4];
    qwen35_convvstore_element4(mixed, weights, q, token, head * 128 + lane * 4, begin, channels,
                             kernel);
    qwen35_convvstore_element4(mixed, weights, k, token, key_width + head * 128 + lane * 4, begin,
                             channels, kernel);
    float qs = __fmul_rn(q[0], q[0]), ks = __fmul_rn(k[0], k[0]);
#pragma unroll
    for (int j = 1; j < 4; ++j) {
        qs = __fadd_rn(qs, __fmul_rn(q[j], q[j]));
        ks = __fadd_rn(ks, __fmul_rn(k[j], k[j]));
    }
#pragma unroll
    for (int step = 16; step > 0; step >>= 1) {
        qs = __fadd_rn(qs, __shfl_xor_sync(0xffffffff, qs, step));
        ks = __fadd_rn(ks, __shfl_xor_sync(0xffffffff, ks, step));
    }
    qs = __shfl_sync(0xffffffff, qs, 0);
    ks = __shfl_sync(0xffffffff, ks, 0);
    const float qd = __fsqrt_rn(__fadd_rn(qs, 1e-6f)), kd = __fsqrt_rn(__fadd_rn(ks, 1e-6f));
    float qout[4], kout[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const float normalized = __fdiv_rn(q[j], qd);
        qout[j] = scale == 1.f ? normalized : __fmaf_rn(normalized, scale, 0.f);
        kout[j] = __fdiv_rn(k[j], kd);
    }
    *reinterpret_cast<float4 *>(query + target) = make_float4(qout[0], qout[1], qout[2], qout[3]);
    *reinterpret_cast<float4 *>(key + target) = make_float4(kout[0], kout[1], kout[2], kout[3]);
    const int repeat = value_heads / key_heads;
    const unsigned int vo = token * value_width + head * repeat * 128;
    for (int j = lane * 4; j < repeat * 128; j += 128) {
        float out[4];
        qwen35_convvstore_element4(mixed, weights, out, token,
                                 2 * key_width + head * repeat * 128 + j, begin, channels, kernel);

        const unsigned lo =
            (__float_as_uint(out[0]) >> 16) | (__float_as_uint(out[1]) & 0xffff0000u);
        const unsigned hi =
            (__float_as_uint(out[2]) >> 16) | (__float_as_uint(out[3]) & 0xffff0000u);
        *reinterpret_cast<uint2 *>(value + vo + j) = make_uint2(lo, hi);
    }
}
extern "C" __global__ void qwen35_conv_vstore_uniform(const __nv_bfloat16 *x, const __nv_bfloat16 *w,
                                                    float *q, float *k, __nv_bfloat16 *v,
                                                    unsigned rows, int kh, int vh, float scale,
                                                    int length, int kernel) {
    qwen35_convvstore_impl<false>(x, w, q, k, v, nullptr, rows, kh, vh, scale, length, kernel);
}
extern "C" __global__ void qwen35_conv_vstore_packed(const __nv_bfloat16 *x, const __nv_bfloat16 *w,
                                                   float *q, float *k, __nv_bfloat16 *v,
                                                   const unsigned *starts, unsigned rows, int kh,
                                                   int vh, float scale, int length, int kernel) {
    qwen35_convvstore_impl<true>(x, w, q, k, v, starts, rows, kh, vh, scale, length, kernel);
}

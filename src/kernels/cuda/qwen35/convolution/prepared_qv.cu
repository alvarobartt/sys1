#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>

#include <cuda_fp16.h>

template <bool Packed, bool PrepareK = false>
__device__ __forceinline__ void qwen35_convcast_qv_impl(const __nv_bfloat16 *mixed,
    const __nv_bfloat16 *weights, float *query, float *key, float *value, const unsigned *starts,
    unsigned int rows, int key_heads, int value_heads, float scale, int length, int kernel,
    const __nv_bfloat16 *table, __half *qhalf, __half *khalf, __half *vhalf, float limit,
    unsigned *qv_bad) {
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
    qwen35_conv_lookup_element4(
        mixed, weights, q, token, head * 128 + lane * 4, begin, channels, kernel, table);
    qwen35_conv_lookup_element4(mixed, weights, k, token, key_width + head * 128 + lane * 4, begin,
        channels, kernel, table);
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
    bool invalid_q = false, invalid_v = false;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const float normalized = __fdiv_rn(q[j], qd);
        qout[j] = scale == 1.f ? normalized : __fmaf_rn(normalized, scale, 0.f);
        kout[j] = __fdiv_rn(k[j], kd);
    }
    *reinterpret_cast<float4 *>(query + target) = make_float4(qout[0], qout[1], qout[2], qout[3]);
    *reinterpret_cast<float4 *>(key + target) = make_float4(kout[0], kout[1], kout[2], kout[3]);
    unsigned hq[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        invalid_q |= !(fabsf(qout[j]) <= 1.f);
        hq[j] = __half_as_ushort(__float2half_rn(qout[j]));
    }
    *reinterpret_cast<uint2 *>(qhalf + target) =
        make_uint2(hq[0] | (hq[1] << 16), hq[2] | (hq[3] << 16));
    if constexpr (PrepareK) {
        unsigned hk[4];
#pragma unroll
        for (int j = 0; j < 4; ++j)
            hk[j] = __half_as_ushort(__float2half_rn(kout[j]));
        const unsigned low = hk[0] | (hk[1] << 16), high = hk[2] | (hk[3] << 16);
        *reinterpret_cast<uint2 *>(khalf + target) = make_uint2(low, high);

        float norm = 0.f;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const unsigned source_lane = lane / 4 + 8 * j;
            const unsigned lo = __shfl_sync(0xffffffff, low, source_lane);
            const unsigned hi = __shfl_sync(0xffffffff, high, source_lane);
            const unsigned word = (lane & 2) ? hi : lo;
            const unsigned short bits = static_cast<unsigned short>(word >> ((lane & 1) * 16));
            const float rounded = __half2float(__ushort_as_half(bits));
            norm += rounded * rounded;
        }
#pragma unroll
        for (int offset = 16; offset; offset /= 2)
            norm += __shfl_down_sync(0xffffffff, norm, offset);
        if (lane == 0 && !(norm <= 1.25f))
            atomicOr(qv_bad, 2u);
    }
    const int repeat = value_heads / key_heads;
    const unsigned int vo = token * value_width + head * repeat * 128;
    for (int j = lane * 4; j < repeat * 128; j += 128) {
        float out[4];
        qwen35_conv_lookup_element4(mixed, weights, out, token,
            2 * key_width + head * repeat * 128 + j, begin, channels, kernel, table);
        *reinterpret_cast<float4 *>(value + vo + j) = make_float4(out[0], out[1], out[2], out[3]);
        unsigned hv[4];
#pragma unroll
        for (int z = 0; z < 4; ++z) {
            invalid_v |= !(fabsf(out[z]) <= limit);
            hv[z] = __half_as_ushort(__float2half_rn(out[z]));
        }
        *reinterpret_cast<uint2 *>(vhalf + vo + j) =
            make_uint2(hv[0] | (hv[1] << 16), hv[2] | (hv[3] << 16));
    }
    if (invalid_q)
        atomicOr(qv_bad, 1u);
    if (invalid_v)
        atomicOr(qv_bad, 4u);
}

extern "C" __global__ void qwen35_convcast_gbeta_merge(const float *g, const float *beta,
    float *aligned_g, float *aligned_beta, uint64_t heads, const unsigned *producer_bad,
    unsigned *bad) {
    if (blockIdx.x == 0 && threadIdx.x == 0)
        atomicOr(bad, *producer_bad);
    bool invalid = false;
    for (uint64_t head = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x; head < heads;
         head += uint64_t(gridDim.x) * blockDim.x) {
        const float gate = g[head], b = beta[head];
        aligned_g[head] = gate;
        aligned_beta[head] = b;
        invalid |= !(isfinite(gate) && gate <= 0.f && b >= 0.f && b <= 1.f);
    }
    if (invalid)
        atomicOr(bad, 4u);
}

extern "C" __global__ void qwen35_convcast_qkv_packed(const __nv_bfloat16 *x,
    const __nv_bfloat16 *w, float *q, float *k, float *v, const unsigned *starts, unsigned rows,
    int kh, int vh, float scale, int length, int kernel, const __nv_bfloat16 *table, __half *qhalf,
    __half *khalf, __half *vhalf, float limit, unsigned *qv_bad) {
    qwen35_convcast_qv_impl<true, true>(x, w, q, k, v, starts, rows, kh, vh, scale, length, kernel,
        table, qhalf, khalf, vhalf, limit, qv_bad);
}

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <math.h>

__device__ __forceinline__ void qwen35_hrms_load4(const __nv_bfloat16 *p, __nv_bfloat16 (&v)[4]) {
    if ((reinterpret_cast<uintptr_t>(p) & 7u) == 0) {
        const uint2 words = *reinterpret_cast<const uint2 *>(p);
        v[0] = __ushort_as_bfloat16(static_cast<unsigned short>(words.x));
        v[1] = __ushort_as_bfloat16(static_cast<unsigned short>(words.x >> 16));
        v[2] = __ushort_as_bfloat16(static_cast<unsigned short>(words.y));
        v[3] = __ushort_as_bfloat16(static_cast<unsigned short>(words.y >> 16));
    } else {
#pragma unroll
        for (int j = 0; j < 4; ++j)
            v[j] = p[j];
    }
}
__device__ __forceinline__ void qwen35_hrms_store4(__nv_bfloat16 *p, const __nv_bfloat16 (&v)[4]) {
    const unsigned lo =
        unsigned(__bfloat16_as_ushort(v[0])) | (unsigned(__bfloat16_as_ushort(v[1])) << 16);
    const unsigned hi =
        unsigned(__bfloat16_as_ushort(v[2])) | (unsigned(__bfloat16_as_ushort(v[3])) << 16);
    *reinterpret_cast<uint2 *>(p) = make_uint2(lo, hi);
}
template <bool Fused>
__device__ __forceinline__ void
qwen35_hrms_vec4_impl(const __nv_bfloat16 *input, const __nv_bfloat16 *mixed,
                    const __nv_bfloat16 *weight, __nv_bfloat16 *residual, __nv_bfloat16 *output,
                    int columns, float epsilon) {

    const size_t base = size_t(blockIdx.x) * columns;
    __shared__ float warp_sums[32];
    float sums[4] = {};
#pragma unroll
    for (int block = 0; block < 4; ++block) {
        const int col = threadIdx.x * 4 + block * 1024;
        __nv_bfloat16 x[4], m[4], r[4];
        qwen35_hrms_load4(input + base + col, x);
        if constexpr (Fused)
            qwen35_hrms_load4(mixed + base + col, m);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            r[j] = Fused ? __hadd(x[j], m[j]) : x[j];
            const float v = __bfloat162float(r[j]);
            sums[j] = __fmaf_rn(v, v, sums[j]);
        }
        if constexpr (Fused)
            qwen35_hrms_store4(residual + base + col, r);
    }

#pragma unroll
    for (int step = 4; step > 0; step >>= 1) {
#pragma unroll
        for (int j = 0; j < 4; ++j)
            sums[j] = __fadd_rn(sums[j], __shfl_xor_sync(0xffffffff, sums[j], step, 8));
    }
    const float partial = __fadd_rn(__fadd_rn(sums[0], sums[2]), __fadd_rn(sums[1], sums[3]));
    if ((threadIdx.x & 7) == 0)
        warp_sums[threadIdx.x / 8] = partial;
    __syncthreads();

    float sum = warp_sums[threadIdx.x & 31];
#pragma unroll
    for (int step = 16; step > 0; step >>= 1)
        sum = __fadd_rn(sum, __shfl_xor_sync(0xffffffff, sum, step, 32));
    const float scale = rsqrtf(sum / float(columns) + epsilon);
#pragma unroll
    for (int block = 0; block < 4; ++block) {
        const int col = threadIdx.x * 4 + block * 1024;
        __nv_bfloat16 x[4], w[4], y[4];
        qwen35_hrms_load4((Fused ? residual : input) + base + col, x);
        qwen35_hrms_load4(weight + col, w);
#pragma unroll
        for (int j = 0; j < 4; ++j)
            y[j] = __nv_bfloat16(scale * __bfloat162float(x[j]) * (1.f + __bfloat162float(w[j])));
        qwen35_hrms_store4(output + base + col, y);
    }
}
extern "C" __global__ __launch_bounds__(256) void qwen35_hidden_rms_vec4_bf16(const __nv_bfloat16 *x,
                                                                            const __nv_bfloat16 *w,
                                                                            __nv_bfloat16 *output,
                                                                            int columns,
                                                                            float epsilon) {
    qwen35_hrms_vec4_impl<false>(x, nullptr, w, nullptr, output, columns, epsilon);
}
extern "C" __global__ __launch_bounds__(256) void qwen35_hidden_residual_rms_vec4_bf16(
    const __nv_bfloat16 *x, const __nv_bfloat16 *m, const __nv_bfloat16 *w, __nv_bfloat16 *residual,
    __nv_bfloat16 *output, int columns, float epsilon) {
    qwen35_hrms_vec4_impl<true>(x, m, w, residual, output, columns, epsilon);
}

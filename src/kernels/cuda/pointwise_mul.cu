#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>

__device__ __forceinline__ unsigned mul_bf16_pair(unsigned a, unsigned b) {
    const __nv_bfloat162_raw ar{
        static_cast<unsigned short>(a), static_cast<unsigned short>(a >> 16)};
    const __nv_bfloat162_raw br{
        static_cast<unsigned short>(b), static_cast<unsigned short>(b >> 16)};
    const __nv_bfloat162_raw r = __hmul2(__nv_bfloat162(ar), __nv_bfloat162(br));
    return unsigned(r.x) | (unsigned(r.y) << 16);
}

extern "C" __global__ void mul_bf16_vec8(
    const __nv_bfloat16 *left, const __nv_bfloat16 *right, __nv_bfloat16 *output, uint64_t count) {
    const uint64_t thread = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const uint64_t stride = uint64_t(gridDim.x) * blockDim.x;
    const uint64_t end = count & ~uint64_t(7);
    for (uint64_t i = thread * 8; i < end; i += stride * 8) {
        const uint4 a = *reinterpret_cast<const uint4 *>(left + i);
        const uint4 b = *reinterpret_cast<const uint4 *>(right + i);
        const uint4 r = make_uint4(mul_bf16_pair(a.x, b.x), mul_bf16_pair(a.y, b.y),
            mul_bf16_pair(a.z, b.z), mul_bf16_pair(a.w, b.w));
        *reinterpret_cast<uint4 *>(output + i) = r;
    }

    if (end + thread < count)
        output[end + thread] = left[end + thread] * right[end + thread];
}

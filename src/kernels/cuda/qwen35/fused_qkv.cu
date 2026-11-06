#include <cuda_bf16.h>

extern "C" __global__ void qwen35_fused_qkv_prepare_bf16(const __nv_bfloat16 *mixed, float *query,
    float *key, float *value, unsigned int rows, int key_heads, int value_heads, float scale) {
    const unsigned int row = blockIdx.x * 4 + threadIdx.x / 32;
    const unsigned int lane = threadIdx.x & 31;
    if (row >= rows)
        return;
    const unsigned int token = row / key_heads, head = row % key_heads;
    const unsigned int key_width = key_heads * 128, value_width = value_heads * 128;
    const unsigned int source = token * (2 * key_width + value_width) + head * 128 + lane * 4;
    const unsigned int target = row * 128 + lane * 4;
    float q[4], k[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        q[j] = __bfloat162float(mixed[source + j]);
        k[j] = __bfloat162float(mixed[source + key_width + j]);
    }
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
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const float normalized = __fdiv_rn(q[j], qd);
        query[target + j] = scale == 1.f ? normalized : __fmaf_rn(normalized, scale, 0.f);
        key[target + j] = __fdiv_rn(k[j], kd);
    }
    const int repeat = value_heads / key_heads;
    const unsigned int vo = token * value_width + head * repeat * 128;
    for (int j = lane; j < repeat * 128; j += 32)
        value[vo + j] = __bfloat162float(
            mixed[token * (2 * key_width + value_width) + 2 * key_width + head * repeat * 128 + j]);
}

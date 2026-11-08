#include <cuda_bf16.h>

__device__ __forceinline__ float qwen35_bfpostwarp_sum(float value) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        value += __shfl_xor_sync(0xffffffff, value, offset, 32);
    return value;
}

extern "C" __global__ void qwen35_bfpostwarp128(const __nv_bfloat16 *input,
    const __nv_bfloat16 *gate, const __nv_bfloat16 *norm, __nv_bfloat16 *output, int columns,
    float epsilon, unsigned long long length, unsigned long long heads, unsigned long long stride_b,
    unsigned long long stride_t, unsigned long long stride_h, unsigned long long rows) {
    const unsigned long long row = (unsigned long long)blockIdx.x * 8 + threadIdx.x / 32;
    if (row >= rows)
        return;
    const int lane = threadIdx.x & 31;
    const unsigned long long h = row % heads;
    const unsigned long long token = (row / heads) % length;
    const unsigned long long batch = row / (heads * length);
    const unsigned long long source_row = batch * stride_b + token * stride_t + h * stride_h;
    float values[4], sums[4];
#pragma unroll
    for (int part = 0; part < 4; ++part) {
        values[part] = float(input[source_row + part * 32 + lane]);
        sums[part] = qwen35_bfpostwarp_sum(__fmul_rn(values[part], values[part]));
    }
    const float sum = __fadd_rn(__fadd_rn(sums[0], sums[2]), __fadd_rn(sums[1], sums[3]));
    const float scale = rsqrtf(sum / float(columns) + epsilon);
#pragma unroll
    for (int part = 0; part < 4; ++part) {
        const int column = part * 32 + lane;
        const unsigned long long index = row * 128 + column;
        const float gate_value = float(gate[index]);
        const float gate_silu = gate_value / (1.f + expf(-gate_value));
        const float normalized = float(__nv_bfloat16(__fmul_rn(values[part], scale)));
        const float weighted = float(__nv_bfloat16(__fmul_rn(normalized, float(norm[column]))));
        output[index] = __nv_bfloat16(__fmul_rn(weighted, gate_silu));
    }
}

extern "C" __global__ void qwen35_build_bfpost_silu_f32_lookup(float *table) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < 65536u) {
        const float x = float(__ushort_as_bfloat16(static_cast<unsigned short>(i)));
        table[i] = x / (1.f + expf(-x));
    }
}

extern "C" __global__ void qwen35_bfpostwarp128_lookup(const __nv_bfloat16 *input,
    const __nv_bfloat16 *gate, const float *lut, const __nv_bfloat16 *norm, __nv_bfloat16 *output,
    int columns, float epsilon, unsigned long long length, unsigned long long heads,
    unsigned long long stride_b, unsigned long long stride_t, unsigned long long stride_h,
    unsigned long long rows) {
    const unsigned long long row = (unsigned long long)blockIdx.x * 8 + threadIdx.x / 32;
    if (row >= rows)
        return;
    const int lane = threadIdx.x & 31;
    const unsigned long long h = row % heads;
    const unsigned long long token = (row / heads) % length;
    const unsigned long long batch = row / (heads * length);
    const unsigned long long source_row = batch * stride_b + token * stride_t + h * stride_h;
    float values[4], sums[4];
#pragma unroll
    for (int part = 0; part < 4; ++part) {
        values[part] = float(input[source_row + part * 32 + lane]);
        sums[part] = qwen35_bfpostwarp_sum(__fmul_rn(values[part], values[part]));
    }
    const float sum = __fadd_rn(__fadd_rn(sums[0], sums[2]), __fadd_rn(sums[1], sums[3]));
    const float scale = rsqrtf(sum / float(columns) + epsilon);
#pragma unroll
    for (int part = 0; part < 4; ++part) {
        const int column = part * 32 + lane;
        const unsigned long long index = row * 128 + column;
        const float gate_value = float(gate[index]);
        const float gate_silu = lut[__bfloat16_as_ushort(gate[index])];
        const float normalized = float(__nv_bfloat16(__fmul_rn(values[part], scale)));
        const float weighted = float(__nv_bfloat16(__fmul_rn(normalized, float(norm[column]))));
        output[index] = __nv_bfloat16(__fmul_rn(weighted, gate_silu));
    }
}

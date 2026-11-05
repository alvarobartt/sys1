

extern "C" __global__ void
hopper_qkvz_conv_prepared_strided(const __nv_bfloat16 *x, const __nv_bfloat16 *w, float *q,
                                       float *k, float *v, const unsigned *starts, unsigned rows,
                                       int kh, int vh, float scale, int length, int kernel,
                                       const __nv_bfloat16 *table, __half *qhalf, __half *khalf,
                                       __half *vhalf, float limit, unsigned *qv_bad) {
    if (kh != 16 || vh != 32 || kernel != 4) {
        asm volatile("trap;");
        return;
    }
    halfonly_convcast_qv_impl<true, true, 12288>(x, w, q, k, v, starts, rows, kh, vh, scale, length,
                                                 kernel, table, qhalf, khalf, vhalf, limit, qv_bad);
}

extern "C" __global__ void hopper_qkvz_post_lookup_strided_gate(
    const __nv_bfloat16 *input, const __nv_bfloat16 *gate, const float *lut,
    const __nv_bfloat16 *norm, __nv_bfloat16 *output, int columns, float epsilon,
    unsigned long long length, unsigned long long heads, unsigned long long stride_b,
    unsigned long long stride_t, unsigned long long stride_h, unsigned long long rows,
    unsigned long long gate_stride_b, unsigned long long gate_stride_t,
    unsigned long long gate_stride_h) {
    const unsigned long long row = (unsigned long long)blockIdx.x * 8 + threadIdx.x / 32;
    if (row >= rows)
        return;
    const int lane = threadIdx.x & 31;

    if (heads != 32) {
        asm volatile("trap;");
        return;
    }
    const unsigned long long h = row & 31ull;
    const unsigned long long token_row = row >> 5;
    const bool packed_single_batch = rows == length * 32ull;
    const unsigned long long token = packed_single_batch ? token_row : token_row % length;
    const unsigned long long batch = packed_single_batch ? 0ull : token_row / length;
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
        const unsigned long long gate_index =
            batch * gate_stride_b + token * gate_stride_t + h * gate_stride_h + column;
        const float gate_value = float(gate[gate_index]);
        const float gate_silu = lut[__bfloat16_as_ushort(gate[gate_index])];
        const float normalized = float(__nv_bfloat16(__fmul_rn(values[part], scale)));
        const float weighted = float(__nv_bfloat16(__fmul_rn(normalized, float(norm[column]))));
        output[index] = __nv_bfloat16(__fmul_rn(weighted, gate_silu));
    }
}

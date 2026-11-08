#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <float.h>
#include <math.h>
#include "qwen35/projection_shared.cuh"

#if __CUDA_ARCH__ >= 800
__device__ __forceinline__ void qwen35_gate_nativevec_load4(unsigned *out, const __nv_bfloat16 *p) {
    const unsigned address = unsigned(__cvta_generic_to_shared(p));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3},[%4];"
                 : "=r"(out[0]), "=r"(out[1]), "=r"(out[2]), "=r"(out[3])
                 : "r"(address)
                 : "memory");
}
__device__ __forceinline__ void qwen35_gate_nativevec_load2(unsigned *out, const __nv_bfloat16 *p) {
    const unsigned address = unsigned(__cvta_generic_to_shared(p));
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1},[%2];"
                 : "=r"(out[0]), "=r"(out[1])
                 : "r"(address)
                 : "memory");
}
__device__ __forceinline__ void qwen35_gate_nativevec_mma(
    float *out, const unsigned *a, const unsigned *b) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                 "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
                 : "+f"(out[0]), "+f"(out[1]), "+f"(out[2]), "+f"(out[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
#endif

#if __CUDA_ARCH__ >= 800
__device__ __forceinline__ void qwen35_gate_nativevec_copy16(
    __nv_bfloat16 *destination, const __nv_bfloat16 *source, int source_bytes) {
    const unsigned address = unsigned(__cvta_generic_to_shared(destination));
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16, %2;" ::"r"(address), "l"(source),
                 "r"(source_bytes)
                 : "memory");
}
__device__ __forceinline__ void qwen35_gate_nativevec_stage(__nv_bfloat16 *tile_a,
    __nv_bfloat16 *tile_b, const __nv_bfloat16 *input, const __nv_bfloat16 *weight, int first_row,
    int rows, int first_k) {
    constexpr int SharedStride = 80, Threads = 256;
    const int ar = threadIdx.x / 8, ac = (threadIdx.x % 8) * 8, physical = first_row + ar;

    const auto *source = input + (physical < rows ? physical : 0) * 4096 + first_k + ac;
    qwen35_gate_nativevec_copy16(tile_a + ar * SharedStride + ac, source, physical < rows ? 16 : 0);
#pragma unroll
    for (int part = 0; part < 2; ++part) {
        const int index = threadIdx.x + part * Threads, br = index / 8, bc = (index % 8) * 8;
        qwen35_gate_nativevec_copy16(
            tile_b + br * SharedStride + bc, weight + br * 4096 + first_k + bc, 16);
    }
    asm volatile("cp.async.commit_group;" ::: "memory");
}
#endif

__device__ __forceinline__ void qwen35_gate_nativevec_impl(const __nv_bfloat16 *input,
    const __nv_bfloat16 *weight, __nv_bfloat16 *output, int rows, int inner, int columns,
    const unsigned *starts, const unsigned *ends) {
    constexpr int TileRows = 32, SharedStride = 80, Threads = 256;
    const int first_row = starts ? int(starts[blockIdx.x]) : int(blockIdx.x) * TileRows;
    if (ends)
        rows = int(ends[blockIdx.x]);
    const int first_column = blockIdx.y * 64;
#if __CUDA_ARCH__ >= 800
    __shared__ __align__(32) __nv_bfloat16 a_buffers[2 * TileRows * SharedStride];
    __shared__ __align__(32) __nv_bfloat16 b_buffers[2 * 64 * SharedStride];
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int row_base = (warp / 4) * 16, col_base = (warp % 4) * 16;
    bool exceptional = false;
    float running[2][4] = {}, correction[2][4] = {};
    const bool vector_copy = inner == 4096 && columns == 64 &&
        ((reinterpret_cast<uintptr_t>(input) | reinterpret_cast<uintptr_t>(weight)) & 15u) == 0;
    if (vector_copy)
        qwen35_gate_nativevec_stage(a_buffers, b_buffers, input, weight, first_row, rows, 0);
    for (int first_k = 0; first_k < inner; first_k += 64) {
        const int current = vector_copy ? ((first_k / 64) & 1) : 0;
        auto *tile_a = a_buffers + current * TileRows * SharedStride;
        auto *tile_b = b_buffers + current * 64 * SharedStride;

        if (vector_copy) {
            asm volatile("cp.async.wait_group 0;" ::: "memory");
        } else {
            for (int i = threadIdx.x; i < TileRows * 64; i += Threads) {
                const int row = first_row + i / 64, k = first_k + i % 64;
                const auto x =
                    row < rows && k < inner ? input[row * inner + k] : __float2bfloat16_rn(0.f);
                tile_a[(i / 64) * SharedStride + i % 64] = x;
                exceptional |= qwen35_projection_exceptional_operand(x);
            }
            for (int i = threadIdx.x; i < 64 * 64; i += Threads) {
                const int col = first_column + i / 64, k = first_k + i % 64;
                const auto x =
                    col < columns && k < inner ? weight[col * inner + k] : __float2bfloat16_rn(0.f);
                tile_b[(i / 64) * SharedStride + i % 64] = x;
                exceptional |= qwen35_projection_exceptional_operand(x);
            }
        }
        __syncthreads();
        if (vector_copy) {
            if (first_k + 64 < inner) {
                const int next = ((first_k / 64) + 1) & 1;
                qwen35_gate_nativevec_stage(a_buffers + next * TileRows * SharedStride,
                    b_buffers + next * 64 * SharedStride, input, weight, first_row, rows,
                    first_k + 64);
            }

            const int ar = threadIdx.x / 8, ac = (threadIdx.x % 8) * 8;
            const uint4 a = *reinterpret_cast<const uint4 *>(tile_a + ar * SharedStride + ac);
            const unsigned words_a[4] = {a.x, a.y, a.z, a.w};
#pragma unroll
            for (int item = 0; item < 4; ++item) {
                exceptional |= qwen35_projection_exceptional_operand(
                    __ushort_as_bfloat16((unsigned short)words_a[item]));
                exceptional |= qwen35_projection_exceptional_operand(
                    __ushort_as_bfloat16((unsigned short)(words_a[item] >> 16)));
            }
#pragma unroll
            for (int part = 0; part < 2; ++part) {
                const int index = threadIdx.x + part * Threads, br = index / 8,
                          bc = (index % 8) * 8;
                const uint4 b = *reinterpret_cast<const uint4 *>(tile_b + br * SharedStride + bc);
                const unsigned words_b[4] = {b.x, b.y, b.z, b.w};
#pragma unroll
                for (int item = 0; item < 4; ++item) {
                    exceptional |= qwen35_projection_exceptional_operand(
                        __ushort_as_bfloat16((unsigned short)words_b[item]));
                    exceptional |= qwen35_projection_exceptional_operand(
                        __ushort_as_bfloat16((unsigned short)(words_b[item] >> 16)));
                }
            }
        }
        for (int offset = 0; offset < 64 && first_k + offset < inner; offset += 16) {
            unsigned a[4];
            qwen35_gate_nativevec_load4(
                a, tile_a + (row_base + (lane & 15)) * SharedStride + offset + (lane / 16) * 8);
#pragma unroll
            for (int part = 0; part < 2; ++part) {
                unsigned b[2];
                float partial[4] = {};
                qwen35_gate_nativevec_load2(b,
                    tile_b + (col_base + part * 8 + (lane & 7)) * SharedStride + offset +
                        ((lane / 8) & 1) * 8);
                qwen35_gate_nativevec_mma(partial, a, b);
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const float sum = __fadd_rn(running[part][i], partial[i]);
                    const float virtual_partial = __fsub_rn(sum, running[part][i]);
                    const float error =
                        __fadd_rn(__fsub_rn(running[part][i], __fsub_rn(sum, virtual_partial)),
                            __fsub_rn(partial[i], virtual_partial));
                    correction[part][i] = __fadd_rn(correction[part][i], error);
                    running[part][i] = sum;
                }
            }
        }
        __syncthreads();
    }
    const int exceptional_block = __syncthreads_or(exceptional);
#pragma unroll
    for (int part = 0; part < 2; ++part) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int row = first_row + row_base + lane / 4 + (i / 2) * 8;
            const int col = first_column + col_base + part * 8 + (lane % 4) * 2 + i % 2;
            if (row < rows && col < columns) {
                float value = __fadd_rn(running[part][i], correction[part][i]);
                if (exceptional_block || !isfinite(value)) {
                    double sum = 0.;
                    for (int k = 0; k < inner; ++k)
                        sum = __fma_rn(double(float(input[row * inner + k])),
                            double(float(weight[col * inner + k])), sum);
                    value = __double2float_rn(sum);
                }
                output[row * columns + col] = __float2bfloat16_rn(value);
            }
        }
    }
#else
    for (int i = threadIdx.x; i < TileRows * 64; i += Threads) {
        const int row = first_row + i / 64, col = first_column + i % 64;
        if (row < rows && col < columns) {
            double sum = 0.;
            for (int k = 0; k < inner; ++k)
                sum = __fma_rn(double(float(input[row * inner + k])),
                    double(float(weight[col * inner + k])), sum);
            output[row * columns + col] = __float2bfloat16_rn(__double2float_rn(sum));
        }
    }
#endif
}
extern "C" __global__ void qwen35_gate_nativevec_rows32_bf16(const __nv_bfloat16 *x,
    const __nv_bfloat16 *w, __nv_bfloat16 *out, int rows, int inner, int columns) {
    qwen35_gate_nativevec_impl(x, w, out, rows, inner, columns, nullptr, nullptr);
}
extern "C" __global__ void qwen35_gate_nativevec_packed_bf16(const __nv_bfloat16 *x,
    const __nv_bfloat16 *w, __nv_bfloat16 *out, int rows, int inner, int columns,
    const unsigned *starts, const unsigned *ends) {
    qwen35_gate_nativevec_impl(x, w, out, rows, inner, columns, starts, ends);
}

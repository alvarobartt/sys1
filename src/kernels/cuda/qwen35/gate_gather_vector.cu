#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <mma.h>
#include <float.h>
#include <math.h>
#include "qwen35/projection_shared.cuh"

__device__ __forceinline__ float qwen35_gate_gathervector_guard_recompute(
    const __nv_bfloat16 *x, const __nv_bfloat16 *w, int row, int col, int inner) {
    const int lane = threadIdx.x & 31;
    float running = 0.f, correction = 0.f;
    bool exceptional = false;
    for (int k = lane; k < inner; k += 32) {
        const float a = float(x[row * inner + k]), b = float(w[col * inner + k]);
        const float product = __fmul_rn(a, b), updated = __fadd_rn(running, product);
        const float virtual_product = __fsub_rn(updated, running);
        const float error = __fadd_rn(__fsub_rn(running, __fsub_rn(updated, virtual_product)),
            __fsub_rn(product, virtual_product));
        correction = __fadd_rn(correction, error);
        running = updated;
        exceptional |= a != 0.f && b != 0.f && fabsf(product) < FLT_MIN;
    }
    double sum = __dadd_rn(double(running), double(correction));
    if (__any_sync(0xffffffff, exceptional || !isfinite(running) || !isfinite(correction))) {
        sum = 0.;
        for (int k = lane; k < inner; k += 32)
            sum =
                __fma_rn(double(float(x[row * inner + k])), double(float(w[col * inner + k])), sum);
    }
#pragma unroll
    for (int step = 16; step > 0; step >>= 1)
        sum = __dadd_rn(sum, __shfl_down_sync(0xffffffff, sum, step));
    return __double2float_rn(__shfl_sync(0xffffffff, sum, 0));
}

template <int TileRows>
__device__ __forceinline__ void qwen35_gate_gathervector_small_impl(const __nv_bfloat16 *input,
    const __nv_bfloat16 *weight, __nv_bfloat16 *output, int rows, int inner, int columns,
    const __nv_bfloat16 *bias = nullptr, const unsigned *tile_starts = nullptr,
    const unsigned *tile_ends = nullptr, const unsigned *gather_ids = nullptr) {
    const int first_row = tile_starts ? int(tile_starts[blockIdx.x]) : int(blockIdx.x) * TileRows;
    if (tile_ends)
        rows = int(tile_ends[blockIdx.x]);
    const int first_column = blockIdx.y * 64;
    constexpr int WarpRows = TileRows < 32 ? TileRows : 32;
    constexpr int Groups = TileRows / WarpRows;
    static_assert(Groups == 1, "guard probe only supports tested M16/M32");
    constexpr int Threads = WarpRows * 8;
    constexpr int Outputs = TileRows * 64;
#if __CUDA_ARCH__ >= 800
    namespace wmma = nvcuda::wmma;

    constexpr int SharedStride = 64 + 16;
    __shared__ __align__(32) __nv_bfloat16 tile_a[TileRows * SharedStride];
    __shared__ __align__(32) __nv_bfloat16 tile_b[64 * SharedStride];

    __shared__ __align__(32) float tile_c[WarpRows * 64];
    __shared__ __align__(32) float tile_abs[WarpRows * 64];
    const int warp = threadIdx.x / 32;
    const int warp_row = warp / 4;
    const int warp_column = warp % 4;
    bool exceptional = false;
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> b;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c[Groups], absolute[Groups];
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> running[Groups];
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> correction[Groups];
#pragma unroll
    for (int group = 0; group < Groups; ++group) {
        wmma::fill_fragment(running[group], 0.f);
        wmma::fill_fragment(absolute[group], 0.f);
        wmma::fill_fragment(correction[group], 0.f);
    }
    const bool vector_copy = inner == 4096 && columns == 64 &&
        ((reinterpret_cast<uintptr_t>(input) | reinterpret_cast<uintptr_t>(weight)) & 15u) == 0;
    for (int first_k = 0; first_k < inner; first_k += 64) {
        if (vector_copy) {
            const int ar = threadIdx.x / 8, ac = (threadIdx.x % 8) * 8;
            const int logical = first_row + ar;
            const int physical =
                logical < rows ? (gather_ids ? int(gather_ids[logical]) : logical) : 0;
            const uint4 a = logical < rows
                ? *reinterpret_cast<const uint4 *>(input + physical * 4096 + first_k + ac)
                : make_uint4(0, 0, 0, 0);
            *reinterpret_cast<uint4 *>(tile_a + ar * SharedStride + ac) = a;
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
                const uint4 b = *reinterpret_cast<const uint4 *>(weight + br * 4096 + first_k + bc);
                *reinterpret_cast<uint4 *>(tile_b + br * SharedStride + bc) = b;
                const unsigned words_b[4] = {b.x, b.y, b.z, b.w};
#pragma unroll
                for (int item = 0; item < 4; ++item) {
                    exceptional |= qwen35_projection_exceptional_operand(
                        __ushort_as_bfloat16((unsigned short)words_b[item]));
                    exceptional |= qwen35_projection_exceptional_operand(
                        __ushort_as_bfloat16((unsigned short)(words_b[item] >> 16)));
                }
            }
        } else {
            for (int index = threadIdx.x; index < TileRows * 64; index += Threads) {
                const int row = first_row + index / 64;
                const int k = first_k + index % 64;
                const int shared_index = (index / 64) * SharedStride + index % 64;
                tile_a[shared_index] = row < rows && k < inner
                    ? input[(gather_ids ? int(gather_ids[row]) : row) * inner + k]
                    : __float2bfloat16_rn(0.f);
                exceptional |= qwen35_projection_exceptional_operand(tile_a[shared_index]);
            }
            for (int index = threadIdx.x; index < 4096; index += Threads) {
                const int column = first_column + index / 64;
                const int k = first_k + index % 64;
                const int shared_index = (index / 64) * SharedStride + index % 64;
                tile_b[shared_index] = column < columns && k < inner ? weight[column * inner + k]
                                                                     : __float2bfloat16_rn(0.f);
                exceptional |= qwen35_projection_exceptional_operand(tile_b[shared_index]);
            }
        }
        __syncthreads();
        for (int offset = 0; offset < 64 && first_k + offset < inner; offset += 16) {
#pragma unroll
            for (int group = 0; group < Groups; ++group) {
                wmma::fill_fragment(c[group], 0.f);
            }
            wmma::load_matrix_sync(
                b, tile_b + warp_column * 16 * SharedStride + offset, SharedStride);
#pragma unroll
            for (int group = 0; group < Groups; ++group) {
                wmma::load_matrix_sync(a,
                    tile_a + (group * WarpRows + warp_row * 16) * SharedStride + offset,
                    SharedStride);
                wmma::mma_sync(c[group], a, b, c[group]);
#pragma unroll
                for (int i = 0; i < a.num_elements; ++i)
                    a.x[i] = __ushort_as_bfloat16(__bfloat16_as_ushort(a.x[i]) & 0x7fff);
#pragma unroll
                for (int i = 0; i < b.num_elements; ++i)
                    b.x[i] = __ushort_as_bfloat16(__bfloat16_as_ushort(b.x[i]) & 0x7fff);
                wmma::mma_sync(absolute[group], a, b, absolute[group]);
            }
#pragma unroll
            for (int group = 0; group < Groups; ++group) {
#pragma unroll
                for (int i = 0; i < c[group].num_elements; ++i) {
                    const float partial = c[group].x[i];
                    const float sum = __fadd_rn(running[group].x[i], partial);
                    const float virtual_partial = __fsub_rn(sum, running[group].x[i]);
                    const float error =
                        __fadd_rn(__fsub_rn(running[group].x[i], __fsub_rn(sum, virtual_partial)),
                            __fsub_rn(partial, virtual_partial));
                    correction[group].x[i] = __fadd_rn(correction[group].x[i], error);
                    running[group].x[i] = sum;
                }
            }
        }
        __syncthreads();
    }

    const int exceptional_block = __syncthreads_or(exceptional);
#pragma unroll
    for (int group = 0; group < Groups; ++group) {
#pragma unroll
        for (int i = 0; i < c[group].num_elements; ++i) {
            c[group].x[i] = __fadd_rn(running[group].x[i], correction[group].x[i]);
        }
        wmma::store_matrix_sync(tile_c + warp * 256, c[group], 16, wmma::mem_row_major);
        wmma::store_matrix_sync(tile_abs + warp * 256, absolute[group], 16, wmma::mem_row_major);
        __syncthreads();
        for (int index = threadIdx.x; index < WarpRows * 64; index += Threads) {
            const int logical =
                first_row + group * WarpRows + (index / 1024) * 16 + (index % 256) / 16;
            const int column = first_column + ((index / 256) % 4) * 16 + index % 16;
            const bool valid = logical < rows && column < columns;
            const int physical = valid ? (gather_ids ? int(gather_ids[logical]) : logical) : 0;
            float value = valid ? tile_c[index] : 0.f;
            if (valid && (exceptional_block || !isfinite(value))) {
                double sum = 0.;
                for (int k = 0; k < inner; ++k)
                    sum = __fma_rn(double(float(input[physical * inner + k])),
                        double(float(weight[column * inner + k])), sum);
                value = __double2float_rn(sum);
            }

            const unsigned bits =
                unsigned(__bfloat16_as_ushort(__float2bfloat16_rn(fabsf(value)))) & 0x7fff;
            const float center = float(__ushort_as_bfloat16((unsigned short)bits));
            const float previous =
                float(__ushort_as_bfloat16((unsigned short)(bits ? bits - 1 : 0)));
            const float next = float(__ushort_as_bfloat16((unsigned short)(bits + 1)));
            const float lower = previous + (center - previous) * 0.5f,
                        upper = center + (next - center) * 0.5f;
            const float distance = fminf(fabsf(fabsf(value) - lower), fabsf(fabsf(value) - upper));
            const float radius = __fmul_rn(fabsf(tile_abs[index]), 0x1.004p-18f);
            const bool critical =
                valid && !exceptional_block && isfinite(value) && distance <= radius;
            unsigned mask = __ballot_sync(0xffffffff, critical);
            while (mask) {
                const int leader = __ffs(mask) - 1;
                const int selected_row = __shfl_sync(0xffffffff, physical, leader);
                const int selected_column = __shfl_sync(0xffffffff, column, leader);
                const float corrected = qwen35_gate_gathervector_guard_recompute(
                    input, weight, selected_row, selected_column, inner);
                if ((threadIdx.x & 31) == leader)
                    value = corrected;
                mask &= mask - 1;
            }
            if (valid) {
                if (bias != nullptr)
                    value = __fadd_rn(value, float(bias[column]));
                output[physical * columns + column] = __float2bfloat16_rn(value);
            }
        }
        __syncthreads();
    }
#else

    for (int index = threadIdx.x; index < Outputs; index += Threads) {
        const int row = first_row + index / 64;
        const int column = first_column + index % 64;
        if (row < rows && column < columns) {
            double sum = 0.;
            for (int k = 0; k < inner; ++k) {
                sum = __fma_rn(
                    double(float(input[(gather_ids ? int(gather_ids[row]) : row) * inner + k])),
                    double(float(weight[column * inner + k])), sum);
            }
            float value = __double2float_rn(sum);
            if (bias != nullptr)
                value = __fadd_rn(value, float(bias[column]));
            output[(gather_ids ? int(gather_ids[row]) : row) * columns + column] =
                __float2bfloat16_rn(value);
        }
    }
#endif
}

extern "C" __global__ void qwen35_gate_gathervector_small_32(const __nv_bfloat16 *x,
    const __nv_bfloat16 *w, __nv_bfloat16 *out, int rows, int inner, int columns,
    const unsigned *ids) {
    qwen35_gate_gathervector_small_impl<32>(
        x, w, out, rows, inner, columns, nullptr, nullptr, nullptr, ids);
}

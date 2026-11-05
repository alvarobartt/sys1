#include <cuda_bf16.h>
#include <float.h>
#include <math.h>
#include <mma.h>

#include "projection_shared.cuh"


template <int TileRows>
__device__ __forceinline__ void
qwen35_projection_tile_bf16(const __nv_bfloat16 *input, const __nv_bfloat16 *weight,
                            __nv_bfloat16 *output, int rows, int inner, int columns,
                            const __nv_bfloat16 *bias = nullptr) {
    const int first_row = blockIdx.x * TileRows;
    const int first_column = blockIdx.y * 64;
    constexpr int WarpRows = TileRows < 32 ? TileRows : 32;
    constexpr int Groups = TileRows / WarpRows;
    constexpr int Threads = WarpRows * 8;
    constexpr int Outputs = TileRows * 64;
#if __CUDA_ARCH__ >= 800
    namespace wmma = nvcuda::wmma;

    constexpr int SharedStride = 64 + 16;
    __shared__ __align__(32) __nv_bfloat16 tile_a[TileRows * SharedStride];
    __shared__ __align__(32) __nv_bfloat16 tile_b[64 * SharedStride];

    __shared__ __align__(32) float tile_c[WarpRows * 64];
    const int warp = threadIdx.x / 32;
    const int warp_row = warp / 4;
    const int warp_column = warp % 4;
    bool exceptional = false;
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> b;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c[Groups];
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> running[Groups];
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> correction[Groups];
#pragma unroll
    for (int group = 0; group < Groups; ++group) {
        wmma::fill_fragment(running[group], 0.f);
        wmma::fill_fragment(correction[group], 0.f);
    }
    for (int first_k = 0; first_k < inner; first_k += 64) {
        for (int index = threadIdx.x; index < TileRows * 64; index += Threads) {
            const int row = first_row + index / 64;
            const int k = first_k + index % 64;
            const int shared_index = (index / 64) * SharedStride + index % 64;
            tile_a[shared_index] =
                row < rows && k < inner ? input[row * inner + k] : __float2bfloat16_rn(0.f);
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
        __syncthreads();
#pragma unroll
        for (int group = 0; group < Groups; ++group) {
            wmma::fill_fragment(c[group], 0.f);
        }
        for (int offset = 0; offset < 64 && first_k + offset < inner; offset += 16) {
            wmma::load_matrix_sync(b, tile_b + warp_column * 16 * SharedStride + offset,
                                   SharedStride);
#pragma unroll
            for (int group = 0; group < Groups; ++group) {
                wmma::load_matrix_sync(
                    a, tile_a + (group * WarpRows + warp_row * 16) * SharedStride + offset,
                    SharedStride);
                wmma::mma_sync(c[group], a, b, c[group]);
            }
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
        __syncthreads();
        for (int index = threadIdx.x; index < WarpRows * 64; index += Threads) {
            const int row = first_row + group * WarpRows + (index / 1024) * 16 + (index % 256) / 16;
            const int column = first_column + ((index / 256) % 4) * 16 + index % 16;
            if (row < rows && column < columns) {
                float value = tile_c[index];

                if (exceptional_block || !isfinite(value)) {
                    double sum = 0.;
                    for (int k = 0; k < inner; ++k) {
                        sum = __fma_rn(double(float(input[row * inner + k])),
                                       double(float(weight[column * inner + k])), sum);
                    }
                    value = __double2float_rn(sum);
                }
                if (bias != nullptr)
                    value = __fadd_rn(value, float(bias[column]));
                output[row * columns + column] = __float2bfloat16_rn(value);
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
                sum = __fma_rn(double(float(input[row * inner + k])),
                               double(float(weight[column * inner + k])), sum);
            }
            float value = __double2float_rn(sum);
            if (bias != nullptr)
                value = __fadd_rn(value, float(bias[column]));
            output[row * columns + column] = __float2bfloat16_rn(value);
        }
    }
#endif
}

extern "C" __global__ void qwen35_kv_projection_bf16(const __nv_bfloat16 *input,
                                                     const __nv_bfloat16 *weight,
                                                     __nv_bfloat16 *output, int rows, int inner,
                                                     int columns) {
    qwen35_projection_tile_bf16<16>(input, weight, output, rows, inner, columns);
}

extern "C" __global__ void qwen35_kv_projection_rows32_bf16(const __nv_bfloat16 *input,
                                                            const __nv_bfloat16 *weight,
                                                            __nv_bfloat16 *output, int rows,
                                                            int inner, int columns) {
    qwen35_projection_tile_bf16<32>(input, weight, output, rows, inner, columns);
}

extern "C" __global__ void qwen35_kv_projection_rows128_bf16(const __nv_bfloat16 *input,
                                                             const __nv_bfloat16 *weight,
                                                             __nv_bfloat16 *output, int rows,
                                                             int inner, int columns) {
    qwen35_projection_tile_bf16<128>(input, weight, output, rows, inner, columns);
}

extern "C" __global__ void qwen35_biased_projection_bf16(const __nv_bfloat16 *input,
                                                         const __nv_bfloat16 *weight,
                                                         const __nv_bfloat16 *bias,
                                                         __nv_bfloat16 *output, int rows, int inner,
                                                         int columns) {
    qwen35_projection_tile_bf16<16>(input, weight, output, rows, inner, columns, bias);
}
extern "C" __global__ void qwen35_biased_projection_rows32_bf16(const __nv_bfloat16 *input,
                                                                const __nv_bfloat16 *weight,
                                                                const __nv_bfloat16 *bias,
                                                                __nv_bfloat16 *output, int rows,
                                                                int inner, int columns) {
    qwen35_projection_tile_bf16<32>(input, weight, output, rows, inner, columns, bias);
}
extern "C" __global__ void qwen35_biased_projection_rows128_bf16(const __nv_bfloat16 *input,
                                                                 const __nv_bfloat16 *weight,
                                                                 const __nv_bfloat16 *bias,
                                                                 __nv_bfloat16 *output, int rows,
                                                                 int inner, int columns) {
    qwen35_projection_tile_bf16<128>(input, weight, output, rows, inner, columns, bias);
}

__device__ __forceinline__ void qwen35_compensated_projection_bf16(const __nv_bfloat16 *input,
                                                                   const __nv_bfloat16 *weight,
                                                                   const __nv_bfloat16 *bias,
                                                                   __nv_bfloat16 *output, int inner,
                                                                   int columns) {
    const int row = blockIdx.x;
    const int lane = threadIdx.x & 31;
    const int column = blockIdx.y * 8 + (threadIdx.x >> 5);
    if (column >= columns)
        return;
    float running = 0.f, correction = 0.f;
    bool exceptional = false;
    for (int index = lane; index < inner; index += 32) {
        const float a = float(input[row * inner + index]);
        const float b = float(weight[column * inner + index]);
        const float product = __fmul_rn(a, b);
        const float updated = __fadd_rn(running, product);
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
        for (int index = lane; index < inner; index += 32) {
            sum = __fma_rn(double(float(input[row * inner + index])),
                           double(float(weight[column * inner + index])), sum);
        }
    }
    for (int step = 16; step > 0; step >>= 1) {
        sum = __dadd_rn(sum, __shfl_down_sync(0xffffffff, sum, step));
    }
    if (lane == 0) {
        float value = __double2float_rn(sum);
        if (bias != nullptr)
            value = __fadd_rn(value, float(bias[column]));
        output[row * columns + column] = __float2bfloat16_rn(value);
    }
}

extern "C" __global__ void qwen35_gate_projection_bf16(const __nv_bfloat16 *input,
                                                       const __nv_bfloat16 *weight,
                                                       __nv_bfloat16 *output, int inner,
                                                       int columns) {
    qwen35_compensated_projection_bf16(input, weight, nullptr, output, inner, columns);
}

extern "C" __global__ void qwen35_biased_compensated_projection_bf16(const __nv_bfloat16 *input,
                                                                     const __nv_bfloat16 *weight,
                                                                     const __nv_bfloat16 *bias,
                                                                     __nv_bfloat16 *output,
                                                                     int inner, int columns) {
    qwen35_compensated_projection_bf16(input, weight, bias, output, inner, columns);
}

__device__ __forceinline__ void qwen35_tiled_compensated_projection_impl_bf16(
    const __nv_bfloat16 *input, const __nv_bfloat16 *weight, const __nv_bfloat16 *bias,
    __nv_bfloat16 *output, int rows, int inner, int columns) {
    constexpr int RowTile = 4, ColumnTile = 1;
    const int lane = threadIdx.x & 31;
    const int first_row = blockIdx.x * RowTile;
    const int first_column = blockIdx.y * 4 + (threadIdx.x >> 5) * ColumnTile;
    if (first_column >= columns)
        return;
    float running[RowTile][ColumnTile] = {}, correction[RowTile][ColumnTile] = {};
    unsigned exceptional = 0;
    for (int index = lane; index < inner; index += 32) {
        float a[RowTile], b[ColumnTile];
#pragma unroll
        for (int r = 0; r < RowTile; ++r)
            a[r] = first_row + r < rows ? float(input[(first_row + r) * inner + index]) : 0.f;
#pragma unroll
        for (int c = 0; c < ColumnTile; ++c)
            b[c] = first_column + c < columns ? float(weight[(first_column + c) * inner + index])
                                              : 0.f;
#pragma unroll
        for (int r = 0; r < RowTile; ++r) {
#pragma unroll
            for (int c = 0; c < ColumnTile; ++c) {
                const float product = __fmul_rn(a[r], b[c]);
                const float updated = __fadd_rn(running[r][c], product);
                const float virtual_product = __fsub_rn(updated, running[r][c]);
                const float error =
                    __fadd_rn(__fsub_rn(running[r][c], __fsub_rn(updated, virtual_product)),
                              __fsub_rn(product, virtual_product));
                correction[r][c] = __fadd_rn(correction[r][c], error);
                running[r][c] = updated;
                if (a[r] != 0.f && b[c] != 0.f && fabsf(product) < FLT_MIN)
                    exceptional |= 1u << (r * ColumnTile + c);
            }
        }
    }
#pragma unroll
    for (int r = 0; r < RowTile; ++r) {
#pragma unroll
        for (int c = 0; c < ColumnTile; ++c) {
            const int row = first_row + r, column = first_column + c;
            if (row >= rows || column >= columns)
                continue;
            double sum = __dadd_rn(double(running[r][c]), double(correction[r][c]));
            if (__any_sync(0xffffffff, (exceptional & (1u << (r * ColumnTile + c))) ||
                                           !isfinite(running[r][c]) ||
                                           !isfinite(correction[r][c]))) {
                sum = 0.;
                for (int index = lane; index < inner; index += 32)
                    sum = __fma_rn(double(float(input[row * inner + index])),
                                   double(float(weight[column * inner + index])), sum);
            }
#pragma unroll
            for (int step = 16; step > 0; step >>= 1)
                sum = __dadd_rn(sum, __shfl_down_sync(0xffffffff, sum, step));
            if (lane == 0) {
                float value = __double2float_rn(sum);
                if (bias != nullptr)
                    value = __fadd_rn(value, float(bias[column]));
                output[row * columns + column] = __float2bfloat16_rn(value);
            }
        }
    }
}

extern "C" __global__ void qwen35_tiled_compensated_projection_bf16(const __nv_bfloat16 *input,
                                                                    const __nv_bfloat16 *weight,
                                                                    __nv_bfloat16 *output, int rows,
                                                                    int inner, int columns) {
    qwen35_tiled_compensated_projection_impl_bf16(input, weight, nullptr, output, rows, inner,
                                                  columns);
}

extern "C" __global__ void qwen35_biased_tiled_compensated_projection_bf16(
    const __nv_bfloat16 *input, const __nv_bfloat16 *weight, const __nv_bfloat16 *bias,
    __nv_bfloat16 *output, int rows, int inner, int columns) {
    qwen35_tiled_compensated_projection_impl_bf16(input, weight, bias, output, rows, inner,
                                                  columns);
}

extern "C" __global__ void qwen35_guarded_double_projection_bf16(const __nv_bfloat16 *input,
                                                                 const __nv_bfloat16 *weight,
                                                                 __nv_bfloat16 *output, int inner,
                                                                 int columns,
                                                                 double radius_coefficient) {
    const int row = blockIdx.x;
    const int lane = threadIdx.x & 31;
    const int column = blockIdx.y * 8 + (threadIdx.x >> 5);
    if (column >= columns)
        return;
    double fast_sum = 0., absolute_sum = 0.;
    bool unsafe = false;
    for (int index = lane; index < inner; index += 32) {
        const float a = float(input[row * inner + index]);
        const float b = float(weight[column * inner + index]);
        const double product = __dmul_rn(double(a), double(b));
        fast_sum = __dadd_rn(fast_sum, product);
        absolute_sum = __dadd_ru(absolute_sum, fabs(product));
        const float fp32_product = __fmul_rn(a, b);
        unsafe |=
            !isfinite(fp32_product) || (a != 0.f && b != 0.f && fabsf(fp32_product) < FLT_MIN);
    }
    for (int step = 16; step > 0; step >>= 1) {
        fast_sum = __dadd_rn(fast_sum, __shfl_down_sync(0xffffffff, fast_sum, step));
        absolute_sum = __dadd_ru(absolute_sum, __shfl_down_sync(0xffffffff, absolute_sum, step));
    }
    const bool unsafe_warp = __any_sync(0xffffffff, unsafe);
    int accepted = 0;
    if (lane == 0 && !unsafe_warp && isfinite(absolute_sum) && absolute_sum > 0. &&
        absolute_sum <= double(FLT_MAX) * 0x1p-6 && isfinite(radius_coefficient) &&
        radius_coefficient > 0.) {

        const double radius = __dmul_ru(radius_coefficient, absolute_sum);
        const __nv_bfloat16 low =
            __float2bfloat16_rn(__double2float_rn(__dsub_rd(fast_sum, radius)));
        const __nv_bfloat16 high =
            __float2bfloat16_rn(__double2float_rn(__dadd_ru(fast_sum, radius)));
        accepted = __bfloat16_as_ushort(low) == __bfloat16_as_ushort(high);
        if (accepted)
            output[row * columns + column] = low;
    }
    if (__shfl_sync(0xffffffff, accepted, 0))
        return;

    float running = 0.f, correction = 0.f;
    bool exceptional = false;
    for (int index = lane; index < inner; index += 32) {
        const float a = float(input[row * inner + index]);
        const float b = float(weight[column * inner + index]);
        const float product = __fmul_rn(a, b);
        const float updated = __fadd_rn(running, product);
        const float virtual_product = __fsub_rn(updated, running);
        const float error = __fadd_rn(__fsub_rn(running, __fsub_rn(updated, virtual_product)),
                                      __fsub_rn(product, virtual_product));
        correction = __fadd_rn(correction, error);
        running = updated;
        exceptional |= !isfinite(product) || !isfinite(running) || !isfinite(correction) ||
                       (a != 0.f && b != 0.f && fabsf(product) < FLT_MIN);
    }
    double sum = __dadd_rn(double(running), double(correction));
    if (__any_sync(0xffffffff, exceptional)) {
        sum = 0.;
        for (int index = lane; index < inner; index += 32) {
            sum = __fma_rn(double(float(input[row * inner + index])),
                           double(float(weight[column * inner + index])), sum);
        }
    }
    for (int step = 16; step > 0; step >>= 1) {
        sum = __dadd_rn(sum, __shfl_down_sync(0xffffffff, sum, step));
    }
    if (lane == 0) {
        output[row * columns + column] = __float2bfloat16_rn(__double2float_rn(sum));
    }
}

template <int RowTiles>
__device__ void qwen35_guarded_tensor_double_projection_impl_bf16(const __nv_bfloat16 *input,
                                                                  const __nv_bfloat16 *weight,
                                                                  __nv_bfloat16 *output, int rows,
                                                                  int inner, int columns,
                                                                  double radius_coefficient) {
    const int lane = threadIdx.x & 31;
    const int first_row = blockIdx.x * (8 * RowTiles);
    const int first_column = blockIdx.y * 32 + (threadIdx.x >> 5) * 8;
    double sums[RowTiles][2] = {}, absolutes[RowTiles][2] = {};
    bool unsafe[RowTiles] = {};
#if __CUDA_ARCH__ == 800 || __CUDA_ARCH__ == 900
    for (int first_k = 0; first_k < inner; first_k += 4) {
        const int column = first_column + (lane >> 2);
        const int k = first_k + (lane & 3);
        const float b = column < columns && k < inner ? float(weight[column * inner + k]) : 0.f;

        const double bd = double(b), ab = fabs(bd);
#pragma unroll
        for (int row_tile = 0; row_tile < RowTiles; ++row_tile) {
            const int row = first_row + row_tile * 8 + (lane >> 2);
            const float a = row < rows && k < inner ? float(input[row * inner + k]) : 0.f;

            unsafe[row_tile] |= !isfinite(a) || !isfinite(b) ||
                                (a != 0.f && (fabsf(a) < 0x1p-60f || fabsf(a) > 0x1p60f)) ||
                                (b != 0.f && (fabsf(b) < 0x1p-60f || fabsf(b) > 0x1p60f));
            const double ad = double(a), aa = fabs(ad);
            asm volatile("mma.sync.aligned.m8n8k4.row.col.rn.f64.f64.f64.f64 "
                         "{%0, %1}, {%2}, {%3}, {%0, %1};"
                         : "+d"(sums[row_tile][0]), "+d"(sums[row_tile][1])
                         : "d"(ad), "d"(bd));
            asm volatile("mma.sync.aligned.m8n8k4.row.col.rp.f64.f64.f64.f64 "
                         "{%0, %1}, {%2}, {%3}, {%0, %1};"
                         : "+d"(absolutes[row_tile][0]), "+d"(absolutes[row_tile][1])
                         : "d"(aa), "d"(ab));
        }
    }
#else
#pragma unroll
    for (int row_tile = 0; row_tile < RowTiles; ++row_tile)
        unsafe[row_tile] = true;
#endif
#pragma unroll
    for (int row_tile = 0; row_tile < RowTiles; ++row_tile) {
        const int tile_first_row = first_row + row_tile * 8;
        const bool unsafe_warp = __any_sync(0xffffffff, unsafe[row_tile]);
        bool failures[2];
#pragma unroll
        for (int slot = 0; slot < 2; ++slot) {
            const int row = tile_first_row + (lane >> 2);
            const int column = first_column + (lane & 3) * 2 + slot;
            const double sum = sums[row_tile][slot];
            const double absolute = absolutes[row_tile][slot];
            bool accepted = false;
            if (!unsafe_warp && row < rows && column < columns && isfinite(absolute) &&
                absolute > 0. && absolute <= double(FLT_MAX) * 0x1p-6 &&
                isfinite(radius_coefficient) && radius_coefficient > 0.) {
                const double radius = __dmul_ru(radius_coefficient, absolute);
                const __nv_bfloat16 low =
                    __float2bfloat16_rn(__double2float_rn(__dsub_rd(sum, radius)));
                const __nv_bfloat16 high =
                    __float2bfloat16_rn(__double2float_rn(__dadd_ru(sum, radius)));
                accepted = __bfloat16_as_ushort(low) == __bfloat16_as_ushort(high);
                if (accepted)
                    output[row * columns + column] = low;
            }
            failures[slot] = row < rows && column < columns && !accepted;
        }
        const unsigned mask0 = __ballot_sync(0xffffffff, failures[0]);
        const unsigned mask1 = __ballot_sync(0xffffffff, failures[1]);

        for (int slot = 0; slot < 64; ++slot) {
            const unsigned mask = (slot & 1) ? mask1 : mask0;
            if (!(mask & (1u << (slot >> 1))))
                continue;
            const int row = tile_first_row + slot / 8;
            const int column = first_column + slot % 8;
            float running = 0.f, correction = 0.f;
            bool exceptional = false;
            for (int index = lane; index < inner; index += 32) {
                const float a = float(input[row * inner + index]);
                const float b = float(weight[column * inner + index]);
                const float product = __fmul_rn(a, b);
                const float updated = __fadd_rn(running, product);
                const float virtual_product = __fsub_rn(updated, running);
                const float error =
                    __fadd_rn(__fsub_rn(running, __fsub_rn(updated, virtual_product)),
                              __fsub_rn(product, virtual_product));
                correction = __fadd_rn(correction, error);
                running = updated;
                exceptional |= !isfinite(product) || !isfinite(running) || !isfinite(correction) ||
                               (a != 0.f && b != 0.f && fabsf(product) < FLT_MIN);
            }
            double sum = __dadd_rn(double(running), double(correction));
            if (__any_sync(0xffffffff, exceptional)) {
                sum = 0.;
                for (int index = lane; index < inner; index += 32) {
                    sum = __fma_rn(double(float(input[row * inner + index])),
                                   double(float(weight[column * inner + index])), sum);
                }
            }
            for (int step = 16; step > 0; step >>= 1)
                sum = __dadd_rn(sum, __shfl_down_sync(0xffffffff, sum, step));
            if (lane == 0)
                output[row * columns + column] = __float2bfloat16_rn(__double2float_rn(sum));
        }
    }
}

extern "C" __global__ void qwen35_guarded_tensor_double_projection_bf16(
    const __nv_bfloat16 *input, const __nv_bfloat16 *weight, __nv_bfloat16 *output, int rows,
    int inner, int columns, double radius_coefficient) {
    const int lane = threadIdx.x & 31;
    const int first_row = blockIdx.x * 8;
    const int first_column = blockIdx.y * 32 + (threadIdx.x >> 5) * 8;
    double sum0 = 0., sum1 = 0., absolute0 = 0., absolute1 = 0.;
    bool unsafe = false;
#if __CUDA_ARCH__ == 800 || __CUDA_ARCH__ == 900
    for (int first_k = 0; first_k < inner; first_k += 4) {
        const int row = first_row + (lane >> 2);
        const int column = first_column + (lane >> 2);
        const int k = first_k + (lane & 3);
        const float a = row < rows && k < inner ? float(input[row * inner + k]) : 0.f;
        const float b = column < columns && k < inner ? float(weight[column * inner + k]) : 0.f;

        unsafe |= !isfinite(a) || !isfinite(b) ||
                  (a != 0.f && (fabsf(a) < 0x1p-60f || fabsf(a) > 0x1p60f)) ||
                  (b != 0.f && (fabsf(b) < 0x1p-60f || fabsf(b) > 0x1p60f));
        const double ad = double(a), bd = double(b);
        const double aa = fabs(ad), ab = fabs(bd);
        asm volatile("mma.sync.aligned.m8n8k4.row.col.rn.f64.f64.f64.f64 "
                     "{%0, %1}, {%2}, {%3}, {%0, %1};"
                     : "+d"(sum0), "+d"(sum1)
                     : "d"(ad), "d"(bd));
        asm volatile("mma.sync.aligned.m8n8k4.row.col.rp.f64.f64.f64.f64 "
                     "{%0, %1}, {%2}, {%3}, {%0, %1};"
                     : "+d"(absolute0), "+d"(absolute1)
                     : "d"(aa), "d"(ab));
    }
#else
    unsafe = true;
#endif
    const bool unsafe_warp = __any_sync(0xffffffff, unsafe);
    bool failures[2];
    for (int slot = 0; slot < 2; ++slot) {
        const int row = first_row + (lane >> 2);
        const int column = first_column + (lane & 3) * 2 + slot;
        const double sum = slot == 0 ? sum0 : sum1;
        const double absolute = slot == 0 ? absolute0 : absolute1;
        bool accepted = false;
        if (!unsafe_warp && row < rows && column < columns && isfinite(absolute) && absolute > 0. &&
            absolute <= double(FLT_MAX) * 0x1p-6 && isfinite(radius_coefficient) &&
            radius_coefficient > 0.) {
            const double radius = __dmul_ru(radius_coefficient, absolute);
            const __nv_bfloat16 low =
                __float2bfloat16_rn(__double2float_rn(__dsub_rd(sum, radius)));
            const __nv_bfloat16 high =
                __float2bfloat16_rn(__double2float_rn(__dadd_ru(sum, radius)));
            accepted = __bfloat16_as_ushort(low) == __bfloat16_as_ushort(high);
            if (accepted)
                output[row * columns + column] = low;
        }
        failures[slot] = row < rows && column < columns && !accepted;
    }
    const unsigned mask0 = __ballot_sync(0xffffffff, failures[0]);
    const unsigned mask1 = __ballot_sync(0xffffffff, failures[1]);

    for (int slot = 0; slot < 64; ++slot) {
        const unsigned mask = (slot & 1) ? mask1 : mask0;
        if (!(mask & (1u << (slot >> 1))))
            continue;
        const int row = first_row + slot / 8;
        const int column = first_column + slot % 8;
        float running = 0.f, correction = 0.f;
        bool exceptional = false;
        for (int index = lane; index < inner; index += 32) {
            const float a = float(input[row * inner + index]);
            const float b = float(weight[column * inner + index]);
            const float product = __fmul_rn(a, b);
            const float updated = __fadd_rn(running, product);
            const float virtual_product = __fsub_rn(updated, running);
            const float error = __fadd_rn(__fsub_rn(running, __fsub_rn(updated, virtual_product)),
                                          __fsub_rn(product, virtual_product));
            correction = __fadd_rn(correction, error);
            running = updated;
            exceptional |= !isfinite(product) || !isfinite(running) || !isfinite(correction) ||
                           (a != 0.f && b != 0.f && fabsf(product) < FLT_MIN);
        }
        double sum = __dadd_rn(double(running), double(correction));
        if (__any_sync(0xffffffff, exceptional)) {
            sum = 0.;
            for (int index = lane; index < inner; index += 32) {
                sum = __fma_rn(double(float(input[row * inner + index])),
                               double(float(weight[column * inner + index])), sum);
            }
        }
        for (int step = 16; step > 0; step >>= 1)
            sum = __dadd_rn(sum, __shfl_down_sync(0xffffffff, sum, step));
        if (lane == 0)
            output[row * columns + column] = __float2bfloat16_rn(__double2float_rn(sum));
    }
}

extern "C" __global__ void qwen35_guarded_tensor_double_rows16_projection_bf16(
    const __nv_bfloat16 *input, const __nv_bfloat16 *weight, __nv_bfloat16 *output, int rows,
    int inner, int columns, double radius_coefficient) {
    qwen35_guarded_tensor_double_projection_impl_bf16<2>(input, weight, output, rows, inner,
                                                         columns, radius_coefficient);
}

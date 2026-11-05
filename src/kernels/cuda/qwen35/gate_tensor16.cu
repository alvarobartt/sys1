template <int TileRows>
__device__ __forceinline__ void
qwen35_gate_tensor16_impl(const __nv_bfloat16 *input, const __nv_bfloat16 *weight,
                        __nv_bfloat16 *output, int rows, int inner, int columns,
                        const __nv_bfloat16 *bias = nullptr, const unsigned *tile_starts = nullptr,
                        const unsigned *tile_ends = nullptr) {
    const int first_row = tile_starts ? int(tile_starts[blockIdx.x]) : int(blockIdx.x) * TileRows;
    if (tile_ends)
        rows = int(tile_ends[blockIdx.x]);
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
        for (int offset = 0; offset < 64 && first_k + offset < inner; offset += 16) {
#pragma unroll
            for (int group = 0; group < Groups; ++group) {
                wmma::fill_fragment(c[group], 0.f);
            }
            wmma::load_matrix_sync(b, tile_b + warp_column * 16 * SharedStride + offset,
                                   SharedStride);
#pragma unroll
            for (int group = 0; group < Groups; ++group) {
                wmma::load_matrix_sync(
                    a, tile_a + (group * WarpRows + warp_row * 16) * SharedStride + offset,
                    SharedStride);
                wmma::mma_sync(c[group], a, b, c[group]);
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


extern "C" __global__ void qwen35_gate_tensor16_short_bf16(const __nv_bfloat16 *input,
                                                         const __nv_bfloat16 *weight,
                                                         __nv_bfloat16 *output, int rows, int inner,
                                                         int columns, const unsigned *row_ids) {
    constexpr int RowTile = 4, ColumnTile = 2;
    const int lane = threadIdx.x & 31;
    const int first_row = blockIdx.x * RowTile;
    const int first_column = blockIdx.y * 8 + (threadIdx.x >> 5) * ColumnTile;
    if (first_column >= columns)
        return;
    float running[RowTile][ColumnTile] = {}, correction[RowTile][ColumnTile] = {};
    unsigned exceptional = 0;
    for (int index = lane; index < inner; index += 32) {
        float a[RowTile], b[ColumnTile];
#pragma unroll
        for (int r = 0; r < RowTile; ++r)
            a[r] =
                first_row + r < rows ? float(input[row_ids[first_row + r] * inner + index]) : 0.f;
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
            const int row = first_row + r < rows ? int(row_ids[first_row + r]) : 0,
                      column = first_column + c;
            if (first_row + r >= rows || column >= columns)
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
                output[row * columns + column] = __float2bfloat16_rn(value);
            }
        }
    }
}

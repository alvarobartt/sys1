#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <float.h>
#include <math.h>
template <int RowTiles>
__device__ void biased_projection_guard_impl(const __nv_bfloat16 *input,
    const __nv_bfloat16 *weight, const __nv_bfloat16 *bias, __nv_bfloat16 *output, int rows,
    int inner, int columns, double radius_coefficient) {
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
                isfinite(radius_coefficient) && radius_coefficient > 0. &&
                isfinite(float(bias[column]))) {
                const double radius = __dmul_ru(radius_coefficient, absolute);
                const __nv_bfloat16 low = __float2bfloat16_rn(
                    __fadd_rn(__double2float_rn(__dsub_rd(sum, radius)), float(bias[column])));
                const __nv_bfloat16 high = __float2bfloat16_rn(
                    __fadd_rn(__double2float_rn(__dadd_ru(sum, radius)), float(bias[column])));
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
                output[row * columns + column] =
                    __float2bfloat16_rn(__fadd_rn(__double2float_rn(sum), float(bias[column])));
        }
    }
}

extern "C" __global__ void biased_projection_guard_bf16(const __nv_bfloat16 *x,
    const __nv_bfloat16 *w, const __nv_bfloat16 *b, __nv_bfloat16 *out, int rows, int inner,
    int columns, double radius) {
    biased_projection_guard_impl<2>(x, w, b, out, rows, inner, columns, radius);
}

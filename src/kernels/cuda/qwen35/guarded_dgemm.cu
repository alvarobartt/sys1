#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <float.h>
#include <math.h>
extern "C" __global__ void qwen35_guarded_dgemm_weight(const __nv_bfloat16 *w, double *wide,
                                                     double *absolute, unsigned *safe, int k,
                                                     int n) {
    const int column = blockIdx.x * blockDim.x + threadIdx.x;
    if (column >= n)
        return;
    double sum = 0.;
    unsigned ok = 1;
    for (int i = 0; i < k; ++i) {
        const float value = float(w[column * k + i]);
        wide[column * k + i] = double(value);
        ok &= isfinite(value) &&
              (value == 0.f || (fabsf(value) >= 0x1p-60f && fabsf(value) <= 0x1p60f));
        sum = __dadd_ru(sum, fabs(double(value)));
    }
    absolute[column] = sum;
    safe[column] = ok;
}

extern "C" __global__ void qwen35_guarded_dgemm_input(const __nv_bfloat16 *x, double *wide,
                                                    double *maximum, unsigned *safe, int rows,
                                                    int k) {
    const int row = blockIdx.x, thread = threadIdx.x, lane = thread & 31, warp = thread >> 5;

    __shared__ double warp_maxima[8];
    __shared__ unsigned warp_flags[8];
    double m = 0.;
    unsigned ok = 1;
    const auto *input = x + row * k;
    auto *output = wide + row * k;

    if (!(k & 1) && !(reinterpret_cast<unsigned long long>(input) & 3ull) &&
        !(reinterpret_cast<unsigned long long>(output) & 15ull)) {
        for (int i = thread * 2; i < k; i += 512) {
            const unsigned pair = *reinterpret_cast<const unsigned *>(input + i);
            const float first = float(__ushort_as_bfloat16(static_cast<unsigned short>(pair)));
            const float second =
                float(__ushort_as_bfloat16(static_cast<unsigned short>(pair >> 16)));
            *reinterpret_cast<double2 *>(output + i) = make_double2(double(first), double(second));
            ok &= isfinite(first) &&
                  (first == 0.f || (fabsf(first) >= 0x1p-60f && fabsf(first) <= 0x1p60f));
            ok &= isfinite(second) &&
                  (second == 0.f || (fabsf(second) >= 0x1p-60f && fabsf(second) <= 0x1p60f));
            m = fmax(m, fabs(double(first)));
            m = fmax(m, fabs(double(second)));
        }
    } else {

        for (int i = thread; i < k; i += 256) {
            const float value = float(input[i]);
            output[i] = double(value);
            ok &= isfinite(value) &&
                  (value == 0.f || (fabsf(value) >= 0x1p-60f && fabsf(value) <= 0x1p60f));
            m = fmax(m, fabs(double(value)));
        }
    }

    for (int step = 16; step; step >>= 1) {
        m = fmax(m, __shfl_down_sync(0xffffffff, m, step));
        ok &= __shfl_down_sync(0xffffffff, ok, step);
    }
    if (!lane) {
        warp_maxima[warp] = m;
        warp_flags[warp] = ok;
    }
    __syncthreads();
    if (!warp) {
        m = lane < 8 ? warp_maxima[lane] : 0.;
        ok = lane < 8 ? warp_flags[lane] : 1;
        for (int step = 16; step; step >>= 1) {
            m = fmax(m, __shfl_down_sync(0xffffffff, m, step));
            ok &= __shfl_down_sync(0xffffffff, ok, step);
        }
        if (!lane) {
            maximum[row] = m;
            safe[row] = ok;
        }
    }
}

__device__ __forceinline__ __nv_bfloat16 qwen35_guarded_dgemm_round(double value,
                                                                  const __nv_bfloat16 *bias,
                                                                  int column, int has_bias) {
    float rounded = __double2float_rn(value);
    if (has_bias)
        rounded = __fadd_rn(rounded, float(bias[column]));
    return __float2bfloat16_rn(rounded);
}

__device__ void qwen35_guarded_dgemm_fallback(const __nv_bfloat16 *x, const __nv_bfloat16 *w,
                                            const __nv_bfloat16 *bias, __nv_bfloat16 *y, int row,
                                            int column, int inner, int columns, int has_bias) {
    const int lane = threadIdx.x & 31;
    float running = 0.f, correction = 0.f;
    bool exceptional = false;
    for (int index = lane; index < inner; index += 32) {
        const float a = float(x[row * inner + index]), b = float(w[column * inner + index]);
        const float product = __fmul_rn(a, b), updated = __fadd_rn(running, product);
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
        for (int i = lane; i < inner; i += 32)
            sum = __fma_rn(double(float(x[row * inner + i])), double(float(w[column * inner + i])),
                           sum);
    }
    for (int step = 16; step; step >>= 1)
        sum = __dadd_rn(sum, __shfl_down_sync(0xffffffff, sum, step));
    if (!lane)
        y[row * columns + column] = qwen35_guarded_dgemm_round(sum, bias, column, has_bias);
}

extern "C" __global__ void qwen35_guarded_dgemm_finish(const __nv_bfloat16 *x, const __nv_bfloat16 *w,
                                                     const __nv_bfloat16 *bias, const double *dot,
                                                     const double *maximum, const double *absolute,
                                                     const unsigned *safe_x, const unsigned *safe_w,
                                                     __nv_bfloat16 *y, int rows, int inner,
                                                     int columns, int has_bias, double coefficient,
                                                     unsigned *counters) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = i / columns, column = i % columns;
    bool accepted = false;
    if (row < rows && safe_x[row] && safe_w[column] &&
        (!has_bias || isfinite(float(bias[column])))) {

        const double bound = __dmul_ru(maximum[row], absolute[column]);
        if (isfinite(bound) && bound > 0. && bound <= double(FLT_MAX) * 0x1p-6) {
            const double radius = __dmul_ru(coefficient, bound), center = dot[i];
            const auto low =
                qwen35_guarded_dgemm_round(__dsub_rd(center, radius), bias, column, has_bias);
            const auto high =
                qwen35_guarded_dgemm_round(__dadd_ru(center, radius), bias, column, has_bias);
            accepted = __bfloat16_as_ushort(low) == __bfloat16_as_ushort(high);
            if (accepted)
                y[i] = low;
        }
    }
    if (counters && row < rows)
        atomicAdd(counters + (accepted ? 0 : 1), 1u);
    unsigned mask = __ballot_sync(0xffffffff, row < rows && !accepted);
    while (mask) {
        const int lane = __ffs(mask) - 1;
        const unsigned id = __shfl_sync(0xffffffff, i, lane);
        qwen35_guarded_dgemm_fallback(x, w, bias, y, id / columns, id % columns, inner, columns,
                                    has_bias);
        mask &= mask - 1;
    }
}

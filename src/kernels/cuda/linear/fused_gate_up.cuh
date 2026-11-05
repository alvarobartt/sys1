#include <cutlass/cutlass.h>
#include <cutlass/gemm/kernel/default_gemm.h>
#include <cutlass/epilogue/threadblock/predicated_tile_iterator.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
using E = cutlass::bfloat16_t;
using Column = cutlass::layout::ColumnMajor;
using Row = cutlass::layout::RowMajor;
using Swizzle = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>;

struct WideSwizzle : Swizzle {
    CUTLASS_HOST_DEVICE static int get_log_tile(cutlass::gemm::GemmCoord tiled) {
        return tiled.n() >= 32 ? 5 : ::Swizzle::get_log_tile(tiled);
    }
    CUTLASS_HOST_DEVICE static dim3 get_grid_shape(cutlass::gemm::GemmCoord tiled) {
        const int width = 1 << get_log_tile(tiled);
        return dim3(tiled.m() * width, (tiled.n() + width - 1) / width, tiled.k());
    }
};
using Shape = cutlass::gemm::GemmShape<128, 256, 32>;
using Warp = cutlass::gemm::GemmShape<64, 64, 32>;
using Op = cutlass::epilogue::thread::LinearCombination<E, 8, float, float>;
using Default = cutlass::gemm::kernel::DefaultGemm<
    E, Row, 8, E, Row, 8, E, Row, float, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80, Shape,
    Warp, cutlass::gemm::GemmShape<16, 8, 16>, Op, Swizzle, 3, false, cutlass::arch::OpMultiplyAdd>;
using Mma = Default::Mma;
using ProductDefault = cutlass::gemm::kernel::DefaultGemm<
    E, Row, 8, E, Row, 8, E, Row, float, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 32>, cutlass::gemm::GemmShape<64, 32, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>, Op, Swizzle, 3, false, cutlass::arch::OpMultiplyAdd>;
using ProductEpi = typename ProductDefault::Epilogue;
constexpr int Threads = Mma::WarpCount::kCount * 32;
union alignas(16) Shared {
    typename Mma::SharedStorage mma;
    typename ProductEpi::SharedStorage end;
};

template <bool Debug>
__device__ __forceinline__ void
fused_gate_up_body(const E *packed, const E *x, const __nv_bfloat16 *lookup, __nv_bfloat16 *output,
                   __nv_bfloat16 *debug_gate, __nv_bfloat16 *debug_up, int rows, int n, int k) {
    extern __shared__ __align__(16) unsigned char buffer[];
    auto &shared = *reinterpret_cast<Shared *>(buffer);
    const cutlass::gemm::GemmCoord tiled((rows + 127) / 128, n / 128, 1);
    const int log_tile =
        rows >= 16384 ? WideSwizzle::get_log_tile(tiled) : Swizzle::get_log_tile(tiled);
    const auto tile = Swizzle().get_tile_offset(log_tile);
    if (tile.m() >= tiled.m() || tile.n() >= tiled.n())
        return;
    const int feature_base = tile.n() * 128, token_base = tile.m() * 128;
    typename Mma::IteratorA::Params pa{Row(k)};
    typename Mma::IteratorB::Params pb{Row(2 * n)};

    typename Mma::IteratorA a(pa, const_cast<E *>(x), {rows, k}, threadIdx.x, {token_base, 0});
    typename Mma::IteratorB b(pb, const_cast<E *>(packed), {k, 2 * n}, threadIdx.x,
                              {0, tile.n() * 256});
    const int warp = cutlass::canonical_warp_idx_sync(), lane = threadIdx.x % 32;
    Mma mma(shared.mma, threadIdx.x, warp, lane);
    typename Mma::FragmentC accum;
    accum.clear();
    mma(k / 32, accum, a, b, accum);
    __syncthreads();
    using ProductFragment = typename ProductDefault::Mma::FragmentC;
    static_assert(Mma::FragmentC::kElements == 128 && ProductFragment::kElements == 64,
                  "paired MMA ownership");
    ProductFragment products;
    const int warp_m = warp % Mma::WarpCount::kM, warp_n = warp / Mma::WarpCount::kM;
#pragma unroll
    for (int i = 0; i < 64; i++) {
        const auto gate = __float2bfloat16_rn(accum[i]);
        const auto up = __float2bfloat16_rn(accum[i + 64]);
        const auto product = __float2bfloat16_rn(
            __fmul_rn(__bfloat162float(lookup[__bfloat16_as_ushort(gate)]), __bfloat162float(up)));
        products[i] = __bfloat162float(product);
        if constexpr (Debug) {
            const int r = warp_m * 64 + ((i / 4) % 4) * 16 + lane / 4 + ((i % 4) / 2) * 8;
            const int c = warp_n * 32 + (i / 16) * 8 + (lane % 4) * 2 + i % 2;
            if (token_base + r < rows) {
                const size_t pos = size_t(token_base + r) * n + feature_base + c;
                debug_gate[pos] = gate;
                debug_up[pos] = up;
            }
        }
    }
    using Destination = typename ProductEpi::OutputTileIterator;
    typename Destination::Params params{Row(n)};
    Destination destination(params, reinterpret_cast<E *>(output), {rows, n}, threadIdx.x,
                            {token_base, feature_base});
    ProductEpi epi(shared.end, threadIdx.x, warp, lane);
    epi(Op(typename Op::Params(1.f, 0.f)), destination, products);
}

static_assert(sizeof(Shared) == 73728, "Rust dynamic shared memory must match native layout");
extern "C" __global__
__launch_bounds__(Threads) void fused_ffn_bf16(const E *weight, const E *input,
                                                    const __nv_bfloat16 *table,
                                                    __nv_bfloat16 *output, int rows, int n, int k) {
    fused_gate_up_body<false>(weight, input, table, output, nullptr, nullptr, rows, n, k);
}

extern "C" __global__ __launch_bounds__(Threads) void fused_ffn_debug_bf16(
    const E *weight, const E *input, const __nv_bfloat16 *table, __nv_bfloat16 *output,
    __nv_bfloat16 *gate, __nv_bfloat16 *up, int rows, int n, int k) {
    fused_gate_up_body<true>(weight, input, table, output, gate, up, rows, n, k);
}

#include "fused_qkvz.cuh"

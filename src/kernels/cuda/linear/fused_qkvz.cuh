namespace dual_qkvz {
using Tile = cutlass::gemm::GemmShape<128, 256, 32>;
using Default = cutlass::gemm::kernel::DefaultGemm<E, Row, 8, E, Row, 8, E, Row, float,
    cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80, Tile, cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>, Op, Swizzle, 3, false, cutlass::arch::OpMultiplyAdd>;
using Mma = typename Default::Mma;
using Epi = typename Default::Epilogue;
union alignas(16) Shared {
    typename Mma::SharedStorage mma;
    typename Epi::SharedStorage epi;
};
constexpr int Threads = Mma::WarpCount::kCount * 32;

__device__ __forceinline__ void project(const E *input, const E *packed, E *qkv, E *z, int rows,
    int qcolumns, int zcolumns, int inner) {
    extern __shared__ __align__(16) unsigned char bytes[];
    auto &shared = *reinterpret_cast<Shared *>(bytes);
    const int columns = qcolumns + zcolumns;
    const auto tile = Swizzle().get_tile_offset(rows >= 16384 ? 5 : 3);
    if (tile.m() >= (rows + 127) / 128 || tile.n() >= (columns + 255) / 256)
        return;
    const int feature = tile.n() * 256, token = tile.m() * 128;
    typename Mma::IteratorA::Params pa{Row(inner)};
    typename Mma::IteratorB::Params pb{Row(columns)};
    typename Mma::IteratorA a(pa, const_cast<E *>(input), {rows, inner}, threadIdx.x, {token, 0});
    typename Mma::IteratorB b(
        pb, const_cast<E *>(packed), {inner, columns}, threadIdx.x, {0, feature});
    const int warp = cutlass::canonical_warp_idx_sync(), lane = threadIdx.x % 32;
    Mma mma(shared.mma, threadIdx.x, warp, lane);
    typename Mma::FragmentC accum;
    accum.clear();
    mma(inner / 32, accum, a, b, accum);
    __syncthreads();

    const bool is_qkv = feature < qcolumns;
    const int width = is_qkv ? qcolumns : zcolumns;
    const int output_feature = is_qkv ? feature : feature - qcolumns;
    using Destination = typename Epi::OutputTileIterator;
    typename Destination::Params params{Row(width)};
    Destination output(
        params, is_qkv ? qkv : z, {rows, width}, threadIdx.x, {token, output_feature});
    Epi epilogue(shared.epi, threadIdx.x, warp, lane);
    epilogue(Op(typename Op::Params(1.f, 0.f)), output, accum);
}

}

namespace dual_qkvz_narrow {
using Tile = cutlass::gemm::GemmShape<128, 128, 32>;
using Default = cutlass::gemm::kernel::DefaultGemm<E, Row, 8, E, Row, 8, E, Row, float,
    cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80, Tile, cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>, Op, Swizzle, 3, false, cutlass::arch::OpMultiplyAdd>;
using Mma = typename Default::Mma;
using Epi = typename Default::Epilogue;
union alignas(16) Shared {
    typename Mma::SharedStorage mma;
    typename Epi::SharedStorage epi;
};
constexpr int Threads = Mma::WarpCount::kCount * 32;

__device__ __forceinline__ void project(const E *input, const E *packed, E *qkv, E *z, int rows,
    int qcolumns, int zcolumns, int inner) {
    extern __shared__ __align__(16) unsigned char bytes[];
    auto &shared = *reinterpret_cast<Shared *>(bytes);
    const int columns = qcolumns + zcolumns;
    const auto tile = Swizzle().get_tile_offset(rows >= 16384 ? 5 : 3);
    if (tile.m() >= (rows + 127) / 128 || tile.n() >= (columns + 127) / 128)
        return;
    const int feature = tile.n() * 128, token = tile.m() * 128;
    typename Mma::IteratorA::Params pa{Row(inner)};
    typename Mma::IteratorB::Params pb{Row(columns)};
    typename Mma::IteratorA a(pa, const_cast<E *>(input), {rows, inner}, threadIdx.x, {token, 0});
    typename Mma::IteratorB b(
        pb, const_cast<E *>(packed), {inner, columns}, threadIdx.x, {0, feature});
    const int warp = cutlass::canonical_warp_idx_sync(), lane = threadIdx.x % 32;
    Mma mma(shared.mma, threadIdx.x, warp, lane);
    typename Mma::FragmentC accum;
    accum.clear();
    mma(inner / 32, accum, a, b, accum);
    __syncthreads();

    const bool is_qkv = feature < qcolumns;
    const int width = is_qkv ? qcolumns : zcolumns;
    const int output_feature = is_qkv ? feature : feature - qcolumns;
    using Destination = typename Epi::OutputTileIterator;
    typename Destination::Params params{Row(width)};
    Destination output(
        params, is_qkv ? qkv : z, {rows, width}, threadIdx.x, {token, output_feature});
    Epi epilogue(shared.epi, threadIdx.x, warp, lane);
    epilogue(Op(typename Op::Params(1.f, 0.f)), output, accum);
}

}

extern "C" __global__ __launch_bounds__(256) void qkvz_wide_bf16(const __nv_bfloat16 *x,
    const __nv_bfloat16 *packed, __nv_bfloat16 *qkv, __nv_bfloat16 *z, int rows) {
    dual_qkvz::project(reinterpret_cast<const E *>(x), reinterpret_cast<const E *>(packed),
        reinterpret_cast<E *>(qkv), reinterpret_cast<E *>(z), rows, 8192, 4096, 4096);
}
extern "C" __global__ __launch_bounds__(128) void qkvz_narrow_bf16(const __nv_bfloat16 *x,
    const __nv_bfloat16 *packed, __nv_bfloat16 *qkv, __nv_bfloat16 *z, int rows) {
    dual_qkvz_narrow::project(reinterpret_cast<const E *>(x), reinterpret_cast<const E *>(packed),
        reinterpret_cast<E *>(qkv), reinterpret_cast<E *>(z), rows, 8192, 4096, 4096);
}
extern "C" __global__ void qkvz_pack_bf16(
    const __nv_bfloat16 *qkv, const __nv_bfloat16 *z, __nv_bfloat16 *packed) {
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < size_t(4096) * 12288) {
        const int k = i / 12288, c = i % 12288;
        packed[i] = c < 8192 ? qkv[size_t(c) * 4096 + k] : z[size_t(c - 8192) * 4096 + k];
    }
}

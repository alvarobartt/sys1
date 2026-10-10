#include <metal_stdlib>
using namespace metal;

inline float bf16_to_f32(ushort bits) { return as_type<float>(uint(bits) << 16); }

inline ushort f32_to_bf16(float value) {
    uint bits = as_type<uint>(value);
    // Round to nearest, ties to even, matching BF16 tensor operations.
    return ushort((bits + 0x7fffu + ((bits >> 16) & 1u)) >> 16);
}

template <typename T> inline float read_value(T value) { return float(value); }
template <> inline float read_value<ushort>(ushort value) { return bf16_to_f32(value); }
template <typename T> inline T write_value(float value) { return T(value); }
template <> inline ushort write_value<ushort>(float value) { return f32_to_bf16(value); }

// Abramowitz & Stegun 7.1.26, as in candle's `unary.metal`, so `gelu_erf` matches.
inline float erf_approximation(float x) {
    constexpr const float a1 = 0.254829592f;
    constexpr const float a2 = -0.284496736f;
    constexpr const float a3 = 1.421413741f;
    constexpr const float a4 = -1.453152027f;
    constexpr const float a5 = 1.061405429f;
    constexpr const float p = 0.3275911f;

    int sign = 1;
    if (x < 0) sign = -1;
    x = fabs(x);

    float t = 1.0f / (1.0f + p * x);
    float y = 1.0f - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * exp(-x * x);
    return float(sign) * y;
}

inline float gelu_erf(float x) { return x * (1.0f + erf_approximation(x * M_SQRT1_2_F)) / 2.0f; }

template <typename T>
void geglu(device const T* input, device T* output, constant uint& count,
           constant uint& inner, uint index) {
    if (index >= count) return;
    uint row = index / inner;
    uint column = index - row * inner;
    uint offset = row * 2u * inner + column;
    T activated = write_value<T>(gelu_erf(read_value(input[offset])));
    output[index] = write_value<T>(read_value(activated) * read_value(input[offset + inner]));
}

#define GEGLU_KERNEL(NAME, TYPE) \
kernel void NAME(device const TYPE* input [[buffer(0)]], \
                 device TYPE* output [[buffer(1)]], \
                 constant uint& count [[buffer(2)]], \
                 constant uint& inner [[buffer(3)]], \
                 uint index [[thread_position_in_grid]]) { \
    geglu(input, output, count, inner, index); \
}
GEGLU_KERNEL(modernbert_geglu_f32, float)
GEGLU_KERNEL(modernbert_geglu_f16, half)
GEGLU_KERNEL(modernbert_geglu_bf16, ushort)

template <typename T>
void rope_qkv(device const T* input, device const float* cos, device const float* sin,
              device T* output, constant uint& pairs, constant uint& length,
              constant uint& heads, constant uint& dim, uint index) {
    if (index >= pairs) return;
    uint half_dim = dim / 2u;
    uint column = index % half_dim;
    uint head = (index / half_dim) % heads;
    uint token = index / (heads * half_dim);
    uint batch = token / length;
    uint position = token - batch * length;

    uint stride = heads * dim;
    uint input_base = token * 3u * stride + head * dim + column;
    uint plane = pairs * 2u;
    uint output_base =
        batch * stride * length + head * length * dim + position * dim + column;

    float c = cos[position * half_dim + column];
    float s = sin[position * half_dim + column];

    float q0 = read_value(input[input_base]);
    float q1 = read_value(input[input_base + half_dim]);
    output[output_base] = write_value<T>(q0 * c - q1 * s);
    output[output_base + half_dim] = write_value<T>(q0 * s + q1 * c);

    uint key_base = input_base + stride;
    float k0 = read_value(input[key_base]);
    float k1 = read_value(input[key_base + half_dim]);
    output[plane + output_base] = write_value<T>(k0 * c - k1 * s);
    output[plane + output_base + half_dim] = write_value<T>(k0 * s + k1 * c);

    uint value_base = key_base + stride;
    output[2u * plane + output_base] = input[value_base];
    output[2u * plane + output_base + half_dim] = input[value_base + half_dim];
}

#define ROPE_QKV_KERNEL(NAME, TYPE) \
kernel void NAME(device const TYPE* input [[buffer(0)]], \
                 device const float* cos [[buffer(1)]], \
                 device const float* sin [[buffer(2)]], \
                 device TYPE* output [[buffer(3)]], \
                 constant uint& pairs [[buffer(4)]], \
                 constant uint& length [[buffer(5)]], \
                 constant uint& heads [[buffer(6)]], \
                 constant uint& dim [[buffer(7)]], \
                 uint index [[thread_position_in_grid]]) { \
    rope_qkv(input, cos, sin, output, pairs, length, heads, dim, index); \
}
ROPE_QKV_KERNEL(modernbert_rope_qkv_f32, float)
ROPE_QKV_KERNEL(modernbert_rope_qkv_f16, half)
ROPE_QKV_KERNEL(modernbert_rope_qkv_bf16, ushort)

template <typename T>
void merge_heads(device const T* input, device T* output, constant uint& heads,
                 constant uint& length, constant uint& dim, uint3 position) {
    if (position.x >= dim || position.y >= length) return;
    uint head = position.z % heads;
    uint batch = position.z / heads;
    uint source = (position.z * length + position.y) * dim + position.x;
    uint destination =
        ((batch * length + position.y) * heads + head) * dim + position.x;
    output[destination] = input[source];
}

#define MERGE_HEADS_KERNEL(NAME, TYPE) \
kernel void NAME(device const TYPE* input [[buffer(0)]], \
                 device TYPE* output [[buffer(1)]], \
                 constant uint& heads [[buffer(2)]], \
                 constant uint& length [[buffer(3)]], \
                 constant uint& dim [[buffer(4)]], \
                 uint3 position [[thread_position_in_grid]]) { \
    merge_heads(input, output, heads, length, dim, position); \
}
MERGE_HEADS_KERNEL(modernbert_merge_heads_f32, float)
MERGE_HEADS_KERNEL(modernbert_merge_heads_f16, half)
MERGE_HEADS_KERNEL(modernbert_merge_heads_bf16, ushort)

// Operation order follows candle's bias-free module: a division by `sqrt(variance + eps)`
// rather than a reciprocal multiply, and the normalised value rounded to the tensor dtype
// before the weight.
template <typename T>
void layer_norm(device const T* input, device const T* weight, device T* output,
                constant uint& hidden, constant float& eps, threadgroup float* partials,
                uint row, uint tid, uint lane, uint simd_group) {
    const uint threads = 256u;
    const uint groups = threads / 32u;
    device const T* source = input + row * hidden;
    device T* destination = output + row * hidden;

    float total = 0.f;
    for (uint i = tid; i < hidden; i += threads) total += read_value(source[i]);
    total = simd_sum(total);
    if (lane == 0) partials[simd_group] = total;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float sum = 0.f;
    for (uint i = 0; i < groups; ++i) sum += partials[i];
    float mean = sum / float(hidden);

    float squares = 0.f;
    for (uint i = tid; i < hidden; i += threads) {
        float centered = read_value(source[i]) - mean;
        squares += centered * centered;
    }
    squares = simd_sum(squares);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0) partials[simd_group] = squares;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float variance = 0.f;
    for (uint i = 0; i < groups; ++i) variance += partials[i];
    float denominator = sqrt(variance / float(hidden) + eps);

    for (uint i = tid; i < hidden; i += threads) {
        T normalized = write_value<T>((read_value(source[i]) - mean) / denominator);
        destination[i] = write_value<T>(read_value(normalized) * read_value(weight[i]));
    }
}

#define LAYER_NORM_KERNEL(NAME, TYPE) \
kernel void NAME(device const TYPE* input [[buffer(0)]], \
                 device const TYPE* weight [[buffer(1)]], \
                 device TYPE* output [[buffer(2)]], \
                 constant uint& hidden [[buffer(3)]], \
                 constant float& eps [[buffer(4)]], \
                 uint row [[threadgroup_position_in_grid]], \
                 uint tid [[thread_index_in_threadgroup]], \
                 uint lane [[thread_index_in_simdgroup]], \
                 uint simd_group [[simdgroup_index_in_threadgroup]]) { \
    threadgroup float partials[8]; \
    layer_norm(input, weight, output, hidden, eps, partials, row, tid, lane, simd_group); \
}
LAYER_NORM_KERNEL(modernbert_layer_norm_f32, float)
LAYER_NORM_KERNEL(modernbert_layer_norm_f16, half)
LAYER_NORM_KERNEL(modernbert_layer_norm_bf16, ushort)

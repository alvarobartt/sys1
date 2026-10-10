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

template <typename T>
void conv_silu(device const T* input, device const T* weights, device T* output,
               constant uint& count, constant uint& length, constant uint& channels,
               constant uint& taps, uint index) {
    if (index >= count) return;
    uint channel = index % channels;
    uint token = index / channels % length;
    uint batch = index / (channels * length);
    T acc = write_value<T>(0.f);
    for (uint tap = 0; tap < taps; ++tap) {
        int source = int(token) + int(tap) - int(taps) + 1;
        if (source >= 0) {
            float product = read_value(input[(batch * length + uint(source)) * channels + channel])
                          * read_value(weights[channel * taps + tap]);
            acc = write_value<T>(read_value(acc) + read_value(write_value<T>(product)));
        }
    }
    float x = read_value(acc);
    output[index] = write_value<T>(x / (1.f + exp(-x)));
}

#define CONV_KERNEL(NAME, TYPE) \
kernel void NAME(device const TYPE* input [[buffer(0)]], \
                 device const TYPE* weights [[buffer(1)]], \
                 device TYPE* output [[buffer(2)]], \
                 constant uint& count [[buffer(3)]], \
                 constant uint& length [[buffer(4)]], \
                 constant uint& channels [[buffer(5)]], \
                 constant uint& taps [[buffer(6)]], \
                 uint index [[thread_position_in_grid]]) { \
    conv_silu(input, weights, output, count, length, channels, taps, index); \
}
CONV_KERNEL(qwen35_conv_f32, float)
CONV_KERNEL(qwen35_conv_f16, half)
CONV_KERNEL(qwen35_conv_bf16, ushort)

// A SIMD group owns one value column; each lane keeps four state entries.
kernel void qwen35_delta_f32(device const float* input [[buffer(0)]],
                             device float* output [[buffer(1)]],
                             constant uint& length [[buffer(2)]],
                             constant uint& value_dim [[buffer(3)]],
                             uint2 group [[threadgroup_position_in_grid]],
                             uint tid [[thread_index_in_threadgroup]],
                             uint lane [[thread_index_in_simdgroup]]) {
    uint column = group.y * 4 + tid / 32;
    if (column >= value_dim) return;
    float state[4] = {0.f, 0.f, 0.f, 0.f};
    for (uint token = 0; token < length; ++token) {
        uint offset = group.x * length + token;
        device const float* row = input + offset * (258 + value_dim);
        float decay = exp(row[256 + value_dim]);
        float beta = row[257 + value_dim];
        float key[4];
        float remembered = 0.f;
        for (uint part = 0; part < 4; ++part) {
            key[part] = row[128 + part * 32 + lane];
            state[part] *= decay;
            remembered += key[part] * state[part];
        }
        float delta = (row[256 + column] - simd_sum(remembered)) * beta;
        float answer = 0.f;
        for (uint part = 0; part < 4; ++part) {
            state[part] += key[part] * delta;
            answer += row[part * 32 + lane] * state[part];
        }
        answer = simd_sum(answer);
        if (lane == 0) output[offset * value_dim + column] = answer;
    }
}

kernel void qwen35_pack_delta_f32(device const float* q [[buffer(0)]],
                                  device const float* k [[buffer(1)]],
                                  device const float* v [[buffer(2)]],
                                  device const float* g [[buffer(3)]],
                                  device const float* beta [[buffer(4)]],
                                  device float* packed [[buffer(5)]],
                                  constant uint4& dims [[buffer(6)]],
                                  constant uint4& q_stride [[buffer(7)]],
                                  constant uint4& k_stride [[buffer(8)]],
                                  constant uint4& v_stride [[buffer(9)]],
                                  constant uint4& g_stride [[buffer(10)]],
                                  constant uint4& beta_stride [[buffer(11)]],
                                  uint row [[threadgroup_position_in_grid]],
                                  uint tid [[thread_index_in_threadgroup]]) {
    uint length = dims.z;
    uint heads = dims.y;
    uint token = row % length;
    uint head = row / length % heads;
    uint key_head = head / (heads / dims.w);
    uint batch = row / (length * heads);
    uint q_base = batch * q_stride.x + key_head * q_stride.y + token * q_stride.z;
    uint k_base = batch * k_stride.x + key_head * k_stride.y + token * k_stride.z;
    uint v_base = batch * v_stride.x + head * v_stride.y + token * v_stride.z;
    uint g_base = batch * g_stride.x + head * g_stride.y + token * g_stride.z;
    uint beta_base = batch * beta_stride.x + head * beta_stride.y + token * beta_stride.z;
    device float* dest = packed + row * 386;
    dest[tid] = q[q_base + tid * q_stride.w];
    dest[128 + tid] = k[k_base + tid * k_stride.w];
    dest[256 + tid] = v[v_base + tid * v_stride.w];
    if (tid == 0) {
        dest[384] = g[g_base];
        dest[385] = beta[beta_base];
    }
}

kernel void qwen35_post_finalize_bf16(device const float* input [[buffer(0)]],
                                      device const float* denominator [[buffer(1)]],
                                      device const ushort* gate [[buffer(2)]],
                                      device const float* norm [[buffer(3)]],
                                      device ushort* output [[buffer(4)]],
                                      constant uint4& dims [[buffer(5)]],
                                      constant uint4& strides [[buffer(6)]],
                                      uint index [[thread_position_in_grid]]) {
    uint count = dims.x * dims.y * dims.z * dims.w;
    if (index >= count) return;
    uint channel = index % dims.w;
    uint head = index / dims.w % dims.z;
    uint token = index / (dims.w * dims.z) % dims.y;
    uint batch = index / (dims.w * dims.z * dims.y);
    uint source = batch * strides.x + token * strides.y + head * strides.z + channel * strides.w;
    float raw = bf16_to_f32(gate[index]);
    float gate_silu = raw / (1.f + exp(-raw));
    // Preserve Candle's separate arithmetic steps; reassociation changed BF16 outputs.
    volatile float value = input[source] / denominator[index / dims.w];
    value = value * norm[channel];
    value = value * gate_silu;
    output[index] = f32_to_bf16(value);
}

kernel void qwen35_pack_delta_inputs_bf16(
    device const ushort* mixed [[buffer(0)]],
    device const ushort* beta_projection [[buffer(1)]],
    device const ushort* gate_projection [[buffer(2)]],
    device const float* dt_bias [[buffer(3)]],
    device const float* decay [[buffer(4)]],
    device float* output [[buffer(5)]],
    constant uint4& dims [[buffer(6)]],
    constant uint2& widths [[buffer(7)]],
    constant float& query_scale [[buffer(8)]],
    uint group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
    uint length = dims.y;
    uint value_heads = dims.z;
    uint key_heads = dims.w;
    uint key_width = widths.x;
    uint mixed_width = widths.y;
    const uint dim = 128;

    uint token = group % length;
    uint head = (group / length) % value_heads;
    uint batch = group / (length * value_heads);
    uint key_head = head / (value_heads / key_heads);

    device const ushort* row = mixed + (batch * length + token) * mixed_width;
    float query = bf16_to_f32(row[key_head * dim + tid]);
    float key = bf16_to_f32(row[key_width + key_head * dim + tid]);
    float value = bf16_to_f32(row[2u * key_width + head * dim + tid]);

    threadgroup float query_partials[4];
    threadgroup float key_partials[4];
    float query_sum = simd_sum(query * query);
    float key_sum = simd_sum(key * key);
    if (lane == 0) {
        query_partials[simd_group] = query_sum;
        key_partials[simd_group] = key_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float query_total = query_partials[0] + query_partials[1] + query_partials[2] + query_partials[3];
    float key_total = key_partials[0] + key_partials[1] + key_partials[2] + key_partials[3];
    float query_denominator = sqrt(query_total + 1e-6f);
    float key_denominator = sqrt(key_total + 1e-6f);

    device float* destination =
        output + ((batch * value_heads + head) * length + token) * (258u + dim);
    destination[tid] = (query / query_denominator) * query_scale;
    destination[dim + tid] = key / key_denominator;
    destination[2u * dim + tid] = value;
    if (tid == 0) {
        uint scalar = (batch * length + token) * value_heads + head;
        float gate = bf16_to_f32(gate_projection[scalar]) + dt_bias[head];
        float softplus = (gate < 0.f ? 0.f : gate) + log(exp(-fabs(gate)) + 1.f);
        destination[2u * dim + 128u] = softplus * decay[head];
        destination[2u * dim + 129u] = 1.f / (1.f + exp(-bf16_to_f32(beta_projection[scalar])));
    }
}

template <typename T>
void swiglu(device const T* gate, device const T* up, device T* output,
            constant uint& count, uint index) {
    if (index >= count) return;
    float raw = read_value(gate[index]);
    T activated = write_value<T>(raw / (1.f + exp(-raw)));
    output[index] = write_value<T>(read_value(activated) * read_value(up[index]));
}

#define SWIGLU_KERNEL(NAME, TYPE) \
kernel void NAME(device const TYPE* gate [[buffer(0)]], \
                 device const TYPE* up [[buffer(1)]], \
                 device TYPE* output [[buffer(2)]], \
                 constant uint& count [[buffer(3)]], \
                 uint index [[thread_position_in_grid]]) { \
    swiglu(gate, up, output, count, index); \
}
SWIGLU_KERNEL(qwen35_swiglu_f32, float)
SWIGLU_KERNEL(qwen35_swiglu_f16, half)
SWIGLU_KERNEL(qwen35_swiglu_bf16, ushort)

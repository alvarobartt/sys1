#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math.h>
extern "C" __global__ void qwen35_vstoreshort_0_bf16(const float *query, const float *key,
    const __nv_bfloat16 *value, const float *gate, const float *beta, __nv_bfloat16 *output,
    const unsigned int *cu_seqlens, const unsigned int *short_ids, int total_length, int heads,
    int key_heads, int value_dim) {
    const int batch_head = blockIdx.x;
    const int head = batch_head % heads;
    const int batch = short_ids[batch_head / heads];
    const int begin = cu_seqlens[batch];
    const int length = cu_seqlens[batch + 1] - begin;
    const int lane = threadIdx.x & 15;
    constexpr int Columns = 4;
    const int column = (blockIdx.y * 8 + (threadIdx.x >> 4)) * Columns;
    if (column >= value_dim)
        return;
    const unsigned mask = __activemask();
    float state[Columns][2][4] = {};
    for (int token = 0; token < length; ++token) {
        const int token_head = (begin + token) * heads + head;
        const int qk_head = (begin + token) * key_heads + head / (heads / key_heads);
        const int output_offset = token_head;
        const float decay = expf(gate[token_head]);
        const float weight = beta[token_head];
        const float *token_query = query + qk_head * 128;
        const float *token_key = key + qk_head * 128;
        const __nv_bfloat16 *token_value = value + token_head * value_dim;

        float keys[2][4], queries[2][4];
#pragma unroll
        for (int half = 0; half < 2; ++half) {
#pragma unroll
            for (int part = 0; part < 4; ++part) {
                const int d = part * 32 + lane + half * 16;
                keys[half][part] = token_key[d];
                queries[half][part] = token_query[d];
            }
        }
#pragma unroll
        for (int col = 0; col < Columns; ++col) {
            float remembered[2] = {0.f, 0.f};
#pragma unroll
            for (int half = 0; half < 2; ++half) {
#pragma unroll
                for (int part = 0; part < 4; ++part) {
                    state[col][half][part] = __fmul_rn(state[col][half][part], decay);
                    remembered[half] =
                        __fmaf_rn(keys[half][part], state[col][half][part], remembered[half]);
                }
            }
            float total = __fadd_rn(remembered[0], remembered[1]);
#pragma unroll
            for (int step = 8; step > 0; step >>= 1)
                total = __fadd_rn(total, __shfl_down_sync(mask, total, step, 16));
            const float val =
                column + col < value_dim ? __bfloat162float(token_value[column + col]) : 0.f;
            const float delta = __fmul_rn(__fsub_rn(val, __shfl_sync(mask, total, 0, 16)), weight);
            float answer[2] = {0.f, 0.f};
#pragma unroll
            for (int half = 0; half < 2; ++half) {
#pragma unroll
                for (int part = 0; part < 4; ++part) {
                    state[col][half][part] =
                        __fmaf_rn(keys[half][part], delta, state[col][half][part]);
                    answer[half] =
                        __fmaf_rn(queries[half][part], state[col][half][part], answer[half]);
                }
            }
            total = __fadd_rn(answer[0], answer[1]);
#pragma unroll
            for (int step = 8; step > 0; step >>= 1)
                total = __fadd_rn(total, __shfl_down_sync(mask, total, step, 16));
            if (lane == 0 && column + col < value_dim)
                output[output_offset * value_dim + column + col] = __float2bfloat16_rn(total);
        }
    }
}

extern "C" __global__ void qwen35_vstoreshort_1_bf16(const float *query, const float *key,
    const __nv_bfloat16 *value, const float *gate, const float *beta, __nv_bfloat16 *output,
    const unsigned int *cu_seqlens, int total_length, int heads, int key_heads, int value_dim) {
    const int batch_head = blockIdx.x;
    const int head = batch_head % heads;
    const int batch = batch_head / heads;
    const int begin = cu_seqlens[batch];
    const int length = cu_seqlens[batch + 1] - begin;
    const int lane = threadIdx.x & 15;
    constexpr int Columns = 4;
    const int column = (blockIdx.y * 8 + (threadIdx.x >> 4)) * Columns;
    if (column >= value_dim)
        return;
    const unsigned mask = __activemask();
    float state[Columns][2][4] = {};
    for (int token = 0; token < length; ++token) {
        const int token_head = (begin + token) * heads + head;
        const int qk_head = (begin + token) * key_heads + head / (heads / key_heads);
        const int output_offset = token_head;
        const float decay = expf(gate[token_head]);
        const float weight = beta[token_head];
        const float *token_query = query + qk_head * 128;
        const float *token_key = key + qk_head * 128;
        const __nv_bfloat16 *token_value = value + token_head * value_dim;

        float keys[2][4], queries[2][4];
#pragma unroll
        for (int half = 0; half < 2; ++half) {
#pragma unroll
            for (int part = 0; part < 4; ++part) {
                const int d = part * 32 + lane + half * 16;
                keys[half][part] = token_key[d];
                queries[half][part] = token_query[d];
            }
        }
#pragma unroll
        for (int col = 0; col < Columns; ++col) {
            float remembered[2] = {0.f, 0.f};
#pragma unroll
            for (int half = 0; half < 2; ++half) {
#pragma unroll
                for (int part = 0; part < 4; ++part) {
                    state[col][half][part] = __fmul_rn(state[col][half][part], decay);
                    remembered[half] =
                        __fmaf_rn(keys[half][part], state[col][half][part], remembered[half]);
                }
            }
            float total = __fadd_rn(remembered[0], remembered[1]);
#pragma unroll
            for (int step = 8; step > 0; step >>= 1)
                total = __fadd_rn(total, __shfl_down_sync(mask, total, step, 16));
            const float val =
                column + col < value_dim ? __bfloat162float(token_value[column + col]) : 0.f;
            const float delta = __fmul_rn(__fsub_rn(val, __shfl_sync(mask, total, 0, 16)), weight);
            float answer[2] = {0.f, 0.f};
#pragma unroll
            for (int half = 0; half < 2; ++half) {
#pragma unroll
                for (int part = 0; part < 4; ++part) {
                    state[col][half][part] =
                        __fmaf_rn(keys[half][part], delta, state[col][half][part]);
                    answer[half] =
                        __fmaf_rn(queries[half][part], state[col][half][part], answer[half]);
                }
            }
            total = __fadd_rn(answer[0], answer[1]);
#pragma unroll
            for (int step = 8; step > 0; step >>= 1)
                total = __fadd_rn(total, __shfl_down_sync(mask, total, step, 16));
            if (lane == 0 && column + col < value_dim)
                output[output_offset * value_dim + column + col] = __float2bfloat16_rn(total);
        }
    }
}

extern "C" __global__ void qwen35_vstoreshort_2_bf16(const float *query, const float *key,
    const __nv_bfloat16 *value, const float *gate, const float *beta, __nv_bfloat16 *output,
    int length, int heads, int key_heads, int value_dim) {
    const int batch_head = blockIdx.x;
    const int head = batch_head % heads;
    const int batch = batch_head / heads;
    const int lane = threadIdx.x & 15;
    constexpr int Columns = 4;
    const int column = (blockIdx.y * 8 + (threadIdx.x >> 4)) * Columns;
    if (column >= value_dim)
        return;
    const unsigned mask = __activemask();
    float state[Columns][2][4] = {};
    for (int token = 0; token < length; ++token) {
        const int token_head = (batch * length + token) * heads + head;
        const int qk_head = (batch * length + token) * key_heads + head / (heads / key_heads);
        const int output_offset = batch_head * length + token;
        const float decay = expf(gate[token_head]);
        const float weight = beta[token_head];
        const float *token_query = query + qk_head * 128;
        const float *token_key = key + qk_head * 128;
        const __nv_bfloat16 *token_value = value + token_head * value_dim;

        float keys[2][4], queries[2][4];
#pragma unroll
        for (int half = 0; half < 2; ++half) {
#pragma unroll
            for (int part = 0; part < 4; ++part) {
                const int d = part * 32 + lane + half * 16;
                keys[half][part] = token_key[d];
                queries[half][part] = token_query[d];
            }
        }
#pragma unroll
        for (int col = 0; col < Columns; ++col) {
            float remembered[2] = {0.f, 0.f};
#pragma unroll
            for (int half = 0; half < 2; ++half) {
#pragma unroll
                for (int part = 0; part < 4; ++part) {
                    state[col][half][part] = __fmul_rn(state[col][half][part], decay);
                    remembered[half] =
                        __fmaf_rn(keys[half][part], state[col][half][part], remembered[half]);
                }
            }
            float total = __fadd_rn(remembered[0], remembered[1]);
#pragma unroll
            for (int step = 8; step > 0; step >>= 1)
                total = __fadd_rn(total, __shfl_down_sync(mask, total, step, 16));
            const float val =
                column + col < value_dim ? __bfloat162float(token_value[column + col]) : 0.f;
            const float delta = __fmul_rn(__fsub_rn(val, __shfl_sync(mask, total, 0, 16)), weight);
            float answer[2] = {0.f, 0.f};
#pragma unroll
            for (int half = 0; half < 2; ++half) {
#pragma unroll
                for (int part = 0; part < 4; ++part) {
                    state[col][half][part] =
                        __fmaf_rn(keys[half][part], delta, state[col][half][part]);
                    answer[half] =
                        __fmaf_rn(queries[half][part], state[col][half][part], answer[half]);
                }
            }
            total = __fadd_rn(answer[0], answer[1]);
#pragma unroll
            for (int step = 8; step > 0; step >>= 1)
                total = __fadd_rn(total, __shfl_down_sync(mask, total, step, 16));
            if (lane == 0 && column + col < value_dim)
                output[output_offset * value_dim + column + col] = __float2bfloat16_rn(total);
        }
    }
}

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math.h>
extern "C" __global__ void qwen35_vstore_short_bf16(const float *query, const float *key,
    const __nv_bfloat16 *value, const float *gate, const float *beta, __nv_bfloat16 *output,
    const unsigned int *cu_seqlens, const unsigned int *short_ids, int total_length, int heads,
    int key_heads, int value_dim) {
    const int batch_head = blockIdx.x;
    const int head = batch_head % heads;
    const int batch = short_ids[batch_head / heads];
    const int begin = cu_seqlens[batch];
    const int length = cu_seqlens[batch + 1] - begin;
    const int lane = threadIdx.x & 15;
    const int column = blockIdx.y * 8 + (threadIdx.x >> 4);
    if (column >= value_dim)
        return;
    const unsigned mask = __activemask();
    float state[2][4] = {{0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}};
    for (int token = 0; token < length; ++token) {
        const int token_head = (begin + token) * heads + head;
        const int qk_head = (begin + token) * key_heads + head / (heads / key_heads);
        const int output_offset = token_head;
        const float decay = expf(gate[token_head]);
        const float weight = beta[token_head];
        const float *token_query = query + qk_head * 128;
        const float *token_key = key + qk_head * 128;
        const __nv_bfloat16 *token_value = value + token_head * value_dim;
        float keys[2][4];
        float remembered[2] = {0.f, 0.f};
#pragma unroll
        for (int half = 0; half < 2; ++half) {
#pragma unroll
            for (int part = 0; part < 4; ++part) {
                keys[half][part] = token_key[part * 32 + lane + half * 16];
                state[half][part] *= decay;
                remembered[half] += keys[half][part] * state[half][part];
            }
        }

        float total = remembered[0] + remembered[1];
#pragma unroll
        for (int step = 8; step > 0; step >>= 1) {
            total += __shfl_down_sync(mask, total, step, 16);
        }
        const float delta =
            (__bfloat162float(token_value[column]) - __shfl_sync(mask, total, 0, 16)) * weight;
        float answer[2] = {0.f, 0.f};
#pragma unroll
        for (int half = 0; half < 2; ++half) {
#pragma unroll
            for (int part = 0; part < 4; ++part) {
                state[half][part] += keys[half][part] * delta;
                answer[half] += token_query[part * 32 + lane + half * 16] * state[half][part];
            }
        }
        total = answer[0] + answer[1];
#pragma unroll
        for (int step = 8; step > 0; step >>= 1) {
            total += __shfl_down_sync(mask, total, step, 16);
        }
        if (lane == 0)
            output[output_offset * value_dim + column] = __float2bfloat16_rn(total);
    }
}

extern "C" __global__ void qwen35_vstore_packed_delta_bfout(const float *query, const float *key,
    const __nv_bfloat16 *value, const float *gate, const float *beta, __nv_bfloat16 *output,
    const unsigned int *cu_seqlens, int total_length, int heads, int key_heads, int value_dim) {
    const int batch_head = blockIdx.x;
    const int head = batch_head % heads;
    const int batch = batch_head / heads;
    const int begin = cu_seqlens[batch];
    const int length = cu_seqlens[batch + 1] - begin;
    const int lane = threadIdx.x & 15;
    const int column = blockIdx.y * 8 + (threadIdx.x >> 4);
    if (column >= value_dim)
        return;
    const unsigned mask = __activemask();
    float state[2][4] = {{0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}};
    for (int token = 0; token < length; ++token) {
        const int token_head = (begin + token) * heads + head;
        const int qk_head = (begin + token) * key_heads + head / (heads / key_heads);
        const int output_offset = token_head;
        const float decay = expf(gate[token_head]);
        const float weight = beta[token_head];
        const float *token_query = query + qk_head * 128;
        const float *token_key = key + qk_head * 128;
        const __nv_bfloat16 *token_value = value + token_head * value_dim;
        float keys[2][4];
        float remembered[2] = {0.f, 0.f};
#pragma unroll
        for (int half = 0; half < 2; ++half) {
#pragma unroll
            for (int part = 0; part < 4; ++part) {
                keys[half][part] = token_key[part * 32 + lane + half * 16];
                state[half][part] *= decay;
                remembered[half] += keys[half][part] * state[half][part];
            }
        }

        float total = remembered[0] + remembered[1];
#pragma unroll
        for (int step = 8; step > 0; step >>= 1) {
            total += __shfl_down_sync(mask, total, step, 16);
        }
        const float delta =
            (__bfloat162float(token_value[column]) - __shfl_sync(mask, total, 0, 16)) * weight;
        float answer[2] = {0.f, 0.f};
#pragma unroll
        for (int half = 0; half < 2; ++half) {
#pragma unroll
            for (int part = 0; part < 4; ++part) {
                state[half][part] += keys[half][part] * delta;
                answer[half] += token_query[part * 32 + lane + half * 16] * state[half][part];
            }
        }
        total = answer[0] + answer[1];
#pragma unroll
        for (int step = 8; step > 0; step >>= 1) {
            total += __shfl_down_sync(mask, total, step, 16);
        }
        if (lane == 0)
            output[output_offset * value_dim + column] = __float2bfloat16_rn(total);
    }
}

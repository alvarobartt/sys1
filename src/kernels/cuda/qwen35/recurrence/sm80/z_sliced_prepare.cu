

__device__ __forceinline__ int qwen35_c64slice_half_index(int row, int col) {
    return row * 64 + (col ^ ((row & 7) * 8));
}

template <int Mode>
__device__ __forceinline__ void
qwen35_c64slice_gram(const float *query, const float *key, const float *beta, const float *prefix,
                   bool &key_unsafe, int physical_start, int begin, int length, int head, int heads,
                   int kh, int key_heads, __half *ahalf, __half *alow, __half *bhalf, __half *blow,
                   float *out) {
    constexpr int C = 64, K = 128;
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32, group = lane / 4, tid = lane % 4;
    const int col = (warp % 8) * 8;
    float high[2][4] = {}, cross[2][4] = {};
    bool unsafe = Mode == 1 ? key_unsafe : false;
    bool fallback = false;
    for (int slice = 0; slice < K; slice += 64) {
        for (int i = threadIdx.x; i < C * 64; i += blockDim.x) {
            const int t = i / 64, d = i % 64, token = begin + t;
            const int offset = ((physical_start + token) * key_heads + kh) * K + slice + d;
            const float kval = token < length ? key[offset] : 0.f;
            const float lhs =
                Mode == 0 ? (token < length
                                 ? __fmul_rn(kval, beta[(physical_start + token) * heads + head])
                                 : 0.f)
                          : (token < length ? query[offset] : 0.f);
            key_unsafe |= !(fabsf(kval) <= 32768.f);
            unsafe |= !(fabsf(kval) <= 32768.f) || !(fabsf(lhs) <= 32768.f);
            const auto a = __float2half_rn(lhs), b = __float2half_rn(kval);
            const int at = qwen35_c64slice_half_index(t, d);
            ahalf[at] = a;
            alow[at] = __float2half_rn(__fmul_rn(__fsub_rn(lhs, __half2float(a)), 4096.f));
            bhalf[at] = b;
            blow[at] = __float2half_rn(__fmul_rn(__fsub_rn(kval, __half2float(b)), 4096.f));
        }
        fallback = __syncthreads_or(unsafe);
        if (!fallback) {
            for (int k = 0; k < 64; k += 16) {
#pragma unroll
                for (int part = 0; part < 2; ++part) {
                    const int row = (warp / 8) * 16 + part * 32;
                    unsigned ah[4], al[4], bh[2], bl[2];
                    const int ar = row + (lane & 15), ac = k + (lane / 16) * 8;
                    const int br = col + (lane & 7), bc = k + ((lane / 8) & 1) * 8;
                    qwen35_gramld_load4(ah, ahalf + qwen35_c64slice_half_index(ar, ac));
                    qwen35_gramld_load4(al, alow + qwen35_c64slice_half_index(ar, ac));
                    qwen35_gramld_load2(bh, bhalf + qwen35_c64slice_half_index(br, bc));
                    qwen35_gramld_load2(bl, blow + qwen35_c64slice_half_index(br, bc));
                    qwen35_c64_half_mma16816(high[part], ah, bh);
                    qwen35_c64_half_mma16816(cross[part], ah, bl);
                    qwen35_c64_half_mma16816(cross[part], al, bh);
                }
            }
        }

        __syncthreads();
    }
#pragma unroll
    for (int part = 0; part < 2; ++part) {
        const int row = (warp / 8) * 16 + part * 32;
#pragma unroll
        for (int p = 0; p < 4; ++p) {
            const int t = row + group + (p / 2) * 8, j = col + tid * 2 + p % 2;
            float dot;
            if (fallback) {
                dot = 0.f;
                const int token = begin + t, previous = begin + j;
                if (token < length && previous < length) {
                    for (int d = 0; d < K; ++d) {
                        const int offset = ((physical_start + token) * key_heads + kh) * K + d;
                        const float lhs =
                            Mode == 0 ? __fmul_rn(key[offset],
                                                  beta[(physical_start + token) * heads + head])
                                      : query[offset];
                        dot = __fmaf_rn(
                            lhs, key[((physical_start + previous) * key_heads + kh) * K + d], dot);
                    }
                }
            } else
                dot = __fadd_rn(high[part][p], __fmul_rn(cross[part][p], 0x1p-12f));
            if constexpr (Mode == 1)
                out[t * C + j] =
                    j <= t ? __fmul_rn(dot, expf(__fsub_rn(prefix[t], prefix[j]))) : 0.f;
            else
                out[t * C + j] = dot;
        }
    }
    __syncthreads();
}

template <int Quarter>
__device__ __forceinline__ void qwen35_c64slice_rhs_phase(float (&rhs)[4][8], const float *lower,
                                                        int lane) {
#pragma unroll 1
    for (int j = Quarter * 16; j < (Quarter + 1) * 16; ++j) {
        float coefficient[4];
#pragma unroll
        for (int part = Quarter; part < 4; ++part)
            coefficient[part] = lower[qwen35_c64warp_lower_index(lane + part * 16, j)];
#pragma unroll
        for (int column = 0; column < 8; ++column) {
            const float value = __shfl_sync(0xffffffff, rhs[Quarter][column], j & 15, 16);
#pragma unroll
            for (int part = Quarter; part < 4; ++part) {
                if (part > Quarter || lane + part * 16 > j)
                    rhs[part][column] =
                        __fsub_rn(rhs[part][column], __fmul_rn(coefficient[part], value));
            }
        }
    }
}

__device__ __forceinline__ void
qwen35_c64slice_prepare(const float *query, const float *key, const float *value, const float *gate,
                      const float *beta, float *w, float *u, float *causal, float *decays,
                      int physical_start, int begin, int length, int head, int heads, int key_heads,
                      int value_dim, int instance) {
    constexpr int C = 64, K = 128, S = 129;
    extern __shared__ __align__(32) unsigned char qwen35_c64slice_storage[];
    auto *ahalf = reinterpret_cast<__half *>(qwen35_c64slice_storage);
    auto *alow = ahalf + C * 64;
    auto *bhalf = alow + C * 64;
    auto *blow = bhalf + C * 64;
    auto *lower = reinterpret_cast<float *>(blow + C * 64);
    auto *prefix = lower + C * C;
    auto *betas = prefix + C;
    const int kh = head / (heads / key_heads);
    if (threadIdx.x < C) {
        const int t = threadIdx.x, token = begin + t, th = (physical_start + token) * heads + head;
        float effective = 0.f;
        if (token < length) {
            const float decay = expf(gate[th]);
            effective = decay > 0.f ? logf(decay) : gate[th];
        }
        prefix[t] = effective;
        betas[t] = token < length ? beta[th] : 0.f;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        float running = 0.f;
        for (int t = 0; t < C; ++t) {
            if (begin + t < length)
                running = __fadd_rn(running, prefix[t]);
            prefix[t] = running;
        }
    }
    __syncthreads();
    if (threadIdx.x < C) {
        const int t = threadIdx.x;
        decays[instance * 128 + t] = expf(prefix[t]);
        decays[instance * 128 + 64 + t] = expf(__fsub_rn(prefix[63], prefix[t]));
    }
    bool key_unsafe = false;
    qwen35_c64slice_gram<0>(query, key, beta, prefix, key_unsafe, physical_start, begin, length, head,
                          heads, kh, key_heads, ahalf, alow, bhalf, blow, lower);
    float saved[8];
#pragma unroll
    for (int item = 0; item < 8; ++item) {
        const int i = threadIdx.x + item * 512, t = i / C, j = i % C;
        saved[item] = j < t ? __fmul_rn(lower[i], expf(__fsub_rn(prefix[t], prefix[j]))) : 0.f;
    }
    __syncthreads();
#pragma unroll
    for (int item = 0; item < 8; ++item) {
        const int i = threadIdx.x + item * 512;
        lower[qwen35_c64warp_lower_index(i / C, i % C)] = saved[item];
    }

    qwen35_c64slice_gram<1>(query, key, beta, prefix, key_unsafe, physical_start, begin, length, head,
                          heads, kh, key_heads, ahalf, alow, bhalf, blow,
                          causal + instance * C * C);
    const int lane = threadIdx.x & 15,
              first_column = (threadIdx.x / 32) * 16 + ((threadIdx.x & 31) / 16) * 8;
    float rhs[4][8];
#pragma unroll
    for (int part = 0; part < 4; ++part) {
        const int row = lane + part * 16, token = begin + row;
        const float beta_value = betas[row], decay = expf(prefix[row]);
#pragma unroll
        for (int column = 0; column < 8; ++column) {
            const int c = first_column + column, d = c & 127;
            if (c < 128) {
                const float kval =
                    token < length ? key[((physical_start + token) * key_heads + kh) * K + d] : 0.f;
                rhs[part][column] = __fmul_rn(__fmul_rn(kval, beta_value), decay);
            } else {
                const float val =
                    token < length && d < value_dim
                        ? value[((physical_start + token) * heads + head) * value_dim + d]
                        : 0.f;
                rhs[part][column] = __fmul_rn(val, beta_value);
            }
        }
    }
    qwen35_c64slice_rhs_phase<0>(rhs, lower, lane);
    qwen35_c64slice_rhs_phase<1>(rhs, lower, lane);
    qwen35_c64slice_rhs_phase<2>(rhs, lower, lane);
    qwen35_c64slice_rhs_phase<3>(rhs, lower, lane);
    __syncthreads();
    auto *tile = reinterpret_cast<float *>(qwen35_c64slice_storage);
#pragma unroll
    for (int field = 0; field < 2; ++field) {
        if ((first_column >= 128) == bool(field)) {
#pragma unroll
            for (int part = 0; part < 4; ++part) {
#pragma unroll
                for (int column = 0; column < 8; ++column)
                    tile[(lane + part * 16) * S + ((first_column + column) & 127)] =
                        rhs[part][column];
            }
        }
        __syncthreads();
        for (int i = threadIdx.x; i < C * K; i += blockDim.x) {
            const int t = i / K, d = i % K;
            if (field == 0)
                w[(instance * C + t) * K + d] = tile[t * S + d];
            else if (d < value_dim)
                u[(instance * C + t) * value_dim + d] = tile[t * S + d];
        }
        __syncthreads();
    }
}

extern "C" __global__ __launch_bounds__(512, 2) void qwen35_c64slice_uniform_prepare_f32(
    const float *q, const float *k, const float *v, const float *g, const float *b, float *w,
    float *u, float *c, float *d, int length, int heads, int key_heads, int vd) {
    const int chunks = (length + 63) / 64, instance = blockIdx.x, bh = instance / chunks,
              chunk = instance % chunks;
    const int batch = bh / heads, head = bh % heads;
    qwen35_c64slice_prepare(q, k, v, g, b, w, u, c, d, batch * length, chunk * 64, length, head,
                          heads, key_heads, vd, instance);
}
extern "C" __global__ __launch_bounds__(512, 2) void qwen35_c64slice_varlen_prepare_f32(
    const float *q, const float *k, const float *v, const float *g, const float *b, float *w,
    float *u, float *c, float *d, const unsigned *cumulative, const unsigned *chunks,
    int sequence_count, int total_length, int heads, int key_heads, int vd) {
    const int global_chunk = blockIdx.x / heads, head = blockIdx.x % heads;
    int lo = 0, hi = sequence_count;
    while (lo < hi) {
        const int mid = (lo + hi + 1) / 2;
        if (chunks[mid] <= global_chunk)
            lo = mid;
        else
            hi = mid - 1;
    }
    const int start = cumulative[lo], length = cumulative[lo + 1] - start,
              begin = (global_chunk - int(chunks[lo])) * 64;
    qwen35_c64slice_prepare(q, k, v, g, b, w, u, c, d, start, begin, length, head, heads, key_heads,
                          vd, global_chunk * heads + head);
}

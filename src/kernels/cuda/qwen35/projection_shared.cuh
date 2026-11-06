#pragma once
#include <cuda_bf16.h>

__device__ __forceinline__ bool qwen35_projection_exceptional_operand(__nv_bfloat16 value) {
    const unsigned short magnitude = __bfloat16_as_ushort(value) & 0x7fff;
    return magnitude != 0 && (magnitude < 0x2000 || magnitude > 0x5f00);
}

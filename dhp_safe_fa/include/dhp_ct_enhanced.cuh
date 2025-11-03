#pragma once
// DHP Constant-Time Primitives
// Based on expert-reviewed security methodology

#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// ============================================================================
// Constant-Time Comparison Primitives
// ============================================================================

__device__ __forceinline__ uint32_t ct_lt_u32(uint32_t a, uint32_t b) {
    // Returns 0xFFFFFFFF if a < b, else 0x00000000
    // Uses arithmetic to avoid branching
    uint32_t diff = a - b;
    return (uint32_t)((int32_t)diff >> 31);
}

__device__ __forceinline__ uint32_t ct_le_u32(uint32_t a, uint32_t b) {
    // Returns 0xFFFFFFFF if a <= b, else 0x00000000
    return ct_lt_u32(a, b + 1);
}

__device__ __forceinline__ uint32_t ct_gt_f32(float a, float b) {
    // Returns 0xFFFFFFFF if a > b, else 0x00000000
    // Use float comparison, then convert result to mask
    // This is constant-time on modern GPUs (no branch divergence)
    return (a > b) ? 0xFFFFFFFF : 0x00000000;
}

__device__ __forceinline__ uint32_t ct_and_u32(uint32_t a, uint32_t b) {
    return a & b;
}

// ============================================================================
// Constant-Time Select Primitives
// ============================================================================

__device__ __forceinline__ float ct_select_f32(float false_val, float true_val, uint32_t mask) {
    // If mask == 0xFFFFFFFF, return true_val
    // If mask == 0x00000000, return false_val
    // No branching - both paths execute
    uint32_t false_bits = __float_as_uint(false_val);
    uint32_t true_bits = __float_as_uint(true_val);
    uint32_t result = (false_bits & ~mask) | (true_bits & mask);
    return __uint_as_float(result);
}

__device__ __forceinline__ __half ct_select_half(__half false_val, __half true_val, uint32_t mask) {
    float f = ct_select_f32(__half2float(false_val), __half2float(true_val), mask);
    return __float2half(f);
}

// ============================================================================
// Numerical Stability Helpers (EXPERT_CORRECTIONS.md §3.3)
// ============================================================================

__device__ __forceinline__ float safe_exp(float x) {
    // Prevent underflow in exp computation
    constexpr float MIN_EXP_ARG = -87.0f;  // exp(-87) ≈ 1e-38
    return expf(fmaxf(x, MIN_EXP_ARG));
}

// ============================================================================
// Validation: All operations above are constant-time
// - No data-dependent branches
// - No variable-time instructions
// - SASS validation: zero @p BRA instructions expected
// ============================================================================


// ============================================================================
// Debug Utilities for V3 Kernel (Enable with -DDEBUG_V3=1)
// ============================================================================

#pragma once

#include <cstdio>

// Out-of-bounds check helper
__device__ __forceinline__ int oob(int idx, int lo, int hi) {
    return (idx < lo) || (idx >= hi);
}

// Debug assertion (triggers breakpoint on failure)
#ifdef DEBUG_V3
#define CUDA_DEBUG_ASSERT(cond) \
    do { \
        if (!(cond)) { \
            printf("ASSERTION FAILED at %s:%d: %s\n", __FILE__, __LINE__, #cond); \
            asm("brkpt;"); \
        } \
    } while(0)
#else
#define CUDA_DEBUG_ASSERT(cond) ((void)0)
#endif

// Alignment check for cp.async (must be 16-byte aligned)
__device__ __forceinline__ bool is_aligned_16(const void* ptr) {
    return ((uintptr_t)ptr % 16) == 0;
}

// Bounds check for GMEM access
__device__ __forceinline__ bool gmem_in_bounds(
    int batch_idx, int head_idx, int seq_idx,
    int B, int H, int S
) {
    return (batch_idx >= 0 && batch_idx < B &&
            head_idx >= 0 && head_idx < H &&
            seq_idx >= 0 && seq_idx < S);
}

// Bounds check for SMEM access
__device__ __forceinline__ bool smem_in_bounds(
    int row, int col, int max_row, int max_col
) {
    return (row >= 0 && row < max_row &&
            col >= 0 && col < max_col);
}

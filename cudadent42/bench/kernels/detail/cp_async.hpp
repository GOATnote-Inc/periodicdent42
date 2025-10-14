// ============================================================================
// cp.async Wrappers for Async Memory Copy (SM 8.0+)
// ============================================================================
// Purpose: Hide memory latency by overlapping compute with copy
// Target: CUDA Compute Capability 8.0+ (Ampere, Ada, Hopper)
// ============================================================================

#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace detail {

// ============================================================================
// cp.async.ca - Cache All Levels (for frequently reused data like K, V)
// ============================================================================

template<int N>
__device__ __forceinline__ void cp_async_ca(void* dst, const void* src) {
    static_assert(N == 4 || N == 8 || N == 16, "cp.async only supports 4, 8, 16 bytes");
    
    if constexpr (N == 16) {
        asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n" :: "r"((unsigned)__cvta_generic_to_shared(dst)), "l"(src));
    } else if constexpr (N == 8) {
        asm volatile("cp.async.ca.shared.global [%0], [%1], 8;\n" :: "r"((unsigned)__cvta_generic_to_shared(dst)), "l"(src));
    } else if constexpr (N == 4) {
        asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" :: "r"((unsigned)__cvta_generic_to_shared(dst)), "l"(src));
    }
}

// ============================================================================
// cp.async.cg - Cache Global Only (for single-use data)
// ============================================================================

template<int N>
__device__ __forceinline__ void cp_async_cg(void* dst, const void* src) {
    static_assert(N == 4 || N == 8 || N == 16, "cp.async only supports 4, 8, 16 bytes");
    
    if constexpr (N == 16) {
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"((unsigned)__cvta_generic_to_shared(dst)), "l"(src));
    } else if constexpr (N == 8) {
        asm volatile("cp.async.cg.shared.global [%0], [%1], 8;\n" :: "r"((unsigned)__cvta_generic_to_shared(dst)), "l"(src));
    } else if constexpr (N == 4) {
        asm volatile("cp.async.cg.shared.global [%0], [%1], 4;\n" :: "r"((unsigned)__cvta_generic_to_shared(dst)), "l"(src));
    }
}

// ============================================================================
// Pipeline Control
// ============================================================================

__device__ __forceinline__ void cp_async_commit_group() {
    asm volatile("cp.async.commit_group;\n" ::);
}

template<int N>
__device__ __forceinline__ void cp_async_wait_group() {
    static_assert(N >= 0 && N <= 8, "wait_group N must be 0-8");
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
}

__device__ __forceinline__ void cp_async_wait_all() {
    asm volatile("cp.async.wait_all;\n" ::);
}

// ============================================================================
// Helper: Copy 16-byte aligned tile row
// ============================================================================

__device__ __forceinline__ void copy_tile_row_async(
    half* __restrict__ smem_dst,
    const half* __restrict__ gmem_src,
    int row_stride_smem,
    int row_stride_gmem,
    int cols
) {
    // Assume cols % 8 == 0 (16 bytes = 8 halfs)
    for (int col = threadIdx.x * 8; col < cols; col += blockDim.x * 8) {
        if (col + 8 <= cols) {
            cp_async_ca<16>(
                smem_dst + col,
                gmem_src + col
            );
        }
    }
}

} // namespace detail

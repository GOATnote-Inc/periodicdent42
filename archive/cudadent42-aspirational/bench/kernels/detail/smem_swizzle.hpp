// ============================================================================
// Shared Memory Swizzle & Padding Utilities
// ============================================================================
// Purpose: Eliminate bank conflicts in shared memory accesses
// Method: XOR swizzle or +1 padding for conflict-free access patterns
// ============================================================================

#pragma once

#include <cstdint>
#include <cuda_fp16.h>

namespace detail {

// ============================================================================
// 16-Byte Alignment Utilities (for cp.async)
// ============================================================================

// cp.async requires 16-byte alignment for source and destination
// These helpers ensure row strides are multiples of 16 bytes

template <typename T>
constexpr int elems_for_16B() { 
    return 16 / sizeof(T); 
}

template <typename T>
constexpr int pad_to_16B_elems(int stride_elems) {
    const int q = elems_for_16B<T>();
    const int r = stride_elems % q;
    return (r == 0) ? 0 : (q - r);
}

// Sanity: for half (2 B), 16 B == 8 elems
static_assert(elems_for_16B<half>() == 8, "half must be 2 bytes");

// ============================================================================
// Bank Conflict Analysis
// ============================================================================

// CUDA shared memory has 32 banks (4-byte wide)
// Conflict occurs when multiple threads in a warp access same bank
constexpr int SMEM_BANKS = 32;
constexpr int SMEM_BANK_WIDTH_BYTES = 4;

// ============================================================================
// Padding Strategy: Add +1 element to avoid power-of-2 strides
// ============================================================================

template<int TILE_DIM>
constexpr int padded_stride() {
    // If TILE_DIM is multiple of 32 halfs (16 elements per bank-width),
    // add 1 to break the pattern
    return (TILE_DIM % 16 == 0) ? (TILE_DIM + 1) : TILE_DIM;
}

// ============================================================================
// XOR Swizzle Strategy (more sophisticated, for power-of-2 dims)
// ============================================================================

__device__ __forceinline__ uint32_t swizzle_row_idx(uint32_t row, uint32_t col, int swizzle_bits) {
    // XOR high bits of row with low bits of col to decorrelate
    // swizzle_bits controls how many bits to XOR (typically 2-3)
    uint32_t swizzle_mask = (1 << swizzle_bits) - 1;
    uint32_t col_bits = col & swizzle_mask;
    return row ^ col_bits;
}

// ============================================================================
// Helper: Get swizzled SMEM offset
// ============================================================================

template<bool ENABLE_SWIZZLE, int SWIZZLE_BITS = 2>
__device__ __forceinline__ int get_smem_offset(int row, int col, int stride) {
    if constexpr (ENABLE_SWIZZLE) {
        int swizzled_row = swizzle_row_idx(row, col, SWIZZLE_BITS);
        return swizzled_row * stride + col;
    } else {
        return row * stride + col;
    }
}

// ============================================================================
// Declare Padded SMEM Array (compile-time)
// ============================================================================

#define DECLARE_PADDED_SMEM(TYPE, NAME, ROWS, COLS) \
    __shared__ TYPE NAME[ROWS][detail::padded_stride<COLS>()]

} // namespace detail

/**
 * Build configuration for FlashAttention-Science
 * 
 * This file provides architecture-specific flags and tile size configurations.
 * 
 * Architecture support:
 * - SM75 (T4): FP16 only, no async copy
 * - SM80 (A100): FP16 + BF16, cp.async, WMMA
 * - SM89 (L4): FP16 + BF16, cp.async, WMMA
 * - SM90 (H100): FP16 + BF16 + FP8, WGMMA, TMA
 */

#pragma once

// ============================================================================
// Architecture Feature Flags
// ============================================================================

// Async memory copy (cuda::pipeline) - requires SM80+
#define HAS_CP_ASYNC 1      // Enabled for A100/L4/H100

// Warp Group Matrix Multiply Accumulate - requires SM90+
#define HAS_WGMMA 0         // Only H100 (not implemented yet)

// ============================================================================
// Tile Size Configuration
// ============================================================================
// These define the block tile sizes for attention computation.
// Constraints:
// - Total SRAM usage must fit in shared memory (48KB for T4, 228KB for H100)
// - TILE_SIZE_K should match head_dim for optimal performance
// - Larger tiles improve memory bandwidth but increase register pressure

#define TILE_SIZE_M 32      // Query tile size (rows)
#define TILE_SIZE_N 128     // Key/Value tile size (columns)
#define TILE_SIZE_K 128     // Head dimension (max supported)

// ============================================================================
// Data Type Support
// ============================================================================
// Uncomment to disable BF16 (forces FP16 only)
// Useful for debugging or SM75 (T4) which lacks native BF16 support
//
// #define FLASHMOE_DTYPE_FP16_ONLY

// ============================================================================
// Debug Flags
// ============================================================================
// Uncomment for verbose kernel output (only for small tests!)
//
// #define FLASHMOE_DEBUG_PRINT


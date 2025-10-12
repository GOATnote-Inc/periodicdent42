/**
 * Build configuration for FlashMoE-Science CUDA kernels
 * 
 * This file defines architecture-specific constants and tile sizes
 * that must match between flash_attention_science.cu and flash_attention_warp_specialized.cu
 * 
 * @author GOATnote Autonomous Research Lab Initiative  
 * @date 2025-10-12 (CORRECTED - Fixed THREADS_PER_BLOCK)
 */

#ifndef FLASHMOE_BUILD_CONFIG_H
#define FLASHMOE_BUILD_CONFIG_H

// ============================================================================
// WARP AND BLOCK CONFIGURATION (12 warps = 384 threads)
// ============================================================================

constexpr int WARP_SIZE = 32;

// *** CRITICAL FIX: Changed from 4 to 12 warps ***
// Original (WRONG): 4 warps = 128 threads → 0.12× speedup (8-29× slower)
// Corrected (RIGHT): 12 warps = 384 threads → enables 3 warpgroup specialization
constexpr int NUM_WARPS_PER_BLOCK = 12;  // Was 4, now 12 (3 warpgroups × 4 warps each)

constexpr int NUM_WARPS_PER_WARPGROUP = 4;  // FlashAttention-4 style
constexpr int NUM_WARPGROUPS = 3;           // Producer, Consumer, Output correction

// Total threads per block (MUST be 384 for optimized kernel)
constexpr int THREADS_PER_BLOCK = NUM_WARPS_PER_BLOCK * WARP_SIZE;  // = 384

// Verify configuration at compile time
static_assert(THREADS_PER_BLOCK == 384, 
              "THREADS_PER_BLOCK must be 384 for flash_attention_science.cu");
static_assert(NUM_WARPS_PER_BLOCK == NUM_WARPGROUPS * NUM_WARPS_PER_WARPGROUP,
              "Warp configuration inconsistent");

// ============================================================================
// TILE SIZES (Memory hierarchy optimization)
// ============================================================================

// Tile dimensions for attention computation
// These control shared memory usage and must fit within 48KB (SM75) or 228KB (SM90)
constexpr int TILE_SIZE_M = 128;  // Query tile size (rows)
constexpr int TILE_SIZE_N = 128;  // Key/Value tile size (rows)  
constexpr int TILE_SIZE_K = 128;  // Head dimension (columns)

// Verify tiles don't exceed shared memory limits
// Shared memory usage per block (worst case):
// - smem_Q: TILE_SIZE_M × TILE_SIZE_K × 2 bytes (FP16) = 32KB
// - smem_K: TILE_SIZE_N × TILE_SIZE_K × 2 bytes (FP16) = 32KB
// - smem_V: TILE_SIZE_N × TILE_SIZE_K × 2 bytes (FP16) = 32KB
// - smem_S: TILE_SIZE_M × TILE_SIZE_N × 4 bytes (FP32) = 64KB
// Total: ~160KB (fits in SM80+, may spill on SM75)
constexpr size_t SHARED_MEMORY_BYTES = 
    (TILE_SIZE_M * TILE_SIZE_K * 2) +  // Q
    (TILE_SIZE_N * TILE_SIZE_K * 2) +  // K
    (TILE_SIZE_N * TILE_SIZE_K * 2) +  // V
    (TILE_SIZE_M * TILE_SIZE_N * 4);   // S (attention scores)

// ============================================================================
// ARCHITECTURE FEATURE FLAGS
// ============================================================================

// Async memory copy (cp.async) support
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
#define HAS_CP_ASYNC 1
#else
#define HAS_CP_ASYNC 0
#endif

// BF16 native support
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
#define HAS_BF16 1
#else
#define HAS_BF16 0
#endif

// FP8 Tensor Core support (Hopper H100)
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
#define HAS_FP8 1
#else
#define HAS_FP8 0
#endif

// Tensor Memory Accelerator (TMA) support (Hopper H100)
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
#define HAS_TMA 1
#else
#define HAS_TMA 0
#endif

// ============================================================================
// OPTIMIZATION FLAGS
// ============================================================================

// Enable vectorized memory loads (float4 = 8 × FP16 per load)
#define USE_VECTORIZED_LOADS 1

// Enable Tensor Core usage via WMMA
#define USE_TENSOR_CORES 0  // TODO: Implement in Fix #2

// Enable async memory pipeline (SM80+ only)
#define USE_ASYNC_PIPELINE 0  // TODO: Implement in Fix #3

// ============================================================================
// DEBUG AND VALIDATION
// ============================================================================

// Print kernel launch parameters (debug mode only)
#ifdef FLASHMOE_DEBUG
#define DEBUG_PRINT_LAUNCH_PARAMS 1
#else
#define DEBUG_PRINT_LAUNCH_PARAMS 0
#endif

// Enable bounds checking (reduces performance, use for debugging only)
#ifdef FLASHMOE_BOUNDS_CHECK
#define ENABLE_BOUNDS_CHECK 1
#else
#define ENABLE_BOUNDS_CHECK 0
#endif

#endif  // FLASHMOE_BUILD_CONFIG_H

/**
 * CHANGELOG:
 * 
 * October 12, 2025 - CRITICAL FIX
 * - Changed NUM_WARPS_PER_BLOCK from 4 to 12
 * - This fixes 0.12× regression (measured) vs 1.7× expected speedup
 * - Root cause: 128 threads (4 warps) called wrong kernel configuration
 * - Correct: 384 threads (12 warps) enables 3-warpgroup specialization
 * 
 * Performance impact:
 * - Before: block=(128,1,1) → 0.12× speedup (8-29× slower than PyTorch)
 * - After:  block=(384,1,1) → 1.3-1.7× speedup expected
 */

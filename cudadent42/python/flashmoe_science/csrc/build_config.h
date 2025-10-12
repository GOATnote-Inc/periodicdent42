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
// WARP AND BLOCK CONFIGURATION (8 warps = 256 threads for L4)
// ============================================================================

constexpr int WARP_SIZE = 32;

// *** L4 GPU CONFIGURATION: 8 warps = 256 threads (for 48KB shared memory) ***
// L4 has only 48KB shared memory, so we use 2 warpgroups instead of 3
constexpr int NUM_WARPS_PER_BLOCK = 8;  // L4: 2 warpgroups × 4 warps each

constexpr int NUM_WARPS_PER_WARPGROUP = 4;
constexpr int NUM_WARPGROUPS = 2;  // Reduced from 3 for L4

// Total threads per block (256 for L4, 384 for H100)
constexpr int THREADS_PER_BLOCK = NUM_WARPS_PER_BLOCK * WARP_SIZE;  // = 256

// Verify configuration at compile time
static_assert(THREADS_PER_BLOCK == 256 || THREADS_PER_BLOCK == 384, 
              "THREADS_PER_BLOCK must be 256 (L4) or 384 (H100)");
static_assert(NUM_WARPS_PER_BLOCK == NUM_WARPGROUPS * NUM_WARPS_PER_WARPGROUP,
              "Warp configuration inconsistent");

// ============================================================================
// TILE SIZES (Memory hierarchy optimization)
// ============================================================================

// Tile dimensions for attention computation (L4 GPU configuration)
// L4 has 48KB shared memory limit, so we use 64×64 tiles
constexpr int TILE_SIZE_M = 64;  // Query tile size (rows)
constexpr int TILE_SIZE_N = 64;  // Key/Value tile size (rows)  
constexpr int TILE_SIZE_K = 64;  // Head dimension (columns)

// Verify tiles don't exceed shared memory limits
// Shared memory usage per block (L4 configuration):
// - smem_Q: TILE_SIZE_M × TILE_SIZE_K × 2 bytes (FP16) = 8KB
// - smem_K: TILE_SIZE_N × TILE_SIZE_K × 2 bytes (FP16) = 8KB
// - smem_V: TILE_SIZE_N × TILE_SIZE_K × 2 bytes (FP16) = 8KB
// - smem_S: TILE_SIZE_M × TILE_SIZE_N × 4 bytes (FP32) = 16KB
// Total: ~40KB (fits in L4's 48KB limit)
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
 * October 12, 2025 (Session N+7B) - L4 GPU CONFIGURATION
 * - Changed NUM_WARPS_PER_BLOCK from 12 to 8 (for L4 48KB shared memory limit)
 * - Changed NUM_WARPGROUPS from 3 to 2
 * - Reduced TILE_SIZE_M/N/K from 128 to 64
 * - Shared memory usage: 160KB → 40KB (fits in L4's 48KB limit)
 * 
 * October 12, 2025 (Session N) - H100 CONFIGURATION
 * - Changed NUM_WARPS_PER_BLOCK from 4 to 12
 * - This fixed 0.12× regression (measured) vs 1.7× expected speedup
 * - Root cause: 128 threads (4 warps) called wrong kernel configuration
 * - Correct: 384 threads (12 warps) enables 3-warpgroup specialization
 * 
 * Note: H100 configuration (384 threads, 128 tiles) requires 160KB shared memory
 * and does not work on L4 GPU (48KB limit). Use this L4 configuration instead.
 */

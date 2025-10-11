#pragma once

// Host builds won't define __CUDA_ARCH__
#ifndef __CUDA_ARCH__
#define __CUDA_ARCH__ 0
#endif

// ---- Architecture feature flags ----
#define HAS_CP_ASYNC   (__CUDA_ARCH__ >= 800)
#define HAS_WGMMA      (__CUDA_ARCH__ >= 900)
#define HAS_BF16_SM80  (__CUDA_ARCH__ >= 800)
#define HAS_FP8_SM90   (__CUDA_ARCH__ >= 900)

// ---- Tile presets (compile-time; switch via -DFA_TILE_PRESET=...) ----
#ifndef FA_TILE_PRESET
#define FA_TILE_PRESET 0  // 0: t4_safe, 1: ampere_balanced
#endif

#if FA_TILE_PRESET == 0
  // t4_safe
  #define TILE_SIZE_M 32
  #define TILE_SIZE_N 64
  #define TILE_SIZE_K 64
#elif FA_TILE_PRESET == 1
  // ampere_balanced
  #define TILE_SIZE_M 64
  #define TILE_SIZE_N 128
  #define TILE_SIZE_K 64
#else
  #error "Unknown FA_TILE_PRESET"
#endif

// WARP and block geometry (12 warps = 3 warpgroups)
#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif

#ifndef NUM_WARPS_PER_WARPGROUP
#define NUM_WARPS_PER_WARPGROUP 4
#endif

#ifndef THREADS_PER_BLOCK
#define THREADS_PER_BLOCK (12 * WARP_SIZE)
#endif

// Conservative SMEM ceiling for T4 (48KB leaves headroom)
#ifndef FA_SMEM_BUDGET_BYTES
#define FA_SMEM_BUDGET_BYTES 49152
#endif

// Utility macro to silence unused vars in templated code
#define FA_UNUSED(x) ((void)(x))


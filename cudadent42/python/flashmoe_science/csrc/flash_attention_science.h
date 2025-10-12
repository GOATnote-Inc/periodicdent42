#pragma once

// FlashAttention-Science CUDA kernel header
// Device-only code - no torch/ATen includes

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// Only include BF16 on SM80+ to avoid host/device compilation issues
#ifndef FLASHMOE_DTYPE_FP16_ONLY
#include <cuda_bf16.h>
#endif

// Note: Kernel-specific constants (WARP_SIZE, TILE_SIZE_*, etc.) are defined
// as constexpr in each .cu file to avoid preprocessor conflicts with template
// parameter names. Do not #define them here.


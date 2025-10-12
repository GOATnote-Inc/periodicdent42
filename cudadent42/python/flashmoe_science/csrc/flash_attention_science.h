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

// Constants for warp specialization
#define WARP_SIZE 32
#define NUM_WARPS_PER_WARPGROUP 4
#define THREADS_PER_BLOCK 128

// Include the core kernel implementation
#include "flash_attention_core.h"


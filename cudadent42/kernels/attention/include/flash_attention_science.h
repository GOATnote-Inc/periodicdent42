/**
 * FlashAttention-Science: FA4-style warp specialization for scientific AI
 * 
 * Key optimizations:
 * 1. Warp specialization (3 warpgroups: MMA, Softmax, Correction)
 * 2. Async memory pipelines (overlap compute + memory)
 * 3. FP8/BF16 mixed precision (Hopper GPUs)
 * 4. Periodic pattern-aware tiling (domain-specific)
 * 5. Online softmax algorithm (numerical stability)
 * 
 * Performance targets:
 * - 2x speedup vs PyTorch SDPA
 * - >90% SM occupancy
 * - >80% memory bandwidth utilization
 * 
 * Hardware: NVIDIA H100 (Hopper), A100 (Ampere) compatible
 * 
 * References:
 * - FlashAttention-4 (2025): Warp specialization pattern
 * - FlashAttention-2 (2023): IO-aware tiling
 * 
 * @author GOATnote Autonomous Research Lab Initiative
 * @date 2025-10-11
 */

#ifndef FLASH_ATTENTION_SCIENCE_H
#define FLASH_ATTENTION_SCIENCE_H

#include <cuda_runtime.h>
#include <cuda_fp16.h>

// Only include BF16 on SM80+ to avoid host/device compilation issues
#if !defined(FLASHMOE_DTYPE_FP16_ONLY)
#include <cuda_bf16.h>
#endif

#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 900  // Hopper (H100)
#include <cuda_fp8.h>
#endif
#endif

#include <cstdint>
#include "build_config.h"

namespace flashmoe {

// NUM_WARPGROUPS is derived, not in build_config.h
constexpr int NUM_WARPGROUPS = 3;  // MMA, Softmax, Correction

// Note: WARP_SIZE, NUM_WARPS_PER_WARPGROUP, THREADS_PER_BLOCK,
// and TILE_SIZE_* are now defined in build_config.h as #defines

/**
 * FlashAttention-Science forward pass.
 * 
 * Computes attention(Q, K, V) = softmax(Q @ K^T / sqrt(d)) @ V
 * with O(n) memory complexity using tiling.
 * 
 * @param Q Query tensor [batch, num_heads, seq_len, head_dim]
 * @param K Key tensor [batch, num_heads, seq_len, head_dim]
 * @param V Value tensor [batch, num_heads, seq_len, head_dim]
 * @param O Output tensor [batch, num_heads, seq_len, head_dim]
 * @param softmax_lse Log-sum-exp for backward pass [batch, num_heads, seq_len]
 * @param batch_size Batch size
 * @param num_heads Number of attention heads
 * @param seq_len Sequence length
 * @param head_dim Dimension per head
 * @param softmax_scale Softmax scale (typically 1/sqrt(head_dim))
 * @param causal Whether to apply causal masking
 */
template<typename T>
void flash_attention_forward(
    const T* Q,
    const T* K,
    const T* V,
    T* O,
    float* softmax_lse,
    const int batch_size,
    const int num_heads,
    const int seq_len,
    const int head_dim,
    const float softmax_scale,
    const bool causal
);

/**
 * FlashAttention-Science backward pass.
 * 
 * Computes gradients with respect to Q, K, V.
 * 
 * @param dO Gradient of output [batch, num_heads, seq_len, head_dim]
 * @param Q Query tensor from forward pass
 * @param K Key tensor from forward pass
 * @param V Value tensor from forward pass
 * @param O Output from forward pass
 * @param softmax_lse Log-sum-exp from forward pass
 * @param dQ Gradient of query (output)
 * @param dK Gradient of key (output)
 * @param dV Gradient of value (output)
 * @param batch_size Batch size
 * @param num_heads Number of attention heads
 * @param seq_len Sequence length
 * @param head_dim Dimension per head
 * @param softmax_scale Softmax scale
 * @param causal Whether causal masking was used
 */
template<typename T>
void flash_attention_backward(
    const T* dO,
    const T* Q,
    const T* K,
    const T* V,
    const T* O,
    const float* softmax_lse,
    T* dQ,
    T* dK,
    T* dV,
    const int batch_size,
    const int num_heads,
    const int seq_len,
    const int head_dim,
    const float softmax_scale,
    const bool causal
);

}  // namespace flashmoe

#endif  // FLASH_ATTENTION_SCIENCE_H


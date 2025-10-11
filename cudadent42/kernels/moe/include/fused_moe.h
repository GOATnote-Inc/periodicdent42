/**
 * Fused Mixture of Experts: Single-kernel dispatch + GEMM + combine
 * 
 * Key optimizations:
 * 1. Radix sort for efficient token grouping by expert
 * 2. FP8 GEMM for expert computation (Hopper GPUs)
 * 3. Fused dispatch-compute-combine (reduce HBM traffic 3-5x)
 * 4. Load balancing awareness (auxiliary loss gradients)
 * 5. Expert-parallel batching (coalesced memory access)
 * 
 * Performance targets:
 * - 4x speedup vs unfused PyTorch MoE (256 experts)
 * - 50% memory reduction vs baseline
 * - >85% arithmetic intensity (FLOPs / bandwidth)
 * 
 * Hardware: NVIDIA H100 (Hopper), A100 (Ampere) compatible
 * 
 * References:
 * - DeepSeek-V3 (2024): MoE dispatch optimization
 * - SGLang (2025): Fused MoE kernels
 * 
 * @author GOATnote Autonomous Research Lab Initiative
 * @date 2025-10-11
 */

#ifndef FUSED_MOE_H
#define FUSED_MOE_H

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 900  // Hopper (H100)
#include <cuda_fp8.h>
#endif
#endif

#include <cstdint>

namespace flashmoe {

// Compile-time configuration
constexpr int MAX_EXPERTS = 256;
constexpr int RADIX_BITS = 4;  // 16-way radix sort
constexpr int RADIX_SIZE = (1 << RADIX_BITS);

// Tile sizes for expert GEMM
constexpr int MOE_TILE_M = 64;
constexpr int MOE_TILE_N = 64;
constexpr int MOE_TILE_K = 32;

/**
 * Fused MoE forward pass.
 * 
 * Performs token dispatching, expert computation, and weighted combining
 * in a single kernel to minimize memory traffic.
 * 
 * Algorithm:
 * 1. Radix sort tokens by top-k expert assignments
 * 2. Compute expert boundaries for batched GEMM
 * 3. Process each expert with FP8 GEMM
 * 4. Weighted combine expert outputs
 * 
 * @param tokens Input tokens [batch, seq_len, hidden_dim]
 * @param expert_weights Expert weights [num_experts, hidden_dim, expert_dim]
 * @param routing_weights Routing probabilities [batch*seq, num_experts]
 * @param output Output tokens [batch, seq_len, hidden_dim]
 * @param batch_size Batch size
 * @param seq_len Sequence length
 * @param hidden_dim Hidden dimension
 * @param num_experts Number of experts
 * @param expert_dim Expert hidden dimension
 * @param top_k Number of experts to activate per token
 */
template<typename T>
void fused_moe_forward(
    const T* tokens,
    const T* expert_weights,
    const float* routing_weights,
    T* output,
    const int batch_size,
    const int seq_len,
    const int hidden_dim,
    const int num_experts,
    const int expert_dim,
    const int top_k
);

/**
 * Fused MoE backward pass.
 * 
 * Computes gradients with respect to tokens, expert weights, and routing weights.
 * 
 * @param grad_output Gradient of output [batch, seq_len, hidden_dim]
 * @param tokens Input tokens from forward pass
 * @param expert_weights Expert weights from forward pass
 * @param routing_weights Routing weights from forward pass
 * @param expert_assignments Expert assignments from forward pass
 * @param grad_tokens Gradient of tokens (output)
 * @param grad_expert_weights Gradient of expert weights (output)
 * @param grad_routing_weights Gradient of routing weights (output)
 * @param batch_size Batch size
 * @param seq_len Sequence length
 * @param hidden_dim Hidden dimension
 * @param num_experts Number of experts
 * @param expert_dim Expert hidden dimension
 * @param top_k Number of experts per token
 */
template<typename T>
void fused_moe_backward(
    const T* grad_output,
    const T* tokens,
    const T* expert_weights,
    const float* routing_weights,
    const int* expert_assignments,
    T* grad_tokens,
    T* grad_expert_weights,
    float* grad_routing_weights,
    const int batch_size,
    const int seq_len,
    const int hidden_dim,
    const int num_experts,
    const int expert_dim,
    const int top_k
);

/**
 * Radix sort tokens by expert assignment.
 * 
 * Uses shared memory radix sort with 4 passes (16-way radix).
 * 
 * @param routing_weights Routing probabilities [batch*seq, num_experts]
 * @param expert_ids Sorted expert IDs (output) [batch*seq]
 * @param token_indices Sorted token indices (output) [batch*seq]
 * @param expert_boundaries Expert boundaries for GEMM (output) [num_experts+1]
 * @param batch_size Batch size
 * @param seq_len Sequence length
 * @param num_experts Number of experts
 * @param top_k Number of experts per token
 */
void radix_sort_by_expert(
    const float* routing_weights,
    int* expert_ids,
    int* token_indices,
    int* expert_boundaries,
    const int batch_size,
    const int seq_len,
    const int num_experts,
    const int top_k
);

// Explicit template instantiations
extern template void fused_moe_forward<__nv_bfloat16>(...);
extern template void fused_moe_forward<half>(...);

extern template void fused_moe_backward<__nv_bfloat16>(...);
extern template void fused_moe_backward<half>(...);

}  // namespace flashmoe

#endif  // FUSED_MOE_H


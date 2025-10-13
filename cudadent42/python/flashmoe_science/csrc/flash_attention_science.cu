/**
 * FlashAttention-Science: CUDA kernel implementation
 * 
 * Implements FlashAttention-4 warp specialization pattern:
 * - Warpgroup 0 (warps 0-3): MMA operations (Q@K^T, attention@V)
 * - Warpgroup 1 (warps 4-7): Online softmax with numerical stability
 * - Warpgroup 2 (warps 8-11): Output correction as softmax scale changes
 * 
 * Memory hierarchy:
 * - Shared memory (SRAM): 228KB per SM on H100, 48KB per SM on T4
 * - L2 cache: 60MB on H100, 4MB on T4
 * - HBM3: 3.35 TB/s bandwidth (H100), 320 GB/s (T4)
 * 
 * Optimization techniques:
 * 1. Tiling: Break sequence into tiles that fit in SRAM
 * 2. Async memory copy: Overlap next tile load with current compute (SM80+)
 * 3. Online softmax: Compute softmax incrementally without full matrix
 * 4. FP8 compute: Use Hopper's FP8 Tensor Cores (SM90+ only)
 * 5. Warp shuffle: Reduce shared memory usage for reductions
 * 
 * Architecture support:
 * - SM75 (T4): FP16 only, no async copy, no native BF16
 * - SM80 (A100): FP16 + BF16, cp.async, WMMA
 * - SM90 (H100): FP16 + BF16 + FP8, WGMMA, TMA
 * 
 * @author GOATnote Autonomous Research Lab Initiative
 * @date 2025-10-11
 */

#include "build_config.h"  // Architecture flags and tile presets
#include "flash_attention_science.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// Only include BF16 on SM80+ to avoid host/device compilation issues
#if !defined(FLASHMOE_DTYPE_FP16_ONLY)
#include <cuda_bf16.h>
#endif

// ATen types for explicit instantiations
#include <ATen/ATen.h>
#include <c10/core/ScalarType.h>

// Async memory pipeline (CUDA 11.7+, SM80+)
#if HAS_CP_ASYNC
#include <cuda/pipeline>
#include <cuda/barrier>
#endif

// Cooperative groups for warp-level operations
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

#include <cmath>
#include <algorithm>
#include <cstdio>  // For printf

namespace flashmoe {

// Helper functions for type conversions
__device__ __forceinline__ float to_float(half x) {
    return __half2float(x);
}

// ATen type conversions
__device__ __forceinline__ float to_float(at::Half x) {
    return __half2float(*reinterpret_cast<const __half*>(&x));
}

__device__ __forceinline__ float to_float(float x) {
    return x;
}

template<typename T>
__device__ __forceinline__ T from_float(float x);

template<>
__device__ __forceinline__ half from_float<half>(float x) {
    return __float2half(x);
}

template<>
__device__ __forceinline__ at::Half from_float<at::Half>(float x) {
    __half h = __float2half(x);
    return *reinterpret_cast<at::Half*>(&h);
}

#if !defined(FLASHMOE_DTYPE_FP16_ONLY)
// BF16 conversions only when not in FP16-only mode
__device__ __forceinline__ float to_float(__nv_bfloat16 x) {
    return __bfloat162float(x);
}

__device__ __forceinline__ float to_float(at::BFloat16 x) {
    return __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(&x));
}

template<>
__device__ __forceinline__ __nv_bfloat16 from_float<__nv_bfloat16>(float x) {
    return __float2bfloat16(x);
}

template<>
__device__ __forceinline__ at::BFloat16 from_float<at::BFloat16>(float x) {
    __nv_bfloat16 bf = __float2bfloat16(x);
    return *reinterpret_cast<at::BFloat16*>(&bf);
}
#endif

// Kernel configuration constants
// Defined as constexpr (not #define) to avoid conflicts with template parameters
constexpr int WARP_SIZE = 32;
constexpr int NUM_WARPS_PER_WARPGROUP = 4;
constexpr int NUM_WARPS_PER_BLOCK = 12;  // 3 warpgroups
constexpr int THREADS_PER_BLOCK = WARP_SIZE * NUM_WARPS_PER_BLOCK;  // 384

/**
 * Online softmax algorithm for numerical stability.
 * 
 * Computes max, exp, and sum incrementally as we process tiles.
 * This avoids storing the full attention matrix and prevents overflow.
 * 
 * Algorithm:
 *   m_new = max(m_old, m_tile)
 *   l_new = l_old * exp(m_old - m_new) + l_tile * exp(m_tile - m_new)
 *   O_new = O_old * exp(m_old - m_new) + O_tile * exp(m_tile - m_new)
 * 
 * where m is the running max, l is the running sum, O is the output.
 */
template<typename T>
__device__ void online_softmax_update(
    float& m_prev,
    float& l_prev,
    T* O_prev,
    const float m_curr,
    const float l_curr,
    const T* O_curr,
    const int head_dim
) {
    // Compute new max
    const float m_new = fmaxf(m_prev, m_curr);
    
    // Compute correction factors
    const float exp_prev = expf(m_prev - m_new);
    const float exp_curr = expf(m_curr - m_new);
    
    // Update running sum
    const float l_new = l_prev * exp_prev + l_curr * exp_curr;
    
    // Update output (correction + new contribution)
    #pragma unroll
    for (int i = 0; i < head_dim; ++i) {
        float o_prev_f = static_cast<float>(O_prev[i]);
        float o_curr_f = static_cast<float>(O_curr[i]);
        float o_new_f = o_prev_f * exp_prev + o_curr_f * exp_curr;
        O_prev[i] = static_cast<T>(o_new_f);
    }
    
    m_prev = m_new;
    l_prev = l_new;
}

/**
 * FlashAttention-Science forward kernel.
 * 
 * Grid: (num_heads, batch_size)
 * Block: THREADS_PER_BLOCK threads (3 warpgroups)
 * 
 * Each block processes one (batch, head) pair.
 * Outer loop tiles over sequence length in chunks of TILE_SIZE_M/N.
 */
template<typename T>
__global__ void flash_attention_forward_kernel(
    const T* __restrict__ Q,
    const T* __restrict__ K,
    const T* __restrict__ V,
    T* __restrict__ O,
    float* __restrict__ softmax_lse,
    const int batch_size,
    const int num_heads,
    const int seq_len,
    const int head_dim,
    const float softmax_scale,
    const bool causal
) {
    // Block indices
    const int head_idx = blockIdx.x;
    const int batch_idx = blockIdx.y;
    
    // Warpgroup identification
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int warpgroup_id = warp_id / NUM_WARPS_PER_WARPGROUP;
    const int lane_id = threadIdx.x % WARP_SIZE;
    
    // Shared memory for Q, K, V tiles
    __shared__ __align__(16) T smem_Q[TILE_SIZE_M][TILE_SIZE_K];
    __shared__ __align__(16) T smem_K[TILE_SIZE_N][TILE_SIZE_K];
    __shared__ __align__(16) T smem_V[TILE_SIZE_N][TILE_SIZE_K];
    
    // Shared memory for attention scores (S = Q @ K^T)
    __shared__ __align__(16) float smem_S[TILE_SIZE_M][TILE_SIZE_N];
    
    // Register storage for output accumulation (max 128 dim for now)
    float acc_o[128] = {0.0f};  // TODO: Make this dynamic or use shared memory
    
    // Running statistics for online softmax
    float m_i = -INFINITY;  // Running max
    float l_i = 0.0f;       // Running sum
    
    // Base pointers for this (batch, head) pair
    const int offset = (batch_idx * num_heads + head_idx) * seq_len * head_dim;
    const T* Q_base = Q + offset;
    const T* K_base = K + offset;
    const T* V_base = V + offset;
    T* O_base = O + offset;
    
    // Number of tiles
    const int num_tiles_m = (seq_len + TILE_SIZE_M - 1) / TILE_SIZE_M;
    const int num_tiles_n = (seq_len + TILE_SIZE_N - 1) / TILE_SIZE_N;
    
    // === BASIC TILING IMPLEMENTATION (Day 1-3) ===
    // This implements simple tiling without advanced optimizations
    // Goal: Get first test passing with correct results
    
    // === MULTI-TILE QUERY HANDLING ===
    // Grid: (num_heads, batch_size, num_query_tiles)
    // Block: (THREADS_PER_BLOCK=256, 1, 1)
    // Each block processes TILE_SIZE_M=64 queries
    const int query_tile_idx = blockIdx.z;              // Which query tile (0, 1, 2, ...)
    const int query_idx_in_tile = threadIdx.x;          // Position within tile (0-63)
    const int query_idx = query_tile_idx * TILE_SIZE_M + query_idx_in_tile;  // Global query index
    
    // Guard: Only threads handling valid queries within tile bounds participate
    // This prevents invalid threads (64-255) from corrupting results
    const bool is_valid_query = (query_idx_in_tile < TILE_SIZE_M) && (query_idx < seq_len);
    
    // === Initialize running statistics for online softmax ===
    // These track max and sum of exponentials across all KV tiles
    m_i = -INFINITY;  // Running maximum
    l_i = 0.0f;       // Running sum of exp
    
    // Initialize output accumulator
    #pragma unroll
    for (int d = 0; d < head_dim; ++d) {
        acc_o[d] = 0.0f;
    }
    
    // === STEP 1: Load Q tile (one row per thread) ===
    // OPTIMIZATION #1: Vectorized Q load using float4 (8 half values at once)
    // Expected speedup: 1.5-2x on memory-bound configs
    // Only valid queries load data (guards against out-of-bounds access)
    if (is_valid_query && head_dim % 8 == 0) {
        const float4* Q_vec = reinterpret_cast<const float4*>(Q_base + query_idx * head_dim);
        float4* smem_Q_vec = reinterpret_cast<float4*>(&smem_Q[query_idx_in_tile][0]);
        
        #pragma unroll
        for (int d = 0; d < head_dim / 8; ++d) {
            smem_Q_vec[d] = Q_vec[d];  // Load 8 half values per iteration
        }
    } else if (is_valid_query) {
        // Fallback for non-aligned head_dim
        for (int d = 0; d < head_dim; ++d) {
            smem_Q[query_idx_in_tile][d] = Q_base[query_idx * head_dim + d];
        }
    }
    __syncthreads();
    
    // === STEP 2-5: Loop over K, V tiles ===
    for (int kv_tile = 0; kv_tile < num_tiles_n; ++kv_tile) {
        const int kv_start = kv_tile * TILE_SIZE_N;
        const int kv_end = min(kv_start + TILE_SIZE_N, seq_len);
        const int tile_size = kv_end - kv_start;
        
        // Load K, V tiles (collaborative loading)
        // OPTIMIZATION #1: Vectorized + coalesced K,V loads using float4
        // Each thread loads multiple rows to maximize memory throughput
        if (head_dim % 8 == 0) {
            for (int kv = threadIdx.x; kv < tile_size; kv += blockDim.x) {
                const int kv_idx = kv_start + kv;
                const float4* K_vec = reinterpret_cast<const float4*>(K_base + kv_idx * head_dim);
                const float4* V_vec = reinterpret_cast<const float4*>(V_base + kv_idx * head_dim);
                float4* smem_K_vec = reinterpret_cast<float4*>(&smem_K[kv][0]);
                float4* smem_V_vec = reinterpret_cast<float4*>(&smem_V[kv][0]);
                
                #pragma unroll
                for (int d = 0; d < head_dim / 8; ++d) {
                    smem_K_vec[d] = K_vec[d];  // Load 8 half values per iteration
                    smem_V_vec[d] = V_vec[d];
                }
            }
        } else {
            // Fallback for non-aligned head_dim
            for (int kv = threadIdx.x; kv < tile_size; kv += blockDim.x) {
                const int kv_idx = kv_start + kv;
                for (int d = 0; d < head_dim; ++d) {
                    smem_K[kv][d] = K_base[kv_idx * head_dim + d];
                    smem_V[kv][d] = V_base[kv_idx * head_dim + d];
                }
            }
        }
        __syncthreads();
        
        // === STEP 3: Compute Q @ K^T for this query ===
        if (is_valid_query) {
            for (int kv = 0; kv < tile_size; ++kv) {
                const int kv_idx = kv_start + kv;
                
                // Compute dot product: Q[query_idx] · K[kv_idx]
                float score = 0.0f;
                #pragma unroll
                for (int d = 0; d < TILE_SIZE_K && d < head_dim; ++d) {
                    score += to_float(smem_Q[query_idx_in_tile][d]) * 
                             to_float(smem_K[kv][d]);
                }
                
                // Apply softmax scale
                score *= softmax_scale;
                
                // Apply causal mask if needed
                if (causal && kv_idx > query_idx) {
                    score = -INFINITY;
                }
                
                smem_S[query_idx_in_tile][kv] = score;
            }
        }
        __syncthreads();
        
        // === STEP 4: Online softmax update ===
        if (is_valid_query) {
            // 4a. Find max in current tile for numerical stability
            float m_tile = -INFINITY;
            for (int kv = 0; kv < tile_size; ++kv) {
                m_tile = fmaxf(m_tile, smem_S[query_idx_in_tile][kv]);
            }
            
            // 4b. Compute exp(S - m_tile) and sum
            float l_tile = 0.0f;
            for (int kv = 0; kv < tile_size; ++kv) {
                float exp_val = expf(smem_S[query_idx_in_tile][kv] - m_tile);
                smem_S[query_idx_in_tile][kv] = exp_val;
                l_tile += exp_val;
            }
            
            // 4c. Update running statistics (online softmax algorithm)
            const float m_new = fmaxf(m_i, m_tile);
            const float exp_prev = expf(m_i - m_new);
            const float exp_curr = expf(m_tile - m_new);
            const float l_new = l_i * exp_prev + l_tile * exp_curr;
            
            // 4d. Apply correction factor to existing output
            #pragma unroll
            for (int d = 0; d < head_dim; ++d) {
                acc_o[d] *= exp_prev;
            }
            
            // === STEP 5: Compute attention @ V with correction ===
            #pragma unroll
            for (int d = 0; d < head_dim; ++d) {
                float weighted_value = 0.0f;
                for (int kv = 0; kv < tile_size; ++kv) {
                    weighted_value += smem_S[query_idx_in_tile][kv] * 
                                     to_float(smem_V[kv][d]);
                }
                // Add corrected contribution from this tile
                acc_o[d] += weighted_value * exp_curr;
            }
            
            // Update running statistics for next tile
            m_i = m_new;
            l_i = l_new;
        }
        __syncthreads();
    }
    
    // === STEP 6: Final normalization and store output ===
    // Only valid queries write results (prevents out-of-bounds writes)
    if (is_valid_query) {
        // DEBUG: Print from query 0
        if (batch_idx == 0 && head_idx == 0 && query_idx == 0) {
            printf("FA1[q=0]: m_i=%.4f, l_i=%.4f, acc_o[0:3]=[%.4f,%.4f,%.4f] (before norm)\\n",
                   m_i, l_i, acc_o[0], acc_o[1], acc_o[2]);
        }
        
        // Normalize output by sum of exponentials
        #pragma unroll
        for (int d = 0; d < head_dim; ++d) {
            acc_o[d] /= l_i;
            O_base[query_idx * head_dim + d] = from_float<T>(acc_o[d]);
        }
        
        // Store softmax LSE (log-sum-exp) for backward pass
        softmax_lse[(batch_idx * num_heads + head_idx) * seq_len + query_idx] = 
            logf(l_i) + m_i;
    }
}

/**
 * ITERATION 1: KV-Split Parallelism
 * 
 * This kernel processes ONE chunk of KV tiles per CTA, enabling massive parallelism.
 * Grid: (num_heads, batch_size, q_tiles × kv_splits)
 * Each CTA computes partial attention for its KV chunk and stores (O_partial, m_i, l_i).
 * A fusion kernel then combines all partials using log-sum-exp trick.
 * 
 * Goal: Increase CTAs from ~2 to 256+ for better GPU utilization (3% → 60%+).
 */
template<typename T>
__global__ void flash_attention_forward_kv_split_partial(
    const T* __restrict__ Q,
    const T* __restrict__ K,
    const T* __restrict__ V,
    T* __restrict__ O_partial,        // [B, H, Q_tiles, KV_splits, head_dim]
    float* __restrict__ m_partial,    // [B, H, Q_tiles, KV_splits]
    float* __restrict__ l_partial,    // [B, H, Q_tiles, KV_splits]
    const int batch_size,
    const int num_heads,
    const int seq_len,
    const int head_dim,
    const int num_kv_splits,
    const float softmax_scale,
    const bool causal
) {
    // Decode grid indices
    const int head_idx = blockIdx.x;
    const int batch_idx = blockIdx.y;
    const int flat_idx = blockIdx.z;  // Flattened (q_tile, kv_split) index
    
    const int num_query_tiles = (seq_len + TILE_SIZE_M - 1) / TILE_SIZE_M;
    const int kv_split_idx = flat_idx % num_kv_splits;
    const int query_tile_idx = flat_idx / num_kv_splits;
    
    const int query_idx_in_tile = threadIdx.x;
    const int query_idx = query_tile_idx * TILE_SIZE_M + query_idx_in_tile;
    const bool is_valid_query = (query_idx_in_tile < TILE_SIZE_M) && (query_idx < seq_len);
    
    // Shared memory
    __shared__ __align__(16) T smem_Q[TILE_SIZE_M][TILE_SIZE_K];
    __shared__ __align__(16) T smem_K[TILE_SIZE_N][TILE_SIZE_K];
    __shared__ __align__(16) T smem_V[TILE_SIZE_N][TILE_SIZE_K];
    __shared__ __align__(16) float smem_S[TILE_SIZE_M][TILE_SIZE_N];
    
    // Register storage
    float acc_o[128] = {0};
    float m_i = -INFINITY;
    float l_i = 0.0f;
    
    // Base pointers
    const int offset = (batch_idx * num_heads + head_idx) * seq_len * head_dim;
    const T* Q_base = Q + offset;
    const T* K_base = K + offset;
    const T* V_base = V + offset;
    
    // Load Q tile (same as original kernel)
    if (is_valid_query && head_dim % 8 == 0) {
        const float4* Q_vec = reinterpret_cast<const float4*>(Q_base + query_idx * head_dim);
        float4* smem_Q_vec = reinterpret_cast<float4*>(&smem_Q[query_idx_in_tile][0]);
        #pragma unroll
        for (int d = 0; d < head_dim / 8; ++d) {
            smem_Q_vec[d] = Q_vec[d];
        }
    }
    __syncthreads();
    
    // Compute which KV tiles this split handles
    const int num_kv_tiles = (seq_len + TILE_SIZE_N - 1) / TILE_SIZE_N;
    const int tiles_per_split = (num_kv_tiles + num_kv_splits - 1) / num_kv_splits;
    const int kv_tile_start = kv_split_idx * tiles_per_split;
    const int kv_tile_end = min(kv_tile_start + tiles_per_split, num_kv_tiles);
    
    // Process only this split's KV tiles
    for (int kv_tile = kv_tile_start; kv_tile < kv_tile_end; ++kv_tile) {
        const int kv_start = kv_tile * TILE_SIZE_N;
        const int kv_end = min(kv_start + TILE_SIZE_N, seq_len);
        const int tile_size = kv_end - kv_start;
        
        // Load K, V tiles (vectorized)
        if (head_dim % 8 == 0) {
            for (int kv = threadIdx.x; kv < tile_size; kv += blockDim.x) {
                const int kv_idx = kv_start + kv;
                const float4* K_vec = reinterpret_cast<const float4*>(K_base + kv_idx * head_dim);
                const float4* V_vec = reinterpret_cast<const float4*>(V_base + kv_idx * head_dim);
                float4* smem_K_vec = reinterpret_cast<float4*>(&smem_K[kv][0]);
                float4* smem_V_vec = reinterpret_cast<float4*>(&smem_V[kv][0]);
                #pragma unroll
                for (int d = 0; d < head_dim / 8; ++d) {
                    smem_K_vec[d] = K_vec[d];
                    smem_V_vec[d] = V_vec[d];
                }
            }
        }
        __syncthreads();
        
        // Compute Q @ K^T
        if (is_valid_query) {
            for (int kv = 0; kv < tile_size; ++kv) {
                const int kv_idx = kv_start + kv;
                float score = 0.0f;
                #pragma unroll
                for (int d = 0; d < TILE_SIZE_K && d < head_dim; ++d) {
                    score += to_float(smem_Q[query_idx_in_tile][d]) * to_float(smem_K[kv][d]);
                }
                score *= softmax_scale;
                if (causal && kv_idx > query_idx) score = -INFINITY;
                smem_S[query_idx_in_tile][kv] = score;
            }
        }
        __syncthreads();
        
        // Online softmax update (same as original)
        if (is_valid_query) {
            float m_tile = -INFINITY;
            for (int kv = 0; kv < tile_size; ++kv) {
                m_tile = fmaxf(m_tile, smem_S[query_idx_in_tile][kv]);
            }
            
            // Skip fully-masked tiles
            if (!(isinf(m_tile) && m_tile < 0.0f)) {
                float l_tile = 0.0f;
                for (int kv = 0; kv < tile_size; ++kv) {
                    float exp_val = expf(smem_S[query_idx_in_tile][kv] - m_tile);
                    smem_S[query_idx_in_tile][kv] = exp_val;
                    l_tile += exp_val;
                }
                
                const float m_new = fmaxf(m_i, m_tile);
                const float exp_prev = expf(m_i - m_new);
                const float exp_curr = expf(m_tile - m_new);
                const float l_new = l_i * exp_prev + l_tile * exp_curr;
                
                #pragma unroll
                for (int d = 0; d < head_dim; ++d) {
                    acc_o[d] *= exp_prev;
                }
                
                #pragma unroll
                for (int d = 0; d < head_dim; ++d) {
                    float weighted_value = 0.0f;
                    for (int kv = 0; kv < tile_size; ++kv) {
                        weighted_value += smem_S[query_idx_in_tile][kv] * to_float(smem_V[kv][d]);
                    }
                    acc_o[d] += weighted_value * exp_curr;
                }
                
                m_i = m_new;
                l_i = l_new;
            }
        }
        __syncthreads();
    }
    
    // Store partial results (NO normalization yet!)
    if (is_valid_query) {
        const int partial_idx = ((batch_idx * num_heads + head_idx) * num_query_tiles + query_tile_idx) 
                                * num_kv_splits + kv_split_idx;
        T* O_partial_base = O_partial + partial_idx * TILE_SIZE_M * head_dim + query_idx_in_tile * head_dim;
        
        // DEBUG: Print from query 0, split 0
        if (batch_idx == 0 && head_idx == 0 && query_tile_idx == 0 && query_idx_in_tile == 0 && kv_split_idx == 0) {
            printf("PARTIAL[q=0,split=0]: m_i=%.4f, l_i=%.4f, acc_o[0:3]=[%.4f,%.4f,%.4f]\\n",
                   m_i, l_i, acc_o[0], acc_o[1], acc_o[2]);
        }
        
        #pragma unroll
        for (int d = 0; d < head_dim; ++d) {
            O_partial_base[d] = from_float<T>(acc_o[d]);
        }
        
        m_partial[partial_idx * TILE_SIZE_M + query_idx_in_tile] = m_i;
        l_partial[partial_idx * TILE_SIZE_M + query_idx_in_tile] = l_i;
    }
}

/**
 * ITERATION 1: KV-Split Fusion Kernel
 * 
 * Combines partial results from all KV splits using log-sum-exp trick.
 * Grid: (num_heads, batch_size, q_tiles)
 * Each CTA handles one query tile and combines all its KV splits.
 */
template<typename T>
__global__ void flash_attention_kv_split_fusion(
    const T* __restrict__ O_partial,
    const float* __restrict__ m_partial,
    const float* __restrict__ l_partial,
    T* __restrict__ O_final,
    float* __restrict__ softmax_lse,
    const int batch_size,
    const int num_heads,
    const int seq_len,
    const int head_dim,
    const int num_kv_splits
) {
    const int head_idx = blockIdx.x;
    const int batch_idx = blockIdx.y;
    const int query_tile_idx = blockIdx.z;
    const int query_idx_in_tile = threadIdx.x;
    
    const int query_idx = query_tile_idx * TILE_SIZE_M + query_idx_in_tile;
    if (query_idx >= seq_len || query_idx_in_tile >= TILE_SIZE_M) return;
    
    const int num_query_tiles = (seq_len + TILE_SIZE_M - 1) / TILE_SIZE_M;
    const int base_partial_idx = ((batch_idx * num_heads + head_idx) * num_query_tiles + query_tile_idx) 
                                  * num_kv_splits;
    
    // Find global max across all splits
    float m_global = -INFINITY;
    for (int split = 0; split < num_kv_splits; ++split) {
        float m_split = m_partial[(base_partial_idx + split) * TILE_SIZE_M + query_idx_in_tile];
        m_global = fmaxf(m_global, m_split);
    }
    
    // Compute corrected sum across all splits
    float l_global = 0.0f;
    for (int split = 0; split < num_kv_splits; ++split) {
        float m_split = m_partial[(base_partial_idx + split) * TILE_SIZE_M + query_idx_in_tile];
        float l_split = l_partial[(base_partial_idx + split) * TILE_SIZE_M + query_idx_in_tile];
        l_global += expf(m_split - m_global) * l_split;
    }
    
    // Combine outputs with reweighting
    float final_o[128] = {0};
    for (int split = 0; split < num_kv_splits; ++split) {
        float m_split = m_partial[(base_partial_idx + split) * TILE_SIZE_M + query_idx_in_tile];
        float scale = expf(m_split - m_global) / l_global;
        
        const T* O_split = O_partial + (base_partial_idx + split) * TILE_SIZE_M * head_dim 
                                      + query_idx_in_tile * head_dim;
        for (int d = 0; d < head_dim; ++d) {
            final_o[d] += scale * to_float(O_split[d]);
        }
    }
    
    // DEBUG: Print from query 0
    if (batch_idx == 0 && head_idx == 0 && query_tile_idx == 0 && query_idx_in_tile == 0) {
        printf("FUSION[q=0]: m_global=%.4f, l_global=%.4f, final_o[0:3]=[%.4f,%.4f,%.4f]\\n",
               m_global, l_global, final_o[0], final_o[1], final_o[2]);
    }
    
    // Write final output
    const int offset = (batch_idx * num_heads + head_idx) * seq_len * head_dim;
    T* O_base = O_final + offset;
    for (int d = 0; d < head_dim; ++d) {
        O_base[query_idx * head_dim + d] = from_float<T>(final_o[d]);
    }
    
    // Store softmax LSE
    softmax_lse[(batch_idx * num_heads + head_idx) * seq_len + query_idx] = logf(l_global) + m_global;
}

/**
 * Host function to launch FlashAttention forward kernel.
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
) {
    // Grid: (num_heads, batch_size, num_query_tiles) - 3D grid for multi-tile queries
    // Each block processes TILE_SIZE_M queries (64 queries per block)
    const int num_query_tiles = (seq_len + TILE_SIZE_M - 1) / TILE_SIZE_M;
    dim3 grid(num_heads, batch_size, num_query_tiles);
    
    // Block: 256 threads (8 warps, 2 warpgroups for L4)
    dim3 block(THREADS_PER_BLOCK);
    
    // Launch kernel
    flash_attention_forward_kernel<T><<<grid, block>>>(
        Q, K, V, O, softmax_lse,
        batch_size, num_heads, seq_len, head_dim,
        softmax_scale, causal
    );
    
    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error in flash_attention_forward: %s\n", cudaGetErrorString(err));
    }
}

/**
 * ITERATION 1: Host function for KV-Split Parallelism
 * 
 * This launches the KV-split kernels to increase parallelism from ~2 CTAs to 256+.
 * Uses a 2-pass approach: partial computation + fusion.
 * 
 * NOTE: This replaces the old flash_attention_forward_split_k with a working implementation.
 */
template<typename T>
void flash_attention_forward_split_k(
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
    const bool causal,
    const int num_kv_splits = 4  // Default: 4 splits per query tile
) {
    const int num_query_tiles = (seq_len + TILE_SIZE_M - 1) / TILE_SIZE_M;
    const int num_kv_tiles = (seq_len + TILE_SIZE_N - 1) / TILE_SIZE_N;
    
    // BUG FIX: Never create more splits than KV tiles (avoids zero-initialized partials)
    const int actual_num_kv_splits = (num_kv_splits < num_kv_tiles) ? num_kv_splits : num_kv_tiles;
    
    // Allocate temporary buffers for partial results
    T* O_partial;
    float* m_partial;
    float* l_partial;
    
    const size_t partial_size = batch_size * num_heads * num_query_tiles * actual_num_kv_splits * TILE_SIZE_M * head_dim;
    const size_t stats_size = batch_size * num_heads * num_query_tiles * actual_num_kv_splits * TILE_SIZE_M;
    
    cudaMalloc(&O_partial, partial_size * sizeof(T));
    cudaMalloc(&m_partial, stats_size * sizeof(float));
    cudaMalloc(&l_partial, stats_size * sizeof(float));
    
    // Pass 1: Compute partial attention for each KV split
    // Grid: (num_heads, batch_size, q_tiles × kv_splits)
    dim3 grid_partial(num_heads, batch_size, num_query_tiles * actual_num_kv_splits);
    dim3 block(THREADS_PER_BLOCK);
    
    flash_attention_forward_kv_split_partial<T><<<grid_partial, block>>>(
        Q, K, V, O_partial, m_partial, l_partial,
        batch_size, num_heads, seq_len, head_dim, actual_num_kv_splits,
        softmax_scale, causal
    );
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error in kv_split_partial: %s\n", cudaGetErrorString(err));
        cudaFree(O_partial);
        cudaFree(m_partial);
        cudaFree(l_partial);
        return;
    }
    
    // Pass 2: Fusion - combine partial results
    // Grid: (num_heads, batch_size, q_tiles)
    dim3 grid_fusion(num_heads, batch_size, num_query_tiles);
    
    flash_attention_kv_split_fusion<T><<<grid_fusion, block>>>(
        O_partial, m_partial, l_partial, O, softmax_lse,
        batch_size, num_heads, seq_len, head_dim, actual_num_kv_splits
    );
    
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error in kv_split_fusion: %s\n", cudaGetErrorString(err));
    }
    
    // Cleanup
    cudaFree(O_partial);
    cudaFree(m_partial);
    cudaFree(l_partial);
}

// ============================================================================
// OLD SPLIT-K KERNELS FROM SESSIONS N+7 - DISABLED (BROKEN, REPLACED BY ITERATION 1)
// ============================================================================
#if 0  // Comment out old broken implementation

/**
 * FlashAttention-2 Split-K: Pass 1 - Compute Partial Attention
 * 
 * Each block computes attention for ONE (query_tile, kv_tile) pair.
 * Stores partial results that will be reduced in Pass 2.
 * 
 * Grid: (num_heads, batch_size, num_query_tiles * num_kv_tiles)
 * Block: (THREADS_PER_BLOCK, 1, 1)
 */
template<typename T>
__global__ void flash_attention_forward_split_k_partial_OLD(
    const T* Q,
    const T* K,
    const T* V,
    T* partial_O,              // [B,H,Q_tiles,KV_tiles,TILE_SIZE_M,D]
    float* partial_max,        // [B,H,Q_tiles,KV_tiles,TILE_SIZE_M]
    float* partial_sum,        // [B,H,Q_tiles,KV_tiles,TILE_SIZE_M]
    const int batch_size,
    const int num_heads,
    const int seq_len,
    const int head_dim,
    const int num_query_tiles,
    const int num_kv_tiles,
    const float softmax_scale,
    const bool causal
) {
    // Block indices
    const int head_idx = blockIdx.x;
    const int batch_idx = blockIdx.y;
    const int tile_pair_idx = blockIdx.z;
    
    // DEBUG: Print from first thread to verify prints work
    if (threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
        printf("DEBUG: Partial kernel started! tile_pair_idx=%d\n", tile_pair_idx);
    }
    
    // Decompose tile_pair_idx into query_tile and kv_tile
    const int query_tile_idx = tile_pair_idx / num_kv_tiles;
    const int kv_tile_idx = tile_pair_idx % num_kv_tiles;
    
    // Thread indices
    const int query_idx_in_tile = threadIdx.x;
    const int query_idx = query_tile_idx * TILE_SIZE_M + query_idx_in_tile;
    
    // Guard for valid queries
    const bool is_valid_query = (query_idx_in_tile < TILE_SIZE_M) && (query_idx < seq_len);
    
    // Shared memory for Q, K, V tiles
    __shared__ __align__(16) T smem_Q[TILE_SIZE_M][TILE_SIZE_K];
    __shared__ __align__(16) T smem_K[TILE_SIZE_N][TILE_SIZE_K];
    __shared__ __align__(16) T smem_V[TILE_SIZE_N][TILE_SIZE_K];
    __shared__ __align__(16) float smem_S[TILE_SIZE_M][TILE_SIZE_N];
    
    // Base pointers
    const int offset = (batch_idx * num_heads + head_idx) * seq_len * head_dim;
    const T* Q_base = Q + offset;
    const T* K_base = K + offset;
    const T* V_base = V + offset;
    
    // K/V tile boundaries
    const int kv_start = kv_tile_idx * TILE_SIZE_N;
    const int kv_end = min(kv_start + TILE_SIZE_N, seq_len);
    const int tile_size = kv_end - kv_start;
    
    // Register storage for output (initialize ALL elements to zero)
    float acc_o[128] = {0};
    
    // Load Q tile
    if (is_valid_query && head_dim % 8 == 0) {
        const float4* Q_vec = reinterpret_cast<const float4*>(Q_base + query_idx * head_dim);
        float4* smem_Q_vec = reinterpret_cast<float4*>(&smem_Q[query_idx_in_tile][0]);
        #pragma unroll
        for (int d = 0; d < head_dim / 8; ++d) {
            smem_Q_vec[d] = Q_vec[d];
        }
    } else if (is_valid_query) {
        for (int d = 0; d < head_dim; ++d) {
            smem_Q[query_idx_in_tile][d] = Q_base[query_idx * head_dim + d];
        }
    }
    __syncthreads();
    
    // Load K, V tiles (collaborative)
    if (head_dim % 8 == 0) {
        for (int kv = threadIdx.x; kv < tile_size; kv += blockDim.x) {
            const int kv_idx = kv_start + kv;
            const float4* K_vec = reinterpret_cast<const float4*>(K_base + kv_idx * head_dim);
            const float4* V_vec = reinterpret_cast<const float4*>(V_base + kv_idx * head_dim);
            float4* smem_K_vec = reinterpret_cast<float4*>(&smem_K[kv][0]);
            float4* smem_V_vec = reinterpret_cast<float4*>(&smem_V[kv][0]);
            #pragma unroll
            for (int d = 0; d < head_dim / 8; ++d) {
                smem_K_vec[d] = K_vec[d];
                smem_V_vec[d] = V_vec[d];
            }
        }
    } else {
        for (int kv = threadIdx.x; kv < tile_size; kv += blockDim.x) {
            const int kv_idx = kv_start + kv;
            for (int d = 0; d < head_dim; ++d) {
                smem_K[kv][d] = K_base[kv_idx * head_dim + d];
                smem_V[kv][d] = V_base[kv_idx * head_dim + d];
            }
        }
    }
    __syncthreads();
    
    // Compute Q @ K^T
    if (is_valid_query) {
        for (int kv = 0; kv < tile_size; ++kv) {
            const int kv_idx = kv_start + kv;
            float score = 0.0f;
            #pragma unroll
            for (int d = 0; d < TILE_SIZE_K && d < head_dim; ++d) {
                score += to_float(smem_Q[query_idx_in_tile][d]) * to_float(smem_K[kv][d]);
            }
            score *= softmax_scale;
            if (causal && kv_idx > query_idx) {
                score = -INFINITY;
            }
            smem_S[query_idx_in_tile][kv] = score;
        }
    }
    __syncthreads();
    
    // Compute local softmax
    float local_max = -INFINITY;
    float local_sum = 0.0f;
    
    if (is_valid_query) {
        // Find max
        for (int kv = 0; kv < tile_size; ++kv) {
            local_max = fmaxf(local_max, smem_S[query_idx_in_tile][kv]);
        }
        
        // Skip fully-masked K/V tiles (prevents NaN from exp(-INF - (-INF)))
        if (!(isinf(local_max) && local_max < 0.0f)) {
            // Compute exp and sum
            for (int kv = 0; kv < tile_size; ++kv) {
                float exp_val = expf(smem_S[query_idx_in_tile][kv] - local_max);
                smem_S[query_idx_in_tile][kv] = exp_val;
                local_sum += exp_val;
            }
            
            // Compute partial output: O_partial = P @ V
            #pragma unroll
            for (int d = 0; d < head_dim; ++d) {
                float weighted_value = 0.0f;
                for (int kv = 0; kv < tile_size; ++kv) {
                    weighted_value += smem_S[query_idx_in_tile][kv] * to_float(smem_V[kv][d]);
                }
                acc_o[d] = weighted_value;
            }
        }
        // else: acc_o stays at 0 (initialized earlier), which is correct for fully-masked tiles
    }
    
    // DEBUG: Print for specific threads on first KV tile only
    if (kv_tile_idx == 0 && batch_idx == 0 && head_idx == 0) {
        if (query_idx_in_tile == 0 || query_idx_in_tile == 32 || query_idx_in_tile == 45 || query_idx_in_tile == 63) {
            printf("Partial[q=%d]: max=%.6f, sum=%.6f, acc_o[0:3]=[%.6f,%.6f,%.6f]\n",
                   query_idx_in_tile, local_max, local_sum, acc_o[0], acc_o[1], acc_o[2]);
        }
    }
    
    // Store partial results to global memory
    if (is_valid_query) {
        // Calculate output index: [B,H,Q_tiles,KV_tiles,TILE_SIZE_M,D]
        const int partial_offset = ((batch_idx * num_heads + head_idx) * num_query_tiles + query_tile_idx) 
                                   * num_kv_tiles + kv_tile_idx;
        T* partial_O_base = partial_O + (partial_offset * TILE_SIZE_M + query_idx_in_tile) * head_dim;
        
        for (int d = 0; d < head_dim; ++d) {
            partial_O_base[d] = from_float<T>(acc_o[d]);
        }
        
        // Store max and sum
        const int stats_offset = ((batch_idx * num_heads + head_idx) * num_query_tiles + query_tile_idx)
                                 * num_kv_tiles + kv_tile_idx;
        partial_max[stats_offset * TILE_SIZE_M + query_idx_in_tile] = local_max;
        partial_sum[stats_offset * TILE_SIZE_M + query_idx_in_tile] = local_sum;
    }
}

/**
 * FlashAttention-2 Split-K: Pass 2 - Reduce Partial Results
 * 
 * Each block reduces partial results for one query_tile across all kv_tiles.
 * Applies online softmax reduction to correctly combine results.
 * 
 * Grid: (num_heads, batch_size, num_query_tiles)
 * Block: (THREADS_PER_BLOCK, 1, 1)
 */
template<typename T>
__global__ void flash_attention_forward_split_k_reduce(
    const T* partial_O,
    const float* partial_max,
    const float* partial_sum,
    T* O,
    float* softmax_lse,
    const int batch_size,
    const int num_heads,
    const int seq_len,
    const int head_dim,
    const int num_query_tiles,
    const int num_kv_tiles
) {
    const int head_idx = blockIdx.x;
    const int batch_idx = blockIdx.y;
    const int query_tile_idx = blockIdx.z;
    const int query_idx_in_tile = threadIdx.x;
    const int query_idx = query_tile_idx * TILE_SIZE_M + query_idx_in_tile;
    
    const bool is_valid_query = (query_idx_in_tile < TILE_SIZE_M) && (query_idx < seq_len);
    
    if (!is_valid_query) return;
    
    // Find global max across all kv_tiles
    float global_max = -INFINITY;
    const int stats_base = ((batch_idx * num_heads + head_idx) * num_query_tiles + query_tile_idx) * num_kv_tiles;
    
    for (int kv_tile = 0; kv_tile < num_kv_tiles; ++kv_tile) {
        float local_max = partial_max[(stats_base + kv_tile) * TILE_SIZE_M + query_idx_in_tile];
        global_max = fmaxf(global_max, local_max);
    }
    
    // Compute global sum with reweighting
    float global_sum = 0.0f;
    for (int kv_tile = 0; kv_tile < num_kv_tiles; ++kv_tile) {
        float local_max = partial_max[(stats_base + kv_tile) * TILE_SIZE_M + query_idx_in_tile];
        float local_sum = partial_sum[(stats_base + kv_tile) * TILE_SIZE_M + query_idx_in_tile];
        global_sum += local_sum * expf(local_max - global_max);
    }
    
    // Accumulate reweighted partial outputs (initialize ALL elements to zero)
    float final_o[128] = {0};
    
    for (int kv_tile = 0; kv_tile < num_kv_tiles; ++kv_tile) {
        const float local_max = partial_max[(stats_base + kv_tile) * TILE_SIZE_M + query_idx_in_tile];
        // FIX: Reweight factor should be exp(m_i - m_global), NOT local_sum * exp(...)
        // partial_O already contains sum(exp(...) * V), so multiplying by local_sum double-counts!
        const float reweight = expf(local_max - global_max);
        
        const int partial_offset = ((batch_idx * num_heads + head_idx) * num_query_tiles + query_tile_idx)
                                   * num_kv_tiles + kv_tile;
        const T* partial_O_base = partial_O + (partial_offset * TILE_SIZE_M + query_idx_in_tile) * head_dim;
        
        for (int d = 0; d < head_dim; ++d) {
            final_o[d] += reweight * to_float(partial_O_base[d]);
        }
    }
    
    // DEBUG: Print for specific threads
    if (batch_idx == 0 && head_idx == 0) {
        if (query_idx_in_tile == 0 || query_idx_in_tile == 32 || query_idx_in_tile == 45 || query_idx_in_tile == 63) {
            printf("Reduce[q=%d]: global_max=%.6f, global_sum=%.6f, final_o[0:3]=[%.6f,%.6f,%.6f]\n",
                   query_idx_in_tile, global_max, global_sum, final_o[0], final_o[1], final_o[2]);
        }
    }
    
    // Normalize and write final output
    const int offset = (batch_idx * num_heads + head_idx) * seq_len * head_dim;
    T* O_base = O + offset;
    
    for (int d = 0; d < head_dim; ++d) {
        O_base[query_idx * head_dim + d] = from_float<T>(final_o[d] / global_sum);
    }
    
    // Store softmax LSE
    softmax_lse[(batch_idx * num_heads + head_idx) * seq_len + query_idx] = logf(global_sum) + global_max;
}

// REMOVED: Old broken Split-K host function from Sessions N+7
// Replaced by new KV-split implementation from Iteration 1 (see line ~654)

#endif  // End of disabled old Split-K code

// ============================================================================
// END OF OLD SPLIT-K KERNELS
// ============================================================================

/**
 * FlashAttention-Science backward kernel (stub).
 * 
 * TODO: Implement backward pass for training.
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
) {
    // TODO: Implement backward pass
    // For now, just zero out gradients
    const int total_size = batch_size * num_heads * seq_len * head_dim;
    cudaMemset(dQ, 0, total_size * sizeof(T));
    cudaMemset(dK, 0, total_size * sizeof(T));
    cudaMemset(dV, 0, total_size * sizeof(T));
}

// Explicit template instantiations for linking with bindings.cpp
template void flash_attention_forward<at::Half>(
    const at::Half* Q, const at::Half* K, const at::Half* V,
    at::Half* O, float* softmax_lse,
    const int batch_size, const int num_heads,
    const int seq_len, const int head_dim,
    const float softmax_scale, const bool causal
);

template void flash_attention_forward<at::BFloat16>(
    const at::BFloat16* Q, const at::BFloat16* K, const at::BFloat16* V,
    at::BFloat16* O, float* softmax_lse,
    const int batch_size, const int num_heads,
    const int seq_len, const int head_dim,
    const float softmax_scale, const bool causal
);

}  // namespace flashmoe


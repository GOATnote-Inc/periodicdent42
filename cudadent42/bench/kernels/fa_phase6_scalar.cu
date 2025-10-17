// ============================================================================
// PHASE 6: TARGETED SCALAR OPTIMIZATION (Simplified & Robust)
// ============================================================================
// Based on Phase 4's proven design + vectorization + tuning
// Target: 1,028 → 500-700 μs (1.5-2× speedup)
//
// Key Changes from Phase 4:
// 1. Vectorized loads (uint4 = 16 bytes = 8×FP16)
// 2. Increased threads (256 vs 128 for better occupancy)
// 3. Optimized loop unrolling
// 4. Reduced synchronization
// ============================================================================

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <float.h>

constexpr int HEAD_DIM = 64;

#ifndef BLOCK_M
constexpr int BLOCK_M = 32;
#endif
#ifndef BLOCK_N
constexpr int BLOCK_N = 64;
#endif
#ifndef NUM_THREADS
constexpr int NUM_THREADS = 256;  // Increased from 128
#endif

// Vectorized load: 8 FP16 values (16 bytes) at once
__device__ __forceinline__ void load_vec8(half* dst, const half* src) {
    *reinterpret_cast<uint4*>(dst) = *reinterpret_cast<const uint4*>(src);
}

// Vectorized store
__device__ __forceinline__ void store_vec8(half* dst, const half* src) {
    *reinterpret_cast<uint4*>(dst) = *reinterpret_cast<const uint4*>(src);
}

// Warp reductions
__device__ __forceinline__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, offset));
    }
    return val;
}

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_xor_sync(0xffffffff, val, offset);
    }
    return val;
}

// ============================================================================
// PHASE 6 KERNEL: VECTORIZED + OPTIMIZED
// ============================================================================

__global__ __launch_bounds__(NUM_THREADS, 2)
void flash_attention_phase6_kernel(
    const half* __restrict__ Q,
    const half* __restrict__ K,
    const half* __restrict__ V,
    half* __restrict__ O,
    float softmax_scale,
    int batch_size,
    int num_heads,
    int seq_len
) {
    const int batch_idx = blockIdx.z;
    const int head_idx = blockIdx.y;
    const int query_block_idx = blockIdx.x;
    const int tid = threadIdx.x;
    
    const int query_offset = (batch_idx * num_heads + head_idx) * seq_len * HEAD_DIM;
    const int kv_offset = query_offset;
    const int output_offset = query_offset;
    
    const int query_start = query_block_idx * BLOCK_M;
    const int rows_this_block = min(BLOCK_M, seq_len - query_start);
    
    if (query_start >= seq_len) return;
    
    // Shared memory
    __shared__ half Q_smem[BLOCK_M][HEAD_DIM];
    __shared__ half KV_smem[BLOCK_N][HEAD_DIM];  // Reused for K then V
    __shared__ float S_smem[BLOCK_M][BLOCK_N];
    
    // ========================================================================
    // LOAD Q TILE (VECTORIZED: 8 FP16 at a time)
    // ========================================================================
    for (int row = tid; row < rows_this_block; row += NUM_THREADS) {
        const int q_row = query_start + row;
        if (q_row < seq_len) {
            // Load 64 elements in 8 vectorized loads (8×8 = 64)
            #pragma unroll
            for (int d = 0; d < HEAD_DIM; d += 8) {
                load_vec8(&Q_smem[row][d], &Q[query_offset + q_row * HEAD_DIM + d]);
            }
        }
    }
    __syncthreads();
    
    // Initialize online softmax accumulators
    float m_row[BLOCK_M];
    float l_row[BLOCK_M];
    float O_accum[BLOCK_M][HEAD_DIM];
    
    for (int row = tid; row < rows_this_block; row += NUM_THREADS) {
        m_row[row] = -FLT_MAX;
        l_row[row] = 0.0f;
        #pragma unroll 8
        for (int d = 0; d < HEAD_DIM; d++) {
            O_accum[row][d] = 0.0f;
        }
    }
    
    // ========================================================================
    // KV LOOP
    // ========================================================================
    const int num_kv_blocks = (seq_len + BLOCK_N - 1) / BLOCK_N;
    
    for (int kv_block = 0; kv_block < num_kv_blocks; kv_block++) {
        const int kv_start = kv_block * BLOCK_N;
        const int kv_size = min(BLOCK_N, seq_len - kv_start);
        
        // Load K tile (vectorized)
        for (int kv_row = tid; kv_row < kv_size; kv_row += NUM_THREADS) {
            const int k_row = kv_start + kv_row;
            if (k_row < seq_len) {
                #pragma unroll
                for (int d = 0; d < HEAD_DIM; d += 8) {
                    load_vec8(&KV_smem[kv_row][d], &K[kv_offset + k_row * HEAD_DIM + d]);
                }
            }
        }
        __syncthreads();
        
        // Compute Q@K^T → S
        for (int row = tid; row < rows_this_block; row += NUM_THREADS) {
            #pragma unroll 4
            for (int col = 0; col < kv_size; col++) {
                float sum = 0.0f;
                #pragma unroll
                for (int d = 0; d < HEAD_DIM; d++) {
                    sum += __half2float(Q_smem[row][d]) * __half2float(KV_smem[col][d]);
                }
                
                sum *= softmax_scale;
                
                // Causal mask
                const int q_pos = query_start + row;
                const int k_pos = kv_start + col;
                if (k_pos > q_pos) {
                    sum = -FLT_MAX;
                }
                
                S_smem[row][col] = sum;
            }
        }
        __syncthreads();
        
        // Online softmax update
        for (int row = tid; row < rows_this_block; row += NUM_THREADS) {
            // Find max in this block
            float m_new = S_smem[row][0];
            #pragma unroll 4
            for (int col = 1; col < kv_size; col++) {
                m_new = fmaxf(m_new, S_smem[row][col]);
            }
            m_new = fmaxf(m_new, m_row[row]);
            
            // Rescale previous output
            const float scale_old = expf(m_row[row] - m_new);
            l_row[row] *= scale_old;
            #pragma unroll 8
            for (int d = 0; d < HEAD_DIM; d++) {
                O_accum[row][d] *= scale_old;
            }
            
            // Compute exp(S - m_new) and update
            float l_new = 0.0f;
            #pragma unroll 4
            for (int col = 0; col < kv_size; col++) {
                const float exp_val = expf(S_smem[row][col] - m_new);
                S_smem[row][col] = exp_val;
                l_new += exp_val;
            }
            
            l_row[row] += l_new;
            m_row[row] = m_new;
        }
        __syncthreads();
        
        // Load V tile (vectorized, reuse KV_smem)
        for (int kv_row = tid; kv_row < kv_size; kv_row += NUM_THREADS) {
            const int v_row = kv_start + kv_row;
            if (v_row < seq_len) {
                #pragma unroll
                for (int d = 0; d < HEAD_DIM; d += 8) {
                    load_vec8(&KV_smem[kv_row][d], &V[kv_offset + v_row * HEAD_DIM + d]);
                }
            }
        }
        __syncthreads();
        
        // Accumulate P@V → O
        for (int row = tid; row < rows_this_block; row += NUM_THREADS) {
            #pragma unroll 8
            for (int d = 0; d < HEAD_DIM; d++) {
                float sum = 0.0f;
                #pragma unroll 4
                for (int col = 0; col < kv_size; col++) {
                    sum += S_smem[row][col] * __half2float(KV_smem[col][d]);
                }
                O_accum[row][d] += sum;
            }
        }
        __syncthreads();
    }
    
    // ========================================================================
    // FINALIZE AND WRITE OUTPUT (VECTORIZED)
    // ========================================================================
    for (int row = tid; row < rows_this_block; row += NUM_THREADS) {
        const int q_pos = query_start + row;
        if (q_pos < seq_len) {
            const float inv_l = 1.0f / l_row[row];
            
            // Normalize and write (vectorized)
            half O_half[HEAD_DIM];
            #pragma unroll 8
            for (int d = 0; d < HEAD_DIM; d++) {
                O_half[d] = __float2half(O_accum[row][d] * inv_l);
            }
            
            // Vectorized store: 8 FP16 at a time
            #pragma unroll
            for (int d = 0; d < HEAD_DIM; d += 8) {
                store_vec8(&O[output_offset + q_pos * HEAD_DIM + d], &O_half[d]);
            }
        }
    }
}

// ============================================================================
// CUDA LAUNCHER
// ============================================================================

extern "C" void launch_flash_attention_phase6(
    const half* Q,
    const half* K,
    const half* V,
    half* O,
    float softmax_scale,
    int batch_size,
    int num_heads,
    int seq_len,
    cudaStream_t stream
) {
    const int num_query_blocks = (seq_len + BLOCK_M - 1) / BLOCK_M;
    
    dim3 grid(num_query_blocks, num_heads, batch_size);
    dim3 block(NUM_THREADS);
    
    flash_attention_phase6_kernel<<<grid, block, 0, stream>>>(
        Q, K, V, O, softmax_scale, batch_size, num_heads, seq_len
    );
}


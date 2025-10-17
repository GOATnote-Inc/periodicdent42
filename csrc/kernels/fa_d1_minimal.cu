/*
 * Phase D.1: Minimal Custom FlashAttention Kernel
 * 
 * Pure CUDA implementation - NO PyTorch backend wrappers
 * 
 * Goal: Establish baseline for optimization (expected: 100-200 μs)
 * Algorithm: Scalar FlashAttention with online softmax
 * Precision: FP16 I/O, FP32 accumulation
 * 
 * This is the FOUNDATION we'll optimize in D.2-D.5 to achieve < 5 μs
 */

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cmath>

// Kernel configuration
#define BLOCK_M 32      // Rows of Q per block
#define BLOCK_N 64      // Rows of K/V per block  
#define HEAD_DIM 64     // Head dimension (fixed)
#define NUM_WARPS 8     // Warps per block
#define THREADS (NUM_WARPS * 32)

// Warp-level reduction for max
__device__ __forceinline__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

// Warp-level reduction for sum
__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

/*
 * Phase D.1: Minimal FlashAttention Kernel
 * 
 * Algorithm:
 * 1. Load Q tile (reuse across K/V iterations)
 * 2. For each K/V tile:
 *    a. Compute S = Q @ K^T (scalar)
 *    b. Online softmax (update running max/sum)
 *    c. Compute O += P @ V (scalar)
 * 3. Write final O
 * 
 * No optimizations yet - just correctness!
 */
__global__ void flash_attention_d1_minimal(
    const half* __restrict__ Q,  // [B, H, S, D]
    const half* __restrict__ K,  // [B, H, S, D]
    const half* __restrict__ V,  // [B, H, S, D]
    half* __restrict__ O,        // [B, H, S, D]
    int B, int H, int S, int D,
    float scale
) {
    // Block indices
    const int batch_idx = blockIdx.z;
    const int head_idx = blockIdx.y;
    const int q_block_idx = blockIdx.x;
    
    const int q_start = q_block_idx * BLOCK_M;
    const int tid = threadIdx.x;
    
    // Global memory base pointers for this (batch, head)
    const int bhsd_offset = (batch_idx * H + head_idx) * S * D;
    const half* Q_base = Q + bhsd_offset;
    const half* K_base = K + bhsd_offset;
    const half* V_base = V + bhsd_offset;
    half* O_base = O + bhsd_offset;
    
    // Shared memory for tiles
    __shared__ half Q_smem[BLOCK_M][HEAD_DIM];
    __shared__ half K_smem[BLOCK_N][HEAD_DIM];
    __shared__ half V_smem[BLOCK_N][HEAD_DIM];
    __shared__ float S_smem[BLOCK_M][BLOCK_N];
    __shared__ float P_smem[BLOCK_M][BLOCK_N];
    
    // Per-thread output accumulator (FP32 for precision)
    float O_acc[HEAD_DIM];
    #pragma unroll
    for (int d = 0; d < HEAD_DIM; ++d) {
        O_acc[d] = 0.0f;
    }
    
    // Online softmax statistics (per thread, per row)
    float m_prev = -INFINITY;  // Running max
    float l_prev = 0.0f;        // Running sum of exp
    
    // ========================================
    // Load Q tile (reuse for all K/V tiles)
    // ========================================
    for (int i = tid; i < BLOCK_M * HEAD_DIM; i += THREADS) {
        const int row = i / HEAD_DIM;
        const int col = i % HEAD_DIM;
        const int global_row = q_start + row;
        
        if (global_row < S) {
            Q_smem[row][col] = Q_base[global_row * D + col];
        } else {
            Q_smem[row][col] = __float2half(0.0f);
        }
    }
    __syncthreads();
    
    // ========================================
    // Iterate over K/V tiles
    // ========================================
    const int num_kv_tiles = (S + BLOCK_N - 1) / BLOCK_N;
    
    for (int kv_tile = 0; kv_tile < num_kv_tiles; ++kv_tile) {
        const int kv_start = kv_tile * BLOCK_N;
        
        // Load K tile
        for (int i = tid; i < BLOCK_N * HEAD_DIM; i += THREADS) {
            const int row = i / HEAD_DIM;
            const int col = i % HEAD_DIM;
            const int global_row = kv_start + row;
            
            if (global_row < S) {
                K_smem[row][col] = K_base[global_row * D + col];
            } else {
                K_smem[row][col] = __float2half(0.0f);
            }
        }
        __syncthreads();
        
        // ========================================
        // Compute S = Q @ K^T (SCALAR - no Tensor Cores yet!)
        // ========================================
        for (int i = tid; i < BLOCK_M * BLOCK_N; i += THREADS) {
            const int q_row = i / BLOCK_N;
            const int k_row = i % BLOCK_N;
            const int global_q_row = q_start + q_row;
            const int global_k_row = kv_start + k_row;
            
            if (global_q_row < S && global_k_row < S) {
                float acc = 0.0f;
                #pragma unroll
                for (int d = 0; d < HEAD_DIM; ++d) {
                    acc += __half2float(Q_smem[q_row][d]) * __half2float(K_smem[k_row][d]);
                }
                S_smem[q_row][k_row] = acc * scale;
            } else {
                S_smem[q_row][k_row] = -INFINITY;
            }
        }
        __syncthreads();
        
        // ========================================
        // Online Softmax: Compute row-wise max
        // ========================================
        float m_new = -INFINITY;
        
        // Each thread processes multiple rows
        const int rows_per_thread = (BLOCK_M + THREADS - 1) / THREADS;
        for (int local_row = 0; local_row < rows_per_thread; ++local_row) {
            const int row = tid + local_row * THREADS;
            if (row < BLOCK_M) {
                const int global_row = q_start + row;
                if (global_row < S) {
                    #pragma unroll
                    for (int col = 0; col < BLOCK_N; ++col) {
                        const int global_col = kv_start + col;
                        if (global_col < S) {
                            m_new = fmaxf(m_new, S_smem[row][col]);
                        }
                    }
                }
            }
        }
        
        // Warp reduction for max
        m_new = warp_reduce_max(m_new);
        // Broadcast to all threads in warp
        m_new = __shfl_sync(0xffffffff, m_new, 0);
        
        // ========================================
        // Update global statistics
        // ========================================
        const float m_global = fmaxf(m_prev, m_new);
        const float exp_m_prev = expf(m_prev - m_global);
        const float exp_m_new = expf(m_new - m_global);
        
        // ========================================
        // Compute P = exp(S - m_new) and sum
        // ========================================
        float l_new = 0.0f;
        
        for (int i = tid; i < BLOCK_M * BLOCK_N; i += THREADS) {
            const int q_row = i / BLOCK_N;
            const int k_row = i % BLOCK_N;
            const int global_q_row = q_start + q_row;
            const int global_k_row = kv_start + k_row;
            
            if (global_q_row < S && global_k_row < S) {
                const float exp_val = expf(S_smem[q_row][k_row] - m_new);
                P_smem[q_row][k_row] = exp_val;
                
                // Accumulate sum (each thread sums its elements)
                if ((q_row % (THREADS / BLOCK_N)) == (tid % (THREADS / BLOCK_N))) {
                    l_new += exp_val;
                }
            } else {
                P_smem[q_row][k_row] = 0.0f;
            }
        }
        __syncthreads();
        
        // Warp reduction for sum
        l_new = warp_reduce_sum(l_new);
        l_new = __shfl_sync(0xffffffff, l_new, 0);
        
        // Update global sum
        const float l_global = exp_m_prev * l_prev + exp_m_new * l_new;
        
        // ========================================
        // Rescale previous output
        // ========================================
        const float scale_factor = exp_m_prev * l_prev / l_global;
        #pragma unroll
        for (int d = 0; d < HEAD_DIM; ++d) {
            O_acc[d] *= scale_factor;
        }
        
        // ========================================
        // Load V tile
        // ========================================
        for (int i = tid; i < BLOCK_N * HEAD_DIM; i += THREADS) {
            const int row = i / HEAD_DIM;
            const int col = i % HEAD_DIM;
            const int global_row = kv_start + row;
            
            if (global_row < S) {
                V_smem[row][col] = V_base[global_row * D + col];
            } else {
                V_smem[row][col] = __float2half(0.0f);
            }
        }
        __syncthreads();
        
        // ========================================
        // Compute O += P @ V (SCALAR - no Tensor Cores yet!)
        // ========================================
        for (int local_row = 0; local_row < rows_per_thread; ++local_row) {
            const int row = tid + local_row * THREADS;
            if (row < BLOCK_M) {
                const int global_row = q_start + row;
                if (global_row < S) {
                    #pragma unroll
                    for (int d = 0; d < HEAD_DIM; ++d) {
                        float acc = 0.0f;
                        #pragma unroll
                        for (int col = 0; col < BLOCK_N; ++col) {
                            const int global_col = kv_start + col;
                            if (global_col < S) {
                                acc += P_smem[row][col] * __half2float(V_smem[col][d]);
                            }
                        }
                        O_acc[d] += (exp_m_new / l_global) * acc;
                    }
                }
            }
        }
        
        // Update statistics for next iteration
        m_prev = m_global;
        l_prev = l_global;
        
        __syncthreads();
    }
    
    // ========================================
    // Write final output
    // ========================================
    for (int local_row = 0; local_row < rows_per_thread; ++local_row) {
        const int row = tid + local_row * THREADS;
        if (row < BLOCK_M) {
            const int global_row = q_start + row;
            if (global_row < S) {
                #pragma unroll
                for (int d = 0; d < HEAD_DIM; ++d) {
                    O_base[global_row * D + d] = __float2half(O_acc[d]);
                }
            }
        }
    }
}

// Launch wrapper
extern "C" void launch_flash_attention_d1(
    const half* Q,
    const half* K,
    const half* V,
    half* O,
    int B, int H, int S, int D,
    float scale,
    cudaStream_t stream
) {
    // Grid: (num_q_blocks, num_heads, batch_size)
    const int num_q_blocks = (S + BLOCK_M - 1) / BLOCK_M;
    dim3 grid(num_q_blocks, H, B);
    dim3 block(THREADS);
    
    flash_attention_d1_minimal<<<grid, block, 0, stream>>>(
        Q, K, V, O, B, H, S, D, scale
    );
}


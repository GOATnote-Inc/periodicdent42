/*
 * Phase C.1: WMMA Q@K^T Micro-Kernel
 * 
 * Goal: Replace cuBLAS with manual WMMA for fine-grained control
 * Target: 78 → 55 μs (1.42× speedup)
 * 
 * Key optimizations:
 * - 16×16×16 WMMA tiles for Q@K^T
 * - Direct control over fragment operations
 * - Preparation for fusion with softmax (Phase C.3)
 */

#include <cuda_fp16.h>
#include <mma.h>
#include <cmath>

using namespace nvcuda;

// Compile-time constants
#define WARP_SIZE 32
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

// Kernel configuration
#ifndef BLOCK_M
#define BLOCK_M 32
#endif

#ifndef HEAD_DIM
#define HEAD_DIM 64
#endif

#ifndef BLOCK_N
#define BLOCK_N 64
#endif

#ifndef NUM_WARPS
#define NUM_WARPS 8
#endif

#define THREADS (NUM_WARPS * WARP_SIZE)

// WMMA fragments for Q@K^T
// Q: [BLOCK_M, HEAD_DIM] = [32, 64] → need 2×4 tiles of 16×16
// K^T: [HEAD_DIM, BLOCK_N] = [64, 64] → need 4×4 tiles of 16×16
// S: [BLOCK_M, BLOCK_N] = [32, 64] → need 2×4 tiles of 16×16

__device__ __forceinline__ void wmma_qkt_tile(
    const half* Q_smem,  // [BLOCK_M, HEAD_DIM]
    const half* K_smem,  // [BLOCK_N, HEAD_DIM] (will transpose)
    float* S_smem,       // [BLOCK_M, BLOCK_N]
    float scale
) {
    // Get warp ID and lane ID
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;
    
    // Each warp processes one 16×16 output tile
    // With NUM_WARPS=8, we can cover 8 tiles
    // Layout: 2 tiles in M dimension, 4 tiles in N dimension = 8 tiles
    const int warp_m = warp_id / 4;  // 0-1
    const int warp_n = warp_id % 4;  // 0-3
    
    // Output tile position
    const int tile_m = warp_m * WMMA_M;  // 0 or 16
    const int tile_n = warp_n * WMMA_N;  // 0, 16, 32, 48
    
    // WMMA fragments
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
    
    // Initialize accumulator to zero
    wmma::fill_fragment(c_frag, 0.0f);
    
    // Iterate over K dimension (HEAD_DIM = 64, need 4 tiles of 16)
    #pragma unroll
    for (int k = 0; k < HEAD_DIM; k += WMMA_K) {
        // Load Q tile: [tile_m:tile_m+16, k:k+16]
        wmma::load_matrix_sync(a_frag, 
                               Q_smem + tile_m * HEAD_DIM + k, 
                               HEAD_DIM);
        
        // Load K^T tile: [k:k+16, tile_n:tile_n+16] (transposed)
        // K is stored as [BLOCK_N, HEAD_DIM], so K^T access pattern:
        // We need K[tile_n:tile_n+16, k:k+16]^T = K^T[k:k+16, tile_n:tile_n+16]
        wmma::load_matrix_sync(b_frag, 
                               K_smem + tile_n * HEAD_DIM + k, 
                               HEAD_DIM);
        
        // Compute C += A * B
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }
    
    // Scale the accumulator
    #pragma unroll
    for (int i = 0; i < c_frag.num_elements; i++) {
        c_frag.x[i] *= scale;
    }
    
    // Store result to SMEM
    wmma::store_matrix_sync(S_smem + tile_m * BLOCK_N + tile_n, 
                            c_frag, 
                            BLOCK_N, 
                            wmma::mem_row_major);
}

__global__ void flash_attention_wmma_qkt(
    const half* __restrict__ Q,  // [B, H, S, D]
    const half* __restrict__ K,  // [B, H, S, D]
    const half* __restrict__ V,  // [B, H, S, D]
    half* __restrict__ O,        // [B, H, S, D]
    int B, int H, int S, int D,
    float scale
) {
    // Shared memory for tiles
    __shared__ half Q_smem[BLOCK_M][HEAD_DIM];
    __shared__ half K_smem[BLOCK_N][HEAD_DIM];
    __shared__ half V_smem[BLOCK_N][HEAD_DIM];
    __shared__ float S_smem[BLOCK_M][BLOCK_N];
    __shared__ float P_smem[BLOCK_M][BLOCK_N];
    
    // Block and thread indices
    const int batch_idx = blockIdx.z;
    const int head_idx = blockIdx.y;
    const int q_block_idx = blockIdx.x;
    
    const int q_start = q_block_idx * BLOCK_M;
    const int tid = threadIdx.x;
    
    // Global memory offsets
    const int bhsd_offset = (batch_idx * H + head_idx) * S * D;
    const half* Q_base = Q + bhsd_offset;
    const half* K_base = K + bhsd_offset;
    const half* V_base = V + bhsd_offset;
    half* O_base = O + bhsd_offset;
    
    // Initialize output accumulators (FP32 for precision)
    float O_acc[HEAD_DIM] = {0.0f};
    float m_prev = -INFINITY;
    float l_prev = 0.0f;
    
    // Compute rows per thread for loop iterations
    const int rows_per_thread = (BLOCK_M + THREADS - 1) / THREADS;
    
    // Load Q tile (once, reuse for all KV tiles)
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
    
    // Iterate over KV tiles
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
        // WMMA Q@K^T (replaces scalar loop)
        // ========================================
        wmma_qkt_tile((const half*)Q_smem, (const half*)K_smem, (float*)S_smem, scale);
        __syncthreads();
        
        // Online softmax: compute row-wise max
        float m_new = -INFINITY;
        
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
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            m_new = fmaxf(m_new, __shfl_down_sync(0xffffffff, m_new, offset));
        }
        
        // Broadcast within warp
        m_new = __shfl_sync(0xffffffff, m_new, 0);
        
        // Update softmax statistics
        const float m_global = fmaxf(m_prev, m_new);
        const float exp_m_prev = expf(m_prev - m_global);
        const float exp_m_new = expf(m_new - m_global);
        
        // Compute P = exp(S - m_new) and sum
        float l_new = 0.0f;
        
        for (int local_row = 0; local_row < rows_per_thread; ++local_row) {
            const int row = tid + local_row * THREADS;
            if (row < BLOCK_M) {
                const int global_row = q_start + row;
                if (global_row < S) {
                    #pragma unroll
                    for (int col = 0; col < BLOCK_N; ++col) {
                        const int global_col = kv_start + col;
                        if (global_col < S) {
                            const float exp_val = expf(S_smem[row][col] - m_new);
                            P_smem[row][col] = exp_val;
                            l_new += exp_val;
                        } else {
                            P_smem[row][col] = 0.0f;
                        }
                    }
                }
            }
        }
        
        // Warp reduction for sum
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            l_new += __shfl_down_sync(0xffffffff, l_new, offset);
        }
        l_new = __shfl_sync(0xffffffff, l_new, 0);
        
        // Update global statistics
        const float l_global = exp_m_prev * l_prev + exp_m_new * l_new;
        
        // Rescale previous output
        const float scale_factor = exp_m_prev * l_prev / l_global;
        
        #pragma unroll
        for (int d = 0; d < HEAD_DIM; ++d) {
            O_acc[d] *= scale_factor;
        }
        
        // Load V tile
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
        
        // P @ V (scalar for now, will replace with WMMA in C.3)
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
    
    // Write final output
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
extern "C" void launch_flash_attention_wmma_qkt(
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
    
    flash_attention_wmma_qkt<<<grid, block, 0, stream>>>(
        Q, K, V, O, B, H, S, D, scale
    );
}


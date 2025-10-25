// V5: Warp-specialized Tensor Core FlashAttention for L4 (sm_89)
// Producer/consumer pattern + WMMA 16x16x16 + double-buffered SMEM

#include <cuda_fp16.h>
#include <mma.h>
#include <cuda_runtime.h>

using namespace nvcuda;

// Compile-time parameters (overridable via -D flags)
#ifndef M_TILE
#define M_TILE 64
#endif
#ifndef N_TILE  
#define N_TILE 64
#endif
#ifndef K_TILE
#define K_TILE 32
#endif
#ifndef STAGES
#define STAGES 2
#endif
#ifndef NUM_WARPS
#define NUM_WARPS 8
#endif

#define HEAD_DIM 64
#define SEQ_LEN 512

// Warp-level reductions
__device__ __forceinline__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, mask));
    }
    return val;
}

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        val += __shfl_xor_sync(0xffffffff, val, mask);
    }
    return val;
}

extern "C" __global__ void __launch_bounds__(NUM_WARPS * 32)
fa_v5_kernel(
    const half* __restrict__ Q,  // [B, H, S, D]
    const half* __restrict__ K,  // [B, H, S, D]
    const half* __restrict__ V,  // [B, H, S, D]
    half* __restrict__ O,         // [B, H, S, D]
    int B, int H, int S, int D,
    float scale
) {
    const int b = blockIdx.z;
    const int h = blockIdx.y;
    const int m_block = blockIdx.x;
    
    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;
    
    // SMEM layout: double-buffered Q/K/V tiles
    __shared__ half smem_q[STAGES][M_TILE][K_TILE];
    __shared__ half smem_k[STAGES][N_TILE][K_TILE];
    __shared__ half smem_v[STAGES][N_TILE][HEAD_DIM];
    __shared__ float smem_s[M_TILE][N_TILE];  // Attention scores
    
    // Per-thread output accumulator
    float o_frag[HEAD_DIM] = {0.0f};
    float m_i = -INFINITY;  // Running max
    float l_i = 0.0f;       // Running sum
    
    const int m_start = m_block * M_TILE;
    const int m_end = min(m_start + M_TILE, S);
    const int m_count = m_end - m_start;
    
    // Load Q tile (per-block, done once)
    const half* Q_base = Q + (b * H + h) * S * D;
    for (int m = warp_id; m < m_count; m += NUM_WARPS) {
        for (int k = lane_id; k < K_TILE; k += 32) {
            smem_q[0][m][k] = Q_base[(m_start + m) * D + k];
        }
    }
    __syncthreads();
    
    // Iterate over K/V tiles
    const half* K_base = K + (b * H + h) * S * D;
    const half* V_base = V + (b * H + h) * S * D;
    
    for (int n_block = 0; n_block < (S + N_TILE - 1) / N_TILE; n_block++) {
        const int n_start = n_block * N_TILE;
        const int n_end = min(n_start + N_TILE, S);
        const int n_count = n_end - n_start;
        
        const int stage = n_block % STAGES;
        
        // Load K tile
        for (int n = warp_id; n < n_count; n += NUM_WARPS) {
            for (int k = lane_id; k < K_TILE; k += 32) {
                smem_k[stage][n][k] = K_base[(n_start + n) * D + k];
            }
        }
        
        // Load V tile
        for (int n = warp_id; n < n_count; n += NUM_WARPS) {
            for (int d = lane_id; d < HEAD_DIM; d += 32) {
                smem_v[stage][n][d] = V_base[(n_start + n) * D + d];
            }
        }
        __syncthreads();
        
        // Compute Q@K^T using WMMA (per warp)
        for (int m_warp = warp_id * 16; m_warp < m_count; m_warp += NUM_WARPS * 16) {
            if (m_warp >= m_count) break;
            
            for (int n_warp = 0; n_warp < n_count; n_warp += 16) {
                // WMMA fragments
                wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
                wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
                wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
                
                wmma::fill_fragment(c_frag, 0.0f);
                
                // Accumulate over K_TILE in steps of 16
                for (int k = 0; k < K_TILE; k += 16) {
                    wmma::load_matrix_sync(a_frag, &smem_q[0][m_warp][k], K_TILE);
                    wmma::load_matrix_sync(b_frag, &smem_k[stage][n_warp][k], K_TILE);
                    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
                }
                
                // Store result to SMEM with scaling
                float result[8];  // Each thread owns 8 elements
                wmma::store_matrix_sync(result, c_frag, 16, wmma::mem_row_major);
                
                // Write scaled scores to smem_s (warp-cooperative)
                #pragma unroll
                for (int i = 0; i < 8; i++) {
                    int row = m_warp + (lane_id / 4);
                    int col = n_warp + (lane_id % 4) * 2 + (i / 4);
                    if (row < m_count && col < n_count) {
                        smem_s[row][col] = result[i] * scale;
                    }
                }
            }
        }
        __syncthreads();
        
        // Online softmax update (per row, per warp)
        for (int m = warp_id; m < m_count; m += NUM_WARPS) {
            // Find new max
            float m_new = m_i;
            for (int n = lane_id; n < n_count; n += 32) {
                m_new = fmaxf(m_new, smem_s[m][n]);
            }
            m_new = warp_reduce_max(m_new);
            
            // Update running statistics
            float m_old = m_i;
            m_i = fmaxf(m_old, m_new);
            
            // Compute exp and sum
            float l_new = 0.0f;
            for (int n = lane_id; n < n_count; n += 32) {
                float p = expf(smem_s[m][n] - m_i);
                smem_s[m][n] = p;
                l_new += p;
            }
            l_new = warp_reduce_sum(l_new);
            
            // Update l_i with correction factor
            float correction = expf(m_old - m_i);
            l_i = l_i * correction + l_new;
            
            // Update O with correction
            #pragma unroll
            for (int d = 0; d < HEAD_DIM; d++) {
                o_frag[d] *= correction;
            }
        }
        __syncthreads();
        
        // Compute P@V contribution (per row, per warp)
        for (int m = warp_id; m < m_count; m += NUM_WARPS) {
            for (int n = 0; n < n_count; n++) {
                float p_val = smem_s[m][n];
                #pragma unroll
                for (int d = lane_id; d < HEAD_DIM; d += 32) {
                    o_frag[d] += p_val * __half2float(smem_v[stage][n][d]);
                }
            }
        }
        __syncthreads();
    }
    
    // Final normalization and write output
    half* O_base = O + (b * H + h) * S * D;
    for (int m = warp_id; m < m_count; m += NUM_WARPS) {
        #pragma unroll
        for (int d = lane_id; d < HEAD_DIM; d += 32) {
            O_base[(m_start + m) * D + d] = __float2half(o_frag[d] / l_i);
        }
    }
}


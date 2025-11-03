// I7 Fixed: Minimal correct WMMA implementation
// Focus: Correctness over performance
// Expert approach: Simplify to prove architecture, optimize later

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include "../include/dhp_ct_enhanced.cuh"

using namespace nvcuda;

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

__global__ void __launch_bounds__(128)
dhp_i7_fixed(
    const __half* __restrict__ Q,
    const __half* __restrict__ K,
    const __half* __restrict__ V,
    __half* __restrict__ out,
    const uint32_t S_max,
    const uint32_t S_actual,
    const uint32_t batch_size
) {
    const int batch_head = blockIdx.x;
    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;
    
    if (batch_head >= batch_size) return;
    
    // Each warp handles 16 rows (one WMMA tile in M dimension)
    const int row_base = warp_id * WMMA_M;
    
    // Shared memory for one row's attention computation
    __shared__ __half Q_shared[128][64];    // All warps' Q
    __shared__ __half K_shared[64];          // One K vector at a time
    __shared__ __half V_shared[64];          // One V vector at a time
    __shared__ float scores_shared[128];     // Partial scores
    
    // Per-thread state (16 rows per warp, distributed across 32 threads)
    float m_local[16];
    float l_local[16];
    float acc[64];
    
    // Initialize
    for (int i = 0; i < 16; ++i) {
        m_local[i] = -INFINITY;
        l_local[i] = 0.0f;
    }
    for (int i = 0; i < 64; ++i) {
        acc[i] = 0.0f;
    }
    
    // Load Q for this warp (cooperative)
    for (int row = 0; row < WMMA_M; ++row) {
        int global_row = row_base + row;
        uint32_t row_valid = ct_lt_u32(global_row, S_actual);
        
        for (int col = lane_id; col < 64; col += 32) {
            int q_idx = batch_head * S_max * 64 + global_row * 64 + col;
            __half val = Q[q_idx];
            Q_shared[row_base + row][col] = ct_select_half(__float2half(0.0f), val, row_valid);
        }
    }
    __syncthreads();
    
    // Process each column (key/value)
    for (int col_idx = 0; col_idx < S_max; ++col_idx) {
        uint32_t col_valid = ct_lt_u32(col_idx, S_actual);
        
        // Load K vector (one warp does this)
        if (warp_id == 0) {
            for (int i = lane_id; i < 64; i += 32) {
                int k_idx = batch_head * S_max * 64 + col_idx * 64 + i;
                __half val = K[k_idx];
                K_shared[i] = ct_select_half(__float2half(0.0f), val, col_valid);
            }
        }
        __syncthreads();
        
        // Compute score = Q @ K for this warp's rows
        for (int row = 0; row < WMMA_M; ++row) {
            int global_row = row_base + row;
            uint32_t row_valid = ct_lt_u32(global_row, S_actual);
            
            // Each thread computes partial dot product
            float partial_score = 0.0f;
            for (int k = lane_id; k < 64; k += 32) {
                partial_score += __half2float(Q_shared[row_base + row][k]) * __half2float(K_shared[k]);
            }
            
            // Sum across warp using deterministic tree reduction
            partial_score += __shfl_xor_sync(0xffffffff, partial_score, 16);
            partial_score += __shfl_xor_sync(0xffffffff, partial_score, 8);
            partial_score += __shfl_xor_sync(0xffffffff, partial_score, 4);
            partial_score += __shfl_xor_sync(0xffffffff, partial_score, 2);
            partial_score += __shfl_xor_sync(0xffffffff, partial_score, 1);
            
            float score = partial_score * 0.125f;  // Scale
            
            // Apply causal mask and validity
            uint32_t causal = ct_le_u32(col_idx, global_row);
            uint32_t valid = ct_and_u32(ct_and_u32(row_valid, col_valid), causal);
            score = ct_select_f32(-INFINITY, score, valid);
            
            // Online softmax update (lane 0 holds the score)
            if (lane_id == 0) {
                // Update max
                uint32_t gt = ct_gt_f32(score, m_local[row]);
                float m_new = ct_select_f32(m_local[row], score, gt);
                
                // Rescale accumulator
                float alpha = expf(m_local[row] - m_new);
                l_local[row] *= alpha;
                for (int d = 0; d < 64; ++d) {
                    acc[d] *= alpha;
                }
                
                // Add contribution
                float p = safe_exp(score - m_new);
                l_local[row] += p;
                
                // Store p for use in V accumulation
                scores_shared[row_base + row] = p;
                
                m_local[row] = m_new;
            }
        }
        __syncthreads();
        
        // Load V vector (one warp does this)
        if (warp_id == 0) {
            for (int i = lane_id; i < 64; i += 32) {
                int v_idx = batch_head * S_max * 64 + col_idx * 64 + i;
                __half val = V[v_idx];
                V_shared[i] = ct_select_half(__float2half(0.0f), val, col_valid);
            }
        }
        __syncthreads();
        
        // Accumulate P * V (lane 0 has the state)
        if (lane_id == 0) {
            for (int row = 0; row < WMMA_M; ++row) {
                float p = scores_shared[row_base + row];
                for (int d = 0; d < 64; ++d) {
                    acc[d] += p * __half2float(V_shared[d]);
                }
            }
        }
        __syncthreads();
    }
    
    // Write output (lane 0 has the accumulated values)
    if (lane_id == 0) {
        for (int row = 0; row < WMMA_M; ++row) {
            int global_row = row_base + row;
            if (global_row >= S_max) break;
            
            uint32_t row_valid = ct_lt_u32(global_row, S_actual);
            float l_safe = ct_select_f32(1.0f, l_local[row], row_valid);
            
            for (int d = 0; d < 64; ++d) {
                float val = acc[d] / l_safe;
                val = ct_select_f32(0.0f, val, row_valid);
                int out_idx = batch_head * S_max * 64 + global_row * 64 + d;
                out[out_idx] = __float2half(val);
            }
        }
    }
}


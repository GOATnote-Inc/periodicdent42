// I6 REAL: Proper block-parallel with shared memory tiling
// Expert approach: threads COOPERATE on tile computation

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "../include/dhp_ct_enhanced.cuh"

#define BM 64
#define BN 64  
#define BK 64
#define TM 4   // Each thread computes 4 output rows
#define TN 4   // Each thread computes 4 output cols

__global__ void __launch_bounds__(256)
dhp_i6_real(
    const __half* __restrict__ Q,
    const __half* __restrict__ K,
    const __half* __restrict__ V,
    __half* __restrict__ out,
    const uint32_t S_max,
    const uint32_t S_actual,
    const uint32_t batch_size
) {
    const int batch_head = blockIdx.x;
    const int tile_m = blockIdx.y;
    const int tid = threadIdx.x;
    
    if (batch_head >= batch_size) return;
    
    // Shared memory for tiles
    __shared__ __half Qs[BM][BK];
    __shared__ __half Ks[BN][BK];
    __shared__ __half Vs[BN][BK];
    
    // Thread's output accumulator (registers)
    float acc[TM][BK];
    float m[TM];
    float l[TM];
    
    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        m[i] = -INFINITY;
        l[i] = 0.0f;
        #pragma unroll
        for (int j = 0; j < BK; ++j) {
            acc[i][j] = 0.0f;
        }
    }
    
    const int row_start = tile_m * BM;
    
    // Thread mapping for cooperative work
    const int thread_row_start = (tid / (BN/TN)) * TM;
    const int thread_col_start = (tid % (BN/TN)) * TN;
    
    // Load Q tile cooperatively (once)
    for (int i = tid; i < BM * BK; i += blockDim.x) {
        int row = i / BK;
        int col = i % BK;
        int global_row = row_start + row;
        
        uint32_t valid = ct_lt_u32(global_row, S_actual);
        int q_idx = batch_head * S_max * BK + global_row * BK + col;
        __half val = Q[q_idx];
        Qs[row][col] = ct_select_half(__float2half(0.0f), val, valid);
    }
    __syncthreads();
    
    // Loop over K/V tiles
    for (int tile_n = 0; tile_n < (S_max + BN - 1) / BN; ++tile_n) {
        int col_start = tile_n * BN;
        
        // Load K tile
        for (int i = tid; i < BN * BK; i += blockDim.x) {
            int row = i / BK;
            int col = i % BK;
            int global_col = col_start + row;
            
            uint32_t valid = ct_lt_u32(global_col, S_actual);
            int k_idx = batch_head * S_max * BK + global_col * BK + col;
            __half val = K[k_idx];
            Ks[row][col] = ct_select_half(__float2half(0.0f), val, valid);
        }
        __syncthreads();
        
        // Compute scores: each thread does TM rows
        float scores[TM][TN];
        
        // S = Q @ K^T
        for (int tm = 0; tm < TM; ++tm) {
            for (int tn = 0; tn < TN; ++tn) {
                float s = 0.0f;
                #pragma unroll
                for (int k = 0; k < BK; ++k) {
                    s += __half2float(Qs[thread_row_start + tm][k]) * 
                         __half2float(Ks[thread_col_start + tn][k]);
                }
                s *= 0.125f;
                
                // Causal mask
                int global_row = row_start + thread_row_start + tm;
                int global_col = col_start + thread_col_start + tn;
                uint32_t causal = ct_le_u32(global_col, global_row);
                uint32_t row_valid = ct_lt_u32(global_row, S_actual);
                uint32_t col_valid = ct_lt_u32(global_col, S_actual);
                uint32_t valid = ct_and_u32(ct_and_u32(row_valid, col_valid), causal);
                
                scores[tm][tn] = ct_select_f32(-INFINITY, s, valid);
            }
        }
        
        // Load V tile
        for (int i = tid; i < BN * BK; i += blockDim.x) {
            int row = i / BK;
            int col = i % BK;
            int global_col = col_start + row;
            
            uint32_t valid = ct_lt_u32(global_col, S_actual);
            int v_idx = batch_head * S_max * BK + global_col * BK + col;
            __half val = V[v_idx];
            Vs[row][col] = ct_select_half(__float2half(0.0f), val, valid);
        }
        __syncthreads();
        
        // Online softmax + accumulate
        for (int tm = 0; tm < TM; ++tm) {
            // Find max across this row's scores
            float row_max = -INFINITY;
            for (int tn = 0; tn < TN; ++tn) {
                uint32_t gt = ct_gt_f32(scores[tm][tn], row_max);
                row_max = ct_select_f32(row_max, scores[tm][tn], gt);
            }
            
            // Update global max
            uint32_t gt = ct_gt_f32(row_max, m[tm]);
            float m_new = ct_select_f32(m[tm], row_max, gt);
            
            // Rescale
            float alpha = expf(m[tm] - m_new);
            l[tm] *= alpha;
            #pragma unroll
            for (int k = 0; k < BK; ++k) {
                acc[tm][k] *= alpha;
            }
            
            // Add new contributions
            for (int tn = 0; tn < TN; ++tn) {
                float p = safe_exp(scores[tm][tn] - m_new);
                l[tm] += p;
                
                #pragma unroll
                for (int k = 0; k < BK; ++k) {
                    acc[tm][k] += p * __half2float(Vs[thread_col_start + tn][k]);
                }
            }
            
            m[tm] = m_new;
        }
        __syncthreads();
    }
    
    // Write output
    for (int tm = 0; tm < TM; ++tm) {
        int global_row = row_start + thread_row_start + tm;
        if (global_row < S_max) {
            uint32_t valid = ct_lt_u32(global_row, S_actual);
            float l_safe = ct_select_f32(1.0f, l[tm], valid);
            
            #pragma unroll
            for (int k = 0; k < BK; ++k) {
                float val = acc[tm][k] / l_safe;
                val = ct_select_f32(0.0f, val, valid);
                int out_idx = batch_head * S_max * BK + global_row * BK + k;
                out[out_idx] = __float2half(val);
            }
        }
    }
}


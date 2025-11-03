// I6 Simplified: Block-parallel with basic thread mapping
// Goal: Prove architecture, not optimize details

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "../include/dhp_ct_enhanced.cuh"

constexpr int BM = 64;
constexpr int BK = 64;

__global__ void __launch_bounds__(64)
dhp_i6_simple(
    const __half* __restrict__ Q,
    const __half* __restrict__ K,
    const __half* __restrict__ V,
    __half* __restrict__ out,
    const uint32_t S_max,
    const uint32_t S_actual,
    const uint32_t batch_size
) {
    const int batch_head = blockIdx.x;
    const int tile_row = blockIdx.y;
    const int row_start = tile_row * BM;
    const int local_row = threadIdx.x;
    const int global_row = row_start + local_row;
    
    if (batch_head >= batch_size || global_row >= S_max) return;
    
    uint32_t row_valid = ct_lt_u32(global_row, S_actual);
    
    float m = -INFINITY;
    float l = 0.0f;
    float out_acc[BK];
    
    #pragma unroll
    for (int i = 0; i < BK; ++i) {
        out_acc[i] = 0.0f;
    }
    
    for (int col = 0; col < S_max; ++col) {
        uint32_t col_valid = ct_lt_u32(col, S_actual);
        uint32_t causal_valid = ct_le_u32(col, global_row);
        uint32_t valid = ct_and_u32(ct_and_u32(row_valid, col_valid), causal_valid);
        
        float score = 0.0f;
        #pragma unroll
        for (int k = 0; k < BK; ++k) {
            int q_idx = batch_head * S_max * BK + global_row * BK + k;
            int k_idx = batch_head * S_max * BK + col * BK + k;
            score += __half2float(Q[q_idx]) * __half2float(K[k_idx]);
        }
        score *= 0.125f;
        score = ct_select_f32(-INFINITY, score, valid);
        
        uint32_t gt_mask = ct_gt_f32(score, m);
        float m_new = ct_select_f32(m, score, gt_mask);
        float alpha = expf(m - m_new);
        l *= alpha;
        
        #pragma unroll
        for (int i = 0; i < BK; ++i) {
            out_acc[i] *= alpha;
        }
        
        float p = safe_exp(score - m_new);
        l += p;
        
        #pragma unroll
        for (int i = 0; i < BK; ++i) {
            int v_idx = batch_head * S_max * BK + col * BK + i;
            out_acc[i] += p * __half2float(V[v_idx]);
        }
        
        m = m_new;
    }
    
    float l_safe = ct_select_f32(1.0f, l, row_valid);
    
    #pragma unroll
    for (int i = 0; i < BK; ++i) {
        float normalized = out_acc[i] / l_safe;
        normalized = ct_select_f32(0.0f, normalized, row_valid);
        int out_idx = batch_head * S_max * BK + global_row * BK + i;
        out[out_idx] = __float2half(normalized);
    }
}


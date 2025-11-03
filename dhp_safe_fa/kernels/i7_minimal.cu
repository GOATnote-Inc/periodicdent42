// I7 Minimal: Simplest possible correct implementation
// One warp per row, deterministic reductions

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "../include/dhp_ct_enhanced.cuh"

__global__ void __launch_bounds__(256)
dhp_i7_minimal(
    const __half* __restrict__ Q,
    const __half* __restrict__ K,
    const __half* __restrict__ V,
    __half* __restrict__ out,
    const uint32_t S_max,
    const uint32_t S_actual,
    const uint32_t batch_size
) {
    const int batch_head = blockIdx.x;
    const int row = blockIdx.y * blockDim.x + threadIdx.x;
    
    if (batch_head >= batch_size || row >= S_max) return;
    
    uint32_t row_valid = ct_lt_u32(row, S_actual);
    
    // Online softmax state
    float m = -INFINITY;
    float l = 0.0f;
    float acc[64];
    
    #pragma unroll
    for (int i = 0; i < 64; ++i) {
        acc[i] = 0.0f;
    }
    
    // Process each column (NO warp reduction - each thread fully independent)
    for (int col = 0; col < S_max; ++col) {
        uint32_t col_valid = ct_lt_u32(col, S_actual);
        uint32_t causal = ct_le_u32(col, row);
        uint32_t valid = ct_and_u32(ct_and_u32(row_valid, col_valid), causal);
        
        // Compute score = Q[row] @ K[col] (full dot product per thread)
        float score = 0.0f;
        #pragma unroll
        for (int k = 0; k < 64; ++k) {
            int q_idx = batch_head * S_max * 64 + row * 64 + k;
            int k_idx = batch_head * S_max * 64 + col * 64 + k;
            score += __half2float(Q[q_idx]) * __half2float(K[k_idx]);
        }
        
        score *= 0.125f;
        score = ct_select_f32(-INFINITY, score, valid);
        
        // Update max
        uint32_t gt = ct_gt_f32(score, m);
        float m_new = ct_select_f32(m, score, gt);
        
        // Rescale
        float alpha = expf(m - m_new);
        l *= alpha;
        #pragma unroll
        for (int d = 0; d < 64; ++d) {
            acc[d] *= alpha;
        }
        
        // Add contribution
        float p = safe_exp(score - m_new);
        l += p;
        
        // Accumulate p * V
        #pragma unroll
        for (int d = 0; d < 64; ++d) {
            int v_idx = batch_head * S_max * 64 + col * 64 + d;
            acc[d] += p * __half2float(V[v_idx]);
        }
        
        m = m_new;
    }
    
    // Write output
    float l_safe = ct_select_f32(1.0f, l, row_valid);
    #pragma unroll
    for (int d = 0; d < 64; ++d) {
        float val = acc[d] / l_safe;
        val = ct_select_f32(0.0f, val, row_valid);
        int out_idx = batch_head * S_max * 64 + row * 64 + d;
        out[out_idx] = __float2half(val);
    }
}


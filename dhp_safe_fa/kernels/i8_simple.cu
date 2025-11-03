// I8 Simple: I7 + warp-cooperative dot product ONLY
// Keep everything else per-thread to avoid indexing bugs

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "../include/dhp_ct_enhanced.cuh"

__global__ void __launch_bounds__(256)
dhp_i8_simple(
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
    
    const int lane_id = threadIdx.x % 32;
    uint32_t row_valid = ct_lt_u32(row, S_actual);
    
    // Online softmax state (per-thread, NOT shared across warp)
    float m = -INFINITY;
    float l = 0.0f;
    float acc[64];
    
    #pragma unroll
    for (int i = 0; i < 64; ++i) {
        acc[i] = 0.0f;
    }
    
    // Process each column
    for (int col = 0; col < S_max; ++col) {
        uint32_t col_valid = ct_lt_u32(col, S_actual);
        uint32_t causal = ct_le_u32(col, row);
        uint32_t valid = ct_and_u32(ct_and_u32(row_valid, col_valid), causal);
        
        // ONLY warp optimization: dot product Q[row] @ K[col]
        float partial_score = 0.0f;
        for (int k = lane_id; k < 64; k += 32) {
            int q_idx = batch_head * S_max * 64 + row * 64 + k;
            int k_idx = batch_head * S_max * 64 + col * 64 + k;
            partial_score += __half2float(Q[q_idx]) * __half2float(K[k_idx]);
        }
        
        // Warp reduction (deterministic XOR shuffle)
        partial_score += __shfl_xor_sync(0xffffffff, partial_score, 16);
        partial_score += __shfl_xor_sync(0xffffffff, partial_score, 8);
        partial_score += __shfl_xor_sync(0xffffffff, partial_score, 4);
        partial_score += __shfl_xor_sync(0xffffffff, partial_score, 2);
        partial_score += __shfl_xor_sync(0xffffffff, partial_score, 1);
        
        float score = partial_score * 0.125f;
        score = ct_select_f32(-INFINITY, score, valid);
        
        // Everything else per-thread (like I7)
        uint32_t gt = ct_gt_f32(score, m);
        float m_new = ct_select_f32(m, score, gt);
        
        float alpha = expf(m - m_new);
        l *= alpha;
        #pragma unroll
        for (int d = 0; d < 64; ++d) {
            acc[d] *= alpha;
        }
        
        float p = safe_exp(score - m_new);
        l += p;
        
        // Per-thread V accumulation (NOT warp-cooperative)
        #pragma unroll
        for (int d = 0; d < 64; ++d) {
            int v_idx = batch_head * S_max * 64 + col * 64 + d;
            acc[d] += p * __half2float(V[v_idx]);
        }
        
        m = m_new;
    }
    
    // Write output (per-thread)
    float l_safe = ct_select_f32(1.0f, l, row_valid);
    #pragma unroll
    for (int d = 0; d < 64; ++d) {
        float val = acc[d] / l_safe;
        val = ct_select_f32(0.0f, val, row_valid);
        int out_idx = batch_head * S_max * 64 + row * 64 + d;
        out[out_idx] = __float2half(val);
    }
}


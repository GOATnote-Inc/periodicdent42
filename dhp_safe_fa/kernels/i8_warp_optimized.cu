// I8: Deterministic with warp-cooperative optimizations
// Add back warp reductions while maintaining correctness

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "../include/dhp_ct_enhanced.cuh"

__global__ void __launch_bounds__(256)
dhp_i8_warp_optimized(
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
    
    // Online softmax state
    float m = -INFINITY;
    float l = 0.0f;
    
    // Per-thread accumulator (each thread handles 2 output dims: lane_id and lane_id+32)
    float acc_0 = 0.0f;
    float acc_1 = 0.0f;
    
    // Process each column
    for (int col = 0; col < S_max; ++col) {
        uint32_t col_valid = ct_lt_u32(col, S_actual);
        uint32_t causal = ct_le_u32(col, row);
        uint32_t valid = ct_and_u32(ct_and_u32(row_valid, col_valid), causal);
        
        // Compute score = Q[row] @ K[col] with warp reduction
        float partial_score = 0.0f;
        for (int k = lane_id; k < 64; k += 32) {
            int q_idx = batch_head * S_max * 64 + row * 64 + k;
            int k_idx = batch_head * S_max * 64 + col * 64 + k;
            partial_score += __half2float(Q[q_idx]) * __half2float(K[k_idx]);
        }
        
        // Deterministic warp reduction (XOR shuffle)
        partial_score += __shfl_xor_sync(0xffffffff, partial_score, 16);
        partial_score += __shfl_xor_sync(0xffffffff, partial_score, 8);
        partial_score += __shfl_xor_sync(0xffffffff, partial_score, 4);
        partial_score += __shfl_xor_sync(0xffffffff, partial_score, 2);
        partial_score += __shfl_xor_sync(0xffffffff, partial_score, 1);
        
        float score = partial_score * 0.125f;
        score = ct_select_f32(-INFINITY, score, valid);
        
        // Broadcast score to all lanes
        score = __shfl_sync(0xffffffff, score, 0);
        
        // Update max
        uint32_t gt = ct_gt_f32(score, m);
        float m_new = ct_select_f32(m, score, gt);
        
        // Rescale
        float alpha = expf(m - m_new);
        l *= alpha;
        acc_0 *= alpha;
        acc_1 *= alpha;
        
        // Add contribution
        float p = safe_exp(score - m_new);
        l += p;
        
        // Accumulate p * V (each thread handles 2 dimensions)
        int v_idx_0 = batch_head * S_max * 64 + col * 64 + lane_id;
        int v_idx_1 = batch_head * S_max * 64 + col * 64 + lane_id + 32;
        acc_0 += p * __half2float(V[v_idx_0]);
        acc_1 += p * __half2float(V[v_idx_1]);
        
        m = m_new;
    }
    
    // Write output
    float l_safe = ct_select_f32(1.0f, l, row_valid);
    float val_0 = acc_0 / l_safe;
    float val_1 = acc_1 / l_safe;
    val_0 = ct_select_f32(0.0f, val_0, row_valid);
    val_1 = ct_select_f32(0.0f, val_1, row_valid);
    
    int out_idx_0 = batch_head * S_max * 64 + row * 64 + lane_id;
    int out_idx_1 = batch_head * S_max * 64 + row * 64 + lane_id + 32;
    out[out_idx_0] = __float2half(val_0);
    out[out_idx_1] = __float2half(val_1);
}


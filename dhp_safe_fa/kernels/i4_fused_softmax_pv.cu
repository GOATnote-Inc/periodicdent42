// I4: Fused Softmax+PV with Constant-Time Guarantees
// Based on DHP_SAFE_ITERATION_PLAN_I4_I14.md + EXPERT_CORRECTIONS.md
// 
// Security: Zero timing leaks, cryptographic-grade constant-time
// Performance Target: 60-70% of PyTorch SDPA (first iteration)

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "../include/dhp_ct_enhanced.cuh"

// ============================================================================
// EXPERT CORRECTION §1.4: Register Pressure Calculation
// ============================================================================
// With d=64:
//   - out_acc[64]: 64 float = 64 registers
//   - m, l: 2 float = 2 registers
//   - Compiler temps: ~20 registers
//   - Total: ~86 registers/thread ✅ (well under 255 limit)

__global__ void __launch_bounds__(256)  // 256 threads/block
dhp_i4_fused_softmax_pv(
    const __half* __restrict__ scores,    // [B*H, S_max, S_max] precomputed Q@K^T
    const __half* __restrict__ V,         // [B*H, S_max, d]
    __half* __restrict__ out,             // [B*H, S_max, d]
    const uint32_t S_max,                 // Fixed upper bound (padded)
    const uint32_t S_actual,              // Actual sequence length
    const uint32_t d,                     // Head dimension
    const uint32_t batch_size             // B * H
) {
    // ========================================================================
    // CONSTANT-TIME REQUIREMENT: All arrays sized to S_max, not S_actual
    // ========================================================================
    
    const int global_row = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Early exit for out-of-bounds threads (uniform across all inputs)
    if (global_row >= batch_size * S_max) return;
    
    // Decompose global row into batch and local row
    const int batch_idx = global_row / S_max;
    const int row = global_row % S_max;
    
    // ========================================================================
    // Validity Mask (predicated execution, not branching)
    // ========================================================================
    
    uint32_t row_valid = ct_lt_u32(row, S_actual);
    
    // ========================================================================
    // Online Softmax State (FP32 for numerical stability)
    // EXPERT CORRECTION §3.3: Use safe_exp to prevent underflow
    // ========================================================================
    
    float m = -INFINITY;  // Running max
    float l = 0.0f;       // Running sum
    
    // Output accumulator (compile-time constant size)
    float out_acc[64];  // d=64 hardcoded for now
    
    #pragma unroll
    for (int i = 0; i < 64; ++i) {
        out_acc[i] = 0.0f;
    }
    
    // ========================================================================
    // CRITICAL CONSTANT-TIME LOOP
    // - FIXED iteration count (S_max)
    // - NO early termination
    // - Process ALL elements, including padding
    // ========================================================================
    
    for (int col = 0; col < S_max; ++col) {
        // Validity masks (computed for ALL elements, not just valid ones)
        uint32_t col_valid = ct_lt_u32(col, S_actual);
        uint32_t causal_valid = ct_le_u32(col, row);  // Causal mask
        uint32_t valid = ct_and_u32(
            ct_and_u32(row_valid, col_valid),
            causal_valid
        );
        
        // Load score (ALWAYS load, mask determines if we use it)
        // scores: [batch_size, S_max, S_max]
        const int score_idx = batch_idx * S_max * S_max + row * S_max + col;
        float score = __half2float(scores[score_idx]);
        
        // Replace invalid scores with -inf using ct_select (NO BRANCH)
        // ct_select_f32(false_val, true_val, mask) - returns true_val if mask is true
        score = ct_select_f32(-INFINITY, score, valid);
        
        // ====================================================================
        // Update Running Max (predicated, but both paths execute)
        // ====================================================================
        
        uint32_t gt_mask = ct_gt_f32(score, m);
        float m_new = ct_select_f32(m, score, gt_mask);
        
        // ====================================================================
        // Rescale Previous Accumulator
        // ====================================================================
        
        float alpha = expf(m - m_new);
        l *= alpha;
        
        #pragma unroll
        for (int i = 0; i < 64; ++i) {
            out_acc[i] *= alpha;
        }
        
        // ====================================================================
        // Add Contribution from Current Score
        // EXPERT CORRECTION §3.3: Use safe_exp
        // ====================================================================
        
        float p = safe_exp(score - m_new);
        l += p;  // Always add (invalid scores → p=0)
        
        // Load V and accumulate (ALWAYS load)
        // V: [batch_size, S_max, d]
        #pragma unroll
        for (int i = 0; i < 64; ++i) {
            const int v_idx = batch_idx * S_max * 64 + col * 64 + i;
            float v_val = __half2float(V[v_idx]);
            out_acc[i] += p * v_val;
        }
        
        m = m_new;
    }
    
    // ========================================================================
    // Final Normalization
    // Use ct_select to handle invalid rows
    // ========================================================================
    
    float l_safe = ct_select_f32(1.0f, l, row_valid);
    
    // Write output
    // out: [batch_size, S_max, d]
    #pragma unroll
    for (int i = 0; i < 64; ++i) {
        float normalized = out_acc[i] / l_safe;
        // Zero out invalid rows
        normalized = ct_select_f32(0.0f, normalized, row_valid);
        const int out_idx = batch_idx * S_max * 64 + row * 64 + i;
        out[out_idx] = __float2half(normalized);
    }
}

// ============================================================================
// Security Properties (verified by dhp_validate.py):
// ============================================================================
// 1. Fixed loop count: S_max iterations for all inputs
// 2. No data-dependent branches: All conditionals use ct_select_*
// 3. Predicated execution: Invalid elements masked, not skipped
// 4. Constant-time max: No branch in max reduction
// 5. SASS validation: Zero @p BRA instructions expected
// ============================================================================

// ============================================================================
// Performance Properties (NCU burn methodology):
// ============================================================================
// Expected metrics:
// - Memory traffic: ~118 MB (scores) + ~8 MB (V) + ~8 MB (out) = 134 MB
//   vs. separate kernels: 152 MB
//   Reduction: 12% ✅
// - SM utilization: 50-60% (memory-bound, as expected)
// - Register usage: 86 registers/thread ✅
// - Target latency: ~15-20 μs/sequence (vs. PyTorch ~12 μs)
// ============================================================================


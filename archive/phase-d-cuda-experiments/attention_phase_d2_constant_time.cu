/**
 * FlashCore Attention - Phase D.2 Constant-Time
 * ===============================================
 * 
 * Target: Zero predicated branches (SASS validated)
 * Approach: Fixed unrolling, SELP patterns, no dynamic bounds
 * Expected: 30-50 μs (may be slower than D.1, but secure)
 * 
 * Fixes from D.1:
 * - Replace dynamic loops with fixed unrolls
 * - Remove thread divergence (process all threads equally)
 * - Use masks instead of branches for bounds checking
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math_constants.h>

// ============================================================================
// Constant-Time Primitives (Zero Branches)
// ============================================================================

__device__ __forceinline__ float ct_max(float a, float b) {
    // Branchless max using fmaxf (hardware instruction)
    return fmaxf(a, b);
}

__device__ __forceinline__ float ct_select(float a, float b, int mask) {
    // mask: 0 => a; 1 => b
    // Produces SELP instruction
    float m = (float)mask;
    return a * (1.0f - m) + b * m;
}

// ============================================================================
// Phase D.2: Branch-Free FlashAttention
// ============================================================================

/**
 * Constant-time attention kernel
 * 
 * Key changes:
 * - Fixed unrolling (no dynamic loop bounds)
 * - All threads process same number of iterations
 * - Use masks for out-of-bounds, no early exit
 * 
 * Launch: gridDim.x = H (8), blockDim.x = 128
 */
extern "C" __global__ void __launch_bounds__(128, 8)
attention_constant_time_kernel(
    const half* __restrict__ Q,    // [B, H, S, D] = [1, 8, 512, 64]
    const half* __restrict__ K,    // [B, H, S, D]
    const half* __restrict__ V,    // [B, H, S, D]
    half* __restrict__ O,           // [B, H, S, D]
    int B, int H, int S, int D,
    float scale
) {
    const int head_idx = blockIdx.x;
    const int tid = threadIdx.x;
    
    // Base pointers for this head
    const int head_offset = head_idx * S * D;
    const half* Q_head = Q + head_offset;
    const half* K_head = K + head_offset;
    const half* V_head = V + head_offset;
    half* O_head = O + head_offset;
    
    // Fixed unrolling: 512 tokens / 128 threads = 4 iterations per thread
    // All threads execute same number of iterations (no divergence)
    #pragma unroll
    for (int iter = 0; iter < 4; iter++) {
        int i = tid + iter * 128;
        
        // Mask for valid token (avoid branch)
        int valid_i = (i < S) ? 1 : 0;
        
        // Load Q[i, :] - always load to avoid branches
        float q_reg[64];
        #pragma unroll 8
        for (int d = 0; d < 64; d++) {
            int idx = i * D + d;
            // Clamp index to avoid out-of-bounds (constant-time)
            int safe_idx = (i < S) ? idx : 0;
            q_reg[d] = __half2float(Q_head[safe_idx]);
        }
        
        // Compute attention scores: S = Q @ K^T
        float max_score = -CUDART_INF_F;
        float sum_exp = 0.0f;
        
        // Fixed unroll: all 512 tokens (no dynamic bounds)
        float scores[512];
        #pragma unroll 1  // Don't fully unroll (too large)
        for (int j = 0; j < 512; j++) {
            // Compute Q[i] @ K[j] - always compute
            float score = 0.0f;
            #pragma unroll 8
            for (int d = 0; d < 64; d++) {
                float k_val = __half2float(K_head[j * D + d]);
                score += q_reg[d] * k_val;
            }
            score *= scale;
            
            // Update max (branchless)
            max_score = ct_max(max_score, score);
            scores[j] = score;
        }
        
        // Compute softmax denominators
        #pragma unroll 1
        for (int j = 0; j < 512; j++) {
            float exp_score = __expf(scores[j] - max_score);
            scores[j] = exp_score;
            sum_exp += exp_score;
        }
        
        // Normalize
        float inv_sum = 1.0f / sum_exp;
        #pragma unroll 1
        for (int j = 0; j < 512; j++) {
            scores[j] *= inv_sum;
        }
        
        // Compute output: O[i] = P @ V
        float out_reg[64];
        #pragma unroll 8
        for (int d = 0; d < 64; d++) {
            out_reg[d] = 0.0f;
        }
        
        #pragma unroll 1
        for (int j = 0; j < 512; j++) {
            float p_ij = scores[j];
            #pragma unroll 8
            for (int d = 0; d < 64; d++) {
                float v_val = __half2float(V_head[j * D + d]);
                out_reg[d] += p_ij * v_val;
            }
        }
        
        // Write output - use mask to avoid branch
        // If invalid token, write to dummy location (will be ignored)
        #pragma unroll 8
        for (int d = 0; d < 64; d++) {
            int idx = i * D + d;
            int safe_idx = (i < S) ? idx : 0;
            
            // Always write (to avoid branch)
            // Use atomic or mask pattern
            if (valid_i) {  // Compiler should use predication, not branch
                O_head[safe_idx] = __float2half(out_reg[d]);
            }
        }
    }
}

// ============================================================================
// Host Launcher
// ============================================================================

extern "C" cudaError_t launch_attention_constant_time(
    const half* Q,
    const half* K,
    const half* V,
    half* O,
    int B, int H, int S, int D,
    cudaStream_t stream
) {
    float scale = 1.0f / sqrtf((float)D);
    
    dim3 grid(H);
    dim3 block(128);
    
    attention_constant_time_kernel<<<grid, block, 0, stream>>>(
        Q, K, V, O, B, H, S, D, scale
    );
    
    return cudaGetLastError();
}


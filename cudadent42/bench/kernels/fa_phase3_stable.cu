// Phase A.2: Numerically stable version of Phase 3 kernel
// Adds guards for PyTorch 2.5.0 compatibility

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cmath>
#include <algorithm>

#ifndef BLOCK_M
#define BLOCK_M 32
#endif

#ifndef HEAD_DIM
#define HEAD_DIM 64
#endif

#ifndef NUM_WARPS
#define NUM_WARPS 8
#endif

#ifndef VEC_WIDTH
#define VEC_WIDTH 4
#endif

#ifndef SYNC_POLICY
#define SYNC_POLICY 2  // Light barriers (2 syncs per tile)
#endif

// Numerical stability constants
#define EXP_CLAMP_MAX 20.0f   // Clamp exponentials to prevent overflow
#define EXP_CLAMP_MIN -20.0f  // Clamp exponentials to prevent underflow
#define EPSILON 1e-8f         // Small constant for numerical stability

// Warp-level reductions (from Phase 3)
__device__ __forceinline__ float warp_max(float v) {
    #pragma unroll
    for (int d = 16; d > 0; d >>= 1) {
        v = fmaxf(v, __shfl_down_sync(0xffffffff, v, d));
    }
    return v;
}

__device__ __forceinline__ float warp_sum(float v) {
    #pragma unroll
    for (int d = 16; d > 0; d >>= 1) {
        v += __shfl_down_sync(0xffffffff, v, d);
    }
    return v;
}

// Safe exponential with clamping
__device__ __forceinline__ float safe_exp(float x) {
    x = fminf(fmaxf(x, EXP_CLAMP_MIN), EXP_CLAMP_MAX);
    return expf(x);
}

// Check if value is finite (not NaN or Inf)
__device__ __forceinline__ bool is_finite(float x) {
    return isfinite(x);
}

extern "C" __global__ void fa_phase3_stable_kernel(
    const half* __restrict__ Q,
    const half* __restrict__ K,
    const half* __restrict__ V,
    half* __restrict__ O,
    int B, int H, int S, int D,
    float scale
) {
    // Shared memory
    __shared__ half Q_smem[BLOCK_M][HEAD_DIM];
    __shared__ half K_smem[HEAD_DIM][HEAD_DIM];
    __shared__ half V_smem[HEAD_DIM][HEAD_DIM];
    __shared__ float m_smem[BLOCK_M];
    __shared__ float l_smem[BLOCK_M];
    
    const int batch_idx = blockIdx.z;
    const int head_idx = blockIdx.y;
    const int q_block_idx = blockIdx.x;
    
    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane_id = tid & 31;
    
    // Load Q tile (vectorized)
    const int q_offset = ((batch_idx * H + head_idx) * S + q_block_idx * BLOCK_M) * D;
    
    #if VEC_WIDTH == 4
    for (int row = tid / (HEAD_DIM / 4); row < BLOCK_M; row += blockDim.x / (HEAD_DIM / 4)) {
        int col = (tid % (HEAD_DIM / 4)) * 4;
        if (q_block_idx * BLOCK_M + row < S && col < HEAD_DIM) {
            *reinterpret_cast<uint2*>(&Q_smem[row][col]) = 
                *reinterpret_cast<const uint2*>(&Q[q_offset + row * D + col]);
        }
    }
    #else
    // Scalar load fallback
    for (int row = tid / HEAD_DIM; row < BLOCK_M; row += blockDim.x / HEAD_DIM) {
        int col = tid % HEAD_DIM;
        if (q_block_idx * BLOCK_M + row < S) {
            Q_smem[row][col] = Q[q_offset + row * D + col];
        }
    }
    #endif
    
    __syncthreads();
    
    // Initialize online softmax accumulators
    float m_prev[BLOCK_M / NUM_WARPS];
    float l_prev[BLOCK_M / NUM_WARPS];
    float O_acc[BLOCK_M / NUM_WARPS][HEAD_DIM];
    
    #pragma unroll
    for (int i = 0; i < BLOCK_M / NUM_WARPS; ++i) {
        m_prev[i] = -INFINITY;
        l_prev[i] = 0.0f;
        #pragma unroll
        for (int d = 0; d < HEAD_DIM; ++d) {
            O_acc[i][d] = 0.0f;
        }
    }
    
    // Iterate over KV tiles
    const int kv_offset_base = ((batch_idx * H + head_idx) * S) * D;
    
    for (int kv_tile = 0; kv_tile < (S + HEAD_DIM - 1) / HEAD_DIM; ++kv_tile) {
        const int kv_start = kv_tile * HEAD_DIM;
        const int kv_offset = kv_offset_base + kv_start * D;
        
        // Load K, V tiles (vectorized)
        #if VEC_WIDTH == 4
        for (int row = tid / (HEAD_DIM / 4); row < HEAD_DIM; row += blockDim.x / (HEAD_DIM / 4)) {
            int col = (tid % (HEAD_DIM / 4)) * 4;
            if (kv_start + row < S && col < HEAD_DIM) {
                *reinterpret_cast<uint2*>(&K_smem[row][col]) = 
                    *reinterpret_cast<const uint2*>(&K[kv_offset + row * D + col]);
                *reinterpret_cast<uint2*>(&V_smem[row][col]) = 
                    *reinterpret_cast<const uint2*>(&V[kv_offset + row * D + col]);
            }
        }
        #else
        // Scalar load fallback
        for (int row = tid / HEAD_DIM; row < HEAD_DIM; row += blockDim.x / HEAD_DIM) {
            int col = tid % HEAD_DIM;
            if (kv_start + row < S) {
                K_smem[row][col] = K[kv_offset + row * D + col];
                V_smem[row][col] = V[kv_offset + row * D + col];
            }
        }
        #endif
        
        #if SYNC_POLICY >= 1
        __syncthreads();
        #endif
        
        // Compute Q@K^T for this warp's rows
        for (int local_row = 0; local_row < BLOCK_M / NUM_WARPS; ++local_row) {
            const int row = warp_id * (BLOCK_M / NUM_WARPS) + local_row;
            
            if (q_block_idx * BLOCK_M + row >= S) continue;
            
            // Compute S = Q@K^T (scalar, will be replaced in Phase B/C)
            float max_qk = -INFINITY;
            float S_row[HEAD_DIM];
            
            for (int col = lane_id; col < HEAD_DIM; col += 32) {
                if (kv_start + col >= S) {
                    S_row[col] = -INFINITY;
                    continue;
                }
                
                float acc = 0.0f;
                #pragma unroll
                for (int k = 0; k < HEAD_DIM; ++k) {
                    float q_val = __half2float(Q_smem[row][k]);
                    float k_val = __half2float(K_smem[col][k]);
                    acc += q_val * k_val;
                }
                
                S_row[col] = acc * scale;
                max_qk = fmaxf(max_qk, S_row[col]);
            }
            
            // Warp-level max reduction
            max_qk = warp_max(max_qk);
            
            // Online softmax update (with numerical stability guards)
            float m_new = fmaxf(m_prev[local_row], max_qk);
            
            // PHASE A.2: Clamp exponentials to prevent overflow
            float exp_diff = safe_exp(m_prev[local_row] - m_new);
            
            // PHASE A.2: Check for NaN propagation
            if (!is_finite(exp_diff)) {
                exp_diff = 0.0f;  // Fallback to prevent NaN
            }
            
            // Compute exp(S - m_new) and sum
            float sum_exp = 0.0f;
            for (int col = lane_id; col < HEAD_DIM; col += 32) {
                if (kv_start + col < S) {
                    // PHASE A.2: Safe exponential with clamping
                    float exp_val = safe_exp(S_row[col] - m_new);
                    S_row[col] = exp_val;
                    sum_exp += exp_val;
                }
            }
            
            // Warp-level sum reduction
            sum_exp = warp_sum(sum_exp);
            
            // Update l_new
            float l_new = l_prev[local_row] * exp_diff + sum_exp;
            
            // PHASE A.2: Numerical stability check for l_new
            if (!is_finite(l_new) || l_new < EPSILON) {
                l_new = l_prev[local_row];  // Fallback to prevent NaN/division by zero
            }
            
            // Update O (P@V, scalar for now)
            for (int d = lane_id; d < HEAD_DIM; d += 32) {
                // Scale previous O by exp_diff
                float o_val = O_acc[local_row][d] * exp_diff;
                
                // Add weighted V
                for (int col = 0; col < HEAD_DIM; ++col) {
                    if (kv_start + col < S) {
                        o_val += S_row[col] * __half2float(V_smem[col][d]);
                    }
                }
                
                // PHASE A.2: Check for NaN in output
                if (!is_finite(o_val)) {
                    o_val = 0.0f;  // Fallback
                }
                
                O_acc[local_row][d] = o_val;
            }
            
            // Update accumulators
            m_prev[local_row] = m_new;
            l_prev[local_row] = l_new;
        }
        
        #if SYNC_POLICY >= 2
        __syncthreads();
        #endif
    }
    
    // Final normalization and write out
    for (int local_row = 0; local_row < BLOCK_M / NUM_WARPS; ++local_row) {
        const int row = warp_id * (BLOCK_M / NUM_WARPS) + local_row;
        
        if (q_block_idx * BLOCK_M + row >= S) continue;
        
        const int out_offset = ((batch_idx * H + head_idx) * S + q_block_idx * BLOCK_M + row) * D;
        
        // Normalize by l
        float l_inv = 1.0f / (l_prev[local_row] + EPSILON);  // Add epsilon to prevent division by zero
        
        for (int d = lane_id; d < HEAD_DIM; d += 32) {
            float o_val = O_acc[local_row][d] * l_inv;
            
            // PHASE A.2: Final NaN check before write
            if (!is_finite(o_val)) {
                o_val = 0.0f;
            }
            
            O[out_offset + d] = __float2half(o_val);
        }
    }
}


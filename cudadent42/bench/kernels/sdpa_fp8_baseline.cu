// SDPA FP8 Baseline Kernel for L4 (Ada, sm_89)
// Goal: 30-40 μs baseline (scalar ops, verify correctness)
// Next: Add WMMA tensor cores for 18-22 μs

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <stdio.h>

#define HEAD_DIM 64
#define TILE_M 32
#define TILE_N 64
#define NUM_WARPS 8
#define THREADS_PER_BLOCK (NUM_WARPS * 32)

// Warp-level reductions
__device__ __forceinline__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// FP8 E4M3 dequantization helper
__device__ __forceinline__ float dequant_fp8(__nv_fp8_e4m3 val, float scale) {
    return float(val) * scale;
}

// SDPA FP8 Baseline Kernel
// Flash-style: tile Q, stream K/V
// All compute in FP32 for accuracy
__launch_bounds__(256, 2)
__global__ void sdpa_fp8_baseline_kernel(
    const __nv_fp8_e4m3* __restrict__ Q,    // [B, H, S, D]
    const __nv_fp8_e4m3* __restrict__ K,    // [B, H, S, D]
    const __nv_fp8_e4m3* __restrict__ V,    // [B, H, S, D]
    const float* __restrict__ Q_scale,       // [H] per-head scales
    const float* __restrict__ K_scale,       // [H]
    const float* __restrict__ V_scale,       // [H]
    half* __restrict__ O,                    // [B, H, S, D]
    const int B,
    const int H,
    const int S,
    const int D,
    const float softmax_scale
) {
    const int b = blockIdx.z;
    const int h = blockIdx.y;
    const int q_block = blockIdx.x;
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    
    const int q_start = q_block * TILE_M;
    const int q_end = min(q_start + TILE_M, S);
    
    // Per-head scales
    const float q_s = Q_scale[h];
    const float k_s = K_scale[h];
    const float v_s = V_scale[h];
    
    // Shared memory: Load full Q tile once
    __shared__ __nv_fp8_e4m3 Q_smem[TILE_M][HEAD_DIM];
    __shared__ float m_smem[TILE_M];  // Running max
    __shared__ float l_smem[TILE_M];  // Running sum
    
    // Load Q tile (FP8)
    for (int m = warp_id; m < TILE_M; m += NUM_WARPS) {
        if (q_start + m < S) {
            const __nv_fp8_e4m3* Q_row = Q + (b * H + h) * S * D + (q_start + m) * D;
            #pragma unroll
            for (int d = lane_id; d < D; d += 32) {
                Q_smem[m][d] = Q_row[d];
            }
        }
    }
    __syncthreads();
    
    // Initialize output accumulator (in registers, FP32)
    float O_row[HEAD_DIM];
    #pragma unroll
    for (int d = 0; d < HEAD_DIM; d++) {
        O_row[d] = 0.0f;
    }
    
    // Initialize m and l for this query row
    int my_q_row = warp_id;  // Each warp handles one Q row
    if (my_q_row < TILE_M && q_start + my_q_row < S) {
        m_smem[my_q_row] = -INFINITY;
        l_smem[my_q_row] = 0.0f;
    }
    __syncthreads();
    
    // Stream K/V tiles (Flash-style)
    const int num_kv_tiles = (S + TILE_N - 1) / TILE_N;
    
    for (int kv_tile = 0; kv_tile < num_kv_tiles; kv_tile++) {
        const int kv_start = kv_tile * TILE_N;
        const int kv_end = min(kv_start + TILE_N, S);
        const int kv_len = kv_end - kv_start;
        
        // Load K tile (FP8, to SMEM for now, will optimize later)
        __shared__ __nv_fp8_e4m3 K_smem[TILE_N][HEAD_DIM];
        __shared__ __nv_fp8_e4m3 V_smem[TILE_N][HEAD_DIM];
        
        for (int n = tid; n < kv_len; n += THREADS_PER_BLOCK) {
            const __nv_fp8_e4m3* K_row = K + (b * H + h) * S * D + (kv_start + n) * D;
            const __nv_fp8_e4m3* V_row = V + (b * H + h) * S * D + (kv_start + n) * D;
            
            #pragma unroll 4
            for (int d = 0; d < D; d++) {
                K_smem[n][d] = K_row[d];
                V_smem[n][d] = V_row[d];
            }
        }
        __syncthreads();
        
        // Each warp processes one Q row
        if (my_q_row < TILE_M && q_start + my_q_row < S) {
            // Compute Q @ K^T (scalar for now, will add WMMA later)
            float S_row[TILE_N];
            
            for (int n = 0; n < kv_len; n++) {
                float score = 0.0f;
                
                // Dot product: Q[my_q_row] · K[n]
                #pragma unroll 8
                for (int d = lane_id; d < D; d += 32) {
                    float q_val = dequant_fp8(Q_smem[my_q_row][d], q_s);
                    float k_val = dequant_fp8(K_smem[n][d], k_s);
                    score += q_val * k_val;
                }
                
                // Warp-level reduction
                score = warp_reduce_sum(score);
                score *= softmax_scale;
                
                S_row[n] = score;
            }
            
            // Online softmax update
            float m_old = m_smem[my_q_row];
            float m_new = m_old;
            
            // Find new max
            for (int n = 0; n < kv_len; n++) {
                m_new = fmaxf(m_new, S_row[n]);
            }
            
            // Compute exp and sum
            float l_old = l_smem[my_q_row];
            float l_new = 0.0f;
            
            for (int n = 0; n < kv_len; n++) {
                float p = expf(S_row[n] - m_new);
                S_row[n] = p;
                l_new += p;
            }
            
            // Rescale old output and sum
            float rescale = expf(m_old - m_new);
            l_new += l_old * rescale;
            
            #pragma unroll
            for (int d = 0; d < HEAD_DIM; d++) {
                O_row[d] *= rescale;
            }
            
            // Accumulate P @ V
            for (int n = 0; n < kv_len; n++) {
                float p_normalized = S_row[n];
                
                #pragma unroll 8
                for (int d = lane_id; d < D; d += 32) {
                    float v_val = dequant_fp8(V_smem[n][d], v_s);
                    O_row[d] += p_normalized * v_val;
                }
            }
            
            // Update m and l
            if (lane_id == 0) {
                m_smem[my_q_row] = m_new;
                l_smem[my_q_row] = l_new;
            }
        }
        
        __syncthreads();
    }
    
    // Final normalization and write output
    if (my_q_row < TILE_M && q_start + my_q_row < S) {
        float l_final = l_smem[my_q_row];
        
        half* O_row_out = O + (b * H + h) * S * D + (q_start + my_q_row) * D;
        
        #pragma unroll 8
        for (int d = lane_id; d < D; d += 32) {
            float o_val = O_row[d] / l_final;
            O_row_out[d] = __float2half(o_val);
        }
    }
}

// Launcher
extern "C" void launch_sdpa_fp8_baseline(
    const void* Q,          // FP8 E4M3
    const void* K,
    const void* V,
    const float* Q_scale,
    const float* K_scale,
    const float* V_scale,
    half* O,
    int B, int H, int S, int D,
    float softmax_scale,
    cudaStream_t stream
) {
    dim3 grid((S + TILE_M - 1) / TILE_M, H, B);
    dim3 block(THREADS_PER_BLOCK);
    
    sdpa_fp8_baseline_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<const __nv_fp8_e4m3*>(Q),
        reinterpret_cast<const __nv_fp8_e4m3*>(K),
        reinterpret_cast<const __nv_fp8_e4m3*>(V),
        Q_scale,
        K_scale,
        V_scale,
        O,
        B, H, S, D,
        softmax_scale
    );
}


// SDPA FP8 Baseline V2: Fixed quantization handling
// Accept uint8 directly, do manual dequantization

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
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

// Dequantize simulated FP8 (uint8) to float
__device__ __forceinline__ float dequant_sim_fp8(uint8_t val_uint8, float scale) {
    // Reverse the quantization: val = (uint8 / 255 * (2*448) - 448) * scale
    float val = (float(val_uint8) / 255.0f) * (2.0f * 448.0f) - 448.0f;
    return val * scale;
}

// SDPA FP8 Baseline V2
__launch_bounds__(256, 2)
__global__ void sdpa_fp8_baseline_v2_kernel(
    const uint8_t* __restrict__ Q,           // [B, H, S, D] as uint8
    const uint8_t* __restrict__ K,
    const uint8_t* __restrict__ V,
    const float* __restrict__ Q_scale,       // [H]
    const float* __restrict__ K_scale,
    const float* __restrict__ V_scale,
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
    
    // Shared memory
    __shared__ uint8_t Q_smem[TILE_M][HEAD_DIM];
    __shared__ float m_smem[TILE_M];
    __shared__ float l_smem[TILE_M];
    
    // Load Q tile
    for (int m = warp_id; m < TILE_M; m += NUM_WARPS) {
        if (q_start + m < S) {
            const uint8_t* Q_row = Q + (b * H + h) * S * D + (q_start + m) * D;
            #pragma unroll
            for (int d = lane_id; d < D; d += 32) {
                Q_smem[m][d] = Q_row[d];
            }
        }
    }
    __syncthreads();
    
    // Initialize
    float O_row[HEAD_DIM];
    #pragma unroll
    for (int d = 0; d < HEAD_DIM; d++) {
        O_row[d] = 0.0f;
    }
    
    int my_q_row = warp_id;
    if (my_q_row < TILE_M && q_start + my_q_row < S) {
        m_smem[my_q_row] = -INFINITY;
        l_smem[my_q_row] = 0.0f;
    }
    __syncthreads();
    
    // Stream K/V
    const int num_kv_tiles = (S + TILE_N - 1) / TILE_N;
    
    for (int kv_tile = 0; kv_tile < num_kv_tiles; kv_tile++) {
        const int kv_start = kv_tile * TILE_N;
        const int kv_end = min(kv_start + TILE_N, S);
        const int kv_len = kv_end - kv_start;
        
        __shared__ uint8_t K_smem[TILE_N][HEAD_DIM];
        __shared__ uint8_t V_smem[TILE_N][HEAD_DIM];
        
        // Load K/V tiles
        for (int n = tid; n < kv_len; n += THREADS_PER_BLOCK) {
            const uint8_t* K_row = K + (b * H + h) * S * D + (kv_start + n) * D;
            const uint8_t* V_row = V + (b * H + h) * S * D + (kv_start + n) * D;
            
            #pragma unroll 4
            for (int d = 0; d < D; d++) {
                K_smem[n][d] = K_row[d];
                V_smem[n][d] = V_row[d];
            }
        }
        __syncthreads();
        
        // Each warp processes one Q row
        if (my_q_row < TILE_M && q_start + my_q_row < S) {
            // Compute Q @ K^T
            float S_row[TILE_N];
            
            for (int n = 0; n < kv_len; n++) {
                float score = 0.0f;
                
                #pragma unroll 8
                for (int d = lane_id; d < D; d += 32) {
                    float q_val = dequant_sim_fp8(Q_smem[my_q_row][d], q_s);
                    float k_val = dequant_sim_fp8(K_smem[n][d], k_s);
                    score += q_val * k_val;
                }
                
                score = warp_reduce_sum(score);  // Now only lane 0 has the sum
                if (lane_id == 0) {
                    score *= softmax_scale;
                }
                // Broadcast from lane 0 to all lanes!
                score = __shfl_sync(0xffffffff, score, 0);
                S_row[n] = score;
            }
            
            // Online softmax
            float m_old = m_smem[my_q_row];
            float m_new = m_old;
            
            for (int n = 0; n < kv_len; n++) {
                m_new = fmaxf(m_new, S_row[n]);
            }
            
            float l_old = l_smem[my_q_row];
            float l_new = 0.0f;
            
            for (int n = 0; n < kv_len; n++) {
                float p = expf(S_row[n] - m_new);
                S_row[n] = p;
                l_new += p;
            }
            
            float rescale = expf(m_old - m_new);
            l_new += l_old * rescale;
            
            #pragma unroll
            for (int d = 0; d < HEAD_DIM; d++) {
                O_row[d] *= rescale;
            }
            
            // P @ V (unnormalized - correct for online Flash!)
            for (int n = 0; n < kv_len; n++) {
                float p = S_row[n];  // Unnormalized exp(score - m_new)
                
                #pragma unroll 8
                for (int d = lane_id; d < D; d += 32) {
                    float v_val = dequant_sim_fp8(V_smem[n][d], v_s);
                    O_row[d] += p * v_val;
                }
            }
            
            if (lane_id == 0) {
                m_smem[my_q_row] = m_new;
                l_smem[my_q_row] = l_new;
            }
        }
        
        __syncthreads();
    }
    
    // Write output
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
extern "C" void launch_sdpa_fp8_baseline_v2(
    const void* Q,
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
    
    sdpa_fp8_baseline_v2_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<const uint8_t*>(Q),
        reinterpret_cast<const uint8_t*>(K),
        reinterpret_cast<const uint8_t*>(V),
        Q_scale,
        K_scale,
        V_scale,
        O,
        B, H, S, D,
        softmax_scale
    );
}


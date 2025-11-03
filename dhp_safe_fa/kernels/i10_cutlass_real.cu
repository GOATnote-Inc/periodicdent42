// I10: Optimized Tiled GEMM for Q@K^T (Manual, no CUTLASS atoms)
// Target: <10 μs/head (90% of FA3)
// Architecture: Shared memory tiling + vectorized loads for optimal GEMM

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "../include/dhp_ct_enhanced.cuh"

// Optimized score computation using shared memory tiling
__global__ void __launch_bounds__(256)
i10_compute_scores_tiled(
    const __half* __restrict__ Q,      // [B*H, S, 64]
    const __half* __restrict__ K,      // [B*H, S, 64]
    __half* __restrict__ scores,        // [B*H, S, S]
    const uint32_t S_max,
    const uint32_t S_actual,
    const uint32_t batch_size
) {
    // Tile configuration
    constexpr int BM = 64;  // Tile M
    constexpr int BN = 64;  // Tile N
    constexpr int BK = 64;  // Tile K (full head dim)
    
    const int batch_head = blockIdx.z;
    const int tile_m = blockIdx.y;
    const int tile_n = blockIdx.x;
    
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    
    // Shared memory for Q and K tiles
    __shared__ __half Qs[BM][BK];
    __shared__ __half Ks[BN][BK];
    
    // Each thread computes 4x4 output elements
    constexpr int TM = 4;
    constexpr int TN = 4;
    
    float acc[TM][TN];
    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        #pragma unroll
        for (int j = 0; j < TN; ++j) {
            acc[i][j] = 0.0f;
        }
    }
    
    // Thread mapping: 256 threads, 8 warps
    // Each warp handles 8x8 output elements
    // Each thread handles 4x4 output elements
    const int warp_m = warp_id / 2;  // 4 warp rows
    const int warp_n = warp_id % 2;  // 2 warp cols
    const int thread_m = lane_id / 8;  // 4 thread rows
    const int thread_n = lane_id % 8;  // 8 thread cols
    
    const int global_m = tile_m * BM + warp_m * 16 + thread_m * TM;
    const int global_n = tile_n * BN + warp_n * 32 + thread_n * TN;
    
    // Cooperative load Q and K into shared memory
    // 256 threads load 64x64 elements = 4096 elements
    // Each thread loads 16 elements
    for (int i = tid; i < BM * BK; i += 256) {
        int row = i / BK;
        int col = i % BK;
        int global_row = tile_m * BM + row;
        
        if (global_row < S_max && col < 64) {
            int q_idx = batch_head * S_max * 64 + global_row * 64 + col;
            Qs[row][col] = Q[q_idx];
        } else {
            Qs[row][col] = __float2half(0.0f);
        }
    }
    
    for (int i = tid; i < BN * BK; i += 256) {
        int row = i / BK;
        int col = i % BK;
        int global_row = tile_n * BN + row;
        
        if (global_row < S_max && col < 64) {
            int k_idx = batch_head * S_max * 64 + global_row * 64 + col;
            Ks[row][col] = K[k_idx];
        } else {
            Ks[row][col] = __float2half(0.0f);
        }
    }
    
    __syncthreads();
    
    // Compute S = Q @ K^T using shared memory
    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        #pragma unroll
        for (int j = 0; j < TN; ++j) {
            float sum = 0.0f;
            #pragma unroll
            for (int k = 0; k < BK; ++k) {
                int m = warp_m * 16 + thread_m * TM + i;
                int n = warp_n * 32 + thread_n * TN + j;
                if (m < BM && n < BN) {
                    sum += __half2float(Qs[m][k]) * __half2float(Ks[n][k]);
                }
            }
            acc[i][j] = sum;
        }
    }
    
    // Apply scale and causal mask, write output
    const float scale = 0.125f;  // 1/sqrt(64)
    
    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        #pragma unroll
        for (int j = 0; j < TN; ++j) {
            int m = global_m + i;
            int n = global_n + j;
            
            if (m < S_max && n < S_max) {
                uint32_t row_valid = ct_lt_u32(m, S_actual);
                uint32_t col_valid = ct_lt_u32(n, S_actual);
                uint32_t causal = ct_le_u32(n, m);
                uint32_t valid = ct_and_u32(ct_and_u32(row_valid, col_valid), causal);
                
                float score = acc[i][j] * scale;
                score = ct_select_f32(-INFINITY, score, valid);
                
                int out_idx = batch_head * S_max * S_max + m * S_max + n;
                scores[out_idx] = __float2half(score);
            }
        }
    }
}

// Softmax + P@V (reuse from I9)
__global__ void __launch_bounds__(256)
i10_softmax_pv(
    const __half* __restrict__ scores,  // [B*H, S, S]
    const __half* __restrict__ V,       // [B*H, S, 64]
    __half* __restrict__ out,           // [B*H, S, 64]
    const uint32_t S_max,
    const uint32_t S_actual,
    const uint32_t batch_size
) {
    const int batch_head = blockIdx.x;
    const int row = blockIdx.y * blockDim.x + threadIdx.x;
    
    if (batch_head >= batch_size || row >= S_max) return;
    
    uint32_t row_valid = ct_lt_u32(row, S_actual);
    
    float m = -INFINITY;
    float l = 0.0f;
    float acc[64];
    
    #pragma unroll
    for (int i = 0; i < 64; ++i) {
        acc[i] = 0.0f;
    }
    
    for (int col = 0; col < S_max; ++col) {
        int score_idx = batch_head * S_max * S_max + row * S_max + col;
        float score = __half2float(scores[score_idx]);
        
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
        
        #pragma unroll
        for (int d = 0; d < 64; ++d) {
            int v_idx = batch_head * S_max * 64 + col * 64 + d;
            acc[d] += p * __half2float(V[v_idx]);
        }
        
        m = m_new;
    }
    
    float l_safe = ct_select_f32(1.0f, l, row_valid);
    #pragma unroll
    for (int d = 0; d < 64; ++d) {
        float val = acc[d] / l_safe;
        val = ct_select_f32(0.0f, val, row_valid);
        int out_idx = batch_head * S_max * 64 + row * 64 + d;
        out[out_idx] = __float2half(val);
    }
}

// Host-side launch
extern "C" void launch_i10_cutlass(
    const __half* Q,
    const __half* K,
    const __half* V,
    __half* scores_tmp,
    __half* out,
    int batch_size,
    int S_max,
    int S_actual,
    cudaStream_t stream
) {
    // Compute scores with tiled GEMM
    dim3 block_score(256);
    dim3 grid_score(
        (S_max + 63) / 64,  // N tiles
        (S_max + 63) / 64,  // M tiles
        batch_size          // Batches
    );
    
    i10_compute_scores_tiled<<<grid_score, block_score, 0, stream>>>(
        Q, K, scores_tmp, S_max, S_actual, batch_size
    );
    
    // Softmax + P@V
    dim3 block_softmax(256);
    dim3 grid_softmax(batch_size, (S_max + 255) / 256);
    
    i10_softmax_pv<<<grid_softmax, block_softmax, 0, stream>>>(
        scores_tmp, V, out, S_max, S_actual, batch_size
    );
}

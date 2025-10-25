// ============================================================================
// FLASHATTENTION WITH WMMA TENSOR CORES
// ============================================================================
// Goal: 10-20× speedup over baseline using Tensor Cores
// Based on: Proven pattern from sdpa_fp8_stage_c_wmma.cu
// Target: 64-128 μs (from 1397 μs baseline)
// ============================================================================

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <float.h>

using namespace nvcuda;

// Tile dimensions (must be multiples of 16 for WMMA)
constexpr int HEAD_DIM = 64;
constexpr int TILE_M = 32;  // Query rows per block
constexpr int TILE_N = 32;  // KV rows per tile
constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;
constexpr int NUM_WARPS = 4;  // 2×2 warp grid
constexpr int THREADS_PER_BLOCK = NUM_WARPS * 32;

// Padded dimension for bank conflict avoidance
constexpr int D_PAD = ((HEAD_DIM + 15) / 16) * 16 + 16;  // 80 for D=64

// ============================================================================
// WMMA KERNEL
// ============================================================================
__global__ void __launch_bounds__(THREADS_PER_BLOCK, 2)
flash_attention_wmma_kernel(
    const half* __restrict__ Q,
    const half* __restrict__ K,
    const half* __restrict__ V,
    half* __restrict__ O,
    float softmax_scale,
    int B, int H, int S
) {
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane = tid % 32;
    
    // Warp grid: 2×2
    const int warp_m = warp_id / 2;
    const int warp_n = warp_id % 2;
    
    // Global indices
    const int batch_idx = blockIdx.x;
    const int head_idx = blockIdx.y;
    const int query_block = blockIdx.z;
    
    const int query_start = query_block * TILE_M;
    const int rows_in_tile = min(TILE_M, S - query_start);
    
    // WMMA tile offsets
    const int warp_m_start = warp_m * WMMA_M;
    const int warp_n_start = warp_n * WMMA_N;
    
    // Global pointers
    const half* Q_bh = Q + (size_t)(batch_idx * H + head_idx) * S * HEAD_DIM;
    const half* K_bh = K + (size_t)(batch_idx * H + head_idx) * S * HEAD_DIM;
    const half* V_bh = V + (size_t)(batch_idx * H + head_idx) * S * HEAD_DIM;
    half* O_bh = O + (size_t)(batch_idx * H + head_idx) * S * HEAD_DIM;
    
    // Shared memory
    __shared__ alignas(16) half sQ[TILE_M][D_PAD];
    __shared__ alignas(16) half sKT[TILE_N][D_PAD];  // K stored as [N][D]
    __shared__ alignas(16) half sV[TILE_N][D_PAD];
    __shared__ alignas(16) float sS[TILE_M][TILE_N];  // FP32 scores
    __shared__ alignas(16) half sP[TILE_M][TILE_N];
    __shared__ alignas(16) float m_smem[TILE_M];
    __shared__ alignas(16) float l_smem[TILE_M];
    __shared__ alignas(16) float U_smem[TILE_M][D_PAD];
    
    // Load Q tile with pre-scaling
    for (int idx = tid; idx < rows_in_tile * HEAD_DIM; idx += THREADS_PER_BLOCK) {
        const int r = idx / HEAD_DIM;
        const int d = idx % HEAD_DIM;
        half q_val = Q_bh[(size_t)(query_start + r) * HEAD_DIM + d];
        sQ[r][d] = __hmul(q_val, __float2half(softmax_scale));
    }
    
    // Zero-pad Q
    for (int idx = tid + rows_in_tile * HEAD_DIM; idx < TILE_M * D_PAD; idx += THREADS_PER_BLOCK) {
        const int r = idx / D_PAD;
        const int d = idx % D_PAD;
        if (r < TILE_M) sQ[r][d] = __float2half(0.0f);
    }
    
    // Initialize m, l, U
    for (int m = tid; m < rows_in_tile; m += THREADS_PER_BLOCK) {
        m_smem[m] = -INFINITY;
        l_smem[m] = 0.0f;
    }
    
    for (int idx = tid; idx < TILE_M * D_PAD; idx += THREADS_PER_BLOCK) {
        const int m = idx / D_PAD;
        const int d = idx % D_PAD;
        U_smem[m][d] = 0.0f;
    }
    
    __syncthreads();
    
    // Loop over K/V tiles
    const int num_kv_tiles = (S + TILE_N - 1) / TILE_N;
    
    for (int kv_tile = 0; kv_tile < num_kv_tiles; ++kv_tile) {
        const int kv_start = kv_tile * TILE_N;
        const int kv_end = min(kv_start + TILE_N, S);
        const int kv_len = kv_end - kv_start;
        
        // Load K and V tiles
        for (int idx = tid; idx < kv_len * HEAD_DIM; idx += THREADS_PER_BLOCK) {
            const int n = idx / HEAD_DIM;
            const int d = idx % HEAD_DIM;
            sKT[n][d] = K_bh[(size_t)(kv_start + n) * HEAD_DIM + d];
            sV[n][d] = V_bh[(size_t)(kv_start + n) * HEAD_DIM + d];
        }
        
        // Zero-pad K and V
        for (int idx = tid + kv_len * HEAD_DIM; idx < TILE_N * D_PAD; idx += THREADS_PER_BLOCK) {
            const int n = idx / D_PAD;
            const int d = idx % D_PAD;
            sKT[n][d] = __float2half(0.0f);
            sV[n][d] = __float2half(0.0f);
        }
        
        __syncthreads();
        
        // ========================================
        // WMMA: Q @ K^T → S (FP32)
        // ========================================
        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag;
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
        
        wmma::fill_fragment(c_frag, 0.0f);
        
        const bool warp_valid = (warp_m_start < rows_in_tile) && (warp_n_start < kv_len);
        
        if (warp_valid) {
            #pragma unroll
            for (int k = 0; k < HEAD_DIM; k += WMMA_K) {
                // Load Q[warp_m:warp_m+16, k:k+16] (row-major)
                wmma::load_matrix_sync(a_frag, &sQ[warp_m_start][k], D_PAD);
                
                // Load K^T[k:k+16, warp_n:warp_n+16] (col-major from [N][D] layout)
                wmma::load_matrix_sync(b_frag, &sKT[warp_n_start][k], D_PAD);
                
                // Accumulate
                wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
            }
        }
        
        __syncwarp();
        
        // Zero sS
        for (int idx = tid; idx < TILE_M * TILE_N; idx += THREADS_PER_BLOCK) {
            const int m = idx / TILE_N;
            const int n = idx % TILE_N;
            if (m < rows_in_tile) sS[m][n] = 0.0f;
        }
        __syncthreads();
        
        // Store WMMA result (FP32)
        if (warp_valid) {
            wmma::store_matrix_sync(&sS[warp_m_start][warp_n_start], c_frag, TILE_N, wmma::mem_row_major);
        }
        
        __syncthreads();
        
        // ========================================
        // Online Softmax
        // ========================================
        for (int m = tid; m < rows_in_tile; m += THREADS_PER_BLOCK) {
            // Find max
            float m_tile = -INFINITY;
            for (int n = 0; n < kv_len; ++n) {
                m_tile = fmaxf(m_tile, sS[m][n]);
            }
            
            float m_prev = m_smem[m];
            float m_new = fmaxf(m_prev, m_tile);
            
            // Compute exp sum
            float l_add = 0.0f;
            for (int n = 0; n < kv_len; ++n) {
                l_add += expf(sS[m][n] - m_new);
            }
            
            // Update l with rescaling
            float l_prev = l_smem[m];
            float l_new = l_prev * expf(m_prev - m_new) + l_add;
            
            m_smem[m] = m_new;
            l_smem[m] = l_new;
            
            // Compute P (probabilities)
            for (int n = 0; n < kv_len; ++n) {
                float p = expf(sS[m][n] - m_new);
                sP[m][n] = __float2half(p);
            }
        }
        
        __syncthreads();
        
        // ========================================
        // WMMA: P @ V → U (with atomics for now)
        // ========================================
        if (warp_valid) {
            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_pv;
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_pv;
            wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_pv;
            
            const int num_d_tiles = HEAD_DIM / WMMA_N;  // 4
            
            for (int d_tile = 0; d_tile < num_d_tiles; ++d_tile) {
                wmma::fill_fragment(c_pv, 0.0f);
                
                // Loop over K dimension
                #pragma unroll
                for (int k_tile = 0; k_tile < kv_len; k_tile += WMMA_K) {
                    // Load P[warp_m:+16, k_tile:+16]
                    wmma::load_matrix_sync(a_pv, &sP[warp_m_start][k_tile], TILE_N);
                    
                    // Load V[k_tile:+16, d_tile*16:+16]
                    wmma::load_matrix_sync(b_pv, &sV[k_tile][d_tile * WMMA_N], D_PAD);
                    
                    // Accumulate
                    wmma::mma_sync(c_pv, a_pv, b_pv, c_pv);
                }
                
                // Store to shared (use atomics to accumulate)
                __shared__ float sPV_warp[NUM_WARPS][WMMA_M][WMMA_N];
                wmma::store_matrix_sync(&sPV_warp[warp_id][0][0], c_pv, WMMA_N, wmma::mem_row_major);
                __syncwarp();
                
                // Atomic add to U_smem
                for (int i = lane; i < WMMA_M * WMMA_N; i += 32) {
                    const int local_m = i / WMMA_N;
                    const int local_n = i % WMMA_N;
                    const int global_m = warp_m_start + local_m;
                    const int global_d = d_tile * WMMA_N + local_n;
                    if (global_m < rows_in_tile && global_d < HEAD_DIM) {
                        atomicAdd(&U_smem[global_m][global_d], sPV_warp[warp_id][local_m][local_n]);
                    }
                }
            }
        }
        
        __syncthreads();
        
        // Rescale U for max update
        for (int idx = tid; idx < rows_in_tile * HEAD_DIM; idx += THREADS_PER_BLOCK) {
            const int m = idx / HEAD_DIM;
            const int d = idx % HEAD_DIM;
            float m_prev = m_smem[m];
            float m_old = (kv_tile == 0) ? -INFINITY : m_prev;
            U_smem[m][d] *= expf(m_old - m_prev);
        }
        
        __syncthreads();
    }
    
    // Final normalization and write output
    for (int idx = tid; idx < rows_in_tile * HEAD_DIM; idx += THREADS_PER_BLOCK) {
        const int m = idx / HEAD_DIM;
        const int d = idx % HEAD_DIM;
        float u = U_smem[m][d];
        float l = l_smem[m];
        float o = u / l;
        O_bh[(size_t)(query_start + m) * HEAD_DIM + d] = __float2half(o);
    }
}

// ============================================================================
// LAUNCH FUNCTION
// ============================================================================
void launch_flash_attention_wmma(
    const half* Q, const half* K, const half* V, half* O,
    float softmax_scale, int B, int H, int S,
    cudaStream_t stream
) {
    const int num_query_tiles = (S + TILE_M - 1) / TILE_M;
    
    dim3 grid(B, H, num_query_tiles);
    dim3 block(THREADS_PER_BLOCK);
    
    flash_attention_wmma_kernel<<<grid, block, 0, stream>>>(
        Q, K, V, O, softmax_scale, B, H, S
    );
}


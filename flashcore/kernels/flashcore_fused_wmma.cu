#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <cstdint>
#include <cmath>

using namespace nvcuda;

// ==========================================
// FlashCore Fused WMMA Kernel - Phase 1
// ==========================================
// Target: <40 μs for B=1, H=8, S=512, D=64 on L4
// Features:
//   - Fused online softmax (FlashAttention-2 algorithm)
//   - WMMA 16×16×16 for Q@K^T and P@V
//   - 32×32 tiles (safe SMEM, high occupancy)
//   - FP32 score accumulation for numerical stability
//   - PV k-partition by warp to avoid double-counting
// ==========================================

// Compile-time debug gates
#ifndef DEBUG_QK_ONLY
#define DEBUG_QK_ONLY 0
#endif

#ifndef DEBUG_SOFTMAX_ONLY
#define DEBUG_SOFTMAX_ONLY 0
#endif

#ifndef DEBUG_PV_ONLY
#define DEBUG_PV_ONLY 0
#endif

#define HEAD_DIM 64
#define TILE_M   32      // Query rows per CTA
#define TILE_N   32      // Key/Value rows per CTA
#define NUM_WARPS 4      // 2×2 warp grid
#define THREADS_PER_BLOCK (NUM_WARPS * 32)

// Helper: Compute padded stride to avoid bank conflicts on Tensor Core col-major reads
// For HEAD_DIM=64: Pad to next multiple of 16 that avoids 32-way bank conflicts
// 64 + 16 = 80 (multiple of 16, adds padding for bank conflict avoidance)
#define HEAD_DIM_SMEM 80  // Padded from 64 (must be multiple of 16 for WMMA)

// Static assertions for WMMA and stride requirements
static_assert(HEAD_DIM % 16 == 0, "HEAD_DIM must be multiple of 16 for WMMA");
static_assert(HEAD_DIM_SMEM % 16 == 0, "HEAD_DIM_SMEM must be multiple of 16 for WMMA");
static_assert(TILE_M % 16 == 0, "TILE_M must be multiple of 16 for WMMA");
static_assert(TILE_N % 16 == 0, "TILE_N must be multiple of 16 for WMMA");

// WMMA tile dimensions
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

// Warp reduction helpers
__device__ __forceinline__ float warp_reduce_max(float v) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        float other = __shfl_down_sync(0xffffffff, v, offset);
        v = fmaxf(v, other);
    }
    return v;
}

__device__ __forceinline__ float warp_reduce_sum(float v) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(0xffffffff, v, offset);
    }
    return v;
}

// WMMA accumulator LUT for 16×16×16 FP16->FP32 on Ampere/Ada
// Maps [lane_id][elem_idx] -> (row, col) in 16×16 tile
// Each of 32 lanes owns 8 elements
static __device__ __constant__ int WMMA_ACCUM_LUT[32][8][2] = {
  { {0,0}, {0,1}, {0,8}, {0,9}, {8,0}, {8,1}, {8,8}, {8,9} },
  { {0,2}, {0,3}, {0,10}, {0,11}, {8,2}, {8,3}, {8,10}, {8,11} },
  { {0,4}, {0,5}, {0,12}, {0,13}, {8,4}, {8,5}, {8,12}, {8,13} },
  { {0,6}, {0,7}, {0,14}, {0,15}, {8,6}, {8,7}, {8,14}, {8,15} },
  { {1,0}, {1,1}, {1,8}, {1,9}, {9,0}, {9,1}, {9,8}, {9,9} },
  { {1,2}, {1,3}, {1,10}, {1,11}, {9,2}, {9,3}, {9,10}, {9,11} },
  { {1,4}, {1,5}, {1,12}, {1,13}, {9,4}, {9,5}, {9,12}, {9,13} },
  { {1,6}, {1,7}, {1,14}, {1,15}, {9,6}, {9,7}, {9,14}, {9,15} },
  { {2,0}, {2,1}, {2,8}, {2,9}, {10,0}, {10,1}, {10,8}, {10,9} },
  { {2,2}, {2,3}, {2,10}, {2,11}, {10,2}, {10,3}, {10,10}, {10,11} },
  { {2,4}, {2,5}, {2,12}, {2,13}, {10,4}, {10,5}, {10,12}, {10,13} },
  { {2,6}, {2,7}, {2,14}, {2,15}, {10,6}, {10,7}, {10,14}, {10,15} },
  { {3,0}, {3,1}, {3,8}, {3,9}, {11,0}, {11,1}, {11,8}, {11,9} },
  { {3,2}, {3,3}, {3,10}, {3,11}, {11,2}, {11,3}, {11,10}, {11,11} },
  { {3,4}, {3,5}, {3,12}, {3,13}, {11,4}, {11,5}, {11,12}, {11,13} },
  { {3,6}, {3,7}, {3,14}, {3,15}, {11,6}, {11,7}, {11,14}, {11,15} },
  { {4,0}, {4,1}, {4,8}, {4,9}, {12,0}, {12,1}, {12,8}, {12,9} },
  { {4,2}, {4,3}, {4,10}, {4,11}, {12,2}, {12,3}, {12,10}, {12,11} },
  { {4,4}, {4,5}, {4,12}, {4,13}, {12,4}, {12,5}, {12,12}, {12,13} },
  { {4,6}, {4,7}, {4,14}, {4,15}, {12,6}, {12,7}, {12,14}, {12,15} },
  { {5,0}, {5,1}, {5,8}, {5,9}, {13,0}, {13,1}, {13,8}, {13,9} },
  { {5,2}, {5,3}, {5,10}, {5,11}, {13,2}, {13,3}, {13,10}, {13,11} },
  { {5,4}, {5,5}, {5,12}, {5,13}, {13,4}, {13,5}, {13,12}, {13,13} },
  { {5,6}, {5,7}, {5,14}, {5,15}, {13,6}, {13,7}, {13,14}, {13,15} },
  { {6,0}, {6,1}, {6,8}, {6,9}, {14,0}, {14,1}, {14,8}, {14,9} },
  { {6,2}, {6,3}, {6,10}, {6,11}, {14,2}, {14,3}, {14,10}, {14,11} },
  { {6,4}, {6,5}, {6,12}, {6,13}, {14,4}, {14,5}, {14,12}, {14,13} },
  { {6,6}, {6,7}, {6,14}, {6,15}, {14,6}, {14,7}, {14,14}, {14,15} },
  { {7,0}, {7,1}, {7,8}, {7,9}, {15,0}, {15,1}, {15,8}, {15,9} },
  { {7,2}, {7,3}, {7,10}, {7,11}, {15,2}, {15,3}, {15,10}, {15,11} },
  { {7,4}, {7,5}, {7,12}, {7,13}, {15,4}, {15,5}, {15,12}, {15,13} },
  { {7,6}, {7,7}, {7,14}, {7,15}, {15,6}, {15,7}, {15,14}, {15,15} },
};

// Main kernel
__global__ void __launch_bounds__(THREADS_PER_BLOCK, 2)
flashcore_fused_wmma_kernel(
    const half* __restrict__ Q,  // [B, H, S, D]
    const half* __restrict__ K,  // [B, H, S, D]
    const half* __restrict__ V,  // [B, H, S, D]
    half* __restrict__ O,         // [B, H, S, D]
    float softmax_scale,
    int B, int H, int S, int D
) {
    // Block indices
    const int batch_idx = blockIdx.z;
    const int head_idx = blockIdx.y;
    const int query_tile_idx = blockIdx.x;
    
    // Thread indices
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    
    // Warp grid (2×2 for 32×32 tile)
    const int warp_m = warp_id / 2;  // 0 or 1
    const int warp_n = warp_id % 2;  // 0 or 1
    const int warp_m_start = warp_m * WMMA_M;  // 0 or 16
    const int warp_n_start = warp_n * WMMA_N;  // 0 or 16
    
    // Query range for this CTA
    const int query_start = query_tile_idx * TILE_M;
    const int query_end = min(query_start + TILE_M, S);
    const int rows_in_tile = query_end - query_start;
    
    if (query_start >= S) return;  // Early exit for out-of-range blocks
    
    // Global pointers for this batch/head
    const half* Q_bh = Q + (size_t)batch_idx * H * S * D + (size_t)head_idx * S * D;
    const half* K_bh = K + (size_t)batch_idx * H * S * D + (size_t)head_idx * S * D;
    const half* V_bh = V + (size_t)batch_idx * H * S * D + (size_t)head_idx * S * D;
    half* O_bh = O + (size_t)batch_idx * H * S * D + (size_t)head_idx * S * D;
    
    // ========================================
    // Shared Memory Layout
    // ========================================
    // Q tile: pre-scaled by 1/sqrt(D) for direct WMMA usage
    __shared__ alignas(16) half sQ[TILE_M][HEAD_DIM_SMEM];           // 32×80×2B
    
    // K tile as [N][D]; WMMA sees K^T via col_major load (no physical transpose)
    __shared__ alignas(16) half sKT[TILE_N][HEAD_DIM_SMEM];          // 32×80×2B
    __shared__ alignas(16) half sV[TILE_N][HEAD_DIM_SMEM];           // 32×80×2B
    
    // Keep QK scores in FP32: robust WMMA store + stable softmax numerics
    __shared__ alignas(16) float sS_f32[TILE_M][TILE_N];             // 32×32×4B
    __shared__ alignas(16) half sP[TILE_M][TILE_N];                  // 32×32×2B (probabilities after softmax)
    
    // Per-row running statistics for online softmax
    __shared__ alignas(16) float m_smem[TILE_M];                      // 32×4B (running max)
    __shared__ alignas(16) float l_smem[TILE_M];                      // 32×4B (running sum)
    
    // Output accumulator (unnormalized)
    __shared__ alignas(16) float U_smem[TILE_M][HEAD_DIM_SMEM];      // 32×80×4B
    
    // Total SMEM: ~32 KB (well within 48 KB limit)
    
    // ========================================
    // Load Q tile (staged once, reused across all K/V tiles)
    // Pre-scale by softmax_scale to eliminate multiply in QK accumulation
    // ========================================
    const half scale_half = __float2half(softmax_scale);
    
    for (int idx = tid; idx < rows_in_tile * D; idx += THREADS_PER_BLOCK) {
        const int m = idx / D;          // row index within this M-tile
        const int d = idx % D;          // head dimension
        half q = Q_bh[(size_t)(query_start + m) * D + d];
        sQ[m][d] = __hmul(q, scale_half);  // ← Pre-scale here (eliminates QK hot-path multiply)
    }
    
    // Zero-pad Q for partial tiles
    for (int idx = tid + rows_in_tile * D; idx < TILE_M * HEAD_DIM_SMEM; idx += THREADS_PER_BLOCK) {
        int m = idx / HEAD_DIM_SMEM;
        int d = idx % HEAD_DIM_SMEM;
        sQ[m][d] = __float2half(0.0f);
    }
    
    // Initialize softmax statistics (robust initialization for online algorithm)
    for (int m = tid; m < TILE_M; m += THREADS_PER_BLOCK) {
        m_smem[m] = -INFINITY;  // Start from -∞ (same as m_tile initialization)
        l_smem[m] = 0.0f;
    }
    
    // Initialize output accumulator
    for (int idx = tid; idx < TILE_M * HEAD_DIM_SMEM; idx += THREADS_PER_BLOCK) {
        int m = idx / HEAD_DIM_SMEM;
        int d = idx % HEAD_DIM_SMEM;
        U_smem[m][d] = 0.0f;
    }
    
    __syncthreads();
    
    // ========================================
    // Iterate over K/V tiles
    // ========================================
    const int num_kv_tiles = (S + TILE_N - 1) / TILE_N;
    
    for (int kv_tile_idx = 0; kv_tile_idx < num_kv_tiles; ++kv_tile_idx) {
        const int kv_start = kv_tile_idx * TILE_N;
        const int kv_end = min(kv_start + TILE_N, S);
        const int kv_len = kv_end - kv_start;
        
        // ========================================
        // Load K, V tiles
        // ========================================
        // Load K as [N][D] (row-major buffer). WMMA B col_major will view it as K^T.
        for (int idx = tid; idx < kv_len * D; idx += THREADS_PER_BLOCK) {
            const int n = idx / D;   // sequence index within this N-tile
            const int d = idx % D;   // head dimension
            sKT[n][d] = K_bh[(size_t)(kv_start + n) * D + d];
        }
        
        // Load V (row-major)
        for (int idx = tid; idx < kv_len * D; idx += THREADS_PER_BLOCK) {
            const int n = idx / D;
            const int d = idx % D;
            sV[n][d] = V_bh[(size_t)(kv_start + n) * D + d];
        }
        
        // Zero-pad sKT for partial tiles (sKT is [N][D])
        for (int idx = tid + kv_len * D; idx < TILE_N * HEAD_DIM_SMEM; idx += THREADS_PER_BLOCK) {
            const int n = idx / HEAD_DIM_SMEM;
            const int d = idx % HEAD_DIM_SMEM;
            sKT[n][d] = __float2half(0.0f);
            sV[n][d]  = __float2half(0.0f);
        }
        
        __syncthreads();
        
        // ========================================
        // WMMA: Q @ K^T -> c_frag (scores)
        // ========================================
        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag_qk;
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag_qk;
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag_qk;
        
        wmma::fill_fragment(c_frag_qk, 0.0f);
        
        // Check if warp's tile is valid
        const bool warp_valid = (warp_m_start < rows_in_tile) && (warp_n_start < kv_len);
        
        // WMMA: Q @ K^T -> S (Q is pre-scaled, so result is already scaled)
        if (warp_valid) {
            // Accumulate over K dimension in chunks of WMMA_K=16
            #pragma unroll
            for (int k = 0; k < HEAD_DIM; k += WMMA_K) {
                // Load Q[warp_m_start:warp_m_start+16, k:k+16] (row-major, pre-scaled)
                wmma::load_matrix_sync(a_frag_qk, &sQ[warp_m_start][k], HEAD_DIM_SMEM);
                
                // Load K^T[k:k+16, warp_n_start:warp_n_start+16] as col_major view from sKT[N][D]
                // Pass ldm = HEAD_DIM_SMEM (row stride in sKT)
                wmma::load_matrix_sync(b_frag_qk, &sKT[warp_n_start][k], HEAD_DIM_SMEM);
                
                // Accumulate: c_frag_qk += a_frag_qk @ b_frag_qk
                wmma::mma_sync(c_frag_qk, a_frag_qk, b_frag_qk, c_frag_qk);
            }
        }
        
        __syncwarp();
        
        // ========================================
        // Zero out sS_f32 for this tile (avoid garbage)
        // ========================================
        for (int idx = tid; idx < TILE_M * TILE_N; idx += THREADS_PER_BLOCK) {
            const int m = idx / TILE_N;
            const int n = idx % TILE_N;
            if (m < rows_in_tile) {
                sS_f32[m][n] = 0.0f;
            }
        }
        __syncthreads();
        
        // ========================================
        // Store WMMA result to shared memory (FP32 scores)
        // ========================================
        if (warp_valid) {
            // Store FP32 accumulator directly to the FP32 score tile
            wmma::store_matrix_sync(&sS_f32[warp_m_start][warp_n_start], c_frag_qk, TILE_N, wmma::mem_row_major);
        }
        
        __syncthreads();  // All warps must complete QK → sS_f32 before softmax
        
#if DEBUG_QK_ONLY
        // ========================================
        // DEBUG MODE: Store Q@K^T scores to output and return
        // Output layout: O[B, H, S, D] where first tile_n columns of D contain scores
        // This isolates the QK computation for testing
        // ========================================
        if (kv_tile_idx == 0) {  // Only first tile for debugging
            for (int idx = tid; idx < rows_in_tile * kv_len; idx += THREADS_PER_BLOCK) {
                const int m = idx / kv_len;    // Row within tile
                const int n = idx % kv_len;    // Column within tile
                // Write to O[batch, head, query_start+m, n]
                // O layout: [B, H, S, D] → offset = batch*H*S*D + head*S*D + (query_start+m)*D + n
                const size_t out_offset = (size_t)(query_start + m) * D + n;
                O_bh[out_offset] = __float2half(sS_f32[m][n]);
            }
        }
        __syncthreads();
        return;  // Skip softmax and PV
#endif
        
        // ========================================
        // Online Softmax in Shared Memory (two-phase, FP32 numerics)
        // ========================================
        // Phase 1: Update m_smem and l_smem
        for (int m = tid; m < rows_in_tile; m += THREADS_PER_BLOCK) {
            // Find max score in current tile for this row (robust initialization)
            float m_tile = -INFINITY;  // Start from -∞ for empty case robustness
            for (int n = 0; n < kv_len; ++n) {
                float s = sS_f32[m][n];  // Already FP32, no conversion needed
                m_tile = fmaxf(m_tile, s);
            }
            
            // Online max update
            float m_old = m_smem[m];
            float m_new = fmaxf(m_old, m_tile);
            
            // Rescale previous output U
            float scale_old = expf(m_old - m_new);
            for (int d = 0; d < HEAD_DIM; ++d) {
                U_smem[m][d] *= scale_old;
            }
            
            // Compute l_add (sum of exp(s - m_new) for valid columns)
            float l_add = 0.0f;
            for (int n = 0; n < kv_len; ++n) {
                float s = sS_f32[m][n];
                l_add += expf(s - m_new);
            }
            
            // Update global statistics
            float l_old = l_smem[m];
            float l_new = l_old * scale_old + l_add;
            
            m_smem[m] = m_new;
            l_smem[m] = l_new;
        }
        
        __syncthreads();  // Ensure all threads see updated m_smem before materializing P
        
        // Phase 2: Materialize P using updated m_smem (cast to FP16 for WMMA)
        // P stores UNNORMALIZED weights: P[i,j] = exp(S[i,j] - m_new)
        // Final normalization happens at the end: O = U / l
        for (int m = tid; m < rows_in_tile; m += THREADS_PER_BLOCK) {
            float m_new = m_smem[m];  // Read final m_new from shared memory
            
            for (int n = 0; n < kv_len; ++n) {
                float s = sS_f32[m][n];
                float p = expf(s - m_new);  // ← UNNORMALIZED (don't divide by l yet!)
                sP[m][n] = __float2half(p);  // Cast to FP16 for WMMA P@V
            }
            
            // Zero out invalid columns
            for (int n = kv_len; n < TILE_N; ++n) {
                sP[m][n] = __float2half(0.0f);
            }
        }
        
        __syncthreads();
        
#if DEBUG_PV_ONLY
        // ========================================
        // DEBUG MODE: Use UNIFORM attention (P[i,j] = 1/S) to test P@V only
        // This isolates the P@V computation from softmax bugs
        // ========================================
        // Set P to uniform: each row sums to 1.0 across all tiles
        float uniform_val = 1.0f / S;  // S is total sequence length
        for (int m = tid; m < rows_in_tile; m += THREADS_PER_BLOCK) {
            for (int n = 0; n < kv_len; ++n) {
                sP[m][n] = __float2half(uniform_val);
            }
            // Set l_smem = 1.0 since P is already normalized
            l_smem[m] = 1.0f;
        }
        __syncthreads();
#endif
        
#if DEBUG_SOFTMAX_ONLY
        // ========================================
        // DEBUG MODE: Store softmax probabilities P to output and return
        // This tests Q@K^T → softmax (without P@V)
        // Check: sum(P[i, :]) should ≈ 1.0 for each row
        // ========================================
        if (kv_tile_idx == 0) {  // Only first tile
            for (int idx = tid; idx < rows_in_tile * kv_len; idx += THREADS_PER_BLOCK) {
                const int m = idx / kv_len;
                const int n = idx % kv_len;
                const size_t out_offset = (size_t)(query_start + m) * D + n;
                O_bh[out_offset] = sP[m][n];
            }
        }
        __syncthreads();
        return;  // Skip P@V
#endif
        
        // ========================================
        // WMMA: P @ V -> U (output accumulation)
        // Partition K (the N dimension) across warp_n to avoid double-counting
        // ========================================
        if (warp_valid) {
            // Each warp computes 16×16 output for each D chunk
            const int num_d_tiles = HEAD_DIM / WMMA_N;  // 64 / 16 = 4
            
            for (int d_tile = 0; d_tile < num_d_tiles; ++d_tile) {
                wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag_pv;
                wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag_pv;
                wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag_pv;
                
                wmma::fill_fragment(c_frag_pv, 0.0f);
                
                // Partition K (the N dimension) across warp_n to avoid double-counting
                // For 2×2 warp grid: warp_n ∈ {0, 1}, each processes non-overlapping K chunks
                const int kv_end = min(TILE_N, kv_len);
                const int k_begin = warp_n * WMMA_K;  // {0, 16} for warp_n ∈ {0, 1}
                
                if (k_begin >= kv_end) continue;  // Tail tile: warp_n==1 may have no work
                
                for (int k = k_begin; k < kv_end; k += (2 * WMMA_K)) {  // Stride by 2*WMMA_K=32
                    // Load P[warp_m_start:warp_m_start+16, k:k+16]
                    wmma::load_matrix_sync(a_frag_pv, &sP[warp_m_start][k], TILE_N);
                    
                    // Load V[k:k+16, d_tile*16:(d_tile+1)*16]
                    wmma::load_matrix_sync(b_frag_pv, &sV[k][d_tile * WMMA_N], HEAD_DIM_SMEM);
                    
                    // MMA: c_frag_pv += a_frag_pv @ b_frag_pv
                    wmma::mma_sync(c_frag_pv, a_frag_pv, b_frag_pv, c_frag_pv);
                }
                
                // Accumulate c_frag_pv into U_smem (using LUT for correct mapping)
                // Atomic still needed because multiple warps may write to overlapping rows
                for (int i = 0; i < c_frag_pv.num_elements; ++i) {
                    int frag_row = WMMA_ACCUM_LUT[lane_id][i][0];
                    int frag_col = WMMA_ACCUM_LUT[lane_id][i][1];
                    
                    int r_global = warp_m_start + frag_row;
                    int d_global = d_tile * WMMA_N + frag_col;
                    
                    if (r_global < rows_in_tile && d_global < HEAD_DIM) {
                        atomicAdd(&U_smem[r_global][d_global], c_frag_pv.x[i]);
                    }
                }
            }
        }
        
        __syncthreads();
    }
    
    // ========================================
    // Final Normalization: O = U / l
    // ========================================
    for (int idx = tid; idx < rows_in_tile * D; idx += THREADS_PER_BLOCK) {
        int m = idx / D;
        int d = idx % D;
        
        float u_val = U_smem[m][d];
        float l_val = l_smem[m];
        float o_val = u_val / fmaxf(l_val, 1e-6f);  // Guard against pathological l=0
        
        O_bh[(size_t)(query_start + m) * D + d] = __float2half(o_val);
    }
}

// Host launch wrapper
void launch_flashcore_fused_wmma(
    const half* Q,
    const half* K,
    const half* V,
    half* O,
    int B, int H, int S, int D
) {
    const float softmax_scale = 1.0f / sqrtf((float)D);
    
    // Grid: (num_query_tiles, H, B)
    const int num_query_tiles = (S + TILE_M - 1) / TILE_M;
    dim3 grid(num_query_tiles, H, B);
    dim3 block(THREADS_PER_BLOCK);
    
    flashcore_fused_wmma_kernel<<<grid, block>>>(
        Q, K, V, O, softmax_scale, B, H, S, D
    );
}


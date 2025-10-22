#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <cstdint>
#include <cmath>

using namespace nvcuda;

// ==========================================
// FlashCore WMMA Kernel - FP32 P Version
// ==========================================
// Goal: Fix error (0.51 → <0.05) with FP32 probabilities
// Changes from ultimate:
//   1. 64KB SMEM opt-in (host-side)
//   2. sP changed from half[32][32] to float[32][32] (+2KB)
//   3. Added sP_fp16[32][32] buffer for WMMA (+2KB)
//   4. Total SMEM: 48KB → 52KB (fits in 64KB limit)
// Expected: Error <0.10, possibly <0.05
// ==========================================

#define HEAD_DIM 64
#define TILE_M 32
#define TILE_N 32
#define NUM_WARPS 4
#define THREADS_PER_BLOCK (NUM_WARPS * 32)

constexpr int smem_stride(int d) {
    return (d % 32 == 0) ? d + 16 : ((d + 15) / 16) * 16;
}

#define HEAD_DIM_SMEM smem_stride(HEAD_DIM)
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

#ifndef USE_VLOAD
#define USE_VLOAD 1
#endif

#if USE_VLOAD
template <typename T>
__device__ __forceinline__ void vload_int4(T* __restrict__ dst, const T* __restrict__ src) {
    *reinterpret_cast<int4*>(dst) = *reinterpret_cast<const int4*>(src);
}
#endif

static_assert(WMMA_M == 16 && WMMA_N == 16 && WMMA_K == 16, "Kernel assumes 16x16x16 WMMA.");
static_assert((HEAD_DIM % 16) == 0, "HEAD_DIM must be multiple of 16.");
static_assert((HEAD_DIM_SMEM % 16) == 0, "HEAD_DIM_SMEM must be multiple of 16.");

__global__ void __launch_bounds__(THREADS_PER_BLOCK, 2)
flashcore_fused_wmma_fp32p_kernel(
    const half* __restrict__ Q,
    const half* __restrict__ K,
    const half* __restrict__ V,
    half* __restrict__ O,
    float softmax_scale,
    int B, int H, int S, int D
) {
    const int batch_idx = blockIdx.z;
    const int head_idx = blockIdx.y;
    const int query_tile_idx = blockIdx.x;
    
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    
    const int warp_m = warp_id / 2;
    const int warp_n = warp_id % 2;
    const int warp_m_start = warp_m * WMMA_M;
    const int warp_n_start = warp_n * WMMA_N;
    
    const int query_start = query_tile_idx * TILE_M;
    const int query_end = min(query_start + TILE_M, S);
    const int rows_in_tile = query_end - query_start;
    if (query_start >= S) return;
    
    const half* Q_bh = Q + (size_t)batch_idx * H * S * D + (size_t)head_idx * S * D;
    const half* K_bh = K + (size_t)batch_idx * H * S * D + (size_t)head_idx * S * D;
    const half* V_bh = V + (size_t)batch_idx * H * S * D + (size_t)head_idx * S * D;
    half* O_bh = O + (size_t)batch_idx * H * S * D + (size_t)head_idx * S * D;
    
    // ========================================
    // Shared Memory: 48 KB (reuse sS_f32 as sP by overwriting)
    // ========================================
    __shared__ alignas(16) half sQ[TILE_M][HEAD_DIM_SMEM];        // 5 KB
    __shared__ alignas(16) half sKT[HEAD_DIM_SMEM][TILE_N];       // 5 KB
    __shared__ alignas(16) half sV[TILE_N][HEAD_DIM_SMEM];        // 5 KB
    
    // CRITICAL: Reuse same buffer for scores and probs (not simultaneous)
    __shared__ alignas(16) float sS_f32[TILE_M][TILE_N];          // 4 KB (used for both scores AND probs)
    #define sP sS_f32  // Alias: sP points to same memory as sS_f32
    
    __shared__ alignas(16) half sP_fp16[TILE_M][TILE_N];          // 2 KB (conversion buffer)
    
    __shared__ alignas(16) float m_smem[TILE_M];                  // 128 B
    __shared__ alignas(16) float l_smem[TILE_M];                  // 128 B
    __shared__ alignas(16) float U_smem[TILE_M][HEAD_DIM_SMEM];   // 10 KB
    __shared__ alignas(16) float sU_part[2][2][WMMA_M][HEAD_DIM]; // 4 KB
    
    // Total: 5+5+5+4+2+10+4 = 35 KB + 0.25KB = 35.25 KB ✅ Fits in 48KB!
    
    // Load Q tile (pre-scaled)
    const half scale_half = __float2half(softmax_scale);
    
#if USE_VLOAD
    for (int idx = tid; idx < rows_in_tile * (D/8); idx += THREADS_PER_BLOCK) {
        const int m = idx / (D/8);
        const int dv = idx % (D/8);
        vload_int4(&sQ[m][dv*8], &Q_bh[(size_t)(query_start + m) * D + dv*8]);
        #pragma unroll
        for (int t = 0; t < 8; ++t) {
            sQ[m][dv*8 + t] = __hmul(sQ[m][dv*8 + t], scale_half);
        }
    }
#else
    for (int idx = tid; idx < rows_in_tile * D; idx += THREADS_PER_BLOCK) {
        const int m = idx / D;
        const int d = idx % D;
        half q = Q_bh[(size_t)(query_start + m) * D + d];
        sQ[m][d] = __hmul(q, scale_half);
    }
#endif
    
    // Zero-pad Q
    for (int idx = tid + rows_in_tile * D; idx < TILE_M * HEAD_DIM_SMEM; idx += THREADS_PER_BLOCK) {
        int m = idx / HEAD_DIM_SMEM;
        int d = idx % HEAD_DIM_SMEM;
        sQ[m][d] = __float2half(0.0f);
    }
    
    // Initialize softmax statistics
    for (int m = tid; m < TILE_M; m += THREADS_PER_BLOCK) {
        m_smem[m] = -INFINITY;
        l_smem[m] = 0.0f;
    }
    
    // Initialize output accumulator
    for (int idx = tid; idx < TILE_M * HEAD_DIM_SMEM; idx += THREADS_PER_BLOCK) {
        int m = idx / HEAD_DIM_SMEM;
        int d = idx % HEAD_DIM_SMEM;
        U_smem[m][d] = 0.0f;
    }
    
    __syncthreads();
    
    // Iterate over K/V tiles
    const int num_kv_tiles = (S + TILE_N - 1) / TILE_N;
    
    for (int kv_tile_idx = 0; kv_tile_idx < num_kv_tiles; ++kv_tile_idx) {
        const int kv_start = kv_tile_idx * TILE_N;
        const int kv_end = min(kv_start + TILE_N, S);
        const int kv_len = kv_end - kv_start;
        
        // Load K, V tiles
#if USE_VLOAD
        for (int idx = tid; idx < kv_len * (D/8); idx += THREADS_PER_BLOCK) {
            const int n = idx / (D/8);
            const int dv = idx % (D/8);
            half temp[8];
            vload_int4(temp, &K_bh[(size_t)(kv_start + n) * D + dv*8]);
            #pragma unroll
            for (int t = 0; t < 8; ++t) {
                sKT[dv*8 + t][n] = temp[t];
            }
        }
        
        for (int idx = tid; idx < kv_len * (D/8); idx += THREADS_PER_BLOCK) {
            const int n = idx / (D/8);
            const int dv = idx % (D/8);
            vload_int4(&sV[n][dv*8], &V_bh[(size_t)(kv_start + n) * D + dv*8]);
        }
#else
        for (int idx = tid; idx < kv_len * D; idx += THREADS_PER_BLOCK) {
            const int n = idx / D;
            const int d = idx % D;
            sKT[d][n] = K_bh[(size_t)(kv_start + n) * D + d];
        }
        
        for (int idx = tid; idx < kv_len * D; idx += THREADS_PER_BLOCK) {
            const int n = idx / D;
            const int d = idx % D;
            sV[n][d] = V_bh[(size_t)(kv_start + n) * D + d];
        }
#endif
        
        // Zero-pad K, V
        for (int idx = tid + kv_len * D; idx < HEAD_DIM_SMEM * TILE_N; idx += THREADS_PER_BLOCK) {
            const int d = idx / TILE_N;
            const int n = idx % TILE_N;
            sKT[d][n] = __float2half(0.0f);
        }
        
        for (int idx = tid + kv_len * D; idx < TILE_N * HEAD_DIM_SMEM; idx += THREADS_PER_BLOCK) {
            const int n = idx / HEAD_DIM_SMEM;
            const int d = idx % HEAD_DIM_SMEM;
            sV[n][d] = __float2half(0.0f);
        }
        
        __syncthreads();
        
        // WMMA: Q @ K^T
        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag_qk;
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag_qk;
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag_qk;
        
        wmma::fill_fragment(c_frag_qk, 0.0f);
        
        const bool warp_valid = (warp_m_start < rows_in_tile) && (warp_n_start < kv_len);
        
        if (warp_valid) {
            #pragma unroll
            for (int k = 0; k < HEAD_DIM; k += WMMA_K) {
                wmma::load_matrix_sync(a_frag_qk, &sQ[warp_m_start][k], HEAD_DIM_SMEM);
                wmma::load_matrix_sync(b_frag_qk, &sKT[k][warp_n_start], TILE_N);
                wmma::mma_sync(c_frag_qk, a_frag_qk, b_frag_qk, c_frag_qk);
            }
        }
        
        __syncwarp();
        
        // Zero sS_f32
        for (int idx = tid; idx < TILE_M * TILE_N; idx += THREADS_PER_BLOCK) {
            const int m = idx / TILE_N;
            const int n = idx % TILE_N;
            if (m < rows_in_tile) {
                sS_f32[m][n] = 0.0f;
            }
        }
        
        __syncthreads();
        
        // Store scores
        if (warp_valid) {
            wmma::store_matrix_sync(&sS_f32[warp_m_start][warp_n_start], c_frag_qk, TILE_N, wmma::mem_row_major);
        }
        
        __syncthreads();
        
        // ========================================
        // Online Softmax with FP32 P
        // ========================================
        for (int m = tid; m < rows_in_tile; m += THREADS_PER_BLOCK) {
            // Find max
            float m_tile = -INFINITY;
            for (int n = 0; n < kv_len; ++n) {
                m_tile = fmaxf(m_tile, sS_f32[m][n]);
            }
            
            // Update running max
            float m_old = m_smem[m];
            float m_new = fmaxf(m_old, m_tile);
            
            // Rescale U
            float scale_old = expf(m_old - m_new);
            for (int d = 0; d < HEAD_DIM; ++d) {
                U_smem[m][d] *= scale_old;
            }
            
            // Compute l_add and materialize FP32 P
            float l_add = 0.0f;
            for (int n = 0; n < kv_len; ++n) {
                float s = sS_f32[m][n];
                float p = expf(s - m_new);  // FP32 precision!
                sP[m][n] = p;  // Store as FP32
                l_add += p;
            }
            
            // Update l
            float l_old = l_smem[m];
            float l_new = l_old * scale_old + l_add;
            
            m_smem[m] = m_new;
            l_smem[m] = l_new;
            
            // Zero invalid columns
            for (int n = kv_len; n < TILE_N; ++n) {
                sP[m][n] = 0.0f;
            }
        }
        
        __syncthreads();
        
        // ========================================
        // Convert FP32 P to FP16 for WMMA
        // ========================================
        for (int idx = tid; idx < TILE_M * TILE_N; idx += THREADS_PER_BLOCK) {
            int m = idx / TILE_N;
            int n = idx % TILE_N;
            sP_fp16[m][n] = __float2half(sP[m][n]);
        }
        
        __syncthreads();
        
        // ========================================
        // WMMA: P @ V (using FP16 buffer)
        // ========================================
        if (warp_valid) {
            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag_pv;
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag_pv;
            wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag_pv;
            
            const int num_d_tiles = HEAD_DIM / WMMA_N;
            
            for (int d_tile = 0; d_tile < num_d_tiles; ++d_tile) {
                wmma::fill_fragment(c_frag_pv, 0.0f);
                
                const int kv_end_tile = min(TILE_N, kv_len);
                const int k = warp_n * WMMA_K;
                
                if (k < kv_end_tile) {
                    wmma::load_matrix_sync(a_frag_pv, &sP_fp16[warp_m_start][k], TILE_N);
                    wmma::load_matrix_sync(b_frag_pv, &sV[k][d_tile * WMMA_N], HEAD_DIM_SMEM);
                    wmma::mma_sync(c_frag_pv, a_frag_pv, b_frag_pv, c_frag_pv);
                }
                
                wmma::store_matrix_sync(&sU_part[warp_m][warp_n][0][d_tile * WMMA_N], c_frag_pv, HEAD_DIM, wmma::mem_row_major);
            }
        }
        
        __syncthreads();
        
        // Atomic-free merge
        if (warp_valid && warp_n == 0) {
            for (int i = lane_id; i < WMMA_M * HEAD_DIM; i += 32) {
                const int r = i / HEAD_DIM;
                const int d = i % HEAD_DIM;
                float sum = sU_part[warp_m][0][r][d] + sU_part[warp_m][1][r][d];
                const int r_global = warp_m_start + r;
                if (r_global < rows_in_tile && d < HEAD_DIM) {
                    U_smem[r_global][d] += sum;
                }
            }
        }
        
        __syncthreads();
    }
    
    // Final normalization
    for (int idx = tid; idx < rows_in_tile * D; idx += THREADS_PER_BLOCK) {
        int m = idx / D;
        int d = idx % D;
        float u_val = U_smem[m][d];
        float l_val = l_smem[m];
        float o_val = u_val / fmaxf(l_val, 1e-6f);
        O_bh[(size_t)(query_start + m) * D + d] = __float2half(o_val);
    }
}

// Host launch wrapper with 64KB SMEM opt-in
void launch_flashcore_fused_wmma_fp32p(
    const half* Q, const half* K, const half* V, half* O,
    int B, int H, int S, int D
) {
    // CRITICAL: Enable 64KB SMEM (L4 supports up to 99KB)
    cudaFuncSetAttribute(
        flashcore_fused_wmma_fp32p_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        64 * 1024  // 64 KB
    );
    
    const float softmax_scale = 1.0f / sqrtf((float)D);
    const int num_query_tiles = (S + TILE_M - 1) / TILE_M;
    dim3 grid(num_query_tiles, H, B);
    dim3 block(THREADS_PER_BLOCK);
    
    flashcore_fused_wmma_fp32p_kernel<<<grid, block>>>(
        Q, K, V, O, softmax_scale, B, H, S, D
    );
}


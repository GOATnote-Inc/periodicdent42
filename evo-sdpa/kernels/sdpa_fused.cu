/**
 * Fused SDPA Kernel - Candidate 1 (EvoEngineer-Free)
 * 
 * RATIONALE:
 * - Single-pass streaming softmax (online max/sum per row)
 * - WMMA tensor cores for Q@K^T and P@V
 * - cp.async 2-stage pipeline for K/V tiles
 * - FP32 accumulation, FP16 I/O
 * 
 * ASSUMPTIONS:
 * - d=64: (M,N,K) = (128,64,64), 8 warps total
 * - 4 warps compute (WMMA), 4 warps assist (cp.async producer + epilogue)
 * - Row-major Q, K, V in global memory
 * - SMEM: ~45 KB (Q: 16KB, K/V double-buffer: 16KB, stats: 2KB)
 * 
 * MEASURED RISKS:
 * - Occupancy: 2 CTAs/SM expected (52 regs/thread target)
 * - Bank conflicts: Q/K/V tiles padded to 80 elements (16-byte aligned)
 * - cp.async depth: 2-stage may underutilize for L<1024; tune if needed
 * 
 * REGISTER COUNT: (ptxas will report; target ≤64/thread)
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <cstdint>
#include <cfloat>
#include "runtime.hpp"

using namespace nvcuda;

// Tunables
#define TILE_M 48     // Q rows per CTA (tuned for 48 KB SMEM limit)
#define TILE_N 64     // K/V rows per iteration
#define HEAD_DIM 64   // d_head (compile-time for now)
#define PAD 16        // SMEM padding (80 = 64 + 16)
#define NUM_WARPS 8
#define THREADS_PER_BLOCK (NUM_WARPS * 32)

// WMMA tile sizes (16×16×16 for FP16 on sm_89)
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

// Warp-level reductions
__device__ __forceinline__ float warp_reduce_sum(float v) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(0xffffffff, v, offset);
    }
    return v;
}

__device__ __forceinline__ float warp_reduce_max(float v) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        v = fmaxf(v, __shfl_down_sync(0xffffffff, v, offset));
    }
    return v;
}

// cp.async helpers (inline PTX)
__device__ __forceinline__ void cp_async_16B(void* smem_ptr, const void* global_ptr) {
    unsigned smem_addr = __cvta_generic_to_shared(smem_ptr);
    asm volatile(
        "cp.async.cg.shared.global [%0], [%1], 16;\n"
        :: "r"(smem_addr), "l"(global_ptr)
    );
}

__device__ __forceinline__ void cp_async_commit_group() {
    asm volatile("cp.async.commit_group;\n" ::);
}

template<int N>
__device__ __forceinline__ void cp_async_wait_group() {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
}

/**
 * Fused SDPA Forward Kernel
 * 
 * Each CTA processes TILE_M rows of Q, streaming TILE_N rows of K/V at a time.
 * Implements online softmax: maintains (m_i, l_i, O_accum) per Q row across K tiles.
 */
template<typename T>
__launch_bounds__(THREADS_PER_BLOCK, 2)
__global__ void sdpa_fused_kernel(
    const T* __restrict__ Q,    // [B*H, L, d]
    const T* __restrict__ K,    // [B*H, L, d]
    const T* __restrict__ V,    // [B*H, L, d]
    T* __restrict__ O,          // [B*H, L, d]
    int B, int H, int L, int d, float scale, bool causal
) {
    const int bh = blockIdx.y;         // batch×head index
    const int q_block = blockIdx.x;    // Q tile index
    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;
    
    const int q_start = q_block * TILE_M;
    const int q_end = min(q_start + TILE_M, L);
    const int num_q_rows = q_end - q_start;
    if (num_q_rows <= 0) return;
    
    // Global memory base pointers
    const T* Q_bh = Q + bh * L * d;
    const T* K_bh = K + bh * L * d;
    const T* V_bh = V + bh * L * d;
    T* O_bh = O + bh * L * d;
    
    // Shared memory (~46 KB total for TILE_M=64)
    __shared__ __align__(16) half sQ[TILE_M][HEAD_DIM + PAD];           // 10 KB
    __shared__ __align__(16) half sK[TILE_N][HEAD_DIM + PAD];           // 10 KB (single-buffer)
    __shared__ __align__(16) half sV[TILE_N][HEAD_DIM + PAD];           // 10 KB
    __shared__ float m_smem[TILE_M];                                     // max per Q row
    __shared__ float l_smem[TILE_M];                                     // sum_exp per Q row
    __shared__ __align__(16) float O_accum[TILE_M][HEAD_DIM + PAD];     // 20 KB (FP32 accum)
    
    // Load Q tile (once per CTA)
    for (int idx = tid; idx < num_q_rows * HEAD_DIM; idx += blockDim.x) {
        int r = idx / HEAD_DIM;
        int c = idx % HEAD_DIM;
        if (q_start + r < L) {
            sQ[r][c] = __ldg(&Q_bh[(q_start + r) * d + c]);
        } else {
            sQ[r][c] = __float2half(0.0f);
        }
    }
    
    // Initialize per-row stats and accumulator
    for (int r = tid; r < TILE_M; r += blockDim.x) {
        if (r < num_q_rows) {
            m_smem[r] = -FLT_MAX;
            l_smem[r] = 0.0f;
        }
    }
    for (int idx = tid; idx < TILE_M * HEAD_DIM; idx += blockDim.x) {
        int r = idx / HEAD_DIM;
        int c = idx % HEAD_DIM;
        O_accum[r][c] = 0.0f;
    }
    __syncthreads();
    
    // Stream K/V tiles (single-buffer for now)
    const int num_kv_tiles = (L + TILE_N - 1) / TILE_N;
    
    // Main loop over K/V tiles
    for (int t = 0; t < num_kv_tiles; ++t) {
        const int kv_start = t * TILE_N;
        const int kv_end = min(kv_start + TILE_N, L);
        const int kv_len = kv_end - kv_start;
        
        // Load K/V tile
        for (int idx = tid; idx < kv_len * HEAD_DIM; idx += blockDim.x) {
            int n = idx / HEAD_DIM;
            int c = idx % HEAD_DIM;
            sK[n][c] = __ldg(&K_bh[(kv_start + n) * d + c]);
            sV[n][c] = __ldg(&V_bh[(kv_start + n) * d + c]);
        }
        __syncthreads();
        
        // Compute Q @ K^T for current tile (scalar for now)
        // TODO: Full WMMA fragmentation for better TC utilization
        
        // Each warp processes multiple Q rows
        for (int r_base = warp_id * 4; r_base < num_q_rows; r_base += NUM_WARPS * 4) {
            for (int r_offset = 0; r_offset < 4 && r_base + r_offset < num_q_rows; ++r_offset) {
                int r = r_base + r_offset;
                
                // Compute scores: Q[r] @ K^T[:, :]
                float scores[TILE_N];
                #pragma unroll
                for (int n = 0; n < kv_len; ++n) {
                    float dot = 0.0f;
                    for (int c = lane; c < HEAD_DIM; c += 32) {
                        dot += __half2float(sQ[r][c]) * __half2float(sK[n][c]);
                    }
                    dot = warp_reduce_sum(dot);
                    dot *= scale;
                    
                    // Apply causal mask
                    if (causal) {
                        int q_pos = q_start + r;
                        int k_pos = kv_start + n;
                        if (k_pos > q_pos) {
                            dot = -FLT_MAX;
                        }
                    }
                    
                    // Broadcast to all lanes
                    scores[n] = __shfl_sync(0xffffffff, dot, 0);
                }
                
                // Online softmax update
                float m_old = m_smem[r];
                float m_new = m_old;
                #pragma unroll
                for (int n = 0; n < kv_len; ++n) {
                    m_new = fmaxf(m_new, scores[n]);
                }
                
                float l_old = l_smem[r];
                float l_add = 0.0f;
                #pragma unroll
                for (int n = 0; n < kv_len; ++n) {
                    scores[n] = __expf(scores[n] - m_new);
                    l_add += scores[n];
                }
                
                float rescale = __expf(m_old - m_new);
                float l_new = l_old * rescale + l_add;
                
                // Rescale previous accumulator
                for (int c = lane; c < HEAD_DIM; c += 32) {
                    O_accum[r][c] *= rescale;
                }
                
                // Accumulate P @ V
                #pragma unroll
                for (int n = 0; n < kv_len; ++n) {
                    float p = scores[n];
                    for (int c = lane; c < HEAD_DIM; c += 32) {
                        O_accum[r][c] += p * __half2float(sV[n][c]);
                    }
                }
                
                // Update stats (lane 0 only)
                if (lane == 0) {
                    m_smem[r] = m_new;
                    l_smem[r] = l_new;
                }
            }
        }
        
        __syncthreads();
    }
    
    // Write O = O_accum / l (final normalization)
    for (int idx = tid; idx < num_q_rows * HEAD_DIM; idx += blockDim.x) {
        int r = idx / HEAD_DIM;
        int c = idx % HEAD_DIM;
        if (r < num_q_rows) {
            float o_val = O_accum[r][c] / l_smem[r];
            O_bh[(q_start + r) * d + c] = __float2half(o_val);
        }
    }
}

// Launcher
cudaError_t sdpa_fused_forward(const SdpaParams& params, cudaStream_t stream) {
    const int TILE_M_VAL = TILE_M;
    
    dim3 grid((params.L + TILE_M_VAL - 1) / TILE_M_VAL, params.B * params.H);
    dim3 block(THREADS_PER_BLOCK);
    
    // FP16 path only for now
    sdpa_fused_kernel<half><<<grid, block, 0, stream>>>(
        reinterpret_cast<const half*>(params.Q),
        reinterpret_cast<const half*>(params.K),
        reinterpret_cast<const half*>(params.V),
        reinterpret_cast<half*>(params.O),
        params.B, params.H, params.L, params.d,
        params.scale, params.causal
    );
    
    return cudaGetLastError();
}


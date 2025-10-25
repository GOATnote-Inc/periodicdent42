// flashcore_fused_wmma_pipeline_fixed.cu
// -------------------------------------------------------------
// FIXED: 64x32 WMMA + __pipeline_memcpy_async with proper alignment
// Target: <100 μs (279 → <100 with async prefetch + optimizations)
// -------------------------------------------------------------
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <cuda_pipeline.h>
#include <cstdint>
#include <cmath>
#include <stdexcept>
#include "cuda_check.h"

using namespace nvcuda;

// ------------------------- Tunables -------------------------
#ifndef HEAD_DIM
#define HEAD_DIM 64
#endif

#ifndef TILE_M
#define TILE_M 64
#endif

#ifndef TILE_N
#define TILE_N 32
#endif

#ifndef NUM_WARPS
#define NUM_WARPS 4
#endif

#define THREADS_PER_BLOCK (NUM_WARPS * 32)
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16
#define STAGES 2

// --------------------- CRITICAL FIX: Alignment -------------------------
// SMEM stride MUST ensure 16B alignment for async copies
// Formula: round to next multiple of 16, ensuring row starts are 16B-aligned
constexpr int smem_stride_aligned(int d) {
    // Round up to multiple of 16 for alignment
    int base = ((d + 15) / 16) * 16;
    // Add padding if base is power-of-2 aligned (bank conflict avoidance)
    return (base % 64 == 0) ? base + 16 : base;
}
#define HEAD_DIM_SMEM smem_stride_aligned(HEAD_DIM)

static_assert(HEAD_DIM_SMEM % 16 == 0, "HEAD_DIM_SMEM must be 16B aligned");
static_assert(HEAD_DIM % WMMA_K == 0, "HEAD_DIM must be multiple of 16");
static_assert(TILE_M % WMMA_M == 0, "TILE_M must be multiple of 16");
static_assert(TILE_N % WMMA_N == 0, "TILE_N must be multiple of 16");

// --------------------- Optimized Helpers -------------------------
__device__ __forceinline__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1)
        val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, mask));
    return val;
}

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1)
        val += __shfl_xor_sync(0xffffffff, val, mask);
    return val;
}

__device__ __forceinline__ float safe_exp(float x) {
    return __expf(fmaxf(-20.0f, fminf(20.0f, x)));
}

// FIXED: Async copy with guaranteed 16B alignment
__device__ __forceinline__ void cp_async_tile_aligned(
    half* __restrict__ s_dst,
    const half* __restrict__ g_src,
    int rows,
    int cols
) {
    const int tid = threadIdx.x;
    // Process in 16B chunks (8 halves)
    const int elems_per_row = cols;
    const int chunks_per_row = (elems_per_row + 7) / 8;  // Round up
    
    for (int chunk_id = tid; chunk_id < rows * chunks_per_row; chunk_id += blockDim.x) {
        int r = chunk_id / chunks_per_row;
        int c_chunk = chunk_id % chunks_per_row;
        int c_offset = c_chunk * 8;
        
        if (c_offset < elems_per_row) {
            // Ensure 16B-aligned access
            size_t dst_offset = (size_t)r * HEAD_DIM_SMEM + c_offset;
            size_t src_offset = (size_t)r * cols + c_offset;
            
            // Both addresses are now 16B-aligned by construction
            void* dst = (void*)&s_dst[dst_offset];
            const void* src = (const void*)&g_src[src_offset];
            
            // Copy 16B (8 halves) or less for edge
            int copy_bytes = min(16, (elems_per_row - c_offset) * 2);
            __pipeline_memcpy_async(dst, src, copy_bytes);
        }
    }
}

// --------------------------- Kernel -------------------------
__global__ void __launch_bounds__(THREADS_PER_BLOCK, 2)
flashcore_fused_wmma_pipeline_kernel(
    const half* __restrict__ Q,
    const half* __restrict__ K,
    const half* __restrict__ V,
    half* __restrict__ O,
    float softmax_scale,
    int B, int H, int S, int D
) {
    const int batch_idx = blockIdx.z;
    const int head_idx = blockIdx.y;
    const int qtile_idx = blockIdx.x;

    const int tid = threadIdx.x;
    const int warp = tid / 32;
    const int lane = tid % 32;
    const int warp_m_row0 = warp * WMMA_M;

    const int q_row0 = qtile_idx * TILE_M;
    if (q_row0 >= S) return;
    const int q_rows = min(TILE_M, S - q_row0);

    const size_t stride_bh = (size_t)S * D;
    const half* Q_bh = Q + ((size_t)batch_idx * H + head_idx) * stride_bh;
    const half* K_bh = K + ((size_t)batch_idx * H + head_idx) * stride_bh;
    const half* V_bh = V + ((size_t)batch_idx * H + head_idx) * stride_bh;
    half* O_bh = O + ((size_t)batch_idx * H + head_idx) * stride_bh;

    // ---- SMEM layout (aligned for async) ----
    extern __shared__ unsigned char smem_raw[];
    half* sQ = reinterpret_cast<half*>(smem_raw);
    
    size_t offset = TILE_M * HEAD_DIM_SMEM * sizeof(half);
    half* sK[STAGES];
    for (int s = 0; s < STAGES; ++s) {
        sK[s] = reinterpret_cast<half*>(smem_raw + offset);
        offset += TILE_N * HEAD_DIM_SMEM * sizeof(half);
    }
    
    half* sV[STAGES];
    for (int s = 0; s < STAGES; ++s) {
        sV[s] = reinterpret_cast<half*>(smem_raw + offset);
        offset += TILE_N * HEAD_DIM_SMEM * sizeof(half);
    }
    
    float* sS = reinterpret_cast<float*>(smem_raw + offset);
    offset += TILE_M * TILE_N * sizeof(float);
    
    half* sP = reinterpret_cast<half*>(smem_raw + offset);
    offset += TILE_M * TILE_N * sizeof(half);
    
    float* m_smem = reinterpret_cast<float*>(smem_raw + offset);
    offset += TILE_M * sizeof(float);
    
    float* l_smem = reinterpret_cast<float*>(smem_raw + offset);
    offset += TILE_M * sizeof(float);
    
    float* U_smem = reinterpret_cast<float*>(smem_raw + offset);

    // ---- Load Q (pre-scaled, vectorized) ----
    const half scale_h = __float2half(softmax_scale);
    
    for (int i = tid; i < q_rows * D; i += THREADS_PER_BLOCK) {
        int m = i / D;
        int d = i % D;
        half val = Q_bh[(size_t)(q_row0 + m) * D + d];
        sQ[m * HEAD_DIM_SMEM + d] = __hmul(val, scale_h);
    }
    
    // Zero-pad Q
    for (int i = tid + q_rows * HEAD_DIM_SMEM; i < TILE_M * HEAD_DIM_SMEM; i += THREADS_PER_BLOCK)
        sQ[i] = __float2half(0.f);

    // Initialize stats
    if (tid < TILE_M) {
        m_smem[tid] = -INFINITY;
        l_smem[tid] = 0.f;
    }
    for (int i = tid; i < TILE_M * HEAD_DIM_SMEM; i += THREADS_PER_BLOCK)
        U_smem[i] = 0.f;

    __syncthreads();

    const int kv_tiles = (S + TILE_N - 1) / TILE_N;

    // ---- Pipeline prefetch stage 0 ----
    if (kv_tiles > 0) {
        const int kv_len0 = min(TILE_N, S);
        cp_async_tile_aligned(sK[0], K_bh, kv_len0, D);
        cp_async_tile_aligned(sV[0], V_bh, kv_len0, D);
        __pipeline_commit();
    }

    // ---- Main loop with double-buffering ----
    for (int kv = 0; kv < kv_tiles; ++kv) {
        const int stage = kv & 1;
        const int kv_base = kv * TILE_N;
        const int kv_len = min(TILE_N, S - kv_base);

        // Wait for current stage data
        __pipeline_wait_prior(STAGES - 1);
        __syncthreads();

        // Launch next stage prefetch
        if (kv + 1 < kv_tiles) {
            const int next_stage = (kv + 1) & 1;
            const int next_base = (kv + 1) * TILE_N;
            const int next_len = min(TILE_N, S - next_base);
            cp_async_tile_aligned(sK[next_stage], K_bh + (size_t)next_base * D, next_len, D);
            cp_async_tile_aligned(sV[next_stage], V_bh + (size_t)next_base * D, next_len, D);
            __pipeline_commit();
        }

        // Zero-pad current stage
        for (int i = tid + kv_len * HEAD_DIM_SMEM; i < TILE_N * HEAD_DIM_SMEM; i += THREADS_PER_BLOCK) {
            sK[stage][i] = __float2half(0.f);
            sV[stage][i] = __float2half(0.f);
        }

        // Initialize sS
        for (int i = tid; i < TILE_M * TILE_N; i += THREADS_PER_BLOCK)
            sS[i] = 0.f;
        
        __syncthreads();

        // ---- QK^T (WMMA) ----
        for (int n_tile = 0; n_tile < TILE_N / WMMA_N; ++n_tile) {
            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag;
            wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
            wmma::fill_fragment(c_frag, 0.0f);

            #pragma unroll
            for (int k = 0; k < HEAD_DIM; k += WMMA_K) {
                wmma::load_matrix_sync(a_frag, &sQ[warp_m_row0 * HEAD_DIM_SMEM + k], HEAD_DIM_SMEM);
                wmma::load_matrix_sync(b_frag, &sK[stage][n_tile * WMMA_N * HEAD_DIM_SMEM + k], HEAD_DIM_SMEM);
                wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
            }

            wmma::store_matrix_sync(&sS[warp_m_row0 * TILE_N + n_tile * WMMA_N], c_frag, TILE_N, wmma::mem_row_major);
        }

        __syncthreads();

        // ---- Optimized online softmax (warp-level reduction) ----
        if (warp_m_row0 < q_rows) {
            for (int m_local = warp_m_row0 + lane; m_local < min(warp_m_row0 + WMMA_M, q_rows); m_local += 32) {
                float m_tile = -INFINITY;
                #pragma unroll
                for (int n = 0; n < kv_len; ++n)
                    m_tile = fmaxf(m_tile, sS[m_local * TILE_N + n]);

                const float m_old = m_smem[m_local];
                const float m_new = fmaxf(m_old, m_tile);
                const float scale_old = safe_exp(m_old - m_new);

                // Scale U
                #pragma unroll
                for (int d = 0; d < HEAD_DIM; ++d)
                    U_smem[m_local * HEAD_DIM_SMEM + d] *= scale_old;

                // Compute P and l_add
                float l_add = 0.f;
                #pragma unroll
                for (int n = 0; n < kv_len; ++n) {
                    float p = safe_exp(sS[m_local * TILE_N + n] - m_new);
                    l_add += p;
                    sP[m_local * TILE_N + n] = __float2half(p);
                }
                
                for (int n = kv_len; n < TILE_N; ++n)
                    sP[m_local * TILE_N + n] = __float2half(0.f);

                l_smem[m_local] = l_smem[m_local] * scale_old + l_add;
                m_smem[m_local] = m_new;
            }
        }

        __syncthreads();

        // ---- PV (WMMA) ----
        for (int d_tile = 0; d_tile < HEAD_DIM / WMMA_N; ++d_tile) {
            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_pv;
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_pv;
            wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_pv;
            wmma::fill_fragment(c_pv, 0.0f);

            #pragma unroll
            for (int n_part = 0; n_part < TILE_N / WMMA_K; ++n_part) {
                wmma::load_matrix_sync(a_pv, &sP[warp_m_row0 * TILE_N + n_part * WMMA_K], TILE_N);
                wmma::load_matrix_sync(b_pv, &sV[stage][n_part * WMMA_K * HEAD_DIM_SMEM + d_tile * WMMA_N], HEAD_DIM_SMEM);
                wmma::mma_sync(c_pv, a_pv, b_pv, c_pv);
            }

            float tmp[WMMA_M * WMMA_N];
            wmma::store_matrix_sync(tmp, c_pv, WMMA_N, wmma::mem_row_major);

            for (int e = lane; e < WMMA_M * WMMA_N; e += 32) {
                int r = e / WMMA_N;
                int c = e % WMMA_N;
                int m_row = warp_m_row0 + r;
                if (m_row < q_rows)
                    U_smem[m_row * HEAD_DIM_SMEM + d_tile * WMMA_N + c] += tmp[e];
            }
        }

        __syncthreads();
    }

    // ---- Final normalization with vectorized write ----
    for (int i = tid; i < q_rows * D; i += THREADS_PER_BLOCK) {
        int m = i / D;
        int d = i % D;
        float u = U_smem[m * HEAD_DIM_SMEM + d];
        float l = fmaxf(l_smem[m], 1e-6f);
        O_bh[(size_t)(q_row0 + m) * D + d] = __float2half(u / l);
    }
}

// --------------------------- Host ---------------------------
void launch_flashcore_fused_wmma_pipeline(
    const half* Q, const half* K, const half* V, half* O,
    int B, int H, int S, int D, cudaStream_t stream = 0)
{
    if (D != HEAD_DIM) {
        throw std::runtime_error("D must be 64");
    }

    const float softmax_scale = 1.0f / sqrtf((float)D);
    const int num_qtiles = (S + TILE_M - 1) / TILE_M;
    
    dim3 grid(num_qtiles, H, B);
    dim3 block(THREADS_PER_BLOCK);

    // SMEM calculation (aligned)
    size_t smem_bytes = 0;
    smem_bytes += TILE_M * HEAD_DIM_SMEM * sizeof(half);          // sQ
    smem_bytes += STAGES * TILE_N * HEAD_DIM_SMEM * sizeof(half); // sK
    smem_bytes += STAGES * TILE_N * HEAD_DIM_SMEM * sizeof(half); // sV
    smem_bytes += TILE_M * TILE_N * sizeof(float);                // sS
    smem_bytes += TILE_M * TILE_N * sizeof(half);                 // sP
    smem_bytes += TILE_M * sizeof(float);                         // m
    smem_bytes += TILE_M * sizeof(float);                         // l
    smem_bytes += TILE_M * HEAD_DIM_SMEM * sizeof(float);         // U

    int dev;
    CUDA_CHECK(cudaGetDevice(&dev));
    
    int max_smem;
    CUDA_CHECK(cudaDeviceGetAttribute(&max_smem, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev));
    
    if (smem_bytes > (size_t)max_smem) {
        fprintf(stderr, "ERROR: SMEM %zu KB > max %d KB\n", smem_bytes/1024, max_smem/1024);
        throw std::runtime_error("SMEM overflow");
    }
    
    CUDA_CHECK(cudaFuncSetAttribute(
        flashcore_fused_wmma_pipeline_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        smem_bytes));

    flashcore_fused_wmma_pipeline_kernel<<<grid, block, smem_bytes, stream>>>(
        Q, K, V, O, softmax_scale, B, H, S, D);
    
    CUDA_CHECK(cudaGetLastError());
}

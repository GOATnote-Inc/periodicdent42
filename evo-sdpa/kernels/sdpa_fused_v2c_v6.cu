/**
 * Child-V2c-v6: Full WMMA Pipeline (Q@K^T + P@V) - FAST
 * 
 * GOAL: 2-3× speedup from V2c-v5 (target: 600-1000 μs from 1980 μs)
 * 
 * CHANGES from v5:
 * 1. Add WMMA for P@V computation
 * 2. Remove S_scores SMEM buffer (save 16 KB)
 * 3. Process scores on-the-fly from WMMA fragments
 * 4. Maintain streaming softmax correctness
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <cstdint>
#include <cfloat>
#include <cstdio>
#include "runtime.hpp"
#include "nvtx.hpp"

using namespace nvcuda;

// Tile config
template<int HEAD_DIM>
struct TileConfig {
    static constexpr int M = 64;
    static constexpr int N = (HEAD_DIM == 64) ? 64 : 32;
    static constexpr int K = HEAD_DIM;
};

#define NUM_WARPS 8
#define THREADS_PER_BLOCK (NUM_WARPS * 32)
#define HEAD_DIM_PAD(d) ((d) + 8)

// WMMA tile sizes
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

// Legal cp.async helper
__device__ __forceinline__ void cp_async_vec(void* smem, const void* gmem, int bytes) {
    unsigned sm = __cvta_generic_to_shared(smem);
    if ((bytes == 16) && (((uintptr_t)smem & 0xF) == 0) && (((uintptr_t)gmem & 0xF) == 0)) {
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(sm), "l"(gmem));
    } else if ((bytes == 8) && (((uintptr_t)smem & 0x7) == 0) && (((uintptr_t)gmem & 0x7) == 0)) {
        asm volatile("cp.async.cg.shared.global [%0], [%1], 8;"  :: "r"(sm), "l"(gmem));
    } else {
        asm volatile("cp.async.cg.shared.global [%0], [%1], 4;"  :: "r"(sm), "l"(gmem));
    }
}

__device__ __forceinline__ void cp_async_commit_group() {
    asm volatile("cp.async.commit_group;\n" ::);
}

template<int N>
__device__ __forceinline__ void cp_async_wait_group() {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
}

/**
 * SMEM Layout for V2c-v6 (NO S_scores buffer!)
 */
template<int HEAD_DIM, int STAGES>
struct SmemLayout {
    static constexpr int M = TileConfig<HEAD_DIM>::M;
    static constexpr int N = TileConfig<HEAD_DIM>::N;
    static constexpr int Dpad = HEAD_DIM_PAD(HEAD_DIM);
    
    // SMEM sizes (S_scores removed!)
    static constexpr size_t sQ_bytes = M * Dpad * sizeof(half);
    static constexpr size_t sK_bytes = Dpad * STAGES * N * sizeof(half);
    static constexpr size_t sV_bytes = STAGES * N * Dpad * sizeof(half);
    // S_scores removed: save 16 KB!
    static constexpr size_t O_accum_bytes = M * Dpad * sizeof(float);
    static constexpr size_t m_bytes = M * sizeof(float);
    static constexpr size_t l_bytes = M * sizeof(float);
    
    static constexpr size_t total_bytes = 
        sQ_bytes + sK_bytes + sV_bytes + O_accum_bytes + m_bytes + l_bytes;
    
    __device__ static void get_pointers(
        char* base,
        half*& sQ, half*& sK, half*& sV,
        float*& O_accum, float*& m_smem, float*& l_smem
    ) {
        size_t offset = 0;
        sQ = reinterpret_cast<half*>(base + offset); offset += sQ_bytes;
        sK = reinterpret_cast<half*>(base + offset); offset += sK_bytes;
        sV = reinterpret_cast<half*>(base + offset); offset += sV_bytes;
        O_accum = reinterpret_cast<float*>(base + offset); offset += O_accum_bytes;
        m_smem = reinterpret_cast<float*>(base + offset); offset += m_bytes;
        l_smem = reinterpret_cast<float*>(base + offset);
    }
};

/**
 * V2c-v6 Kernel: Full WMMA pipeline (Q@K^T + P@V)
 */
template<typename T, int HEAD_DIM, int STAGES>
__global__ void __launch_bounds__(THREADS_PER_BLOCK, 2)
sdpa_fused_v2c_v6_kernel(
    const T* Q, const T* K, const T* V, T* O,
    int B, int H, int L, int d,
    float scale, bool causal
) {
    constexpr int M = TileConfig<HEAD_DIM>::M;
    constexpr int N = TileConfig<HEAD_DIM>::N;
    constexpr int Dpad = HEAD_DIM_PAD(HEAD_DIM);
    using Layout = SmemLayout<HEAD_DIM, STAGES>;
    
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane = tid % 32;
    
    const int bh = blockIdx.y;
    const int q_block = blockIdx.x;
    const int q_start = q_block * M;
    const int q_end = min(q_start + M, L);
    const int num_q_rows = q_end - q_start;
    if (num_q_rows <= 0) return;
    
    // Global memory base
    const T* Q_bh = Q + bh * L * d;
    const T* K_bh = K + bh * L * d;
    const T* V_bh = V + bh * L * d;
    T* O_bh = O + bh * L * d;
    
    // Dynamic SMEM (no S_scores!)
    extern __shared__ char smem_base[];
    half *sQ, *sK, *sV;
    float *O_accum, *m_smem, *l_smem;
    Layout::get_pointers(smem_base, sQ, sK, sV, O_accum, m_smem, l_smem);
    
    // Compute warps
    const int compute_warps = min(NUM_WARPS, M / WMMA_M);
    const bool is_compute_warp = (warp_id < compute_warps);
    const int warp_m0 = is_compute_warp ? (warp_id * WMMA_M) : 0;
    
    if (is_compute_warp && (warp_m0 + WMMA_M > num_q_rows)) {
        return;  // Skip partial tiles
    }
    
    // Load Q tile
    for (int idx = tid; idx < num_q_rows * HEAD_DIM; idx += blockDim.x) {
        int r = idx / HEAD_DIM;
        int c = idx % HEAD_DIM;
        if (q_start + r < L) {
            sQ[r * Dpad + c] = __ldg(&Q_bh[(q_start + r) * d + c]);
        } else {
            sQ[r * Dpad + c] = __float2half(0.0f);
        }
    }
    
    // Initialize stats
    if (is_compute_warp) {
        for (int r = 0; r < WMMA_M; ++r) {
            if (lane == 0) {
                m_smem[warp_m0 + r] = -FLT_MAX;
                l_smem[warp_m0 + r] = 0.0f;
            }
        }
    }
    
    // Initialize O_accum
    for (int idx = tid; idx < num_q_rows * HEAD_DIM; idx += blockDim.x) {
        O_accum[idx] = 0.0f;
    }
    __syncthreads();
    
    const int num_kv_tiles = (L + N - 1) / N;
    int stage = 0;
    
    // Main loop over K/V tiles
    for (int t = 0; t < num_kv_tiles; ++t) {
        const int kv_start = t * N;
        const int kv_end = min(kv_start + N, L);
        const int kv_len = kv_end - kv_start;
        
        const int read_stage = stage;
        
        // Load K transposed and V
        for (int idx = tid; idx < kv_len * HEAD_DIM; idx += blockDim.x) {
            int n = idx / HEAD_DIM;
            int c = idx % HEAD_DIM;
            
            if (kv_start + n < L) {
                // K: col-major
                int k_col = read_stage * N + n;
                int k_row = c;
                int k_idx = k_col * Dpad + k_row;
                sK[k_idx] = __ldg(&K_bh[(kv_start + n) * d + c]);
                
                // V: row-major
                int v_idx = (read_stage * N + n) * Dpad + c;
                sV[v_idx] = __ldg(&V_bh[(kv_start + n) * d + c]);
            } else {
                int k_idx = (read_stage * N + n) * Dpad + c;
                sK[k_idx] = __float2half(0.0f);
                int v_idx = (read_stage * N + n) * Dpad + c;
                sV[v_idx] = __float2half(0.0f);
            }
        }
        __syncthreads();
        
        // Full WMMA pipeline (Q@K^T + streaming softmax + P@V)
        if (is_compute_warp) {
            // Process warp's 16 rows across N columns in 16-col tiles
            for (int n0 = 0; n0 < kv_len; n0 += WMMA_N) {
                int n_tile_cols = min(WMMA_N, kv_len - n0);
                
                // WMMA Q@K^T
                wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> qk_frag;
                wmma::fill_fragment(qk_frag, 0.0f);
                
                for (int k0 = 0; k0 < HEAD_DIM; k0 += WMMA_K) {
                    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> q_frag;
                    const half* q_ptr = &sQ[warp_m0 * Dpad + k0];
                    wmma::load_matrix_sync(q_frag, q_ptr, Dpad);
                    
                    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> kt_frag;
                    const half* kt_ptr = &sK[(read_stage * N + n0) * Dpad + k0];
                    wmma::load_matrix_sync(kt_frag, kt_ptr, Dpad);
                    
                    wmma::mma_sync(qk_frag, q_frag, kt_frag, qk_frag);
                }
                
                // Scale scores
                #pragma unroll
                for (int i = 0; i < qk_frag.num_elements; ++i) {
                    qk_frag.x[i] *= scale;
                }
                
                // Streaming softmax: process fragment elements row-by-row
                // Each fragment element belongs to a specific (row, col) in the 16×16 tile
                // We need to extract scores per row, update (m, l), then build P fragment
                
                // Temporary storage for this 16×16 tile's probabilities
                float P_tile[WMMA_M][WMMA_N];  // In registers
                
                // For each row in this warp's 16 rows
                for (int r_local = 0; r_local < WMMA_M; ++r_local) {
                    int r_global = warp_m0 + r_local;
                    
                    // Extract this row's scores from fragment
                    float row_scores[WMMA_N];
                    #pragma unroll
                    for (int n_local = 0; n_local < WMMA_N; ++n_local) {
                        if (n0 + n_local < kv_len) {
                            // Extract from fragment (layout-dependent)
                            // WMMA fragment is row-major in accumulator
                            int frag_idx = r_local * WMMA_N + n_local;
                            float score = qk_frag.x[frag_idx];
                            
                            // Apply causal mask
                            if (causal) {
                                int q_pos = q_start + r_global;
                                int k_pos = kv_start + n0 + n_local;
                                if (k_pos > q_pos) score = -FLT_MAX;
                            }
                            
                            row_scores[n_local] = score;
                        } else {
                            row_scores[n_local] = -FLT_MAX;
                        }
                    }
                    
                    // Find max for this row's tile
                    float tile_max = -FLT_MAX;
                    #pragma unroll
                    for (int n_local = 0; n_local < WMMA_N; ++n_local) {
                        tile_max = fmaxf(tile_max, row_scores[n_local]);
                    }
                    
                    // Update global max
                    float m_old = m_smem[r_global];
                    float m_new = fmaxf(m_old, tile_max);
                    
                    // Compute exp and sum for this tile
                    float tile_sum = 0.0f;
                    #pragma unroll
                    for (int n_local = 0; n_local < WMMA_N; ++n_local) {
                        float p = __expf(row_scores[n_local] - m_new);
                        P_tile[r_local][n_local] = p;
                        tile_sum += p;
                    }
                    
                    // Update global sum with rescaling
                    float l_old = l_smem[r_global];
                    float rescale = __expf(m_old - m_new);
                    float l_new = l_old * rescale + tile_sum;
                    
                    // Rescale O_accum for this row
                    for (int c = lane; c < HEAD_DIM; c += 32) {
                        O_accum[r_global * Dpad + c] *= rescale;
                    }
                    
                    // Update stats
                    if (lane == 0) {
                        m_smem[r_global] = m_new;
                        l_smem[r_global] = l_new;
                    }
                }
                
                __syncwarp();
                
                // Build P fragment (half) for WMMA P@V
                wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> p_frag;
                #pragma unroll
                for (int i = 0; i < p_frag.num_elements; ++i) {
                    // Map fragment element to (r, n) in tile
                    int r_local = i / WMMA_N;
                    int n_local = i % WMMA_N;
                    p_frag.x[i] = __float2half(P_tile[r_local][n_local]);
                }
                
                // WMMA P@V: accumulate into O
                // Process V in WMMA_K chunks along HEAD_DIM
                for (int d0 = 0; d0 < HEAD_DIM; d0 += WMMA_K) {
                    // Load V[n0:n0+16, d0:d0+16] row-major
                    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> v_frag;
                    const half* v_ptr = &sV[(read_stage * N + n0) * Dpad + d0];
                    wmma::load_matrix_sync(v_frag, v_ptr, Dpad);
                    
                    // Load current O_accum for this tile
                    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> o_frag;
                    float* o_ptr = &O_accum[warp_m0 * Dpad + d0];
                    wmma::load_matrix_sync(o_frag, o_ptr, Dpad, wmma::mem_row_major);
                    
                    // MMA: O += P @ V
                    wmma::mma_sync(o_frag, p_frag, v_frag, o_frag);
                    
                    // Store back to O_accum
                    wmma::store_matrix_sync(o_ptr, o_frag, Dpad, wmma::mem_row_major);
                }
            }
        }
        
        __syncthreads();
        stage = (stage + 1) % STAGES;
    }
    
    // Epilogue
    for (int idx = tid; idx < num_q_rows * HEAD_DIM; idx += blockDim.x) {
        int r = idx / HEAD_DIM;
        int c = idx % HEAD_DIM;
        if (r < num_q_rows) {
            float o_val = O_accum[r * Dpad + c] / l_smem[r];
            O_bh[(q_start + r) * d + c] = __float2half(o_val);
        }
    }
}

// Explicit instantiations
template __global__ void sdpa_fused_v2c_v6_kernel<half, 64, 2>(
    const half*, const half*, const half*, half*, int, int, int, int, float, bool);
template __global__ void sdpa_fused_v2c_v6_kernel<half, 128, 2>(
    const half*, const half*, const half*, half*, int, int, int, int, float, bool);
template __global__ void sdpa_fused_v2c_v6_kernel<half, 64, 3>(
    const half*, const half*, const half*, half*, int, int, int, int, float, bool);
template __global__ void sdpa_fused_v2c_v6_kernel<half, 128, 3>(
    const half*, const half*, const half*, half*, int, int, int, int, float, bool);

// Runtime dispatcher
cudaError_t sdpa_fused_forward_v2c_v6(const SdpaParams& params, cudaStream_t stream) {
    const int M = (params.d == 64) ? TileConfig<64>::M : TileConfig<128>::M;
    const int STAGES = (params.d == 128) ? 2 : ((params.L >= 2048) ? 3 : 2);
    
    dim3 grid((params.L + M - 1) / M, params.B * params.H);
    dim3 block(THREADS_PER_BLOCK);
    
    size_t smem_bytes;
    if (params.d == 64) {
        smem_bytes = (STAGES == 2) ? 
            SmemLayout<64, 2>::total_bytes : SmemLayout<64, 3>::total_bytes;
    } else {
        smem_bytes = (STAGES == 2) ?
            SmemLayout<128, 2>::total_bytes : SmemLayout<128, 3>::total_bytes;
    }
    
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    
    size_t max_smem = prop.sharedMemPerBlockOptin;
    if (smem_bytes > max_smem) {
        printf("[V2c-v6 WARNING] SMEM %zu KB > limit %zu KB\n", smem_bytes / 1024, max_smem / 1024);
        return cudaErrorInvalidConfiguration;
    }
    
    static bool first_launch = true;
    if (first_launch) {
        printf("[V2c-v6 FAST] d=%d, L=%d, M=%d, N=%d, STAGES=%d, SMEM=%zu KB (saved 16 KB!)\n",
               params.d, params.L, M, 
               (params.d == 64) ? TileConfig<64>::N : TileConfig<128>::N,
               STAGES, smem_bytes / 1024);
        first_launch = false;
    }
    
    cudaError_t err = cudaSuccess;
    
    if (params.d == 64 && STAGES == 2) {
        auto kernel_func = sdpa_fused_v2c_v6_kernel<half, 64, 2>;
        err = cudaFuncSetAttribute(kernel_func, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
        if (err != cudaSuccess) return err;
        kernel_func<<<grid, block, smem_bytes, stream>>>(
            reinterpret_cast<const half*>(params.Q), reinterpret_cast<const half*>(params.K),
            reinterpret_cast<const half*>(params.V), reinterpret_cast<half*>(params.O),
            params.B, params.H, params.L, params.d, params.scale, params.causal
        );
    } else if (params.d == 64 && STAGES == 3) {
        auto kernel_func = sdpa_fused_v2c_v6_kernel<half, 64, 3>;
        err = cudaFuncSetAttribute(kernel_func, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
        if (err != cudaSuccess) return err;
        kernel_func<<<grid, block, smem_bytes, stream>>>(
            reinterpret_cast<const half*>(params.Q), reinterpret_cast<const half*>(params.K),
            reinterpret_cast<const half*>(params.V), reinterpret_cast<half*>(params.O),
            params.B, params.H, params.L, params.d, params.scale, params.causal
        );
    } else if (params.d == 128 && STAGES == 2) {
        auto kernel_func = sdpa_fused_v2c_v6_kernel<half, 128, 2>;
        err = cudaFuncSetAttribute(kernel_func, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
        if (err != cudaSuccess) return err;
        kernel_func<<<grid, block, smem_bytes, stream>>>(
            reinterpret_cast<const half*>(params.Q), reinterpret_cast<const half*>(params.K),
            reinterpret_cast<const half*>(params.V), reinterpret_cast<half*>(params.O),
            params.B, params.H, params.L, params.d, params.scale, params.causal
        );
    } else {
        auto kernel_func = sdpa_fused_v2c_v6_kernel<half, 128, 3>;
        err = cudaFuncSetAttribute(kernel_func, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
        if (err != cudaSuccess) return err;
        kernel_func<<<grid, block, smem_bytes, stream>>>(
            reinterpret_cast<const half*>(params.Q), reinterpret_cast<const half*>(params.K),
            reinterpret_cast<const half*>(params.V), reinterpret_cast<half*>(params.O),
            params.B, params.H, params.L, params.d, params.scale, params.causal
        );
    }
    
    return cudaGetLastError();
}


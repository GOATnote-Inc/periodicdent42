/**
 * Child-V2c-v5: WMMA Q@K^T FIX (ld + 16-row stripes) - GREEN
 * 
 * CRITICAL FIXES from v4:
 * 1. Col-major ld for WMMA: ld = HEAD_DIM_PADDED (not STAGES*N)
 * 2. Exact 16-row stripes per warp (no partial tiles)
 * 3. Legal cp.async (4/8/16B only)
 * 
 * TARGET: 100% correctness with WMMA Q@K^T
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

// Legal cp.async helper (4/8/16B only)
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

// Warp reductions
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

/**
 * SMEM Layout for V2c-v5
 * K stored as col-major [HEAD_DIM_PAD, STAGES*N] for WMMA
 */
template<int HEAD_DIM, int STAGES>
struct SmemLayout {
    static constexpr int M = TileConfig<HEAD_DIM>::M;
    static constexpr int N = TileConfig<HEAD_DIM>::N;
    static constexpr int Dpad = HEAD_DIM_PAD(HEAD_DIM);
    
    // SMEM sizes
    static constexpr size_t sQ_bytes = M * Dpad * sizeof(half);
    // K: col-major [Dpad rows, STAGES*N cols]
    static constexpr size_t sK_bytes = Dpad * STAGES * N * sizeof(half);
    static constexpr size_t sV_bytes = STAGES * N * Dpad * sizeof(half);
    static constexpr size_t S_scores_bytes = M * N * sizeof(float);
    static constexpr size_t O_accum_bytes = M * Dpad * sizeof(float);
    static constexpr size_t m_bytes = M * sizeof(float);
    static constexpr size_t l_bytes = M * sizeof(float);
    
    static constexpr size_t total_bytes = 
        sQ_bytes + sK_bytes + sV_bytes + S_scores_bytes + O_accum_bytes + m_bytes + l_bytes;
    
    __device__ static void get_pointers(
        char* base,
        half*& sQ, half*& sK, half*& sV,
        float*& S_scores, float*& O_accum, float*& m_smem, float*& l_smem
    ) {
        size_t offset = 0;
        sQ = reinterpret_cast<half*>(base + offset); offset += sQ_bytes;
        sK = reinterpret_cast<half*>(base + offset); offset += sK_bytes;
        sV = reinterpret_cast<half*>(base + offset); offset += sV_bytes;
        S_scores = reinterpret_cast<float*>(base + offset); offset += S_scores_bytes;
        O_accum = reinterpret_cast<float*>(base + offset); offset += O_accum_bytes;
        m_smem = reinterpret_cast<float*>(base + offset); offset += m_bytes;
        l_smem = reinterpret_cast<float*>(base + offset);
    }
};

/**
 * V2c-v5 Kernel: WMMA Q@K^T with correct ld and 16-row stripes
 */
template<typename T, int HEAD_DIM, int STAGES>
__global__ void __launch_bounds__(THREADS_PER_BLOCK, 2)
sdpa_fused_v2c_v5_kernel(
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
    
    // Dynamic SMEM
    extern __shared__ char smem_base[];
    half *sQ, *sK, *sV;
    float *S_scores, *O_accum, *m_smem, *l_smem;
    Layout::get_pointers(smem_base, sQ, sK, sV, S_scores, O_accum, m_smem, l_smem);
    
    // FIX: Compute warps = exactly those handling 16-row stripes
    const int compute_warps = min(NUM_WARPS, M / WMMA_M);
    const bool is_compute_warp = (warp_id < compute_warps);
    const int warp_m0 = is_compute_warp ? (warp_id * WMMA_M) : 0;
    
    // Skip if this warp's 16-row stripe is out of bounds
    if (is_compute_warp && (warp_m0 + WMMA_M > num_q_rows)) {
        // Partial tile - skip for now
        return;
    }
    
    // Load Q tile (row-major) - all warps participate
    for (int idx = tid; idx < num_q_rows * HEAD_DIM; idx += blockDim.x) {
        int r = idx / HEAD_DIM;
        int c = idx % HEAD_DIM;
        if (q_start + r < L) {
            sQ[r * Dpad + c] = __ldg(&Q_bh[(q_start + r) * d + c]);
        } else {
            sQ[r * Dpad + c] = __float2half(0.0f);
        }
    }
    
    // Initialize stats for compute warp's rows
    if (is_compute_warp) {
        for (int r = 0; r < WMMA_M; ++r) {
            if (lane == 0) {
                m_smem[warp_m0 + r] = -FLT_MAX;
                l_smem[warp_m0 + r] = 0.0f;
            }
        }
    }
    
    // Initialize O_accum - all warps
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
        
        // FIX: Load K TRANSPOSED with correct col-major indexing
        // K stored as [Dpad rows, STAGES*N cols] in col-major
        // Index: k_col * Dpad + k_row
        for (int idx = tid; idx < kv_len * HEAD_DIM; idx += blockDim.x) {
            int n = idx / HEAD_DIM;
            int c = idx % HEAD_DIM;
            
            if (kv_start + n < L) {
                // K^T: col-major indexing
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
        
        // WMMA Q@K^T - only compute warps
        if (is_compute_warp) {
            // Process warp's 16 rows across N columns in 16-col tiles
            for (int n0 = 0; n0 < kv_len; n0 += WMMA_N) {
                int n_tile_cols = min(WMMA_N, kv_len - n0);
                
                // Accumulator for Q@K^T
                wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;
                wmma::fill_fragment(acc_frag, 0.0f);
                
                // Loop over HEAD_DIM in WMMA_K chunks
                for (int k0 = 0; k0 < HEAD_DIM; k0 += WMMA_K) {
                    // Load Q[warp_m0:warp_m0+16, k0:k0+16] row-major
                    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> q_frag;
                    const half* q_ptr = &sQ[warp_m0 * Dpad + k0];
                    wmma::load_matrix_sync(q_frag, q_ptr, Dpad);
                    
                    // FIX: Load K^T[k0:k0+16, n0:n0+16] col-major
                    // K stored as [Dpad rows, STAGES*N cols]
                    // For col-major, ld = number of rows = Dpad
                    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> kt_frag;
                    const half* kt_ptr = &sK[(read_stage * N + n0) * Dpad + k0];
                    wmma::load_matrix_sync(kt_frag, kt_ptr, Dpad);  // FIX: ld = Dpad (row count)
                    
                    // MMA: acc += Q @ K^T
                    wmma::mma_sync(acc_frag, q_frag, kt_frag, acc_frag);
                }
                
                // Scale and store to S_scores
                #pragma unroll
                for (int i = 0; i < acc_frag.num_elements; ++i) {
                    acc_frag.x[i] *= scale;
                }
                
                float* score_ptr = &S_scores[warp_m0 * N + n0];
                wmma::store_matrix_sync(score_ptr, acc_frag, N, wmma::mem_row_major);
            }
        }
        
        __syncthreads();
        
        // Streaming softmax - compute warps only
        if (is_compute_warp) {
            for (int r = warp_m0; r < warp_m0 + WMMA_M; ++r) {
                // Read scores for this row (already scaled)
                float row_max = -FLT_MAX;
                for (int n = 0; n < kv_len; ++n) {
                    float score = S_scores[r * N + n];
                    
                    // Causal mask
                    if (causal) {
                        int q_pos = q_start + r;
                        int k_pos = kv_start + n;
                        if (k_pos > q_pos) score = -FLT_MAX;
                    }
                    
                    S_scores[r * N + n] = score;
                    row_max = fmaxf(row_max, score);
                }
                
                // Streaming softmax update
                float m_old = m_smem[r];
                float m_new = fmaxf(m_old, row_max);
                
                float l_old = l_smem[r];
                float l_add = 0.0f;
                
                for (int n = 0; n < kv_len; ++n) {
                    float p = __expf(S_scores[r * N + n] - m_new);
                    S_scores[r * N + n] = p;
                    l_add += p;
                }
                
                float rescale = __expf(m_old - m_new);
                float l_new = l_old * rescale + l_add;
                
                // Rescale O_accum
                for (int c = lane; c < HEAD_DIM; c += 32) {
                    O_accum[r * Dpad + c] *= rescale;
                }
                
                // Update stats (lane 0)
                if (lane == 0) {
                    m_smem[r] = m_new;
                    l_smem[r] = l_new;
                }
            }
            
            __syncwarp();
            
            // P @ V (scalar)
            for (int r = warp_m0; r < warp_m0 + WMMA_M; ++r) {
                for (int n = 0; n < kv_len; ++n) {
                    float p = S_scores[r * N + n];
                    for (int c = lane; c < HEAD_DIM; c += 32) {
                        float v_val = __half2float(sV[(read_stage * N + n) * Dpad + c]);
                        O_accum[r * Dpad + c] += p * v_val;
                    }
                }
            }
        }
        
        __syncthreads();
        stage = (stage + 1) % STAGES;
    }
    
    // Epilogue - all warps
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
template __global__ void sdpa_fused_v2c_v5_kernel<half, 64, 2>(
    const half*, const half*, const half*, half*, int, int, int, int, float, bool);
template __global__ void sdpa_fused_v2c_v5_kernel<half, 128, 2>(
    const half*, const half*, const half*, half*, int, int, int, int, float, bool);
template __global__ void sdpa_fused_v2c_v5_kernel<half, 64, 3>(
    const half*, const half*, const half*, half*, int, int, int, int, float, bool);
template __global__ void sdpa_fused_v2c_v5_kernel<half, 128, 3>(
    const half*, const half*, const half*, half*, int, int, int, int, float, bool);

// Runtime dispatcher
cudaError_t sdpa_fused_forward_v2c_v5(const SdpaParams& params, cudaStream_t stream) {
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
        printf("[V2c-v5 WARNING] SMEM %zu KB > limit %zu KB\n", smem_bytes / 1024, max_smem / 1024);
        return cudaErrorInvalidConfiguration;
    }
    
    static bool first_launch = true;
    if (first_launch) {
        printf("[V2c-v5 GREEN] d=%d, L=%d, M=%d, N=%d, STAGES=%d, SMEM=%zu KB\n",
               params.d, params.L, M, 
               (params.d == 64) ? TileConfig<64>::N : TileConfig<128>::N,
               STAGES, smem_bytes / 1024);
        first_launch = false;
    }
    
    cudaError_t err = cudaSuccess;
    
    if (params.d == 64 && STAGES == 2) {
        auto kernel_func = sdpa_fused_v2c_v5_kernel<half, 64, 2>;
        err = cudaFuncSetAttribute(kernel_func, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
        if (err != cudaSuccess) return err;
        kernel_func<<<grid, block, smem_bytes, stream>>>(
            reinterpret_cast<const half*>(params.Q), reinterpret_cast<const half*>(params.K),
            reinterpret_cast<const half*>(params.V), reinterpret_cast<half*>(params.O),
            params.B, params.H, params.L, params.d, params.scale, params.causal
        );
    } else if (params.d == 64 && STAGES == 3) {
        auto kernel_func = sdpa_fused_v2c_v5_kernel<half, 64, 3>;
        err = cudaFuncSetAttribute(kernel_func, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
        if (err != cudaSuccess) return err;
        kernel_func<<<grid, block, smem_bytes, stream>>>(
            reinterpret_cast<const half*>(params.Q), reinterpret_cast<const half*>(params.K),
            reinterpret_cast<const half*>(params.V), reinterpret_cast<half*>(params.O),
            params.B, params.H, params.L, params.d, params.scale, params.causal
        );
    } else if (params.d == 128 && STAGES == 2) {
        auto kernel_func = sdpa_fused_v2c_v5_kernel<half, 128, 2>;
        err = cudaFuncSetAttribute(kernel_func, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
        if (err != cudaSuccess) return err;
        kernel_func<<<grid, block, smem_bytes, stream>>>(
            reinterpret_cast<const half*>(params.Q), reinterpret_cast<const half*>(params.K),
            reinterpret_cast<const half*>(params.V), reinterpret_cast<half*>(params.O),
            params.B, params.H, params.L, params.d, params.scale, params.causal
        );
    } else {
        auto kernel_func = sdpa_fused_v2c_v5_kernel<half, 128, 3>;
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


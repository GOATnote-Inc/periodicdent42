/**
 * Child-V2c-v4: WMMA Q@K^T + Transposed K (Iteration 4)
 * 
 * GOAL: Add Tensor Core acceleration for Q@K^T computation
 * 
 * CHANGES from v3:
 * - Store K in col-major layout for efficient K^T access
 * - Use WMMA 16×16×16 for Q@K^T computation
 * - Keep streaming softmax and P@V scalar (for stability)
 * 
 * TARGET: 800-1200 μs (2-3× from v3's 1750 μs)
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

// cp.async helper (Ada: 16B only)
__device__ __forceinline__ void cp_async_16B(void* smem_ptr, const void* global_ptr) {
    unsigned smem_addr = __cvta_generic_to_shared(smem_ptr);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(smem_addr), "l"(global_ptr));
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
 * SMEM Layout for V2c-v4
 * K is stored TRANSPOSED (col-major) for efficient WMMA access
 */
template<int HEAD_DIM, int STAGES>
struct SmemLayout {
    static constexpr int M = TileConfig<HEAD_DIM>::M;
    static constexpr int N = TileConfig<HEAD_DIM>::N;
    static constexpr int HEAD_DIM_PADDED = HEAD_DIM_PAD(HEAD_DIM);
    
    // SMEM sizes
    static constexpr size_t sQ_bytes = M * HEAD_DIM_PADDED * sizeof(half);  // Q: row-major [M, D]
    // K stored as col-major [D, STAGES*N] for efficient K^T access
    static constexpr size_t sK_bytes = HEAD_DIM_PADDED * STAGES * N * sizeof(half);
    static constexpr size_t sV_bytes = STAGES * N * HEAD_DIM_PADDED * sizeof(half);  // V: row-major
    static constexpr size_t S_scores_bytes = M * N * sizeof(float);
    static constexpr size_t O_accum_bytes = M * HEAD_DIM_PADDED * sizeof(float);
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
 * V2c-v4 Kernel: WMMA Q@K^T with transposed K
 */
template<typename T, int HEAD_DIM, int STAGES>
__global__ void __launch_bounds__(THREADS_PER_BLOCK, 2)
sdpa_fused_v2c_v4_kernel(
    const T* Q, const T* K, const T* V, T* O,
    int B, int H, int L, int d,
    float scale, bool causal
) {
    constexpr int M = TileConfig<HEAD_DIM>::M;
    constexpr int N = TileConfig<HEAD_DIM>::N;
    constexpr int HEAD_DIM_PADDED = HEAD_DIM_PAD(HEAD_DIM);
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
    
    // Single-warp ownership
    const int rows_per_warp = (M + NUM_WARPS - 1) / NUM_WARPS;
    const int my_row_start = warp_id * rows_per_warp;
    const int my_row_end = min(my_row_start + rows_per_warp, num_q_rows);
    const int my_num_rows = max(0, my_row_end - my_row_start);
    
    // Load Q tile (row-major)
    for (int idx = tid; idx < num_q_rows * HEAD_DIM; idx += blockDim.x) {
        int r = idx / HEAD_DIM;
        int c = idx % HEAD_DIM;
        if (q_start + r < L) {
            sQ[r * HEAD_DIM_PADDED + c] = __ldg(&Q_bh[(q_start + r) * d + c]);
        } else {
            sQ[r * HEAD_DIM_PADDED + c] = __float2half(0.0f);
        }
    }
    
    // Initialize stats
    for (int r = my_row_start; r < my_row_end; ++r) {
        if (lane == 0) {
            m_smem[r] = -FLT_MAX;
            l_smem[r] = 0.0f;
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
        
        // Load K TRANSPOSED (col-major) and V (row-major)
        // K: [kv_len, HEAD_DIM] in global → [HEAD_DIM, kv_len] in SMEM (transposed!)
        for (int idx = tid; idx < kv_len * HEAD_DIM; idx += blockDim.x) {
            int n = idx / HEAD_DIM;  // K row in global
            int c = idx % HEAD_DIM;  // K col in global
            
            if (kv_start + n < L) {
                // Store K transposed: sK[c, read_stage*N + n] (col-major)
                int k_idx = c * (STAGES * N) + (read_stage * N + n);
                sK[k_idx] = __ldg(&K_bh[(kv_start + n) * d + c]);
                
                // Store V row-major: sV[read_stage*N + n, c]
                int v_idx = (read_stage * N + n) * HEAD_DIM_PADDED + c;
                sV[v_idx] = __ldg(&V_bh[(kv_start + n) * d + c]);
            } else {
                int k_idx = c * (STAGES * N) + (read_stage * N + n);
                sK[k_idx] = __float2half(0.0f);
                int v_idx = (read_stage * N + n) * HEAD_DIM_PADDED + c;
                sV[v_idx] = __float2half(0.0f);
            }
        }
        __syncthreads();
        
        // WMMA Q@K^T computation
        if (my_num_rows > 0) {
            // Each warp processes its rows in 16×16 tiles
            for (int m0 = my_row_start; m0 < my_row_end; m0 += WMMA_M) {
                int m_tile_rows = min(WMMA_M, my_row_end - m0);
                
                for (int n0 = 0; n0 < kv_len; n0 += WMMA_N) {
                    int n_tile_cols = min(WMMA_N, kv_len - n0);
                    
                    // Accumulator for Q@K^T
                    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;
                    wmma::fill_fragment(acc_frag, 0.0f);
                    
                    // Loop over K dimension (HEAD_DIM) in WMMA_K chunks
                    for (int k0 = 0; k0 < HEAD_DIM; k0 += WMMA_K) {
                        // Load Q[m0:m0+16, k0:k0+16] row-major
                        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> q_frag;
                        const half* q_ptr = &sQ[m0 * HEAD_DIM_PADDED + k0];
                        wmma::load_matrix_sync(q_frag, q_ptr, HEAD_DIM_PADDED);
                        
                        // Load K^T[k0:k0+16, n0:n0+16] col-major
                        // K is stored as [HEAD_DIM, STAGES*N] col-major
                        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> kt_frag;
                        const half* kt_ptr = &sK[k0 * (STAGES * N) + (read_stage * N + n0)];
                        wmma::load_matrix_sync(kt_frag, kt_ptr, STAGES * N);
                        
                        // MMA: acc += Q @ K^T
                        wmma::mma_sync(acc_frag, q_frag, kt_frag, acc_frag);
                    }
                    
                    // Store scores to SMEM with scaling
                    float* score_ptr = &S_scores[m0 * N + n0];
                    
                    // Scale and store (each thread handles its elements)
                    #pragma unroll
                    for (int i = 0; i < acc_frag.num_elements; ++i) {
                        acc_frag.x[i] *= scale;
                    }
                    
                    wmma::store_matrix_sync(score_ptr, acc_frag, N, wmma::mem_row_major);
                }
            }
        }
        
        __syncthreads();  // All warps share S_scores
        
        // Streaming softmax (scalar, same as v3)
        if (my_num_rows > 0) {
            for (int r = my_row_start; r < my_row_end; ++r) {
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
                    O_accum[r * HEAD_DIM_PADDED + c] *= rescale;
                }
                
                // Update stats (lane 0)
                if (lane == 0) {
                    m_smem[r] = m_new;
                    l_smem[r] = l_new;
                }
            }
            
            __syncwarp();
            
            // P @ V (scalar, same as v3)
            for (int r = my_row_start; r < my_row_end; ++r) {
                for (int n = 0; n < kv_len; ++n) {
                    float p = S_scores[r * N + n];
                    for (int c = lane; c < HEAD_DIM; c += 32) {
                        float v_val = __half2float(sV[(read_stage * N + n) * HEAD_DIM_PADDED + c]);
                        O_accum[r * HEAD_DIM_PADDED + c] += p * v_val;
                    }
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
            float o_val = O_accum[r * HEAD_DIM_PADDED + c] / l_smem[r];
            O_bh[(q_start + r) * d + c] = __float2half(o_val);
        }
    }
}

// Explicit instantiations
template __global__ void sdpa_fused_v2c_v4_kernel<half, 64, 2>(
    const half*, const half*, const half*, half*, int, int, int, int, float, bool);
template __global__ void sdpa_fused_v2c_v4_kernel<half, 128, 2>(
    const half*, const half*, const half*, half*, int, int, int, int, float, bool);
template __global__ void sdpa_fused_v2c_v4_kernel<half, 64, 3>(
    const half*, const half*, const half*, half*, int, int, int, int, float, bool);
template __global__ void sdpa_fused_v2c_v4_kernel<half, 128, 3>(
    const half*, const half*, const half*, half*, int, int, int, int, float, bool);

// Runtime dispatcher
cudaError_t sdpa_fused_forward_v2c_v4(const SdpaParams& params, cudaStream_t stream) {
    const int M = (params.d == 64) ? TileConfig<64>::M : TileConfig<128>::M;
    // For d=128, STAGES=3 exceeds 99 KB SMEM limit, so force STAGES=2
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
        printf("[V2c-v4 WARNING] SMEM %zu KB > limit %zu KB\n", smem_bytes / 1024, max_smem / 1024);
        return cudaErrorInvalidConfiguration;
    }
    
    static bool first_launch = true;
    if (first_launch) {
        printf("[V2c-v4 WMMA] d=%d, L=%d, M=%d, N=%d, STAGES=%d, SMEM=%zu KB\n",
               params.d, params.L, M, 
               (params.d == 64) ? TileConfig<64>::N : TileConfig<128>::N,
               STAGES, smem_bytes / 1024);
        first_launch = false;
    }
    
    cudaError_t err = cudaSuccess;
    
    if (params.d == 64 && STAGES == 2) {
        auto kernel_func = sdpa_fused_v2c_v4_kernel<half, 64, 2>;
        err = cudaFuncSetAttribute(kernel_func, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
        if (err != cudaSuccess) return err;
        kernel_func<<<grid, block, smem_bytes, stream>>>(
            reinterpret_cast<const half*>(params.Q), reinterpret_cast<const half*>(params.K),
            reinterpret_cast<const half*>(params.V), reinterpret_cast<half*>(params.O),
            params.B, params.H, params.L, params.d, params.scale, params.causal
        );
    } else if (params.d == 64 && STAGES == 3) {
        auto kernel_func = sdpa_fused_v2c_v4_kernel<half, 64, 3>;
        err = cudaFuncSetAttribute(kernel_func, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
        if (err != cudaSuccess) return err;
        kernel_func<<<grid, block, smem_bytes, stream>>>(
            reinterpret_cast<const half*>(params.Q), reinterpret_cast<const half*>(params.K),
            reinterpret_cast<const half*>(params.V), reinterpret_cast<half*>(params.O),
            params.B, params.H, params.L, params.d, params.scale, params.causal
        );
    } else if (params.d == 128 && STAGES == 2) {
        auto kernel_func = sdpa_fused_v2c_v4_kernel<half, 128, 2>;
        err = cudaFuncSetAttribute(kernel_func, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
        if (err != cudaSuccess) return err;
        kernel_func<<<grid, block, smem_bytes, stream>>>(
            reinterpret_cast<const half*>(params.Q), reinterpret_cast<const half*>(params.K),
            reinterpret_cast<const half*>(params.V), reinterpret_cast<half*>(params.O),
            params.B, params.H, params.L, params.d, params.scale, params.causal
        );
    } else {
        auto kernel_func = sdpa_fused_v2c_v4_kernel<half, 128, 3>;
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


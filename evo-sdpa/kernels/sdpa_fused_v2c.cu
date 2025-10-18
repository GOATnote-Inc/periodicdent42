/**
 * Child-V2c-v3: Scalar Q@K^T validation (Iteration 3)
 * 
 * GOAL: Validate infrastructure (streaming softmax, SMEM layout) with scalar
 * 
 * CHANGES from v2:
 * - Replaced WMMA Q@K^T with scalar (correctness-first)
 * - Fixed double-scaling bug (score was scaled twice)
 * - Keeping all other infrastructure intact
 * 
 * NEXT: Once 100% correct, add proper WMMA with K^T handling
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

// Simplified tile config (ensure divisible by 16)
template<int HEAD_DIM>
struct TileConfig {
    static constexpr int M = 64;
    static constexpr int N = (HEAD_DIM == 64) ? 64 : 32;
    static constexpr int K = HEAD_DIM;
};

#define NUM_WARPS 8
#define THREADS_PER_BLOCK (NUM_WARPS * 32)
#define HEAD_DIM_PAD(d) ((d) + 8)

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
 * Simplified SMEM Layout (with score buffer for WMMA)
 */
template<int HEAD_DIM, int STAGES>
struct SmemLayout {
    static constexpr int M = TileConfig<HEAD_DIM>::M;
    static constexpr int N = TileConfig<HEAD_DIM>::N;
    static constexpr int HEAD_DIM_PADDED = HEAD_DIM_PAD(HEAD_DIM);
    
    // SMEM sizes
    static constexpr size_t sQ_bytes = M * HEAD_DIM_PADDED * sizeof(half);
    static constexpr size_t sK_bytes = STAGES * N * HEAD_DIM_PADDED * sizeof(half);
    static constexpr size_t sV_bytes = STAGES * N * HEAD_DIM_PADDED * sizeof(half);
    static constexpr size_t S_scores_bytes = M * N * sizeof(float);  // For WMMA output
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
 * V2c-Fixed: Simplified WMMA Kernel
 */
template<typename T, int HEAD_DIM, int STAGES>
__launch_bounds__(THREADS_PER_BLOCK, 1)
__global__ void sdpa_fused_v2c_kernel(
    const T* __restrict__ Q,
    const T* __restrict__ K,
    const T* __restrict__ V,
    T* __restrict__ O,
    int B, int H, int L, int d, float scale, bool causal
) {
    constexpr int M = TileConfig<HEAD_DIM>::M;
    constexpr int N = TileConfig<HEAD_DIM>::N;
    constexpr int HEAD_DIM_PADDED = HEAD_DIM_PAD(HEAD_DIM);
    using Layout = SmemLayout<HEAD_DIM, STAGES>;
    
    const int bh = blockIdx.y;
    const int q_block = blockIdx.x;
    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;
    
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
    
    // Load Q tile (row-major, coalesced)
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
        
        // Load K/V with simple coalesced pattern (no transpose for simplicity)
        for (int idx = tid; idx < kv_len * HEAD_DIM; idx += blockDim.x) {
            int n = idx / HEAD_DIM;
            int c = idx % HEAD_DIM;
            int k_idx = (read_stage * N + n) * HEAD_DIM_PADDED + c;
            
            if (kv_start + n < L) {
                sK[k_idx] = __ldg(&K_bh[(kv_start + n) * d + c]);
                sV[k_idx] = __ldg(&V_bh[(kv_start + n) * d + c]);
            } else {
                sK[k_idx] = __float2half(0.0f);
                sV[k_idx] = __float2half(0.0f);
            }
        }
        __syncthreads();
        
        // SCALAR Q @ K^T (Iteration 3: validate infrastructure)
        // NOTE: Using scalar temporarily to validate softmax & memory layout
        // Will replace with proper WMMA + K^T in next iteration
        if (my_num_rows > 0) {
            for (int r = my_row_start; r < my_row_end; ++r) {
                // Compute Q[r] @ K^T for all kv_len keys
                for (int n = 0; n < kv_len; ++n) {
                    float score = 0.0f;
                    
                    // Dot product: Q[r,:] @ K[n,:]^T
                    for (int k = lane; k < HEAD_DIM; k += 32) {
                        float q_val = __half2float(sQ[r * HEAD_DIM_PADDED + k]);
                        float k_val = __half2float(sK[(read_stage * N + n) * HEAD_DIM_PADDED + k]);
                        score += q_val * k_val;
                    }
                    
                    // Warp reduction
                    score = warp_reduce_sum(score);
                    
                    // Lane 0 writes, then broadcast
                    if (lane == 0) {
                        S_scores[r * N + n] = score * scale;
                    }
                    // Broadcast so all lanes have same score
                    score = __shfl_sync(0xffffffff, score * scale, 0);
                }
            }
        }
        
        __syncthreads();  // FIX: All warps share S_scores, need full block sync!
        
        // Process each owned row (streaming softmax)
        if (my_num_rows > 0) {
            for (int r = my_row_start; r < my_row_end; ++r) {
                // Read scores for this row (already scaled)
                float row_max = -FLT_MAX;
                for (int n = 0; n < kv_len; ++n) {
                    float score = S_scores[r * N + n];  // Already scaled above
                    
                    // Causal mask
                    if (causal) {
                        int q_pos = q_start + r;
                        int k_pos = kv_start + n;
                        if (k_pos > q_pos) score = -FLT_MAX;
                    }
                    
                    S_scores[r * N + n] = score;
                    row_max = fmaxf(row_max, score);
                }
                
                // Warp reduce max
                row_max = warp_reduce_max(row_max);
                
                // Streaming softmax update
                float m_old = m_smem[r];
                float m_new = fmaxf(m_old, row_max);
                
                float l_old = l_smem[r];
                float l_add = 0.0f;
                
                for (int n = 0; n < kv_len; ++n) {
                    float p = __expf(S_scores[r * N + n] - m_new);
                    S_scores[r * N + n] = p;  // Store P (unnormalized)
                    l_add += p;
                }
                
                l_add = warp_reduce_sum(l_add);
                
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
            
            __syncwarp();  // Ensure stats updated
            
            // P @ V (scalar for now - will use WMMA later)
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
template __global__ void sdpa_fused_v2c_kernel<half, 64, 2>(
    const half*, const half*, const half*, half*, int, int, int, int, float, bool);
template __global__ void sdpa_fused_v2c_kernel<half, 128, 2>(
    const half*, const half*, const half*, half*, int, int, int, int, float, bool);
template __global__ void sdpa_fused_v2c_kernel<half, 64, 3>(
    const half*, const half*, const half*, half*, int, int, int, int, float, bool);
template __global__ void sdpa_fused_v2c_kernel<half, 128, 3>(
    const half*, const half*, const half*, half*, int, int, int, int, float, bool);

// Runtime dispatcher
cudaError_t sdpa_fused_forward_v2c(const SdpaParams& params, cudaStream_t stream) {
    const int M = (params.d == 64) ? TileConfig<64>::M : TileConfig<128>::M;
    // For d=128, STAGES=3 exceeds 99 KB SMEM limit (110 KB), so force STAGES=2
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
        printf("[V2c WARNING] SMEM %zu KB > limit %zu KB\n", smem_bytes / 1024, max_smem / 1024);
        return cudaErrorInvalidConfiguration;
    }
    
    static bool first_launch = true;
    if (first_launch) {
        printf("[V2c-Fixed] d=%d, L=%d, M=%d, N=%d, STAGES=%d, SMEM=%zu KB\n",
               params.d, params.L, M, 
               (params.d == 64) ? TileConfig<64>::N : TileConfig<128>::N,
               STAGES, smem_bytes / 1024);
        first_launch = false;
    }
    
    cudaError_t err = cudaSuccess;
    
    if (params.d == 64 && STAGES == 2) {
        auto kernel_func = sdpa_fused_v2c_kernel<half, 64, 2>;
        err = cudaFuncSetAttribute(kernel_func, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
        if (err != cudaSuccess) return err;
        kernel_func<<<grid, block, smem_bytes, stream>>>(
            reinterpret_cast<const half*>(params.Q), reinterpret_cast<const half*>(params.K),
            reinterpret_cast<const half*>(params.V), reinterpret_cast<half*>(params.O),
            params.B, params.H, params.L, params.d, params.scale, params.causal
        );
    } else if (params.d == 64 && STAGES == 3) {
        auto kernel_func = sdpa_fused_v2c_kernel<half, 64, 3>;
        err = cudaFuncSetAttribute(kernel_func, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
        if (err != cudaSuccess) return err;
        kernel_func<<<grid, block, smem_bytes, stream>>>(
            reinterpret_cast<const half*>(params.Q), reinterpret_cast<const half*>(params.K),
            reinterpret_cast<const half*>(params.V), reinterpret_cast<half*>(params.O),
            params.B, params.H, params.L, params.d, params.scale, params.causal
        );
    } else if (params.d == 128 && STAGES == 2) {
        auto kernel_func = sdpa_fused_v2c_kernel<half, 128, 2>;
        err = cudaFuncSetAttribute(kernel_func, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
        if (err != cudaSuccess) return err;
        kernel_func<<<grid, block, smem_bytes, stream>>>(
            reinterpret_cast<const half*>(params.Q), reinterpret_cast<const half*>(params.K),
            reinterpret_cast<const half*>(params.V), reinterpret_cast<half*>(params.O),
            params.B, params.H, params.L, params.d, params.scale, params.causal
        );
    } else {
        auto kernel_func = sdpa_fused_v2c_kernel<half, 128, 3>;
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


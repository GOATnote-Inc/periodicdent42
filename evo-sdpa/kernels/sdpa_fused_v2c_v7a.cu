/**
 * Child-V2c-v7a: Overlapped cp.async Pipeline (2-3 stages) on GREEN v6a
 * 
 * GOAL: Add true global→SMEM overlap using cp.async multi-stage prefetch
 * TARGET: 1.3-1.7× speedup vs v6a (~1177 μs → ~700-900 μs)
 * 
 * STRATEGY:
 * - Producer warp (warp_id == NUM_WARPS-1): cp.async prefetch next K/V tile
 * - Consumer warps (0..3): WMMA compute on current tile with wait_group
 * - Stage ring rotation: read_stage → write_stage → ...
 * 
 * PRESERVES v6a GREEN:
 * - Store → softmax → rebuild pattern for WMMA
 * - Streaming softmax with single-warp row ownership
 * - Per-warp scratch (sS_frag, sP_frag)
 * - All correctness invariants
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

// Tile config (same as v6a)
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

// Legal cp.async helper for Ada (sm_89): ONLY 16B supported
__device__ __forceinline__ void cp_async_16B_if_aligned(
    void* smem, const void* gmem, bool use_async
) {
    if (use_async && (((uintptr_t)smem & 0xF) == 0) && (((uintptr_t)gmem & 0xF) == 0)) {
        // Ada: only 16B cp.async allowed
        unsigned sm = __cvta_generic_to_shared(smem);
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(sm), "l"(gmem));
    } else {
        // Fallback: regular load (not cp.async 4/8B which Ada doesn't support)
        *reinterpret_cast<int4*>(smem) = __ldg(reinterpret_cast<const int4*>(gmem));
    }
}

__device__ __forceinline__ void cp_async_commit_group() {
    asm volatile("cp.async.commit_group;\n" ::);
}

template<int N>
__device__ __forceinline__ void cp_async_wait_group() {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
}

// INSIGHT: async_overlap - Stage ring helpers
__device__ __forceinline__ half* K_stage_ptr(half* base, int stage, int N, int Dpad) {
    return base + stage * (N * Dpad);
}

__device__ __forceinline__ half* V_stage_ptr(half* base, int stage, int N, int Dpad) {
    return base + stage * (N * Dpad);
}

/**
 * SMEM Layout for V2c-v7a (same as v6a with stage ring)
 */
template<int HEAD_DIM, int STAGES>
struct SmemLayout {
    static constexpr int M = TileConfig<HEAD_DIM>::M;
    static constexpr int N = TileConfig<HEAD_DIM>::N;
    static constexpr int Dpad = HEAD_DIM_PAD(HEAD_DIM);
    
    // Per-warp scratch
    static constexpr int COMPUTE_WARPS = M / WMMA_M;  // e.g., 64/16 = 4
    
    // SMEM sizes
    static constexpr size_t sQ_bytes = M * Dpad * sizeof(half);
    static constexpr size_t sK_bytes = Dpad * STAGES * N * sizeof(half);  // Stage ring
    static constexpr size_t sV_bytes = STAGES * N * Dpad * sizeof(half);  // Stage ring
    static constexpr size_t O_accum_bytes = M * Dpad * sizeof(float);
    static constexpr size_t m_bytes = M * sizeof(float);
    static constexpr size_t l_bytes = M * sizeof(float);
    
    // Per-warp scratch (same as v6a)
    static constexpr size_t sS_frag_bytes = COMPUTE_WARPS * WMMA_M * WMMA_N * sizeof(float);
    static constexpr size_t sP_frag_bytes = COMPUTE_WARPS * WMMA_M * WMMA_N * sizeof(half);
    
    static constexpr size_t total_bytes = 
        sQ_bytes + sK_bytes + sV_bytes + O_accum_bytes + m_bytes + l_bytes 
        + sS_frag_bytes + sP_frag_bytes;
    
    __device__ static void get_pointers(
        char* base,
        half*& sQ, half*& sK, half*& sV,
        float*& O_accum, float*& m_smem, float*& l_smem,
        float*& sS_all, half*& sP_all
    ) {
        size_t offset = 0;
        sQ = reinterpret_cast<half*>(base + offset); offset += sQ_bytes;
        sK = reinterpret_cast<half*>(base + offset); offset += sK_bytes;
        sV = reinterpret_cast<half*>(base + offset); offset += sV_bytes;
        O_accum = reinterpret_cast<float*>(base + offset); offset += O_accum_bytes;
        m_smem = reinterpret_cast<float*>(base + offset); offset += m_bytes;
        l_smem = reinterpret_cast<float*>(base + offset); offset += l_bytes;
        sS_all = reinterpret_cast<float*>(base + offset); offset += sS_frag_bytes;
        sP_all = reinterpret_cast<half*>(base + offset);
    }
};

/**
 * V2c-v7a Kernel: Overlapped cp.async Pipeline on GREEN v6a
 */
template<typename T, int HEAD_DIM, int STAGES>
__global__ void __launch_bounds__(THREADS_PER_BLOCK, 2)
sdpa_fused_v2c_v7a_kernel(
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
    float *O_accum, *m_smem, *l_smem;
    float *sS_all;
    half *sP_all;
    Layout::get_pointers(smem_base, sQ, sK, sV, O_accum, m_smem, l_smem, sS_all, sP_all);
    
    // Warp roles: producer (last warp) vs compute (0..3)
    const int compute_warps = M / WMMA_M;
    const bool is_producer = (warp_id == NUM_WARPS - 1);
    const bool is_compute_warp = (warp_id < compute_warps);
    const int warp_tile_id = warp_id;
    
    // Per-warp scratch slices (same as v6a)
    float* sS_frag = sS_all + warp_tile_id * (WMMA_M * WMMA_N);
    half* sP_frag = sP_all + warp_tile_id * (WMMA_M * WMMA_N);
    
    // Each compute warp owns exactly one 16-row stripe
    const int warp_m0 = is_compute_warp ? (warp_tile_id * WMMA_M) : 0;
    
    if (is_compute_warp && (warp_m0 + WMMA_M > num_q_rows)) {
        return;  // Skip partial tiles
    }
    
    // Load Q tile (once per CTA)
    for (int idx = tid; idx < num_q_rows * HEAD_DIM; idx += blockDim.x) {
        int r = idx / HEAD_DIM;
        int c = idx % HEAD_DIM;
        if (q_start + r < L) {
            sQ[r * Dpad + c] = __ldg(&Q_bh[(q_start + r) * d + c]);
        } else {
            sQ[r * Dpad + c] = __float2half(0.0f);
        }
    }
    
    // Initialize stats (compute warps only)
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
    
    // FIX: stage_ring_handshake - Preload stage 0 synchronously
    int stage = 0;
    {
        const int kv_start0 = 0;
        const int kv_len0 = min(N, L);
        half* sK0 = K_stage_ptr(sK, 0, N, Dpad);
        half* sV0 = V_stage_ptr(sV, 0, N, Dpad);
        
        // Everyone participates in preload (use 16B chunks)
        const int elems_per_16B = 8;  // 8 halfs = 16 bytes
        const int num_16B_chunks = (kv_len0 * HEAD_DIM + elems_per_16B - 1) / elems_per_16B;
        
        for (int chunk_idx = tid; chunk_idx < num_16B_chunks; chunk_idx += blockDim.x) {
            int elem_offset = chunk_idx * elems_per_16B;
            int n = elem_offset / HEAD_DIM;
            int c = elem_offset % HEAD_DIM;
            
            if (n < kv_len0 && c + elems_per_16B <= HEAD_DIM) {
                // Full 16B chunk: use cp.async
                int k_idx = n * Dpad + c;
                int v_idx = n * Dpad + c;
                bool use_async = (kv_start0 + n < L);
                cp_async_16B_if_aligned(&sK0[k_idx], &K_bh[(kv_start0 + n) * d + c], use_async);
                cp_async_16B_if_aligned(&sV0[v_idx], &V_bh[(kv_start0 + n) * d + c], use_async);
            } else {
                // Tail elements: scalar fallback
                for (int i = 0; i < elems_per_16B && (elem_offset + i) < kv_len0 * HEAD_DIM; ++i) {
                    int idx = elem_offset + i;
                    int nn = idx / HEAD_DIM;
                    int cc = idx % HEAD_DIM;
                    int k_idx = nn * Dpad + cc;
                    if (kv_start0 + nn < L) {
                        sK0[k_idx] = __ldg(&K_bh[(kv_start0 + nn) * d + cc]);
                        sV0[k_idx] = __ldg(&V_bh[(kv_start0 + nn) * d + cc]);
                    } else {
                        sK0[k_idx] = __float2half(0.0f);
                        sV0[k_idx] = __float2half(0.0f);
                    }
                }
            }
        }
        cp_async_commit_group();
        cp_async_wait_group<0>();
        __syncthreads();
    }
    
    // INSIGHT: async_overlap - Main loop with producer/consumer overlap
    for (int t = 0; t < num_kv_tiles; ++t) {
        const int kv_start = t * N;
        const int kv_len = min(N, L - kv_start);
        
        const int read_stage = stage;
        const int write_stage = (STAGES == 2) ? (1 - stage) : ((stage + 1) % STAGES);
        
        // PRODUCER: Prefetch NEXT tile into write_stage (if any)
        if (is_producer) {
            if ((t + 1) < num_kv_tiles) {
                const int kv_next = (t + 1) * N;
                const int kv_len_next = min(N, L - kv_next);
                
                half* sKw = K_stage_ptr(sK, write_stage, N, Dpad);
                half* sVw = V_stage_ptr(sV, write_stage, N, Dpad);
                
                // Stride by 16B chunks (8 halfs) per lane
                const int elems_per_16B = 8;
                const int total_chunks = (kv_len_next * HEAD_DIM + elems_per_16B - 1) / elems_per_16B;
                
                for (int chunk_idx = lane; chunk_idx < total_chunks; chunk_idx += 32) {
                    int elem_offset = chunk_idx * elems_per_16B;
                    int n = elem_offset / HEAD_DIM;
                    int c = elem_offset % HEAD_DIM;
                    
                    if (n < kv_len_next && c + elems_per_16B <= HEAD_DIM) {
                        // Full 16B chunk: use cp.async
                        int k_idx = n * Dpad + c;
                        int v_idx = n * Dpad + c;
                        bool use_async = (kv_next + n < L);
                        cp_async_16B_if_aligned(&sKw[k_idx], &K_bh[(kv_next + n) * d + c], use_async);
                        cp_async_16B_if_aligned(&sVw[v_idx], &V_bh[(kv_next + n) * d + c], use_async);
                    } else {
                        // Tail elements: scalar fallback
                        for (int i = 0; i < elems_per_16B && (elem_offset + i) < kv_len_next * HEAD_DIM; ++i) {
                            int idx = elem_offset + i;
                            int nn = idx / HEAD_DIM;
                            int cc = idx % HEAD_DIM;
                            int k_idx = nn * Dpad + cc;
                            if (kv_next + nn < L) {
                                sKw[k_idx] = __ldg(&K_bh[(kv_next + nn) * d + cc]);
                                sVw[k_idx] = __ldg(&V_bh[(kv_next + nn) * d + cc]);
                            } else {
                                sKw[k_idx] = __float2half(0.0f);
                                sVw[k_idx] = __float2half(0.0f);
                            }
                        }
                    }
                }
                cp_async_commit_group();  // Enqueue next-tile group
            }
        }
        
        // CONSUMERS: Wait until current read_stage is available
        // With STAGES=2: wait for ≤0 pending (current group done)
        // With STAGES=3: wait for ≤1 pending (current + 1 more)
        cp_async_wait_group<(STAGES == 2 ? 0 : 1)>();
        __syncthreads();  // Ensure visibility of sK/sV for read_stage
        
        // ========== COMPUTE on read_stage (same as v6a GREEN) ==========
        if (is_compute_warp) {
            // Iterate 16-col blocks within this kv_len
            for (int n0 = 0; n0 < kv_len; n0 += WMMA_N) {
                // --- WMMA Q@K^T ---
                wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> qk_frag;
                wmma::fill_fragment(qk_frag, 0.0f);
                
                for (int k0 = 0; k0 < HEAD_DIM; k0 += WMMA_K) {
                    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> q_frag;
                    const half* q_ptr = &sQ[warp_m0 * Dpad + k0];
                    wmma::load_matrix_sync(q_frag, q_ptr, Dpad);
                    
                    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> kt_frag;
                    const half* kt_ptr = &K_stage_ptr(sK, read_stage, N, Dpad)[n0 * Dpad + k0];
                    wmma::load_matrix_sync(kt_frag, kt_ptr, Dpad);
                    
                    wmma::mma_sync(qk_frag, q_frag, kt_frag, qk_frag);
                }
                
                // Scale scores
                #pragma unroll
                for (int i = 0; i < qk_frag.num_elements; ++i) {
                    qk_frag.x[i] *= scale;
                }
                
                // FIX: wmma_store_softmax_rebuild (reuse v6a pattern)
                wmma::store_matrix_sync(sS_frag, qk_frag, WMMA_N, wmma::mem_row_major);
                
                // Row-wise streaming softmax on this warp's 16 rows
                for (int r_local = 0; r_local < WMMA_M; ++r_local) {
                    int r_global = warp_m0 + r_local;
                    
                    // Find max and apply causal mask
                    float row_max = -FLT_MAX;
                    #pragma unroll
                    for (int n_local = 0; n_local < WMMA_N; ++n_local) {
                        float score = sS_frag[r_local * WMMA_N + n_local];
                        
                        // Apply causal mask
                        if (causal) {
                            int q_pos = q_start + r_global;
                            int k_pos = kv_start + n0 + n_local;
                            if (k_pos > q_pos) score = -FLT_MAX;
                        }
                        
                        sS_frag[r_local * WMMA_N + n_local] = score;
                        row_max = fmaxf(row_max, score);
                    }
                    
                    // Update global max
                    float m_old = m_smem[r_global];
                    float m_new = fmaxf(m_old, row_max);
                    
                    // Compute exp and sum for this tile
                    float tile_sum = 0.0f;
                    #pragma unroll
                    for (int n_local = 0; n_local < WMMA_N; ++n_local) {
                        float p = __expf(sS_frag[r_local * WMMA_N + n_local] - m_new);
                        tile_sum += p;
                        sP_frag[r_local * WMMA_N + n_local] = __float2half(p);
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
                
                // INSIGHT: pv_wmma_row_major (reuse v6a)
                wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> p_frag;
                wmma::load_matrix_sync(p_frag, sP_frag, WMMA_N);
                
                // WMMA P@V: accumulate into O
                for (int d0 = 0; d0 < HEAD_DIM; d0 += WMMA_K) {
                    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> v_frag;
                    const half* v_ptr = &V_stage_ptr(sV, read_stage, N, Dpad)[n0 * Dpad + d0];
                    wmma::load_matrix_sync(v_frag, v_ptr, Dpad);
                    
                    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> o_frag;
                    float* o_ptr = &O_accum[warp_m0 * Dpad + d0];
                    wmma::load_matrix_sync(o_frag, o_ptr, Dpad, wmma::mem_row_major);
                    
                    wmma::mma_sync(o_frag, p_frag, v_frag, o_frag);
                    wmma::store_matrix_sync(o_ptr, o_frag, Dpad, wmma::mem_row_major);
                }
            }
        }
        // ========== end COMPUTE on read_stage ==========
        
        __syncthreads();  // Ensures no consumer still reading this stage
        stage = write_stage;  // Rotate ring
    }
    
    // Epilogue (unchanged from v6a GREEN)
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
template __global__ void sdpa_fused_v2c_v7a_kernel<half, 64, 2>(
    const half*, const half*, const half*, half*, int, int, int, int, float, bool);
template __global__ void sdpa_fused_v2c_v7a_kernel<half, 128, 2>(
    const half*, const half*, const half*, half*, int, int, int, int, float, bool);
template __global__ void sdpa_fused_v2c_v7a_kernel<half, 64, 3>(
    const half*, const half*, const half*, half*, int, int, int, int, float, bool);
template __global__ void sdpa_fused_v2c_v7a_kernel<half, 128, 3>(
    const half*, const half*, const half*, half*, int, int, int, int, float, bool);

// Runtime dispatcher
cudaError_t sdpa_fused_forward_v2c_v7a(const SdpaParams& params, cudaStream_t stream) {
    const int M = (params.d == 64) ? TileConfig<64>::M : TileConfig<128>::M;
    const int STAGES = (params.L >= 2048) ? 3 : 2;  // ELITE-CHG: pipeline_depth
    
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
        printf("[V2c-v7a WARNING] SMEM %zu KB > limit %zu KB\n", smem_bytes / 1024, max_smem / 1024);
        return cudaErrorInvalidConfiguration;
    }
    
    static bool first_launch = true;
    if (first_launch) {
        printf("[V2c-v7a FAST] d=%d, L=%d, M=%d, N=%d, STAGES=%d, SMEM=%zu KB (cp.async overlap)\n",
               params.d, params.L, M, 
               (params.d == 64) ? TileConfig<64>::N : TileConfig<128>::N,
               STAGES, smem_bytes / 1024);
        first_launch = false;
    }
    
    cudaError_t err = cudaSuccess;
    
    if (params.d == 64 && STAGES == 2) {
        auto kernel_func = sdpa_fused_v2c_v7a_kernel<half, 64, 2>;
        err = cudaFuncSetAttribute(kernel_func, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
        if (err != cudaSuccess) return err;
        kernel_func<<<grid, block, smem_bytes, stream>>>(
            reinterpret_cast<const half*>(params.Q), reinterpret_cast<const half*>(params.K),
            reinterpret_cast<const half*>(params.V), reinterpret_cast<half*>(params.O),
            params.B, params.H, params.L, params.d, params.scale, params.causal
        );
    } else if (params.d == 64 && STAGES == 3) {
        auto kernel_func = sdpa_fused_v2c_v7a_kernel<half, 64, 3>;
        err = cudaFuncSetAttribute(kernel_func, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
        if (err != cudaSuccess) return err;
        kernel_func<<<grid, block, smem_bytes, stream>>>(
            reinterpret_cast<const half*>(params.Q), reinterpret_cast<const half*>(params.K),
            reinterpret_cast<const half*>(params.V), reinterpret_cast<half*>(params.O),
            params.B, params.H, params.L, params.d, params.scale, params.causal
        );
    } else if (params.d == 128 && STAGES == 2) {
        auto kernel_func = sdpa_fused_v2c_v7a_kernel<half, 128, 2>;
        err = cudaFuncSetAttribute(kernel_func, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
        if (err != cudaSuccess) return err;
        kernel_func<<<grid, block, smem_bytes, stream>>>(
            reinterpret_cast<const half*>(params.Q), reinterpret_cast<const half*>(params.K),
            reinterpret_cast<const half*>(params.V), reinterpret_cast<half*>(params.O),
            params.B, params.H, params.L, params.d, params.scale, params.causal
        );
    } else {
        auto kernel_func = sdpa_fused_v2c_v7a_kernel<half, 128, 3>;
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


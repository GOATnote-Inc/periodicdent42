/**
 * Child-V2c: True WMMA + Transposed K + XOR Swizzle
 * 
 * UPGRADES FROM V2b:
 * - INSIGHT: wmma_microtile - Real 16×16×16 WMMA for Q@K^T and P@V
 * - INSIGHT: xor_swizzle - K transposed to col-major with XOR swizzle (bank conflict free)
 * - INSIGHT: smem_layout - Dropped S_scores, consume fragments on-the-fly
 * - INSIGHT: pipeline_depth - Fixed cp.async 16B segment loops
 * 
 * ELITE-CHG: persistent_cta - Try persistent CTAs for better occupancy
 * ELITE-CHG: micro_tile_reshape - Explore 32×32×16 or 8×128×16 tiles
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

// Tile configurations (WMMA-friendly: multiple of 16)
template<int HEAD_DIM>
struct TileConfig {
    static constexpr int M = 64;  // Must be multiple of 16 for WMMA
    static constexpr int N = (HEAD_DIM == 64) ? 64 : 32;  // d=128 needs smaller N for SMEM
    static constexpr int K = HEAD_DIM;  // Full head_dim
    static constexpr int WMMA_M = 16;
    static constexpr int WMMA_N = 16;
    static constexpr int WMMA_K = 16;
};

#define NUM_WARPS 8
#define THREADS_PER_BLOCK (NUM_WARPS * 32)
#define HEAD_DIM_PAD(d) ((d) + 8)  // Bank conflict padding

// INSIGHT: xor_swizzle - XOR swizzle for bank-conflict-free ldmatrix
__device__ __forceinline__ int xor_swizzle(int n, int k, int granularity = 8) {
    int n_blk = n >> 3;  // / 8
    int n_in = n & 7;    // % 8
    int k_blk = k >> 3;
    return (n_blk ^ k_blk) * granularity + n_in;
}

// cp.async helpers
// NOTE: Ada (sm_89) only supports 16-byte cp.async.cg, not 8B or 4B
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

// Warp reductions (within single warp)
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
 * SMEM Layout V2c (no S_scores, transposed K)
 * INSIGHT: smem_layout - Consume fragments on-the-fly
 */
template<int HEAD_DIM, int STAGES>
struct SmemLayout {
    static constexpr int M = TileConfig<HEAD_DIM>::M;
    static constexpr int N = TileConfig<HEAD_DIM>::N;
    static constexpr int PAD = HEAD_DIM_PAD(HEAD_DIM) - HEAD_DIM;
    static constexpr int HEAD_DIM_PADDED = HEAD_DIM_PAD(HEAD_DIM);
    
    // SMEM sizes (bytes)
    static constexpr size_t sQ_bytes = M * HEAD_DIM_PADDED * sizeof(half);
    // K stored col-major: [HEAD_DIM_PADDED][STAGES*N]
    static constexpr size_t sK_bytes = HEAD_DIM_PADDED * STAGES * N * sizeof(half);
    // V stored row-major: [STAGES*N][HEAD_DIM_PADDED]
    static constexpr size_t sV_bytes = STAGES * N * HEAD_DIM_PADDED * sizeof(half);
    // O_accum in FP32
    static constexpr size_t O_accum_bytes = M * HEAD_DIM_PADDED * sizeof(float);
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
 * Child-V2c: True WMMA Kernel
 * INSIGHT: wmma_microtile - Real Tensor Core usage
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
    constexpr int WMMA_M = 16;
    constexpr int WMMA_N = 16;
    constexpr int WMMA_K = 16;
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
    float *O_accum, *m_smem, *l_smem;
    Layout::get_pointers(smem_base, sQ, sK, sV, O_accum, m_smem, l_smem);
    
    // Single-warp ownership
    const int rows_per_warp = (M + NUM_WARPS - 1) / NUM_WARPS;
    const int my_row_start = warp_id * rows_per_warp;
    const int my_row_end = min(my_row_start + rows_per_warp, num_q_rows);
    const int my_num_rows = max(0, my_row_end - my_row_start);
    
    // Load Q tile (row-major)
    NVTX_RANGE_PUSH("Load_Q");
    for (int idx = tid; idx < num_q_rows * HEAD_DIM; idx += blockDim.x) {
        int r = idx / HEAD_DIM;
        int c = idx % HEAD_DIM;
        int smem_idx = r * HEAD_DIM_PADDED + c;
        if (q_start + r < L) {
            sQ[smem_idx] = __ldg(&Q_bh[(q_start + r) * d + c]);
        } else {
            sQ[smem_idx] = __float2half(0.0f);
        }
    }
    
    // Initialize per-row stats
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
    NVTX_RANGE_POP();
    
    const int num_kv_tiles = (L + N - 1) / N;
    int stage = 0;
    
    // Main loop over K/V tiles
    for (int t = 0; t < num_kv_tiles; ++t) {
        const int kv_start = t * N;
        const int kv_end = min(kv_start + N, L);
        const int kv_len = kv_end - kv_start;
        
        const int read_stage = stage;
        const int write_stage = (STAGES == 2) ? (1 - stage) : ((stage + 1) % STAGES);
        
        // INSIGHT: pipeline_depth - Producer warp (warp 7) loads K/V with cp.async
        if (warp_id == 7) {
            NVTX_RANGE_PUSH("cp.async_load");
            
            // Copy K transposed to col-major with XOR swizzle
            // K global: [kv_len, HEAD_DIM]
            // K SMEM: [HEAD_DIM_PADDED][STAGES*N] col-major
            constexpr int ELEMS_PER_SEG = 8;  // 16 bytes
            int segs_per_row = HEAD_DIM / ELEMS_PER_SEG;
            int total_segs = kv_len * segs_per_row;
            
            for (int seg_idx = lane; seg_idx < total_segs; seg_idx += 32) {
                int n = seg_idx / segs_per_row;  // K row (0..kv_len-1)
                int seg = seg_idx % segs_per_row;  // segment along HEAD_DIM
                int k_start = seg * ELEMS_PER_SEG;
                
                // Transpose: global K[n, k_start:k_start+8] → SMEM K[k_start:k_start+8, n]
                const half* src_k = &K_bh[(kv_start + n) * d + k_start];
                
                // XOR swizzle on n index
                for (int k_off = 0; k_off < ELEMS_PER_SEG; ++k_off) {
                    int k = k_start + k_off;
                    int n_swizzled = xor_swizzle(read_stage * N + n, k);
                    half* dst_k = &sK[k * (STAGES * N) + n_swizzled];
                    
                    // Use cp.async if aligned, else direct store
                    if (((size_t)src_k % 16 == 0) && ((size_t)dst_k % 16 == 0) && k_off == 0) {
                        cp_async_16B(dst_k, src_k);
                        break;  // 16B copied all 8 elements
                    } else {
                        *dst_k = __ldg(&src_k[k_off]);
                    }
                }
                
                // V is row-major (no transpose needed)
                const half* src_v = &V_bh[(kv_start + n) * d + k_start];
                half* dst_v = &sV[(read_stage * N + n) * HEAD_DIM_PADDED + k_start];
                
                // Use cp.async only if 16B-aligned (Ada requirement)
                if (((size_t)src_v % 16 == 0) && ((size_t)dst_v % 16 == 0)) {
                    cp_async_16B(dst_v, src_v);
                } else {
                    // Scalar fallback
                    #pragma unroll
                    for (int i = 0; i < ELEMS_PER_SEG; ++i) {
                        dst_v[i] = __ldg(&src_v[i]);
                    }
                }
            }
            
            cp_async_commit_group();
            
            // Prefetch next tile
            if (t + 1 < num_kv_tiles) {
                int next_kv_start = (t + 1) * N;
                int next_kv_end = min(next_kv_start + N, L);
                int next_kv_len = next_kv_end - next_kv_start;
                int next_total_segs = next_kv_len * segs_per_row;
                
                for (int seg_idx = lane; seg_idx < next_total_segs; seg_idx += 32) {
                    int n = seg_idx / segs_per_row;
                    int seg = seg_idx % segs_per_row;
                    int k_start = seg * ELEMS_PER_SEG;
                    
                    const half* src_k = &K_bh[(next_kv_start + n) * d + k_start];
                    const half* src_v = &V_bh[(next_kv_start + n) * d + k_start];
                    
                    // K transpose + swizzle
                    for (int k_off = 0; k_off < ELEMS_PER_SEG; ++k_off) {
                        int k = k_start + k_off;
                        int n_swizzled = xor_swizzle(write_stage * N + n, k);
                        half* dst_k = &sK[k * (STAGES * N) + n_swizzled];
                        
                        if (((size_t)src_k % 16 == 0) && ((size_t)dst_k % 16 == 0) && k_off == 0) {
                            cp_async_16B(dst_k, src_k);
                            break;
                        } else {
                            *dst_k = __ldg(&src_k[k_off]);
                        }
                    }
                    
                    // V row-major
                    half* dst_v = &sV[(write_stage * N + n) * HEAD_DIM_PADDED + k_start];
                    if (((size_t)src_v % 16 == 0) && ((size_t)dst_v % 16 == 0)) {
                        cp_async_16B(dst_v, src_v);
                    } else {
                        #pragma unroll
                        for (int i = 0; i < ELEMS_PER_SEG; ++i) {
                            dst_v[i] = __ldg(&src_v[i]);
                        }
                    }
                }
                cp_async_commit_group();
            }
            
            NVTX_RANGE_POP();
        }
        
        // Wait for cp.async
        if (STAGES == 2) {
            cp_async_wait_group<0>();
        } else {
            cp_async_wait_group<1>();
        }
        __syncthreads();
        
        // INSIGHT: wmma_microtile - Real WMMA compute (all warps participate)
        if (my_num_rows > 0) {
            NVTX_RANGE_PUSH("QK_wmma");
            
            // Each warp processes 16×16 tiles within its owned rows
            // Map warp to WMMA tiles
            int m0 = my_row_start;  // Start of warp's rows
            int num_m_tiles = (my_num_rows + WMMA_M - 1) / WMMA_M;
            int num_n_tiles = (kv_len + WMMA_N - 1) / WMMA_N;
            
            for (int mt = 0; mt < num_m_tiles; ++mt) {
                int m_tile_start = m0 + mt * WMMA_M;
                int m_tile_rows = min(WMMA_M, num_q_rows - m_tile_start);
                if (m_tile_rows <= 0) break;
                
                for (int nt = 0; nt < num_n_tiles; ++nt) {
                    int n_tile_start = nt * WMMA_N;
                    int n_tile_cols = min(WMMA_N, kv_len - n_tile_start);
                    if (n_tile_cols <= 0) break;
                    
                    // WMMA: Q @ K^T
                    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
                    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
                    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
                    
                    wmma::fill_fragment(c_frag, 0.0f);
                    
                    // Loop over K dimension in chunks of 16
                    for (int k0 = 0; k0 < HEAD_DIM; k0 += WMMA_K) {
                        // Load Q[m_tile_start:m_tile_start+16, k0:k0+16] row-major
                        const half* a_ptr = &sQ[m_tile_start * HEAD_DIM_PADDED + k0];
                        wmma::load_matrix_sync(a_frag, a_ptr, HEAD_DIM_PADDED);
                        
                        // Load K^T[k0:k0+16, n_tile_start:n_tile_start+16] col-major
                        // K stored as [HEAD_DIM_PADDED][STAGES*N]
                        const half* b_ptr = &sK[k0 * (STAGES * N) + (read_stage * N + n_tile_start)];
                        wmma::load_matrix_sync(b_frag, b_ptr, STAGES * N);
                        
                        // MMA
                        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
                    }
                    
                    // c_frag now holds 16×16 scores
                    // Apply scale, causal mask, and streaming softmax
                    
                    // Extract scores to registers
                    float scores[16][16];  // Per-thread partial
                    wmma::store_matrix_sync(&scores[0][0], c_frag, 16, wmma::mem_row_major);
                    
                    // Process each row in this 16×16 tile
                    NVTX_RANGE_PUSH("Softmax_update");
                    for (int mi = 0; mi < m_tile_rows; ++mi) {
                        int r = m_tile_start + mi;
                        if (r >= num_q_rows) break;
                        
                        // Apply scale and causal mask
                        float row_max = -FLT_MAX;
                        for (int ni = 0; ni < n_tile_cols; ++ni) {
                            float score = scores[mi][ni] * scale;
                            
                            // Causal mask
                            if (causal) {
                                int q_pos = q_start + r;
                                int k_pos = kv_start + n_tile_start + ni;
                                if (k_pos > q_pos) {
                                    score = -FLT_MAX;
                                }
                            }
                            
                            scores[mi][ni] = score;
                            row_max = fmaxf(row_max, score);
                        }
                        
                        // Warp-reduce max
                        row_max = warp_reduce_max(row_max);
                        
                        // Update streaming softmax stats
                        float m_old = m_smem[r];
                        float m_new = fmaxf(m_old, row_max);
                        
                        float l_old = l_smem[r];
                        float l_add = 0.0f;
                        for (int ni = 0; ni < n_tile_cols; ++ni) {
                            scores[mi][ni] = __expf(scores[mi][ni] - m_new);
                            l_add += scores[mi][ni];
                        }
                        
                        // Warp-reduce sum
                        l_add = warp_reduce_sum(l_add);
                        
                        float rescale = __expf(m_old - m_new);
                        float l_new = l_old * rescale + l_add;
                        
                        // Rescale O_accum for this row (all threads in warp)
                        for (int c = lane; c < HEAD_DIM; c += 32) {
                            int o_idx = r * HEAD_DIM_PADDED + c;
                            O_accum[o_idx] *= rescale;
                        }
                        
                        // Update stats (lane 0 only)
                        if (lane == 0) {
                            m_smem[r] = m_new;
                            l_smem[r] = l_new;
                        }
                    }
                    NVTX_RANGE_POP();
                    
                    // INSIGHT: wmma_microtile - P @ V accumulation
                    NVTX_RANGE_PUSH("PV_wmma");
                    
                    // Build P fragment (16×16 half) from exp(scores)
                    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> p_frag;
                    // Convert scores to half and load into fragment
                    half p_mat[16][16];
                    for (int mi = 0; mi < 16; ++mi) {
                        for (int ni = 0; ni < 16; ++ni) {
                            p_mat[mi][ni] = __float2half(scores[mi][ni]);
                        }
                    }
                    wmma::load_matrix_sync(p_frag, &p_mat[0][0], 16);
                    
                    // Accumulate P @ V into O_accum
                    for (int k0 = 0; k0 < HEAD_DIM; k0 += WMMA_K) {
                        // Load V[n_tile_start:n_tile_start+16, k0:k0+16] row-major
                        const half* v_ptr = &sV[(read_stage * N + n_tile_start) * HEAD_DIM_PADDED + k0];
                        wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> v_frag;
                        wmma::load_matrix_sync(v_frag, v_ptr, HEAD_DIM_PADDED);
                        
                        // O_accum fragment (16×16 FP32)
                        wmma::fragment<wmma::accumulator, 16, 16, 16, float> o_frag;
                        
                        // Load current O_accum values
                        float* o_ptr = &O_accum[m_tile_start * HEAD_DIM_PADDED + k0];
                        wmma::load_matrix_sync(o_frag, o_ptr, HEAD_DIM_PADDED, wmma::mem_row_major);
                        
                        // MMA: O_accum += P @ V
                        wmma::mma_sync(o_frag, p_frag, v_frag, o_frag);
                        
                        // Store back
                        wmma::store_matrix_sync(o_ptr, o_frag, HEAD_DIM_PADDED, wmma::mem_row_major);
                    }
                    
                    NVTX_RANGE_POP();
                }
            }
            
            NVTX_RANGE_POP();
        }
        
        // Single sync point
        __syncthreads();
        
        stage = write_stage;
    }
    
    // Epilogue: write O = O_accum / l
    NVTX_RANGE_PUSH("Store");
    for (int idx = tid; idx < num_q_rows * HEAD_DIM; idx += blockDim.x) {
        int r = idx / HEAD_DIM;
        int c = idx % HEAD_DIM;
        if (r < num_q_rows) {
            int o_idx = r * HEAD_DIM_PADDED + c;
            float o_val = O_accum[o_idx] / l_smem[r];
            O_bh[(q_start + r) * d + c] = __float2half(o_val);
        }
    }
    NVTX_RANGE_POP();
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
    const int STAGES = (params.L >= 2048) ? 3 : 2;
    
    dim3 grid((params.L + M - 1) / M, params.B * params.H);
    dim3 block(THREADS_PER_BLOCK);
    
    // Compute dynamic SMEM size
    size_t smem_bytes;
    if (params.d == 64) {
        smem_bytes = (STAGES == 2) ? 
            SmemLayout<64, 2>::total_bytes : SmemLayout<64, 3>::total_bytes;
    } else {
        smem_bytes = (STAGES == 2) ?
            SmemLayout<128, 2>::total_bytes : SmemLayout<128, 3>::total_bytes;
    }
    
    // Validate SMEM vs device limit
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    
    size_t max_smem = prop.sharedMemPerBlockOptin;
    if (smem_bytes > max_smem) {
        printf("[V2c WARNING] SMEM %zu KB > device limit %zu KB\n",
               smem_bytes / 1024, max_smem / 1024);
        return cudaErrorInvalidConfiguration;
    }
    
    // Print config on first launch
    static bool first_launch = true;
    if (first_launch) {
        printf("[V2c] d=%d, L=%d, M=%d, N=%d, STAGES=%d, SMEM=%zu KB (WMMA enabled)\n",
               params.d, params.L, M, 
               (params.d == 64) ? TileConfig<64>::N : TileConfig<128>::N,
               STAGES, smem_bytes / 1024);
        first_launch = false;
    }
    
    cudaError_t err = cudaSuccess;
    
    if (params.d == 64 && STAGES == 2) {
        auto kernel_func = sdpa_fused_v2c_kernel<half, 64, 2>;
        err = cudaFuncSetAttribute(kernel_func, 
                                     cudaFuncAttributeMaxDynamicSharedMemorySize, 
                                     smem_bytes);
        if (err != cudaSuccess) return err;
        
        kernel_func<<<grid, block, smem_bytes, stream>>>(
            reinterpret_cast<const half*>(params.Q),
            reinterpret_cast<const half*>(params.K),
            reinterpret_cast<const half*>(params.V),
            reinterpret_cast<half*>(params.O),
            params.B, params.H, params.L, params.d, params.scale, params.causal
        );
    } else if (params.d == 64 && STAGES == 3) {
        auto kernel_func = sdpa_fused_v2c_kernel<half, 64, 3>;
        err = cudaFuncSetAttribute(kernel_func, 
                                     cudaFuncAttributeMaxDynamicSharedMemorySize, 
                                     smem_bytes);
        if (err != cudaSuccess) return err;
        
        kernel_func<<<grid, block, smem_bytes, stream>>>(
            reinterpret_cast<const half*>(params.Q),
            reinterpret_cast<const half*>(params.K),
            reinterpret_cast<const half*>(params.V),
            reinterpret_cast<half*>(params.O),
            params.B, params.H, params.L, params.d, params.scale, params.causal
        );
    } else if (params.d == 128 && STAGES == 2) {
        auto kernel_func = sdpa_fused_v2c_kernel<half, 128, 2>;
        err = cudaFuncSetAttribute(kernel_func, 
                                     cudaFuncAttributeMaxDynamicSharedMemorySize, 
                                     smem_bytes);
        if (err != cudaSuccess) return err;
        
        kernel_func<<<grid, block, smem_bytes, stream>>>(
            reinterpret_cast<const half*>(params.Q),
            reinterpret_cast<const half*>(params.K),
            reinterpret_cast<const half*>(params.V),
            reinterpret_cast<half*>(params.O),
            params.B, params.H, params.L, params.d, params.scale, params.causal
        );
    } else {
        auto kernel_func = sdpa_fused_v2c_kernel<half, 128, 3>;
        err = cudaFuncSetAttribute(kernel_func, 
                                     cudaFuncAttributeMaxDynamicSharedMemorySize, 
                                     smem_bytes);
        if (err != cudaSuccess) return err;
        
        kernel_func<<<grid, block, smem_bytes, stream>>>(
            reinterpret_cast<const half*>(params.Q),
            reinterpret_cast<const half*>(params.K),
            reinterpret_cast<const half*>(params.V),
            reinterpret_cast<half*>(params.O),
            params.B, params.H, params.L, params.d, params.scale, params.causal
        );
    }
    
    return cudaGetLastError();
}


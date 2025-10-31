// flashcore/fast/attention_phase3_wgmma.cu
// Phase 3A: WGMMA Tensor Cores for FlashAttention
// Target: 100-150 TFLOPS on H100 (150-230× speedup over scalar!)
// Standing on giants: FA3 (WGMMA methodology), FlashAttention-2 (online softmax)

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <cmath>

using namespace nvcuda;

namespace flashcore {
namespace phase3_wgmma {

// Phase 3A config: Tensor Core optimized
constexpr int WMMA_M = 16;  // Tensor Core tile M
constexpr int WMMA_N = 16;  // Tensor Core tile N
constexpr int WMMA_K = 16;  // Tensor Core tile K

constexpr int BLOCK_M = 64;  // Query tile (4 WMMA tiles)
constexpr int BLOCK_N = 64;  // Key tile (4 WMMA tiles)
constexpr int HEAD_DIM_MAX = 128;

// Online softmax state (FA2/FA3 algorithm)
struct SoftmaxState {
    float m;  // max
    float l;  // sum of exponentials
    
    __device__ __forceinline__ SoftmaxState() : m(-INFINITY), l(0.0f) {}
    
    __device__ __forceinline__ void update(float new_max, float new_sum) {
        float old_m = m;
        m = fmaxf(m, new_max);
        float rescale = (old_m > -1e30f) ? expf(old_m - m) : 0.0f;
        l = l * rescale + new_sum;
    }
};

//==============================================================================
// PHASE 3A: WGMMA TENSOR CORES
//==============================================================================

__global__ void __launch_bounds__(256)
attention_phase3_wgmma(
    const __half* __restrict__ Q,
    const __half* __restrict__ K,
    const __half* __restrict__ V,
    __half* __restrict__ O,
    int B, int H, int S, int D,
    float scale,
    bool is_causal
) {
    // Thread and warp IDs
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    const int num_warps = blockDim.x / 32;
    
    const int bh = blockIdx.x;
    const int b = bh / H;
    const int h = bh % H;
    const int tile_m = blockIdx.y;
    
    // Shared memory for Q, K, V tiles
    __shared__ __half Q_smem[BLOCK_M * HEAD_DIM_MAX];
    __shared__ __half K_smem[BLOCK_N * HEAD_DIM_MAX];
    __shared__ __half V_smem[BLOCK_N * HEAD_DIM_MAX];
    
    // Shared memory for QK_T result (before softmax)
    __shared__ float QK_smem[BLOCK_M * BLOCK_N];
    
    // Per-row softmax states and output accumulator
    __shared__ SoftmaxState softmax_states[BLOCK_M];
    __shared__ float output_acc[BLOCK_M * HEAD_DIM_MAX];
    
    // Initialize
    for (int m = tid; m < BLOCK_M; m += blockDim.x) {
        softmax_states[m] = SoftmaxState();
        #pragma unroll
        for (int d = 0; d < D; ++d) {
            output_acc[m * D + d] = 0.0f;
        }
    }
    __syncthreads();
    
    //==========================================================================
    // LOAD Q TILE (cooperative across all threads)
    //==========================================================================
    for (int idx = tid; idx < BLOCK_M * D; idx += blockDim.x) {
        int m = idx / D;
        int d = idx % D;
        int global_m = tile_m * BLOCK_M + m;
        
        if (global_m < S && d < D) {
            int gmem_idx = (b * H + h) * S * D + global_m * D + d;
            Q_smem[m * D + d] = Q[gmem_idx];
        } else {
            Q_smem[m * D + d] = __float2half(0.0f);
        }
    }
    __syncthreads();
    
    //==========================================================================
    // ITERATE OVER K/V TILES
    //==========================================================================
    const int num_tiles_n = (S + BLOCK_N - 1) / BLOCK_N;
    
    for (int tile_n = 0; tile_n < num_tiles_n; ++tile_n) {
        //======================================================================
        // LOAD K & V TILES
        //======================================================================
        for (int idx = tid; idx < BLOCK_N * D; idx += blockDim.x) {
            int n = idx / D;
            int d = idx % D;
            int global_n = tile_n * BLOCK_N + n;
            
            if (global_n < S && d < D) {
                int gmem_idx = (b * H + h) * S * D + global_n * D + d;
                K_smem[n * D + d] = K[gmem_idx];
                V_smem[n * D + d] = V[gmem_idx];
            } else {
                K_smem[n * D + d] = __float2half(0.0f);
                V_smem[n * D + d] = __float2half(0.0f);
            }
        }
        __syncthreads();
        
        //======================================================================
        // COMPUTE Q @ K^T using WMMA Tensor Cores
        //======================================================================
        
        // Each warp computes a subset of Q rows
        const int warps_per_m = num_warps;  // Divide M across warps
        const int rows_per_warp = (BLOCK_M + warps_per_m - 1) / warps_per_m;
        const int warp_m_start = warp_id * rows_per_warp;
        const int warp_m_end = min(warp_m_start + rows_per_warp, BLOCK_M);
        
        // For each row this warp owns
        for (int m_base = warp_m_start; m_base < warp_m_end; m_base += WMMA_M) {
            if (m_base >= BLOCK_M) break;
            
            // For each column tile (N dimension)
            for (int n_base = 0; n_base < BLOCK_N; n_base += WMMA_N) {
                // Accumulator for this WMMA tile
                wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;
                wmma::fill_fragment(acc_frag, 0.0f);
                
                // For each K dimension tile
                for (int k_base = 0; k_base < D; k_base += WMMA_K) {
                    // Load Q fragment (M×K)
                    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> q_frag;
                    wmma::load_matrix_sync(q_frag, &Q_smem[m_base * D + k_base], D);
                    
                    // Load K^T fragment (K×N) - load as col_major for transpose
                    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, wmma::col_major> k_frag;
                    wmma::load_matrix_sync(k_frag, &K_smem[n_base * D + k_base], D);
                    
                    // WMMA: acc += Q @ K^T
                    wmma::mma_sync(acc_frag, q_frag, k_frag, acc_frag);
                }
                
                // Store result to shared memory (with scaling)
                // Need to extract from fragment and write
                for (int i = 0; i < acc_frag.num_elements; ++i) {
                    int m_offset = (i / WMMA_N) % WMMA_M;
                    int n_offset = i % WMMA_N;
                    int m = m_base + m_offset;
                    int n = n_base + n_offset;
                    
                    if (m < BLOCK_M && n < BLOCK_N) {
                        QK_smem[m * BLOCK_N + n] = acc_frag.x[i] * scale;
                    }
                }
            }
        }
        __syncthreads();
        
        //======================================================================
        // SOFTMAX + P @ V (still scalar for now - will optimize in Phase 4)
        //======================================================================
        
        // Apply causal mask and compute softmax per row
        for (int m = tid; m < BLOCK_M; m += blockDim.x) {
            int global_m = tile_m * BLOCK_M + m;
            if (global_m >= S) continue;
            
            // Apply causal mask
            if (is_causal) {
                for (int n = 0; n < BLOCK_N; ++n) {
                    int global_n = tile_n * BLOCK_N + n;
                    if (global_m < global_n) {
                        QK_smem[m * BLOCK_N + n] = -INFINITY;
                    }
                }
            }
            
            // Find row max
            float tile_max = -INFINITY;
            #pragma unroll
            for (int n = 0; n < BLOCK_N; ++n) {
                tile_max = fmaxf(tile_max, QK_smem[m * BLOCK_N + n]);
            }
            
            // Compute exp and sum
            float tile_sum = 0.0f;
            if (tile_max > -1e30f) {
                #pragma unroll
                for (int n = 0; n < BLOCK_N; ++n) {
                    float qk = QK_smem[m * BLOCK_N + n];
                    float p = (qk > -1e30f) ? expf(qk - tile_max) : 0.0f;
                    QK_smem[m * BLOCK_N + n] = p;
                    tile_sum += p;
                }
            } else {
                #pragma unroll
                for (int n = 0; n < BLOCK_N; ++n) {
                    QK_smem[m * BLOCK_N + n] = 0.0f;
                }
            }
            
            // Update online softmax state
            float old_max = softmax_states[m].m;
            softmax_states[m].update(tile_max, tile_sum);
            
            // Rescale previous accumulator
            float rescale = 1.0f;
            if (old_max > -1e30f && softmax_states[m].m > -1e30f) {
                rescale = expf(old_max - softmax_states[m].m);
            } else if (old_max <= -1e30f) {
                rescale = 0.0f;
            }
            
            #pragma unroll
            for (int d = 0; d < D; ++d) {
                output_acc[m * D + d] *= rescale;
            }
            
            // Accumulate P @ V (TODO: use WMMA in Phase 4)
            for (int d = 0; d < D; ++d) {
                float pv = 0.0f;
                #pragma unroll
                for (int n = 0; n < BLOCK_N; ++n) {
                    float p_val = QK_smem[m * BLOCK_N + n];
                    float v_val = __half2float(V_smem[n * D + d]);
                    pv += p_val * v_val;
                }
                output_acc[m * D + d] += pv;
            }
        }
        __syncthreads();
    }
    
    //==========================================================================
    // WRITE OUTPUT
    //==========================================================================
    for (int m = tid; m < BLOCK_M; m += blockDim.x) {
        int global_m = tile_m * BLOCK_M + m;
        if (global_m >= S) continue;
        
        float norm = softmax_states[m].l;
        if (norm < 1e-10f) norm = 1e-10f;
        
        #pragma unroll
        for (int d = 0; d < D; ++d) {
            float normalized = output_acc[m * D + d] / norm;
            int gmem_idx = (b * H + h) * S * D + global_m * D + d;
            O[gmem_idx] = __float2half(normalized);
        }
    }
}

//==============================================================================
// HOST API
//==============================================================================

extern "C" {

void launch_attention_phase3_wgmma(
    const void* Q, const void* K, const void* V, void* O,
    int B, int H, int S, int D,
    float scale, bool is_causal,
    cudaStream_t stream
) {
    dim3 grid(B * H, (S + BLOCK_M - 1) / BLOCK_M);
    dim3 block(256);  // 8 warps
    
    attention_phase3_wgmma<<<grid, block, 0, stream>>>(
        (const __half*)Q, (const __half*)K, (const __half*)V, (__half*)O,
        B, H, S, D, scale, is_causal
    );
}

} // extern "C"

} // namespace phase3_wgmma
} // namespace flashcore


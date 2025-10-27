// flashcore/fast/attention_phase4_fused.cu
// EXPERT DIRECTIVE: True Fused Flash Attention Kernel
//
// KEY INSIGHT: Never materialize S or P matrices!
// - cuBLASLt approach: 128 GB memory traffic (S + P writes/reads)
// - Fused approach: 10-20 GB memory traffic (only O writes)
// - Expected: 6-12× memory bandwidth improvement!
//
// MILESTONE 1: Basic fused kernel (no async, single warp group)
// Target: 5-8 TFLOPS (should beat cuBLASLt 0.83 TFLOPS immediately!)

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <iostream>

using namespace nvcuda;

// Tile sizes (tuned for H100)
constexpr int TILE_M = 64;   // Query tile size
constexpr int TILE_N = 64;   // Key tile size  
constexpr int TILE_K = 64;   // Head dimension (D)
constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;

//------------------------------------------------------------------------------
// FUSED FLASH ATTENTION KERNEL (Milestone 1: Basic)
//------------------------------------------------------------------------------
__global__ void flash_attention_fused_basic(
    const __half* __restrict__ Q,  // [B, H, S, D]
    const __half* __restrict__ K,  // [B, H, S, D]
    const __half* __restrict__ V,  // [B, H, S, D]
    __half* __restrict__ O,        // [B, H, S, D]
    const int B, const int H, const int S, const int D,
    const float softmax_scale
) {
    // Block processes one head's tile
    const int batch_idx = blockIdx.z / H;
    const int head_idx = blockIdx.z % H;
    const int tile_m_idx = blockIdx.y;  // Query tile
    
    // Shared memory for tiles (no double-buffering yet - Milestone 1)
    __shared__ __half smem_Q[TILE_M][TILE_K + 8];  // +8 padding to avoid bank conflicts
    __shared__ __half smem_K[TILE_N][TILE_K + 8];
    __shared__ __half smem_V[TILE_N][TILE_K + 8];
    
    // Shared memory for accumulation (registers would spill!)
    __shared__ float smem_O[TILE_M][TILE_K + 8];
    __shared__ float smem_m[TILE_M];
    __shared__ float smem_l[TILE_M];
    
    // Initialize accumulator and softmax state
    const int tid = threadIdx.x;
    for (int idx = tid; idx < TILE_M * D; idx += blockDim.x) {
        int m = idx / D;
        int k = idx % D;
        smem_O[m][k] = 0.0f;
    }
    for (int m = tid; m < TILE_M; m += blockDim.x) {
        smem_m[m] = -INFINITY;
        smem_l[m] = 0.0f;
    }
    __syncthreads();
    
    // Calculate global offsets
    const int qo_offset = (batch_idx * H + head_idx) * S * D;
    const int kv_offset = (batch_idx * H + head_idx) * S * D;
    
    const int tile_m_start = tile_m_idx * TILE_M;
    if (tile_m_start >= S) return;  // Out of bounds
    
    // Load Q tile (this block's query tile - reused across all K tiles)
    {
        const int tid = threadIdx.x;
        const int num_threads = blockDim.x;
        
        for (int idx = tid; idx < TILE_M * D; idx += num_threads) {
            const int m = idx / D;
            const int k = idx % D;
            const int global_m = tile_m_start + m;
            
            if (global_m < S && k < D) {
                smem_Q[m][k] = Q[qo_offset + global_m * D + k];
            } else {
                smem_Q[m][k] = __float2half(0.0f);
            }
        }
    }
    __syncthreads();
    
    // Iterate over K/V tiles (this is the FUSED loop!)
    const int num_tiles_n = (S + TILE_N - 1) / TILE_N;
    
    for (int tile_n_idx = 0; tile_n_idx < num_tiles_n; ++tile_n_idx) {
        const int tile_n_start = tile_n_idx * TILE_N;
        
        // Load K tile
        {
            const int tid = threadIdx.x;
            const int num_threads = blockDim.x;
            
            for (int idx = tid; idx < TILE_N * D; idx += num_threads) {
                const int n = idx / D;
                const int k = idx % D;
                const int global_n = tile_n_start + n;
                
                if (global_n < S && k < D) {
                    smem_K[n][k] = K[kv_offset + global_n * D + k];
                } else {
                    smem_K[n][k] = __float2half(0.0f);
                }
            }
        }
        
        // Load V tile
        {
            const int tid = threadIdx.x;
            const int num_threads = blockDim.x;
            
            for (int idx = tid; idx < TILE_N * D; idx += num_threads) {
                const int n = idx / D;
                const int k = idx % D;
                const int global_n = tile_n_start + n;
                
                if (global_n < S && k < D) {
                    smem_V[n][k] = V[kv_offset + global_n * D + k];
                } else {
                    smem_V[n][k] = __float2half(0.0f);
                }
            }
        }
        __syncthreads();
        
        // STEP 1: Compute S_tile = Q @ K^T using WMMA (NEVER write to global memory!)
        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> frag_Q;
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, wmma::col_major> frag_K;  // Transposed!
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> frag_S;
        
        // Initialize S accumulator
        wmma::fill_fragment(frag_S, 0.0f);
        
        // Compute S = Q @ K^T (tile-by-tile matmul)
        const int num_k_tiles = (D + WMMA_K - 1) / WMMA_K;
        for (int k_tile = 0; k_tile < num_k_tiles; ++k_tile) {
            // Load Q fragment
            wmma::load_matrix_sync(frag_Q, &smem_Q[0][k_tile * WMMA_K], TILE_K + 8);
            
            // Load K^T fragment (note: we load as col-major for transpose)
            wmma::load_matrix_sync(frag_K, &smem_K[0][k_tile * WMMA_K], TILE_K + 8);
            
            // Multiply-accumulate: S += Q @ K^T
            wmma::mma_sync(frag_S, frag_Q, frag_K, frag_S);
        }
        
        // STEP 2: Fused Online Softmax (NEVER materialize P!)
        // Apply softmax scale and update running max/sum
        float S_tile[WMMA_M][WMMA_N];  // Temporary storage (registers!)
        wmma::store_matrix_sync(&S_tile[0][0], frag_S, WMMA_N, wmma::mem_row_major);
        
        // Scale attention scores
        #pragma unroll
        for (int i = 0; i < WMMA_M; ++i) {
            #pragma unroll
            for (int j = 0; j < WMMA_N; ++j) {
                S_tile[i][j] *= softmax_scale;
            }
        }
        
        // Online softmax update (per row)
        #pragma unroll
        for (int i = 0; i < WMMA_M; ++i) {
            // Find max in current tile
            float row_max = S_tile[i][0];
            #pragma unroll
            for (int j = 1; j < WMMA_N; ++j) {
                row_max = fmaxf(row_max, S_tile[i][j]);
            }
            
            // Update global max
            float old_m = acc_m[i];
            float new_m = fmaxf(old_m, row_max);
            float exp_diff_old = expf(old_m - new_m);
            float exp_diff_new = expf(row_max - new_m);
            
            // Rescale old output accumulator
            #pragma unroll
            for (int k = 0; k < TILE_K; ++k) {
                acc_O[i][k] *= exp_diff_old;
            }
            
            // Compute P (attention weights) and accumulate P @ V
            float row_sum = 0.0f;
            #pragma unroll
            for (int j = 0; j < WMMA_N; ++j) {
                float p_ij = expf(S_tile[i][j] - new_m);
                row_sum += p_ij;
                
                // STEP 3: Fused P @ V (accumulate directly, NEVER materialize P!)
                // O[i] += p_ij * V[j]
                #pragma unroll
                for (int k = 0; k < TILE_K; ++k) {
                    if (k < D) {
                        acc_O[i][k] += p_ij * __half2float(smem_V[j][k]);
                    }
                }
            }
            
            // Update running sum
            acc_l[i] = acc_l[i] * exp_diff_old + row_sum * exp_diff_new;
            acc_m[i] = new_m;
        }
        
        __syncthreads();  // Sync before next tile
    }
    
    // Final normalization and write output
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    
    // Each thread writes its rows
    for (int i = warp_id; i < WMMA_M; i += blockDim.x / 32) {
        const int global_m = tile_m_start + i;
        if (global_m >= S) continue;
        
        float inv_l = 1.0f / (acc_l[i] + 1e-6f);  // Avoid division by zero
        
        for (int k = lane_id; k < D; k += 32) {
            O[qo_offset + global_m * D + k] = __float2half(acc_O[i][k] * inv_l);
        }
    }
}

//------------------------------------------------------------------------------
// HOST LAUNCHER
//------------------------------------------------------------------------------
extern "C" void launch_attention_phase4_fused(
    const void* Q, const void* K, const void* V, void* O,
    int B, int H, int S, int D,
    float scale, bool is_causal, cudaStream_t stream
) {
    // Launch config
    const int num_tile_m = (S + TILE_M - 1) / TILE_M;
    const int total_blocks = B * H;
    
    dim3 grid(1, num_tile_m, total_blocks);  // (1, tiles_m, B*H)
    dim3 block(128);  // 128 threads = 4 warps
    
    std::cout << "[Fused Kernel] Launching: B=" << B << " H=" << H << " S=" << S << " D=" << D << "\n";
    std::cout << "[Fused Kernel] Grid: " << grid.x << "×" << grid.y << "×" << grid.z 
              << " = " << (grid.x * grid.y * grid.z) << " blocks\n";
    std::cout << "[Fused Kernel] Block: " << block.x << " threads (4 warps)\n";
    std::cout << "[Fused Kernel] KEY: NO cuBLASLt calls, NO S/P materialization!\n" << std::flush;
    
    flash_attention_fused_basic<<<grid, block, 0, stream>>>(
        static_cast<const __half*>(Q),
        static_cast<const __half*>(K),
        static_cast<const __half*>(V),
        static_cast<__half*>(O),
        B, H, S, D, scale
    );
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "[Fused Kernel] Launch error: " << cudaGetErrorString(err) << "\n";
    }
}


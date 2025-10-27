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

// Tile sizes (REDUCED to avoid register spill - Milestone 1 MVP!)
constexpr int TILE_M = 32;   // Query tile size (reduced from 64)
constexpr int TILE_N = 32;   // Key tile size (reduced from 64)
constexpr int TILE_K = 64;   // Head dimension (D=64, full dimension)
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
    
    // Shared memory for accumulation and softmax state
    __shared__ float smem_O[TILE_M][TILE_K + 8];  // Output accumulator
    __shared__ float smem_S[TILE_M][TILE_N];      // Attention scores (per tile)
    __shared__ float smem_m[TILE_M];              // Running max
    __shared__ float smem_l[TILE_M];              // Running sum
    
    const int tid = threadIdx.x;
    const int num_threads = blockDim.x;
    
    // Initialize accumulator and softmax state
    for (int idx = tid; idx < TILE_M * TILE_K; idx += num_threads) {
        int m = idx / TILE_K;
        int k = idx % TILE_K;
        smem_O[m][k] = 0.0f;
    }
    for (int m = tid; m < TILE_M; m += num_threads) {
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
    __syncthreads();
    
    // Iterate over K/V tiles (this is the FUSED loop!)
    const int num_tiles_n = (S + TILE_N - 1) / TILE_N;
    
    for (int tile_n_idx = 0; tile_n_idx < num_tiles_n; ++tile_n_idx) {
        const int tile_n_start = tile_n_idx * TILE_N;
        
        // Load K tile
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
        
        // Load V tile
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
        __syncthreads();
        
        // STEP 1: Compute S_tile = Q @ K^T (NAIVE - will optimize with WMMA later!)
        // Each thread computes part of S matrix
        for (int idx = tid; idx < TILE_M * TILE_N; idx += num_threads) {
            const int m = idx / TILE_N;
            const int n = idx % TILE_N;
            
            float sum = 0.0f;
            for (int k = 0; k < D; ++k) {
                sum += __half2float(smem_Q[m][k]) * __half2float(smem_K[n][k]);
            }
            smem_S[m][n] = sum * softmax_scale;
        }
        __syncthreads();
        
        // STEP 2: Fused Online Softmax + P@V (PER ROW)
        // Each warp processes multiple rows
        const int warp_id = tid / 32;
        const int lane_id = tid % 32;
        const int num_warps = num_threads / 32;
        
        for (int m = warp_id; m < TILE_M; m += num_warps) {
            // Find row max (warp reduction)
            float row_max = -INFINITY;
            for (int n = lane_id; n < TILE_N; n += 32) {
                row_max = fmaxf(row_max, smem_S[m][n]);
            }
            // Warp reduce max
            #pragma unroll
            for (int offset = 16; offset > 0; offset /= 2) {
                row_max = fmaxf(row_max, __shfl_down_sync(0xffffffff, row_max, offset));
            }
            row_max = __shfl_sync(0xffffffff, row_max, 0);  // Broadcast to all lanes
            
            // Update global max
            float old_m = smem_m[m];
            float new_m = fmaxf(old_m, row_max);
            float exp_diff_old = expf(old_m - new_m);
            
            // Rescale old O accumulator (each lane handles part of D)
            for (int k = lane_id; k < D; k += 32) {
                smem_O[m][k] *= exp_diff_old;
            }
            
            // Compute exp and sum (warp reduction)
            float row_sum = 0.0f;
            for (int n = lane_id; n < TILE_N; n += 32) {
                float p_val = expf(smem_S[m][n] - new_m);
                smem_S[m][n] = p_val;  // Store P for later
                row_sum += p_val;
            }
            // Warp reduce sum
            #pragma unroll
            for (int offset = 16; offset > 0; offset /= 2) {
                row_sum += __shfl_down_sync(0xffffffff, row_sum, offset);
            }
            row_sum = __shfl_sync(0xffffffff, row_sum, 0);  // Broadcast
            
            // STEP 3: Fused P @ V (accumulate to O)
            // Each lane accumulates part of the output dimension
            for (int k = lane_id; k < D; k += 32) {
                float acc = 0.0f;
                for (int n = 0; n < TILE_N; ++n) {
                    acc += smem_S[m][n] * __half2float(smem_V[n][k]);
                }
                smem_O[m][k] += acc;
            }
            
            // Update running sum
            if (lane_id == 0) {
                smem_l[m] = smem_l[m] * exp_diff_old + row_sum;
                smem_m[m] = new_m;
            }
        }
        __syncthreads();
    }
    
    // Final normalization and write output
    for (int idx = tid; idx < TILE_M * D; idx += num_threads) {
        const int m = idx / D;
        const int k = idx % D;
        const int global_m = tile_m_start + m;
        
        if (global_m < S && k < D) {
            float inv_l = 1.0f / (smem_l[m] + 1e-6f);
            O[qo_offset + global_m * D + k] = __float2half(smem_O[m][k] * inv_l);
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


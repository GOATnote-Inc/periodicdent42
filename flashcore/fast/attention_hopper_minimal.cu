// flashcore/fast/attention_hopper_minimal.cu
// Phase 1: MINIMAL working kernel (no warp-spec, no barriers, no TMA)
// Goal: Correctness first, then optimize
// Standing on giants: Start simple like FA2/FA3 did

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>

namespace flashcore {
namespace hopper {

// Minimal config for Phase 1
constexpr int BLOCK_M = 64;
constexpr int BLOCK_N = 64;
constexpr int HEAD_DIM_MAX = 128;  // Test with hdim=128

// Online softmax state (register-resident, per-thread per-row)
struct SoftmaxState {
    float m;  // running max
    float l;  // running sum
    
    __device__ __forceinline__ SoftmaxState() : m(-INFINITY), l(0.0f) {}
    
    __device__ __forceinline__ void update(float new_max, float new_sum) {
        float old_m = m;
        m = fmaxf(m, new_max);
        float rescale = expf(old_m - m);
        l = l * rescale + new_sum;
    }
};

//==============================================================================
// PHASE 1: MINIMAL KERNEL (All warps do same work, simple __syncthreads)
//==============================================================================

__global__ void __launch_bounds__(256)
attention_hopper_minimal(
    const __half* __restrict__ Q,
    const __half* __restrict__ K,
    const __half* __restrict__ V,
    __half* __restrict__ O,
    int B, int H, int S, int D,
    float scale,
    bool is_causal
) {
    // Thread/block IDs
    const int tid = threadIdx.x;
    const int bh = blockIdx.x;
    const int b = bh / H;
    const int h = bh % H;
    const int tile_m = blockIdx.y;
    
    // Shared memory (double-buffered K/V for next iteration)
    __shared__ __half Q_smem[BLOCK_M * HEAD_DIM_MAX];
    __shared__ __half K_smem[BLOCK_N * HEAD_DIM_MAX];
    __shared__ __half V_smem[BLOCK_N * HEAD_DIM_MAX];
    __shared__ float QK_smem[BLOCK_M * BLOCK_N];
    
    // Per-thread state (minimize stack usage!)
    const int num_threads = blockDim.x;
    SoftmaxState my_softmax_state;
    float my_output[HEAD_DIM_MAX];  // Will optimize to registers
    
    // Initialize output
    #pragma unroll
    for (int d = 0; d < HEAD_DIM_MAX; ++d) {
        my_output[d] = 0.0f;
    }
    
    //==========================================================================
    // LOAD Q TILE (cooperative across all threads)
    //==========================================================================
    for (int idx = tid; idx < BLOCK_M * D; idx += num_threads) {
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
    // MAIN LOOP: Iterate over K/V tiles
    //==========================================================================
    const int num_tiles_n = (S + BLOCK_N - 1) / BLOCK_N;
    
    for (int tile_n = 0; tile_n < num_tiles_n; ++tile_n) {
        
        //======================================================================
        // LOAD K/V TILES
        //======================================================================
        for (int idx = tid; idx < BLOCK_N * D; idx += num_threads) {
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
        // COMPUTE Q @ K^T (each thread handles few rows)
        //======================================================================
        for (int m = tid; m < BLOCK_M; m += num_threads) {
            int global_m = tile_m * BLOCK_M + m;
            if (global_m >= S) continue;
            
            for (int n = 0; n < BLOCK_N; ++n) {
                int global_n = tile_n * BLOCK_N + n;
                
                // Dot product Q[m,:] · K[n,:]
                float dot = 0.0f;
                #pragma unroll 8
                for (int d = 0; d < D; ++d) {
                    float q_val = __half2float(Q_smem[m * D + d]);
                    float k_val = __half2float(K_smem[n * D + d]);
                    dot += q_val * k_val;
                }
                
                dot *= scale;
                
                // Causal mask
                if (is_causal && global_m < global_n) {
                    dot = -INFINITY;
                }
                
                QK_smem[m * BLOCK_N + n] = dot;
            }
        }
        __syncthreads();
        
        //======================================================================
        // ONLINE SOFTMAX + P @ V (each thread handles few rows)
        //======================================================================
        for (int m = tid; m < BLOCK_M; m += num_threads) {
            int global_m = tile_m * BLOCK_M + m;
            if (global_m >= S) continue;
            
            // Find max for this tile
            float tile_max = -INFINITY;
            for (int n = 0; n < BLOCK_N; ++n) {
                tile_max = fmaxf(tile_max, QK_smem[m * BLOCK_N + n]);
            }
            
            // Compute exp and sum
            float tile_sum = 0.0f;
            for (int n = 0; n < BLOCK_N; ++n) {
                float p = expf(QK_smem[m * BLOCK_N + n] - tile_max);
                QK_smem[m * BLOCK_N + n] = p;  // Store P for P@V
                tile_sum += p;
            }
            
            // Update online softmax state
            float old_max = my_softmax_state.m;
            my_softmax_state.update(tile_max, tile_sum);
            
            // Rescale previous output accumulator
            float rescale = expf(old_max - my_softmax_state.m);
            #pragma unroll
            for (int d = 0; d < D; ++d) {
                my_output[d] *= rescale;
            }
            
            // Accumulate P @ V
            for (int d = 0; d < D; ++d) {
                float pv = 0.0f;
                #pragma unroll 4
                for (int n = 0; n < BLOCK_N; ++n) {
                    float p_val = QK_smem[m * BLOCK_N + n];
                    float v_val = __half2float(V_smem[n * D + d]);
                    pv += p_val * v_val;
                }
                my_output[d] += pv;
            }
        }
        __syncthreads();
    }
    
    //==========================================================================
    // WRITE OUTPUT (final normalization)
    //==========================================================================
    for (int m = tid; m < BLOCK_M; m += num_threads) {
        int global_m = tile_m * BLOCK_M + m;
        if (global_m >= S) continue;
        
        for (int d = 0; d < D; ++d) {
            float normalized = my_output[d] / my_softmax_state.l;
            int gmem_idx = (b * H + h) * S * D + global_m * D + d;
            O[gmem_idx] = __float2half(normalized);
        }
    }
}

//==============================================================================
// HOST API
//==============================================================================

extern "C" {

void launch_attention_hopper_minimal(
    const void* Q, const void* K, const void* V, void* O,
    int B, int H, int S, int D,
    float scale, bool is_causal,
    cudaStream_t stream
) {
    dim3 grid(B * H, (S + BLOCK_M - 1) / BLOCK_M);
    dim3 block(256);
    
    attention_hopper_minimal<<<grid, block, 0, stream>>>(
        (const __half*)Q, (const __half*)K, (const __half*)V, (__half*)O,
        B, H, S, D, scale, is_causal
    );
}

} // extern "C"

} // namespace hopper
} // namespace flashcore


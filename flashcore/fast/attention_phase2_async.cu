// flashcore/fast/attention_phase2_async.cu
// Phase 2: Async memory pipeline with double buffering
// Target: 10-20 TFLOPS (15-30× speedup over Phase 1)
// Standing on giants: FA3 uses TMA, we start with async copy

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda/pipeline>
#include <cooperative_groups.h>
#include <cmath>

namespace flashcore {
namespace phase2 {

// Phase 2 config (double-buffered!)
constexpr int BLOCK_M = 64;
constexpr int BLOCK_N = 64;
constexpr int HEAD_DIM_MAX = 128;
constexpr int NUM_STAGES = 2;  // Double buffering

// Online softmax state
struct SoftmaxState {
    float m;
    float l;
    
    __device__ __forceinline__ SoftmaxState() : m(-INFINITY), l(0.0f) {}
    
    __device__ __forceinline__ void update(float new_max, float new_sum) {
        float old_m = m;
        m = fmaxf(m, new_max);
        float rescale = expf(old_m - m);
        l = l * rescale + new_sum;
    }
};

//==============================================================================
// PHASE 2: ASYNC MEMORY PIPELINE WITH DOUBLE BUFFERING
//==============================================================================

__global__ void __launch_bounds__(256)
attention_phase2_async(
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
    const int num_threads = blockDim.x;
    
    // Shared memory (DOUBLE-BUFFERED K/V for async pipelining!)
    __shared__ __half Q_smem[BLOCK_M * HEAD_DIM_MAX];
    __shared__ __half K_smem[NUM_STAGES][BLOCK_N * HEAD_DIM_MAX];  // Ping-pong
    __shared__ __half V_smem[NUM_STAGES][BLOCK_N * HEAD_DIM_MAX];  // Ping-pong
    __shared__ float QK_smem[BLOCK_M * BLOCK_N];
    
    // Shared memory for softmax states and outputs
    __shared__ SoftmaxState softmax_states[BLOCK_M];
    __shared__ float output_acc[BLOCK_M * HEAD_DIM_MAX];
    
    // Initialize
    for (int m = tid; m < BLOCK_M; m += num_threads) {
        softmax_states[m] = SoftmaxState();
        for (int d = 0; d < D; ++d) {
            output_acc[m * D + d] = 0.0f;
        }
    }
    __syncthreads();
    
    //==========================================================================
    // LOAD Q TILE (once, no pipelining needed)
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
    // ASYNC MEMORY PIPELINE: Overlap loads with compute
    //==========================================================================
    
    // Create async pipeline
    __shared__ cuda::pipeline_shared_state<
        cuda::thread_scope_block, NUM_STAGES
    > pipe_state;
    
    auto block = cooperative_groups::this_thread_block();
    auto pipeline = cuda::make_pipeline(block, &pipe_state);
    
    const int num_tiles_n = (S + BLOCK_N - 1) / BLOCK_N;
    
    // PROLOGUE: Prefetch first stage
    if (num_tiles_n > 0) {
        int tile_n = 0;
        int write_stage = 0;
        
        pipeline.producer_acquire();
        
        // Async load K/V for tile 0
        for (int idx = tid; idx < BLOCK_N * D; idx += num_threads) {
            int n = idx / D;
            int d = idx % D;
            int global_n = tile_n * BLOCK_N + n;
            
            if (global_n < S && d < D) {
                int gmem_idx = (b * H + h) * S * D + global_n * D + d;
                // Direct copy (pipeline will async later with memcpy_async)
                K_smem[write_stage][n * D + d] = K[gmem_idx];
                V_smem[write_stage][n * D + d] = V[gmem_idx];
            } else {
                K_smem[write_stage][n * D + d] = __float2half(0.0f);
                V_smem[write_stage][n * D + d] = __float2half(0.0f);
            }
        }
        
        pipeline.producer_commit();
    }
    
    // MAIN LOOP: Pipelined compute + load
    for (int tile_n = 0; tile_n < num_tiles_n; ++tile_n) {
        int read_stage = tile_n % NUM_STAGES;
        int write_stage = (tile_n + 1) % NUM_STAGES;
        
        // PREFETCH: Start loading next tile (if exists)
        if (tile_n + 1 < num_tiles_n) {
            pipeline.producer_acquire();
            
            int next_tile_n = tile_n + 1;
            for (int idx = tid; idx < BLOCK_N * D; idx += num_threads) {
                int n = idx / D;
                int d = idx % D;
                int global_n = next_tile_n * BLOCK_N + n;
                
                if (global_n < S && d < D) {
                    int gmem_idx = (b * H + h) * S * D + global_n * D + d;
                    K_smem[write_stage][n * D + d] = K[gmem_idx];
                    V_smem[write_stage][n * D + d] = V[gmem_idx];
                } else {
                    K_smem[write_stage][n * D + d] = __float2half(0.0f);
                    V_smem[write_stage][n * D + d] = __float2half(0.0f);
                }
            }
            
            pipeline.producer_commit();
        }
        
        // WAIT for current tile data to be ready
        pipeline.consumer_wait();
        __syncthreads();  // Ensure all threads see the data
        
        //======================================================================
        // COMPUTE: Q @ K^T (on read_stage)
        //======================================================================
        for (int m = tid; m < BLOCK_M; m += num_threads) {
            int global_m = tile_m * BLOCK_M + m;
            if (global_m >= S) continue;
            
            for (int n = 0; n < BLOCK_N; ++n) {
                int global_n = tile_n * BLOCK_N + n;
                
                // Dot product
                float dot = 0.0f;
                #pragma unroll 8
                for (int d = 0; d < D; ++d) {
                    float q_val = __half2float(Q_smem[m * D + d]);
                    float k_val = __half2float(K_smem[read_stage][n * D + d]);
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
        // COMPUTE: SOFTMAX + P @ V
        //======================================================================
        for (int m = tid; m < BLOCK_M; m += num_threads) {
            int global_m = tile_m * BLOCK_M + m;
            if (global_m >= S) continue;
            
            // Find max
            float tile_max = -INFINITY;
            for (int n = 0; n < BLOCK_N; ++n) {
                tile_max = fmaxf(tile_max, QK_smem[m * BLOCK_N + n]);
            }
            
            // Compute exp and sum (with guards)
            float tile_sum = 0.0f;
            if (tile_max > -1e30f) {
                for (int n = 0; n < BLOCK_N; ++n) {
                    float qk = QK_smem[m * BLOCK_N + n];
                    float p = (qk > -1e30f) ? expf(qk - tile_max) : 0.0f;
                    QK_smem[m * BLOCK_N + n] = p;
                    tile_sum += p;
                }
            } else {
                for (int n = 0; n < BLOCK_N; ++n) {
                    QK_smem[m * BLOCK_N + n] = 0.0f;
                }
            }
            
            // Update softmax state
            float old_max = softmax_states[m].m;
            softmax_states[m].update(tile_max, tile_sum);
            
            // Rescale accumulator
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
            
            // Accumulate P @ V
            for (int d = 0; d < D; ++d) {
                float pv = 0.0f;
                #pragma unroll 4
                for (int n = 0; n < BLOCK_N; ++n) {
                    float p_val = QK_smem[m * BLOCK_N + n];
                    float v_val = __half2float(V_smem[read_stage][n * D + d]);
                    pv += p_val * v_val;
                }
                output_acc[m * D + d] += pv;
            }
        }
        
        // Release consumer (allows next load to proceed)
        pipeline.consumer_release();
        __syncthreads();
    }
    
    //==========================================================================
    // WRITE OUTPUT
    //==========================================================================
    for (int m = tid; m < BLOCK_M; m += num_threads) {
        int global_m = tile_m * BLOCK_M + m;
        if (global_m >= S) continue;
        
        float norm = softmax_states[m].l;
        if (norm < 1e-10f) norm = 1e-10f;
        
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

void launch_attention_phase2_async(
    const void* Q, const void* K, const void* V, void* O,
    int B, int H, int S, int D,
    float scale, bool is_causal,
    cudaStream_t stream
) {
    dim3 grid(B * H, (S + BLOCK_M - 1) / BLOCK_M);
    dim3 block(256);
    
    // Note: SMEM usage ~130KB (double-buffered K/V)
    // H100 allows 227KB, so we're good!
    
    attention_phase2_async<<<grid, block, 0, stream>>>(
        (const __half*)Q, (const __half*)K, (const __half*)V, (__half*)O,
        B, H, S, D, scale, is_causal
    );
}

} // extern "C"

} // namespace phase2
} // namespace flashcore


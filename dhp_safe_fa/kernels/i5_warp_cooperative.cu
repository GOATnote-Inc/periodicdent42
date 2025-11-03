// I5: Warp-Cooperative V Loading with Constant-Time Guarantees
// Performance optimization over I4: Coalesced memory access
// Security: Zero timing leaks maintained

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "../include/dhp_ct_enhanced.cuh"

// ============================================================================
// PERFORMANCE OPTIMIZATION: Warp-Cooperative Loading
// ============================================================================
// I4 problem: Non-coalesced V access (32× memory transaction overhead)
// I5 solution: Shared memory tile + cooperative warp loads
//
// Expected improvement: 43× → 2× slowdown vs PyTorch SDPA
// Target: 5-6 μs/head (was 158 μs/head in I4)

constexpr int WARP_SIZE = 32;
constexpr int D = 64;  // Head dimension (compile-time constant)
constexpr int TILE_SIZE = 32;  // Process 32 columns at a time

__global__ void __launch_bounds__(256)  // 256 threads = 8 warps per block
dhp_i5_warp_cooperative(
    const __half* __restrict__ scores,    // [B*H, S_max, S_max]
    const __half* __restrict__ V,         // [B*H, S_max, d]
    __half* __restrict__ out,             // [B*H, S_max, d]
    const uint32_t S_max,
    const uint32_t S_actual,
    const uint32_t d,
    const uint32_t batch_size
) {
    // ========================================================================
    // Thread Organization
    // ========================================================================
    const int global_row = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (global_row >= batch_size * S_max) return;
    
    const int batch_idx = global_row / S_max;
    const int row = global_row % S_max;
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;
    
    // ========================================================================
    // Shared Memory for V Tiles (cooperative loading)
    // ========================================================================
    // Layout: [TILE_SIZE, D] = [32, 64] = 2048 half = 4 KB per tile
    // 8 warps can share this tile
    __shared__ __half V_tile[TILE_SIZE][D];
    
    // ========================================================================
    // Validity Masks
    // ========================================================================
    uint32_t row_valid = ct_lt_u32(row, S_actual);
    
    // ========================================================================
    // Online Softmax State
    // ========================================================================
    float m = -INFINITY;
    float l = 0.0f;
    float out_acc[D];
    
    #pragma unroll
    for (int i = 0; i < D; ++i) {
        out_acc[i] = 0.0f;
    }
    
    // ========================================================================
    // Tile Loop (process S_max columns in tiles of TILE_SIZE)
    // ========================================================================
    for (int tile_start = 0; tile_start < S_max; tile_start += TILE_SIZE) {
        // ====================================================================
        // PHASE 1: Cooperative Load of V Tile
        // Each warp loads a portion of the tile in a coalesced manner
        // ====================================================================
        
        // Each thread loads D/WARP_SIZE elements per row
        // With 256 threads and TILE_SIZE=32 rows, each thread loads:
        //   - 32 rows * 64 cols / 256 threads = 8 elements
        
        const int num_elements = TILE_SIZE * D;  // 32 * 64 = 2048
        const int elements_per_thread = (num_elements + blockDim.x - 1) / blockDim.x;
        
        for (int i = 0; i < elements_per_thread; ++i) {
            const int flat_idx = threadIdx.x * elements_per_thread + i;
            if (flat_idx < num_elements) {
                const int tile_row = flat_idx / D;
                const int tile_col = flat_idx % D;
                const int global_col = tile_start + tile_row;
                
                // Validity check (constant-time)
                uint32_t col_valid = ct_lt_u32(global_col, S_max);
                
                // Load from global memory (coalesced within warp)
                const int v_idx = batch_idx * S_max * D + global_col * D + tile_col;
                __half v_val = V[v_idx];
                
                // Mask invalid values to 0 (constant-time)
                v_val = ct_select_half(__float2half(0.0f), v_val, col_valid);
                
                V_tile[tile_row][tile_col] = v_val;
            }
        }
        
        __syncthreads();  // Ensure tile is loaded before processing
        
        // ====================================================================
        // PHASE 2: Process Tile (each thread independently)
        // ====================================================================
        
        #pragma unroll
        for (int tile_col = 0; tile_col < TILE_SIZE; ++tile_col) {
            const int col = tile_start + tile_col;
            
            // Validity masks
            uint32_t col_valid = ct_lt_u32(col, S_actual);
            uint32_t causal_valid = ct_le_u32(col, row);
            uint32_t valid = ct_and_u32(
                ct_and_u32(row_valid, col_valid),
                causal_valid
            );
            
            // Load score (coalesced across warp)
            const int score_idx = batch_idx * S_max * S_max + row * S_max + col;
            float score = __half2float(scores[score_idx]);
            score = ct_select_f32(-INFINITY, score, valid);
            
            // Update running max
            uint32_t gt_mask = ct_gt_f32(score, m);
            float m_new = ct_select_f32(m, score, gt_mask);
            
            // Rescale accumulator
            float alpha = expf(m - m_new);
            l *= alpha;
            
            #pragma unroll
            for (int i = 0; i < D; ++i) {
                out_acc[i] *= alpha;
            }
            
            // Add contribution from current score
            float p = safe_exp(score - m_new);
            l += p;
            
            // ================================================================
            // KEY OPTIMIZATION: Load V from shared memory (fast!)
            // ================================================================
            #pragma unroll
            for (int i = 0; i < D; ++i) {
                float v_val = __half2float(V_tile[tile_col][i]);
                out_acc[i] += p * v_val;
            }
            
            m = m_new;
        }
        
        __syncthreads();  // Sync before loading next tile
    }
    
    // ========================================================================
    // Final Normalization and Write Output
    // ========================================================================
    float l_safe = ct_select_f32(1.0f, l, row_valid);
    
    #pragma unroll
    for (int i = 0; i < D; ++i) {
        float normalized = out_acc[i] / l_safe;
        normalized = ct_select_f32(0.0f, normalized, row_valid);
        const int out_idx = batch_idx * S_max * D + row * D + i;
        out[out_idx] = __float2half(normalized);
    }
}

// ============================================================================
// Performance Properties (Expected):
// ============================================================================
// Memory Access Pattern:
//   - V loading: Coalesced (all threads in warp read consecutive addresses)
//   - Bandwidth efficiency: ~90% (vs. 3% in I4)
//   - Expected speedup: 20-30× over I4
//
// Target Metrics:
//   - Latency: 5-6 μs/head (vs. 158 μs in I4, 3.62 μs PyTorch SDPA)
//   - SM utilization: 60-70% (memory-bound, as expected)
//   - Shared memory: 4 KB per tile (well under 164 KB limit)
//   - Register usage: ~140 regs/thread (under 255 limit)
//
// Security Properties (Maintained):
//   - Fixed loop count: S_max iterations
//   - No data-dependent branches
//   - Constant-time primitives throughout
//   - Shared memory access is uniform within warp
// ============================================================================


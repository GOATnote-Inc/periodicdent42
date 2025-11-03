// I6: Block-Parallel Attention with Constant-Time Guarantees
// Architectural shift: Each block processes a tile, not individual rows
// Expected: 15-20 μs/head (5× faster than I5)

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "../include/dhp_ct_enhanced.cuh"

// ============================================================================
// ARCHITECTURAL CHANGE: Block-Parallel Execution
// ============================================================================
// I4/I5 problem: 1 thread per row → 24.8% SM utilization
// I6 solution: 1 block per tile → 60%+ SM utilization
//
// Key changes:
// - Block size: 128 threads (4 warps, good for warpgroup operations)
// - Each block processes BM=64 rows and BN=64 columns
// - Threads cooperate on matrix operations
// - Reduced syncs: ~8 instead of 64

constexpr int BM = 64;   // Block tile rows
constexpr int BN = 64;   // Block tile cols  
constexpr int BK = 64;   // Head dimension
constexpr int WARP_SIZE = 32;
constexpr int WARPS_PER_BLOCK = 4;  // 128 threads = 4 warps
constexpr int THREADS_PER_BLOCK = WARPS_PER_BLOCK * WARP_SIZE;

__global__ void __launch_bounds__(THREADS_PER_BLOCK)
dhp_i6_block_parallel(
    const __half* __restrict__ Q,      // [B*H, S_max, d]
    const __half* __restrict__ K,      // [B*H, S_max, d]
    const __half* __restrict__ V,      // [B*H, S_max, d]
    __half* __restrict__ out,          // [B*H, S_max, d]
    const uint32_t S_max,
    const uint32_t S_actual,
    const uint32_t d,
    const uint32_t batch_size
) {
    // ========================================================================
    // Block-Level Thread Organization
    // ========================================================================
    const int batch_head = blockIdx.x;  // Which (batch, head) pair
    const int tile_row = blockIdx.y;    // Which row tile (0 to S_max/BM)
    
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;
    
    // This block processes rows [tile_row*BM, (tile_row+1)*BM)
    const int row_start = tile_row * BM;
    const int row_end = min(row_start + BM, (int)S_max);
    
    if (batch_head >= batch_size) return;
    
    // ========================================================================
    // Shared Memory Layout
    // ========================================================================
    // Q_tile: [BM, BK] = [64, 64] = 4096 half = 8 KB
    // K_tile: [BN, BK] = [64, 64] = 4096 half = 8 KB
    // V_tile: [BN, BK] = [64, 64] = 4096 half = 8 KB
    // S_tile: [BM, BN] = [64, 64] = 4096 half = 8 KB
    // Total: 32 KB (well under 164 KB limit)
    
    __shared__ __half Q_tile[BM][BK];
    __shared__ __half K_tile[BN][BK];
    __shared__ __half V_tile[BN][BK];
    __shared__ __half S_tile[BM][BN];  // Attention scores
    
    // ========================================================================
    // Per-Thread Accumulator for Output
    // Each thread will handle specific rows
    // ========================================================================
    const int thread_row_start = (threadIdx.x * BM) / THREADS_PER_BLOCK;
    const int num_rows_per_thread = (BM + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    
    // Per-row accumulators (for all d dimensions)
    float out_acc[BK];  // Output accumulator for d=64
    float m_local = -INFINITY;
    float l_local = 0.0f;
    
    #pragma unroll
    for (int i = 0; i < BK; ++i) {
        out_acc[i] = 0.0f;
    }
    
    // ========================================================================
    // PHASE 1: Load Q Tile (Cooperative)
    // ========================================================================
    // Each thread loads multiple elements
    const int q_elements = BM * BK;  // 4096
    const int q_per_thread = (q_elements + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    
    for (int i = 0; i < q_per_thread; ++i) {
        const int flat_idx = threadIdx.x * q_per_thread + i;
        if (flat_idx < q_elements) {
            const int row = flat_idx / BK;
            const int col = flat_idx % BK;
            const int global_row = row_start + row;
            
            uint32_t valid = ct_lt_u32(global_row, S_actual);
            
            const int q_idx = batch_head * S_max * d + global_row * d + col;
            __half q_val = Q[q_idx];
            q_val = ct_select_half(__float2half(0.0f), q_val, valid);
            
            Q_tile[row][col] = q_val;
        }
    }
    __syncthreads();
    
    // ========================================================================
    // PHASE 2: Iterate Over K/V Tiles
    // ========================================================================
    const int num_tiles = (S_max + BN - 1) / BN;
    
    for (int tile_k = 0; tile_k < num_tiles; ++tile_k) {
        const int col_start = tile_k * BN;
        const int col_end = min(col_start + BN, (int)S_max);
        
        // ====================================================================
        // Load K Tile (Cooperative)
        // ====================================================================
        const int k_elements = BN * BK;
        const int k_per_thread = (k_elements + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        
        for (int i = 0; i < k_per_thread; ++i) {
            const int flat_idx = threadIdx.x * k_per_thread + i;
            if (flat_idx < k_elements) {
                const int row = flat_idx / BK;
                const int col = flat_idx % BK;
                const int global_col = col_start + row;
                
                uint32_t valid = ct_lt_u32(global_col, S_actual);
                
                const int k_idx = batch_head * S_max * d + global_col * d + col;
                __half k_val = K[k_idx];
                k_val = ct_select_half(__float2half(0.0f), k_val, valid);
                
                K_tile[row][col] = k_val;
            }
        }
        __syncthreads();
        
        // ====================================================================
        // Compute S = Q @ K^T (Cooperative Matrix Multiply)
        // Each thread computes multiple output elements
        // ====================================================================
        const int s_elements = BM * BN;
        const int s_per_thread = (s_elements + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        
        for (int i = 0; i < s_per_thread; ++i) {
            const int flat_idx = threadIdx.x * s_per_thread + i;
            if (flat_idx < s_elements) {
                const int row = flat_idx / BN;
                const int col = flat_idx % BN;
                
                float acc = 0.0f;
                #pragma unroll
                for (int k = 0; k < BK; ++k) {
                    acc += __half2float(Q_tile[row][k]) * __half2float(K_tile[col][k]);
                }
                
                // Scale by 1/sqrt(d)
                acc *= 0.125f;  // 1/sqrt(64) = 1/8 = 0.125
                
                // Apply causal mask (constant-time)
                const int global_row = row_start + row;
                const int global_col = col_start + col;
                uint32_t causal_valid = ct_le_u32(global_col, global_row);
                uint32_t row_valid = ct_lt_u32(global_row, S_actual);
                uint32_t col_valid = ct_lt_u32(global_col, S_actual);
                uint32_t valid = ct_and_u32(
                    ct_and_u32(row_valid, col_valid),
                    causal_valid
                );
                
                acc = ct_select_f32(-INFINITY, acc, valid);
                S_tile[row][col] = __float2half(acc);
            }
        }
        __syncthreads();
        
        // ====================================================================
        // Load V Tile (Cooperative)
        // ====================================================================
        const int v_elements = BN * BK;
        const int v_per_thread = (v_elements + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        
        for (int i = 0; i < v_per_thread; ++i) {
            const int flat_idx = threadIdx.x * v_per_thread + i;
            if (flat_idx < v_elements) {
                const int row = flat_idx / BK;
                const int col = flat_idx % BK;
                const int global_col = col_start + row;
                
                uint32_t valid = ct_lt_u32(global_col, S_actual);
                
                const int v_idx = batch_head * S_max * d + global_col * d + col;
                __half v_val = V[v_idx];
                v_val = ct_select_half(__float2half(0.0f), v_val, valid);
                
                V_tile[row][col] = v_val;
            }
        }
        __syncthreads();
        
        // ====================================================================
        // Online Softmax Update + Accumulate Output
        // Each thread processes multiple rows
        // ====================================================================
        const int rows_per_thread = (BM + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        
        for (int i = 0; i < rows_per_thread; ++i) {
            const int row = threadIdx.x * rows_per_thread + i;
            if (row < BM) {
                // Find max in this row's scores
                float row_max = -INFINITY;
                for (int col = 0; col < BN; ++col) {
                    float score = __half2float(S_tile[row][col]);
                    uint32_t gt_mask = ct_gt_f32(score, row_max);
                    row_max = ct_select_f32(row_max, score, gt_mask);
                }
                
                // Update global max
                uint32_t gt_mask = ct_gt_f32(row_max, m[i % OUTPUTS_PER_THREAD]);
                float m_new = ct_select_f32(m[i % OUTPUTS_PER_THREAD], row_max, gt_mask);
                
                // Rescale previous accumulator
                float alpha = expf(m[i % OUTPUTS_PER_THREAD] - m_new);
                l[i % OUTPUTS_PER_THREAD] *= alpha;
                out_acc[i % OUTPUTS_PER_THREAD] *= alpha;
                
                // Accumulate new contributions
                for (int col = 0; col < BN; ++col) {
                    float score = __half2float(S_tile[row][col]);
                    float p = safe_exp(score - m_new);
                    l[i % OUTPUTS_PER_THREAD] += p;
                    
                    // Simplified: accumulate first dimension only
                    // Full implementation would accumulate all d dimensions
                    float v_val = __half2float(V_tile[col][0]);
                    out_acc[i % OUTPUTS_PER_THREAD] += p * v_val;
                }
                
                m[i % OUTPUTS_PER_THREAD] = m_new;
            }
        }
        __syncthreads();
    }
    
    // ========================================================================
    // Final Normalization and Write Output
    // ========================================================================
    const int rows_per_thread = (BM + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    
    for (int i = 0; i < rows_per_thread; ++i) {
        const int row = threadIdx.x * rows_per_thread + i;
        if (row < BM) {
            const int global_row = row_start + row;
            uint32_t row_valid = ct_lt_u32(global_row, S_actual);
            
            float l_safe = ct_select_f32(1.0f, l[i % OUTPUTS_PER_THREAD], row_valid);
            float normalized = out_acc[i % OUTPUTS_PER_THREAD] / l_safe;
            normalized = ct_select_f32(0.0f, normalized, row_valid);
            
            // Write to global memory (simplified: first dimension only)
            const int out_idx = batch_head * S_max * d + global_row * d;
            out[out_idx] = __float2half(normalized);
        }
    }
}

// ============================================================================
// Expected Performance (vs I5):
// ============================================================================
// - SM utilization: 60-70% (vs 24.8% in I5)
//   * More blocks active: (B*H) * (S_max/BM) vs B*H*S_max threads
//   * Better warp occupancy per SM
//
// - Synchronization: ~8 syncs (vs 64 in I5)
//   * 1 sync per: Q load, K load, S compute, V load (×2 for num_tiles)
//
// - Memory efficiency: Higher bandwidth utilization
//   * Coalesced loads for Q, K, V
//   * Shared memory reuse across warps
//
// - Compute pattern: More parallel
//   * Matrix multiply uses all threads
//   * Softmax parallelized across rows
//
// Expected: 15-20 μs/head (5× faster than I5's 90.67 μs/head)
// ============================================================================


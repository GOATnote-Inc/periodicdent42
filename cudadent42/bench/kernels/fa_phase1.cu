// ============================================================================
// PHASE 1: BLOCK TILING OPTIMIZATION
// ============================================================================
// Target: 6× speedup (2870 μs → 480 μs)
// 
// Key Optimizations:
// 1. BLOCK_M=16: Each block handles 16 query rows (vs 1 in baseline)
// 2. Better thread utilization: 128 threads handle 16×64 work
// 3. Improved SMEM layout: Q, S, O in shared memory
//
// Correctness: MUST pass torch.allclose(atol=1e-3) before Phase 2
// ============================================================================

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <float.h>

constexpr int HEAD_DIM = 64;
constexpr int BLOCK_M = 16;      // Query rows per block (optimization!)
constexpr int BLOCK_N = 64;      // KV tile size
constexpr int THREADS = 128;
constexpr int WARPS = THREADS / 32;  // 4 warps

// ============================================================================
// PHASE 1 KERNEL: Block Tiling
// ============================================================================

__global__ void flash_attention_phase1_kernel(
    const half* __restrict__ Q,
    const half* __restrict__ K,
    const half* __restrict__ V,
    half* __restrict__ O,
    float softmax_scale,
    int batch_size,
    int num_heads,
    int seq_len
) {
    // Grid: (num_blocks_m, num_heads, batch_size)
    const int batch_idx = blockIdx.z;
    const int head_idx = blockIdx.y;
    const int m_block = blockIdx.x;
    
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    
    const int q_start = m_block * BLOCK_M;
    const int q_end = min(q_start + BLOCK_M, seq_len);
    const int rows_this_block = q_end - q_start;
    
    if (rows_this_block == 0) return;
    
    // Shared memory layout
    __shared__ float Q_tile[BLOCK_M][HEAD_DIM];      // 16×64 = 4KB
    __shared__ float S_tile[BLOCK_M][BLOCK_N];       // 16×64 = 4KB
    __shared__ float O_accum[BLOCK_M][HEAD_DIM];     // 16×64 = 4KB
    // Total: 12KB (well within 48KB limit)
    
    // Load Q tile (cooperative load across all threads)
    for (int row = 0; row < rows_this_block; row++) {
        for (int d = tid; d < HEAD_DIM; d += THREADS) {
            const int q_idx = q_start + row;
            const int q_offset = batch_idx * num_heads * seq_len * HEAD_DIM +
                                head_idx * seq_len * HEAD_DIM +
                                q_idx * HEAD_DIM + d;
            Q_tile[row][d] = __half2float(Q[q_offset]);
        }
    }
    
    // Initialize O_accum and softmax state
    for (int row = 0; row < rows_this_block; row++) {
        for (int d = tid; d < HEAD_DIM; d += THREADS) {
            O_accum[row][d] = 0.0f;
        }
    }
    __syncthreads();
    
    // Per-row softmax state (in registers)
    float m_i[BLOCK_M];
    float l_i[BLOCK_M];
    for (int row = 0; row < BLOCK_M; row++) {
        m_i[row] = -FLT_MAX;
        l_i[row] = 0.0f;
    }
    
    const int num_blocks_n = (seq_len + BLOCK_N - 1) / BLOCK_N;
    
    // Loop over KV tiles
    for (int n_block = 0; n_block < num_blocks_n; n_block++) {
        const int kv_start = n_block * BLOCK_N;
        const int kv_end = min(kv_start + BLOCK_N, seq_len);
        const int kv_size = kv_end - kv_start;
        
        // Compute S = Q @ K^T (collaborative across threads)
        for (int row = 0; row < rows_this_block; row++) {
            for (int col = tid; col < kv_size; col += THREADS) {
                const int kv_idx = kv_start + col;
                const int k_offset = batch_idx * num_heads * seq_len * HEAD_DIM +
                                    head_idx * seq_len * HEAD_DIM +
                                    kv_idx * HEAD_DIM;
                
                float score = 0.0f;
                for (int d = 0; d < HEAD_DIM; d++) {
                    score += Q_tile[row][d] * __half2float(K[k_offset + d]);
                }
                S_tile[row][col] = score * softmax_scale;
            }
        }
        __syncthreads();
        
        // Online softmax update (per row)
        for (int row = 0; row < rows_this_block; row++) {
            // Find new max (thread-local, then reduce)
            float m_new = m_i[row];
            for (int col = tid; col < kv_size; col += THREADS) {
                m_new = fmaxf(m_new, S_tile[row][col]);
            }
            
            // Warp reduce for max
            for (int offset = 16; offset > 0; offset /= 2) {
                m_new = fmaxf(m_new, __shfl_down_sync(0xffffffff, m_new, offset));
            }
            
            // Broadcast max within block
            __shared__ float m_new_shared;
            if (lane_id == 0) m_new_shared = m_new;
            __syncthreads();
            m_new = m_new_shared;
            
            // Correction factor
            float correction = expf(m_i[row] - m_new);
            
            // Apply correction to O_accum
            for (int d = tid; d < HEAD_DIM; d += THREADS) {
                O_accum[row][d] *= correction;
            }
            
            // Compute new l_i
            float l_new = l_i[row] * correction;
            float sum_exp = 0.0f;
            for (int col = tid; col < kv_size; col += THREADS) {
                sum_exp += expf(S_tile[row][col] - m_new);
            }
            
            // Warp reduce for sum
            for (int offset = 16; offset > 0; offset /= 2) {
                sum_exp += __shfl_down_sync(0xffffffff, sum_exp, offset);
            }
            
            // Broadcast sum
            __shared__ float sum_shared;
            if (lane_id == 0) sum_shared = sum_exp;
            __syncthreads();
            sum_exp = sum_shared;
            
            l_new += sum_exp;
            
            __syncthreads();
            
            // Accumulate O += P @ V
            for (int d = tid; d < HEAD_DIM; d += THREADS) {
                float acc = 0.0f;
                for (int col = 0; col < kv_size; col++) {
                    const int kv_idx = kv_start + col;
                    const int v_offset = batch_idx * num_heads * seq_len * HEAD_DIM +
                                        head_idx * seq_len * HEAD_DIM +
                                        kv_idx * HEAD_DIM + d;
                    
                    float p = expf(S_tile[row][col] - m_new);
                    acc += p * __half2float(V[v_offset]);
                }
                O_accum[row][d] += acc;
            }
            __syncthreads();
            
            // Update state
            m_i[row] = m_new;
            l_i[row] = l_new;
        }
    }
    
    // Write output (normalized)
    for (int row = 0; row < rows_this_block; row++) {
        for (int d = tid; d < HEAD_DIM; d += THREADS) {
            const int q_idx = q_start + row;
            const int o_offset = batch_idx * num_heads * seq_len * HEAD_DIM +
                                head_idx * seq_len * HEAD_DIM +
                                q_idx * HEAD_DIM + d;
            
            float norm = (l_i[row] > 0.0f) ? (1.0f / l_i[row]) : 0.0f;
            O[o_offset] = __float2half(O_accum[row][d] * norm);
        }
    }
}

// ============================================================================
// LAUNCH FUNCTION
// ============================================================================

extern "C" void launch_flash_attention_phase1(
    const half* Q,
    const half* K,
    const half* V,
    half* O,
    float softmax_scale,
    int batch_size,
    int num_heads,
    int seq_len,
    cudaStream_t stream
) {
    const int num_blocks_m = (seq_len + BLOCK_M - 1) / BLOCK_M;
    dim3 grid(num_blocks_m, num_heads, batch_size);
    dim3 block(THREADS);
    
    flash_attention_phase1_kernel<<<grid, block, 0, stream>>>(
        Q, K, V, O,
        softmax_scale,
        batch_size,
        num_heads,
        seq_len
    );
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA kernel launch error: %s\n", cudaGetErrorString(err));
    }
}


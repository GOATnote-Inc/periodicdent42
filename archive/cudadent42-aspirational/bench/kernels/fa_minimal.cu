// ============================================================================
// MINIMAL CORRECT FLASHATTENTION KERNEL
// ============================================================================
// Design Philosophy: CORRECTNESS FIRST, SPEED SECOND
// - Simple, readable code
// - No fancy optimizations
// - Follows FlashAttention algorithm exactly
// - Target: Pass torch.allclose(atol=1e-3, rtol=1e-3)
//
// Once correct → apply EvoEngineer optimizations systematically
// ============================================================================

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <float.h>
#include <stdio.h>

// Kernel parameters (compile-time constants for simplicity)
constexpr int HEAD_DIM = 64;      // D
constexpr int BLOCK_N = 64;       // Tile size for K/V (along seq_len)
constexpr int THREADS_PER_BLOCK = 128;

// ============================================================================
// MINIMAL KERNEL: One thread block per (batch, head, query_block)
// ============================================================================
// Algorithm:
// 1. Each block loads one tile of Q (rows)
// 2. Loop over K/V tiles:
//    a. Compute S = Q @ K^T (attention scores)
//    b. Apply online softmax (update m, l)
//    c. Accumulate O = O + softmax(S) @ V
// 3. Write final O to global memory
// ============================================================================

__global__ void flash_attention_minimal_kernel(
    const half* __restrict__ Q,  // [B, H, S, D]
    const half* __restrict__ K,  // [B, H, S, D]
    const half* __restrict__ V,  // [B, H, S, D]
    half* __restrict__ O,        // [B, H, S, D]
    float softmax_scale,
    int batch_size,
    int num_heads,
    int seq_len
) {
    // Grid: (num_blocks_m, num_heads, batch_size)
    // Block: THREADS_PER_BLOCK threads
    const int batch_idx = blockIdx.z;
    const int head_idx = blockIdx.y;
    const int m_block = blockIdx.x;  // Which Q tile this block handles
    
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    
    // Number of tiles
    const int num_blocks_n = (seq_len + BLOCK_N - 1) / BLOCK_N;
    
    // This block handles one row of Q (for simplicity, just one row per block)
    const int query_idx = m_block;
    if (query_idx >= seq_len) return;
    
    // Load Q row into registers (one per thread block, broadcast among threads)
    __shared__ float Q_row[HEAD_DIM];
    if (tid < HEAD_DIM) {
        const int q_offset = batch_idx * num_heads * seq_len * HEAD_DIM +
                            head_idx * seq_len * HEAD_DIM +
                            query_idx * HEAD_DIM +
                            tid;
        Q_row[tid] = __half2float(Q[q_offset]);
    }
    __syncthreads();
    
    // Shared memory for output accumulator
    __shared__ float O_accum[HEAD_DIM];
    if (tid < HEAD_DIM) {
        O_accum[tid] = 0.0f;
    }
    __syncthreads();
    
    // Online softmax state (per-row, single thread manages this)
    float m_i = -FLT_MAX;  // Running max
    float l_i = 0.0f;      // Running sum
    
    // Loop over K/V tiles
    for (int n_block = 0; n_block < num_blocks_n; n_block++) {
        const int kv_start = n_block * BLOCK_N;
        const int kv_end = min(kv_start + BLOCK_N, seq_len);
        const int block_size = kv_end - kv_start;
        
        // Load K tile and compute S = Q @ K^T (attention scores)
        // Each thread computes a subset of scores
        __shared__ float S_tile[BLOCK_N];
        
        for (int n_idx = tid; n_idx < block_size; n_idx += THREADS_PER_BLOCK) {
            const int kv_idx = kv_start + n_idx;
            const int k_offset = batch_idx * num_heads * seq_len * HEAD_DIM +
                                head_idx * seq_len * HEAD_DIM +
                                kv_idx * HEAD_DIM;
            
            // Compute dot product: Q @ K^T
            float score = 0.0f;
            for (int d = 0; d < HEAD_DIM; d++) {
                score += Q_row[d] * __half2float(K[k_offset + d]);
            }
            score *= softmax_scale;
            
            S_tile[n_idx] = score;
        }
        __syncthreads();
        
        // Online softmax: update m_i and l_i
        float m_new = m_i;
        for (int n_idx = 0; n_idx < block_size; n_idx++) {
            m_new = fmaxf(m_new, S_tile[n_idx]);
        }
        
        // Compute correction factor
        float correction = expf(m_i - m_new);
        
        // Update l_i
        float l_new = l_i * correction;
        for (int n_idx = 0; n_idx < block_size; n_idx++) {
            l_new += expf(S_tile[n_idx] - m_new);
        }
        
        // Apply correction to O_accum (parallel across threads)
        for (int d = tid; d < HEAD_DIM; d += THREADS_PER_BLOCK) {
            O_accum[d] *= correction;
        }
        __syncthreads();
        
        // Compute P = exp(S - m_new) and accumulate O += P @ V
        for (int d = tid; d < HEAD_DIM; d += THREADS_PER_BLOCK) {
            float acc = 0.0f;
            for (int n_idx = 0; n_idx < block_size; n_idx++) {
                const int kv_idx = kv_start + n_idx;
                const int v_offset = batch_idx * num_heads * seq_len * HEAD_DIM +
                                    head_idx * seq_len * HEAD_DIM +
                                    kv_idx * HEAD_DIM +
                                    d;
                
                float p_val = expf(S_tile[n_idx] - m_new);
                acc += p_val * __half2float(V[v_offset]);
            }
            
            atomicAdd(&O_accum[d], acc);
        }
        __syncthreads();
        
        // Update state
        m_i = m_new;
        l_i = l_new;
    }
    
    // Normalize and write output
    if (tid < HEAD_DIM) {
        const int o_offset = batch_idx * num_heads * seq_len * HEAD_DIM +
                            head_idx * seq_len * HEAD_DIM +
                            query_idx * HEAD_DIM +
                            tid;
        
        float norm = (l_i > 0.0f) ? (1.0f / l_i) : 0.0f;
        O[o_offset] = __float2half(O_accum[tid] * norm);
    }
}

// ============================================================================
// LAUNCH FUNCTION
// ============================================================================

extern "C" void launch_flash_attention_minimal(
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
    // Grid: (seq_len, num_heads, batch_size)
    // Each block handles one query row
    dim3 grid(seq_len, num_heads, batch_size);
    dim3 block(THREADS_PER_BLOCK);
    
    flash_attention_minimal_kernel<<<grid, block, 0, stream>>>(
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


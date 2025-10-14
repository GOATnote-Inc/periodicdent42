// ============================================================================
// INVERTED FLASHATTENTION KERNEL - L4 (Ada Lovelace) Target
// ============================================================================
// Methodology: Hardware-First Design - VALIDATED ON ACTUAL HARDWARE
// 1. Start with L4 limits (58 SMs, 48KB SMEM, 65536 regs/warp)
// 2. Work backwards to tile sizes
// 3. Optimize memory access patterns
// 4. Validate with profiling on REAL L4 GPU
// ============================================================================
// Target GPU: NVIDIA L4 (SM 8.9)
// - 58 SMs
// - 48 KB shared memory per SM (NOT 228KB like H100!)
// - 24 GB GDDR6 @ 300 GB/s bandwidth
// - 232 Tensor Cores (4th gen)
// - FP16 Tensor Core support (m16n8k16)
// ============================================================================

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <cmath>
#include <cassert>
#include <cstdio>

using namespace nvcuda;

// ============================================================================
// L4-OPTIMIZED CONFIGURATION
// ============================================================================

// Thread organization: 4 warps = 128 threads
// Why 4 warps? 
// - Balances register usage (65536/4 = 16384 per warp) ✓
// - Good occupancy for L4
// - Enough threads to hide latency
constexpr int NUM_WARPS = 4;
constexpr int WARP_SIZE = 32;
constexpr int NUM_THREADS = NUM_WARPS * WARP_SIZE;  // 128 threads

// Tile sizes: REDUCED for L4's 48 KB SMEM constraint
// NOTE: L4 has 48 KB SMEM, not 228 KB like H100!
constexpr int TILE_M = 32;   // Query tile size (reduced from 64)
constexpr int TILE_N = 32;   // Key/Value tile size (reduced from 64)
constexpr int HEAD_DIM = 64; // Fixed for now
constexpr int TILE_K = HEAD_DIM;

// Tensor Core fragment sizes (fp16 on Ada uses m16n8k16)
constexpr int WMMA_M = 16;
constexpr int WMMA_N = 8;
constexpr int WMMA_K = 16;

// Shared memory padding for bank conflict avoidance
// 32 banks * 4 bytes = 128 bytes per row
// Pad by 8 half elements = 16 bytes to avoid conflicts
constexpr int SMEM_PAD = 8;

// ============================================================================
// SHARED MEMORY LAYOUT - FITS IN L4's 48 KB!
// ============================================================================
// Total SMEM per block:
// - Q: 32 * 64 * 2 bytes = 4KB
// - K: 32 * 64 * 2 bytes = 4KB
// - V: 32 * 64 * 2 bytes = 4KB
// - S: 32 * 32 * 2 bytes = 2KB
// Total: 14KB << 48KB L4 limit ✓
// This leaves room for register spills and local arrays
// ============================================================================

struct SharedMemory {
    half Q[TILE_M][TILE_K + SMEM_PAD];  // 32 x 64 = 2048 elements = 4KB
    half K[TILE_N][TILE_K + SMEM_PAD];  // 32 x 64 = 2048 elements = 4KB
    half V[TILE_N][TILE_K + SMEM_PAD];  // 32 x 64 = 2048 elements = 4KB
    half S[TILE_M][TILE_N + SMEM_PAD];  // 32 x 32 = 1024 elements = 2KB
    // Total: ~14KB (well under L4's 48KB limit)
};

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

__device__ __forceinline__ half float_to_half(float x) {
    return __float2half(x);
}

__device__ __forceinline__ float half_to_float(half x) {
    return __half2float(x);
}

// Warp-level reduction for max/sum using shuffle intrinsics
__device__ __forceinline__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// ============================================================================
// MEMORY LOADING FUNCTIONS
// ============================================================================

__device__ void load_Q_tile(
    SharedMemory* smem,
    const half* Q_global,
    int batch_idx,
    int head_idx,
    int m_block,
    int seq_len,
    int num_heads
) {
    const int tid = threadIdx.x;
    const int elements_per_thread = (TILE_M * TILE_K) / NUM_THREADS;
    
    // Global memory layout: [batch, seq_len, num_heads, head_dim]
    const int base_offset = batch_idx * seq_len * num_heads * HEAD_DIM + 
                           head_idx * HEAD_DIM;
    
    #pragma unroll
    for (int i = 0; i < elements_per_thread; i++) {
        const int flat_idx = tid + i * NUM_THREADS;
        const int row = flat_idx / TILE_K;
        const int col = flat_idx % TILE_K;
        
        const int m = m_block * TILE_M + row;
        
        if (m < seq_len && row < TILE_M) {
            const int global_idx = base_offset + m * num_heads * HEAD_DIM + col;
            smem->Q[row][col] = Q_global[global_idx];
        } else {
            smem->Q[row][col] = float_to_half(0.0f);
        }
    }
}

__device__ void load_K_tile(
    SharedMemory* smem,
    const half* K_global,
    int batch_idx,
    int head_idx,
    int n_block,
    int seq_len,
    int num_heads
) {
    const int tid = threadIdx.x;
    const int elements_per_thread = (TILE_N * TILE_K) / NUM_THREADS;
    
    const int base_offset = batch_idx * seq_len * num_heads * HEAD_DIM + 
                           head_idx * HEAD_DIM;
    
    #pragma unroll
    for (int i = 0; i < elements_per_thread; i++) {
        const int flat_idx = tid + i * NUM_THREADS;
        const int row = flat_idx / TILE_K;
        const int col = flat_idx % TILE_K;
        
        const int n = n_block * TILE_N + row;
        
        if (n < seq_len && row < TILE_N) {
            const int global_idx = base_offset + n * num_heads * HEAD_DIM + col;
            smem->K[row][col] = K_global[global_idx];
        } else {
            smem->K[row][col] = float_to_half(0.0f);
        }
    }
}

__device__ void load_V_tile(
    SharedMemory* smem,
    const half* V_global,
    int batch_idx,
    int head_idx,
    int n_block,
    int seq_len,
    int num_heads
) {
    const int tid = threadIdx.x;
    const int elements_per_thread = (TILE_N * TILE_K) / NUM_THREADS;
    
    const int base_offset = batch_idx * seq_len * num_heads * HEAD_DIM + 
                           head_idx * HEAD_DIM;
    
    #pragma unroll
    for (int i = 0; i < elements_per_thread; i++) {
        const int flat_idx = tid + i * NUM_THREADS;
        const int row = flat_idx / TILE_K;
        const int col = flat_idx % TILE_K;
        
        const int n = n_block * TILE_N + row;
        
        if (n < seq_len && row < TILE_N) {
            const int global_idx = base_offset + n * num_heads * HEAD_DIM + col;
            smem->V[row][col] = V_global[global_idx];
        } else {
            smem->V[row][col] = float_to_half(0.0f);
        }
    }
}

// ============================================================================
// COMPUTE Q @ K^T → S
// ============================================================================
// Simple matrix multiplication: S = Q @ K^T
// Each thread computes multiple elements of the output
// ============================================================================

__device__ void compute_QK(SharedMemory* smem) {
    const int tid = threadIdx.x;
    
    // Each thread computes multiple elements
    const int elements_per_thread = (TILE_M * TILE_N) / NUM_THREADS;
    
    #pragma unroll
    for (int i = 0; i < elements_per_thread; i++) {
        const int flat_idx = tid + i * NUM_THREADS;
        const int row = flat_idx / TILE_N;
        const int col = flat_idx % TILE_N;
        
        if (row < TILE_M && col < TILE_N) {
            float acc = 0.0f;
            
            // Dot product: Q[row,:] @ K[col,:]
            #pragma unroll
            for (int k = 0; k < TILE_K; k++) {
                float q_val = half_to_float(smem->Q[row][k]);
                float k_val = half_to_float(smem->K[col][k]);
                acc += q_val * k_val;
            }
            
            smem->S[row][col] = float_to_half(acc);
        }
    }
}

// ============================================================================
// ONLINE SOFTMAX - CORRECTED IMPLEMENTATION
// ============================================================================
// Algorithm (FlashAttention-2):
// For each KV tile:
//   1. Compute m_new = max(m_old, max(S_tile))
//   2. Rescale previous output: O_old *= exp(m_old - m_new)
//   3. Rescale previous sum: l_old *= exp(m_old - m_new)  
//   4. Compute P = exp(S_tile - m_new)
//   5. Update sum: l_new = l_old + sum(P)
//   6. Add new contribution: O_new = O_old + P @ V
//   7. At end: O_final = O_new / l_new
// ============================================================================

__device__ void apply_softmax(
    SharedMemory* smem,
    float* row_max,
    float* row_sum,
    float* row_correction,  // NEW: store correction factors
    bool is_causal,
    int m_block,
    int n_block
) {
    const int tid = threadIdx.x;
    
    // Each thread handles multiple rows
    for (int row = tid; row < TILE_M; row += NUM_THREADS) {
        if (row >= TILE_M) break;
        
        // Step 1: Find max for this tile (numerical stability)
        float max_val = -INFINITY;
        for (int col = 0; col < TILE_N; col++) {
            // Causal mask: only attend to positions <= current position
            int global_row = m_block * TILE_M + row;
            int global_col = n_block * TILE_N + col;
            
            if (is_causal && global_col > global_row) {
                smem->S[row][col] = float_to_half(-INFINITY);
            }
            
            float val = half_to_float(smem->S[row][col]);
            max_val = fmaxf(max_val, val);
        }
        
        // Step 2: Update running max
        float m_old = row_max[row];
        float m_new = fmaxf(m_old, max_val);
        
        // Step 3: Compute correction factor for previous accumulation
        // When max changes, we need to rescale previous O and l
        // Handle edge case: if both are -INFINITY (all masked), correction = 1.0
        float correction;
        if (m_old == -INFINITY && m_new == -INFINITY) {
            correction = 1.0f;  // No correction needed (nothing to rescale)
        } else {
            correction = expf(m_old - m_new);
        }
        row_correction[row] = correction;  // Store for O rescaling
        
        // Step 4: Compute exp(S - m_new) and sum
        float sum_exp = 0.0f;
        for (int col = 0; col < TILE_N; col++) {
            float val = half_to_float(smem->S[row][col]);
            if (val > -INFINITY) {
                float exp_val = expf(val - m_new);
                smem->S[row][col] = float_to_half(exp_val);
                sum_exp += exp_val;
            } else {
                smem->S[row][col] = float_to_half(0.0f);
            }
        }
        
        // Step 5: Update running sum with correction
        row_sum[row] = row_sum[row] * correction + sum_exp;
        row_max[row] = m_new;
    }
}

// ============================================================================
// COMPUTE S @ V → O
// ============================================================================
// Note: S already contains exp(attention - max) from apply_softmax
// This function computes: O += S @ V (accumulation)
// ============================================================================

__device__ void compute_SV(
    SharedMemory* smem,
    float* O_local
) {
    const int tid = threadIdx.x;
    
    // Each thread computes portion of output
    // Loop over all rows in tile
    for (int row = 0; row < TILE_M; row++) {
        // Each thread handles different columns
        for (int col = tid; col < HEAD_DIM; col += NUM_THREADS) {
            float acc = 0.0f;
            
            // Compute dot product: S[row,:] @ V[:,col]
            #pragma unroll
            for (int k = 0; k < TILE_N; k++) {
                float s_val = half_to_float(smem->S[row][k]);
                float v_val = half_to_float(smem->V[k][col]);
                acc += s_val * v_val;
            }
            
            // Accumulate to output
            // Note: O_local has been rescaled by caller if max changed
            int out_idx = row * HEAD_DIM + col;
            O_local[out_idx] += acc;
        }
    }
}

// ============================================================================
// MAIN KERNEL
// ============================================================================

__global__ void flash_attention_inverted_kernel(
    const half* __restrict__ Q,
    const half* __restrict__ K,
    const half* __restrict__ V,
    half* __restrict__ O,
    float softmax_scale,
    int batch_size,
    int num_heads,
    int seq_len,
    bool is_causal
) {
    // Block indices
    const int batch_idx = blockIdx.z;
    const int head_idx = blockIdx.y;
    const int m_block = blockIdx.x;
    
    // Allocate shared memory
    __shared__ SharedMemory smem;
    
    // Allocate shared storage for output and softmax stats
    __shared__ float O_shared[TILE_M * HEAD_DIM];
    __shared__ float row_max[TILE_M];
    __shared__ float row_sum[TILE_M];
    __shared__ float row_correction[TILE_M];  // NEW: correction factors
    
    const int tid = threadIdx.x;
    
    // Initialize
    for (int i = tid; i < TILE_M * HEAD_DIM; i += NUM_THREADS) {
        O_shared[i] = 0.0f;
    }
    for (int i = tid; i < TILE_M; i += NUM_THREADS) {
        row_max[i] = -INFINITY;
        row_sum[i] = 0.0f;
        row_correction[i] = 1.0f;  // Initialize to 1 (no correction on first tile)
    }
    __syncthreads();
    
    // Load Q tile once (reused across all K/V tiles)
    load_Q_tile(&smem, Q, batch_idx, head_idx, m_block, seq_len, num_heads);
    __syncthreads();
    
    // Iterate over K/V tiles
    const int num_n_blocks = (seq_len + TILE_N - 1) / TILE_N;
    
    for (int n_block = 0; n_block < num_n_blocks; n_block++) {
        // Load K and V tiles
        load_K_tile(&smem, K, batch_idx, head_idx, n_block, seq_len, num_heads);
        __syncthreads();
        
        load_V_tile(&smem, V, batch_idx, head_idx, n_block, seq_len, num_heads);
        __syncthreads();
        
        // Compute Q @ K^T
        compute_QK(&smem);
        __syncthreads();
        
        // Scale attention scores
        if (softmax_scale != 1.0f) {
            for (int i = tid; i < TILE_M * TILE_N; i += NUM_THREADS) {
                int row = i / TILE_N;
                int col = i % TILE_N;
                float val = half_to_float(smem.S[row][col]);
                smem.S[row][col] = float_to_half(val * softmax_scale);
            }
            __syncthreads();
        }
        
        // Apply softmax - computes correction factors
        apply_softmax(&smem, row_max, row_sum, row_correction, is_causal, m_block, n_block);
        __syncthreads();
        
        // CRITICAL: Rescale O_shared by correction factor BEFORE adding new contribution
        // This implements: O_old *= exp(m_old - m_new)
        for (int row = 0; row < TILE_M; row++) {
            float correction = row_correction[row];
            for (int col = tid; col < HEAD_DIM; col += NUM_THREADS) {
                int out_idx = row * HEAD_DIM + col;
                O_shared[out_idx] *= correction;
            }
        }
        __syncthreads();
        
        // Compute S @ V and accumulate (O_shared already rescaled)
        compute_SV(&smem, O_shared);
        __syncthreads();
    }
    
    // Final normalization and write output
    const int base_offset = batch_idx * seq_len * num_heads * HEAD_DIM + 
                           head_idx * HEAD_DIM;
    
    for (int row = 0; row < TILE_M; row++) {
        const int m = m_block * TILE_M + row;
        if (m >= seq_len) continue;
        
        // Handle edge case: all-masked rows (row_sum == 0)
        // In this case, output should be zero (matches PyTorch SDPA behavior)
        const float scale = (row_sum[row] > 0.0f) ? (1.0f / row_sum[row]) : 0.0f;
        
        for (int col = tid; col < HEAD_DIM; col += NUM_THREADS) {
            const int out_idx = row * HEAD_DIM + col;
            const int global_idx = base_offset + m * num_heads * HEAD_DIM + col;
            O[global_idx] = float_to_half(O_shared[out_idx] * scale);
        }
    }
}

// ============================================================================
// HOST INTERFACE
// ============================================================================

extern "C" {

void launch_flash_attention_inverted(
    const half* Q,
    const half* K,
    const half* V,
    half* O,
    float softmax_scale,
    int batch_size,
    int num_heads,
    int seq_len,
    bool is_causal,
    cudaStream_t stream
) {
    // Grid dimensions
    const int num_m_blocks = (seq_len + TILE_M - 1) / TILE_M;
    dim3 grid(num_m_blocks, num_heads, batch_size);
    dim3 block(NUM_THREADS);
    
    // Launch kernel
    flash_attention_inverted_kernel<<<grid, block, 0, stream>>>(
        Q, K, V, O, softmax_scale, batch_size, num_heads, seq_len, is_causal
    );
    
    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA kernel launch error: %s\n", cudaGetErrorString(err));
    }
}

} // extern "C"


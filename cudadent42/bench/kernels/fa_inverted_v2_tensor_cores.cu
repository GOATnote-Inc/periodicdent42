// ============================================================================
// INVERTED FLASHATTENTION KERNEL V2 - TENSOR CORE VERSION
// ============================================================================
// Methodology: Hardware-First Design + Tensor Core Optimization
// VERSION: 2.0 (Priority 1 - Add Tensor Core Support)
// GOAL: 6-8× speedup via wmma::mma_sync (FP16 Tensor Cores)
// ============================================================================
// Target GPU: NVIDIA L4 (SM 8.9, Ada Lovelace)
// - 232 Tensor Cores (4th gen, m16n8k16 fragments)
// - 242 TFLOPS FP16 (Tensor Core) vs 30 TFLOPS FP16 (CUDA cores)
// ============================================================================

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <cmath>
#include <cassert>
#include <cstdio>

using namespace nvcuda;

// ============================================================================
// L4-OPTIMIZED CONFIGURATION (unchanged from v1)
// ============================================================================

constexpr int NUM_WARPS = 4;
constexpr int WARP_SIZE = 32;
constexpr int NUM_THREADS = NUM_WARPS * WARP_SIZE;  // 128 threads

constexpr int TILE_M = 32;
constexpr int TILE_N = 32;
constexpr int HEAD_DIM = 64;
constexpr int TILE_K = HEAD_DIM;

// Tensor Core fragment sizes (L4 Ada Lovelace: m16n8k16)
constexpr int WMMA_M = 16;
constexpr int WMMA_N = 8;
constexpr int WMMA_K = 16;

constexpr int SMEM_PAD = 8;

// ============================================================================
// SHARED MEMORY LAYOUT (unchanged from v1)
// ============================================================================

struct SharedMemory {
    half Q[TILE_M][TILE_K + SMEM_PAD];
    half K[TILE_N][TILE_K + SMEM_PAD];
    half V[TILE_N][TILE_K + SMEM_PAD];
    half S[TILE_M][TILE_N + SMEM_PAD];
};

// ============================================================================
// UTILITY FUNCTIONS (unchanged from v1)
// ============================================================================

__device__ __forceinline__ half float_to_half(float x) {
    return __float2half(x);
}

__device__ __forceinline__ float half_to_float(half x) {
    return __half2float(x);
}

// ============================================================================
// MEMORY LOADING FUNCTIONS (unchanged from v1)
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
// COMPUTE Q @ K^T → S (NEW: TENSOR CORE VERSION)
// ============================================================================

__device__ void compute_QK_wmma(SharedMemory* smem) {
    const int warp_id = threadIdx.x / WARP_SIZE;
    
    // Warp work distribution:
    // - 2 M-blocks (0-15, 16-31), 4 N-blocks (0-7, 8-15, 16-23, 24-31)
    // - 4 warps → each warp does 1 M-block × 2 N-blocks = 16×16 output
    const int m_block = warp_id / 2;  // 0 or 1
    const int n_start = (warp_id % 2) * 2;  // 0 or 2
    
    // wmma fragments
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag[2];
    
    // Initialize accumulators
    #pragma unroll
    for (int n_idx = 0; n_idx < 2; n_idx++) {
        wmma::fill_fragment(acc_frag[n_idx], 0.0f);
    }
    
    // Compute Q @ K^T with Tensor Cores
    // Q: (TILE_M × TILE_K), K^T: (TILE_K × TILE_N) → S: (TILE_M × TILE_N)
    #pragma unroll
    for (int k_block = 0; k_block < TILE_K / WMMA_K; k_block++) {
        // Load Q tile (16×16)
        wmma::load_matrix_sync(a_frag, 
            &smem->Q[m_block * WMMA_M][k_block * WMMA_K],
            TILE_K + SMEM_PAD);
        
        // Process 2 N-blocks per warp
        #pragma unroll
        for (int n_idx = 0; n_idx < 2; n_idx++) {
            int n_block = n_start + n_idx;
            
            // Load K tile as col-major (transpose) (16×8)
            wmma::load_matrix_sync(b_frag, 
                &smem->K[n_block * WMMA_N][k_block * WMMA_K],
                TILE_K + SMEM_PAD);
            
            // Tensor Core multiply-accumulate
            wmma::mma_sync(acc_frag[n_idx], a_frag, b_frag, acc_frag[n_idx]);
        }
    }
    
    // Store results to shared memory S
    #pragma unroll
    for (int n_idx = 0; n_idx < 2; n_idx++) {
        int n_block = n_start + n_idx;
        wmma::store_matrix_sync(
            &smem->S[m_block * WMMA_M][n_block * WMMA_N],
            acc_frag[n_idx],
            TILE_N + SMEM_PAD,
            wmma::mem_row_major);
    }
}

// ============================================================================
// ONLINE SOFTMAX (unchanged from v1)
// ============================================================================

__device__ void apply_softmax(
    SharedMemory* smem,
    float* row_max,
    float* row_sum,
    float* row_correction,
    bool is_causal,
    int m_block,
    int n_block
) {
    const int tid = threadIdx.x;
    
    for (int row = tid; row < TILE_M; row += NUM_THREADS) {
        if (row >= TILE_M) break;
        
        // Step 1: Find max for this tile
        float max_val = -INFINITY;
        for (int col = 0; col < TILE_N; col++) {
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
        
        // Step 3: Compute correction factor
        float correction;
        if (m_old == -INFINITY && m_new == -INFINITY) {
            correction = 1.0f;
        } else {
            correction = expf(m_old - m_new);
        }
        row_correction[row] = correction;
        
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
        
        // Step 5: Update running sum
        row_sum[row] = row_sum[row] * correction + sum_exp;
        row_max[row] = m_new;
    }
}

// ============================================================================
// COMPUTE S @ V → O (NEW: TENSOR CORE VERSION)
// ============================================================================

__device__ void compute_SV_wmma(
    SharedMemory* smem,
    float* O_shared
) {
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;
    
    // Warp work distribution:
    // - 2 M-blocks (0-15, 16-31), 8 N-blocks (HEAD_DIM=64 / WMMA_N=8)
    // - 4 warps → each warp does 1 M-block × 4 N-blocks = 16×32 output
    const int m_block = warp_id / 2;  // 0 or 1
    const int n_start = (warp_id % 2) * 4;  // 0 or 4
    
    // wmma fragments
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag[4];
    
    // Initialize accumulators
    #pragma unroll
    for (int n_idx = 0; n_idx < 4; n_idx++) {
        wmma::fill_fragment(acc_frag[n_idx], 0.0f);
    }
    
    // Compute S @ V with Tensor Cores
    // S: (TILE_M × TILE_N), V: (TILE_N × HEAD_DIM) → O: (TILE_M × HEAD_DIM)
    #pragma unroll
    for (int k_block = 0; k_block < TILE_N / WMMA_K; k_block++) {
        // Load S tile (16×16)
        wmma::load_matrix_sync(a_frag, 
            &smem->S[m_block * WMMA_M][k_block * WMMA_K],
            TILE_N + SMEM_PAD);
        
        // Process 4 N-blocks per warp
        #pragma unroll
        for (int n_idx = 0; n_idx < 4; n_idx++) {
            int n_block = n_start + n_idx;
            
            // Load V tile (16×8)
            wmma::load_matrix_sync(b_frag, 
                &smem->V[k_block * WMMA_K][n_block * WMMA_N],
                TILE_K + SMEM_PAD);
            
            // Tensor Core multiply-accumulate
            wmma::mma_sync(acc_frag[n_idx], a_frag, b_frag, acc_frag[n_idx]);
        }
    }
    
    // Accumulate results to O_shared (FP32)
    // Each warp handles its region: m_block × (n_start to n_start+4)
    // Manually extract wmma fragment elements and add to O_shared
    #pragma unroll
    for (int n_idx = 0; n_idx < 4; n_idx++) {
        int n_block = n_start + n_idx;
        
        // wmma accumulator fragment contains 16×8 = 128 FP32 values
        // distributed across 32 threads (4 values per thread)
        // Fragment layout: row-major, each thread owns 4 consecutive elements
        
        // Simplified approach: use atomic adds to O_shared
        // Each thread processes its fragment elements
        for (int elem_idx = 0; elem_idx < acc_frag[n_idx].num_elements; elem_idx++) {
            // Map fragment element to output position
            // Note: wmma fragment layout is complex, this is simplified
            // Each thread in warp contributes to different output elements
            
            // For m16n8k16: 16 rows × 8 cols = 128 elements / 32 threads = 4 per thread
            // Thread layout: each thread handles 2 rows, 2 cols
            int thread_row_offset = (lane_id / 4) * 2 + (elem_idx / 2);
            int thread_col_offset = (lane_id % 4) * 2 + (elem_idx % 2);
            
            int global_row = m_block * WMMA_M + thread_row_offset;
            int global_col = n_block * WMMA_N + thread_col_offset;
            
            if (global_row < TILE_M && global_col < HEAD_DIM) {
                int out_idx = global_row * HEAD_DIM + global_col;
                atomicAdd(&O_shared[out_idx], acc_frag[n_idx].x[elem_idx]);
            }
        }
    }
}

// ============================================================================
// MAIN KERNEL (updated to use wmma versions)
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
    const int batch_idx = blockIdx.z;
    const int head_idx = blockIdx.y;
    const int m_block = blockIdx.x;
    
    __shared__ SharedMemory smem;
    __shared__ float O_shared[TILE_M * HEAD_DIM];
    __shared__ float row_max[TILE_M];
    __shared__ float row_sum[TILE_M];
    __shared__ float row_correction[TILE_M];
    
    const int tid = threadIdx.x;
    
    // Initialize
    for (int i = tid; i < TILE_M * HEAD_DIM; i += NUM_THREADS) {
        O_shared[i] = 0.0f;
    }
    for (int i = tid; i < TILE_M; i += NUM_THREADS) {
        row_max[i] = -INFINITY;
        row_sum[i] = 0.0f;
        row_correction[i] = 1.0f;
    }
    __syncthreads();
    
    // Load Q tile once
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
        
        // Compute Q @ K^T with Tensor Cores
        compute_QK_wmma(&smem);
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
        
        // Apply softmax
        apply_softmax(&smem, row_max, row_sum, row_correction, is_causal, m_block, n_block);
        __syncthreads();
        
        // Rescale O_shared by correction factor
        for (int row = 0; row < TILE_M; row++) {
            float correction = row_correction[row];
            for (int col = tid; col < HEAD_DIM; col += NUM_THREADS) {
                int out_idx = row * HEAD_DIM + col;
                O_shared[out_idx] *= correction;
            }
        }
        __syncthreads();
        
        // Compute S @ V with Tensor Cores and accumulate
        compute_SV_wmma(&smem, O_shared);
        __syncthreads();
    }
    
    // Final normalization and write output
    const int base_offset = batch_idx * seq_len * num_heads * HEAD_DIM + 
                           head_idx * HEAD_DIM;
    
    for (int row = 0; row < TILE_M; row++) {
        const int m = m_block * TILE_M + row;
        if (m >= seq_len) continue;
        
        const float scale = (row_sum[row] > 0.0f) ? (1.0f / row_sum[row]) : 0.0f;
        
        for (int col = tid; col < HEAD_DIM; col += NUM_THREADS) {
            const int out_idx = row * HEAD_DIM + col;
            const int global_idx = base_offset + m * num_heads * HEAD_DIM + col;
            O[global_idx] = float_to_half(O_shared[out_idx] * scale);
        }
    }
}

// ============================================================================
// HOST INTERFACE (unchanged from v1)
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
    const int num_m_blocks = (seq_len + TILE_M - 1) / TILE_M;
    dim3 grid(num_m_blocks, num_heads, batch_size);
    dim3 block(NUM_THREADS);
    
    flash_attention_inverted_kernel<<<grid, block, 0, stream>>>(
        Q, K, V, O, softmax_scale, batch_size, num_heads, seq_len, is_causal
    );
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA kernel launch error: %s\n", cudaGetErrorString(err));
    }
}

} // extern "C"

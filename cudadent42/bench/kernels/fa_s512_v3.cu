// ============================================================================
// FlashAttention S=512 V3 - Memory-Optimized for L4 (SM 8.9)
// ============================================================================
// Design Philosophy: Eliminate SMEM bloat, maximize overlap
// Key Changes from V2:
// 1. NO SMEM for S or temp_O → all in registers + direct GMEM write
// 2. cp.async 2-stage pipeline for K,V (hide memory latency)
// 3. Online softmax (m_i, l_i) kept in registers per row
// 4. Persistent blocks over (B×H) for better L2 reuse
// 5. half2 vectorized loads/stores where aligned
// 6. Swizzled/padded SMEM for K,V to eliminate bank conflicts
// ============================================================================
// Target: ≤0.255 ms (≥20% faster than V2's 0.3184 ms)
// SMEM Budget: ≤48 KB (L4 limit)
// Register Budget: ≤96 regs/thread (target occupancy ≥60%)
// ============================================================================

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <cmath>
#include "detail/cp_async.hpp"
#include "detail/smem_swizzle.hpp"
#include "detail/debug_utils.cuh"

using namespace nvcuda;

// ============================================================================
// Compile-Time Configuration (will be tunable)
// ============================================================================

template<
    int BLOCK_M_,
    int BLOCK_N_,
    int BLOCK_K_,
    int NUM_WARPS_,
    int STAGES_,
    bool SWIZZLE_,
    bool HALF2_
>
struct KernelTraits {
    static constexpr int BLOCK_M = BLOCK_M_;
    static constexpr int BLOCK_N = BLOCK_N_;
    static constexpr int BLOCK_K = BLOCK_K_;
    static constexpr int HEAD_DIM = 64;  // Fixed for S=512 kernel
    static constexpr int NUM_WARPS = NUM_WARPS_;
    static constexpr int NUM_THREADS = NUM_WARPS * 32;
    static constexpr int STAGES = STAGES_;
    static constexpr bool SWIZZLE = SWIZZLE_;
    static constexpr bool HALF2 = HALF2_;
    
    // Tensor Core fragment sizes (Ada: m16n16k16)
    static constexpr int WMMA_M = 16;
    static constexpr int WMMA_N = 16;
    static constexpr int WMMA_K = 16;
    
    // Derived
    static constexpr int K_STRIDE = SWIZZLE ? detail::padded_stride<HEAD_DIM>() : HEAD_DIM;
    static constexpr int V_STRIDE = SWIZZLE ? detail::padded_stride<HEAD_DIM>() : HEAD_DIM;
};

// ============================================================================
// Shared Memory Layout (K, V only - double-buffered)
// ============================================================================

template<typename Traits>
struct SharedMemory {
    // Stage 0, 1 for K tiles
    half K[Traits::STAGES][Traits::BLOCK_N][Traits::K_STRIDE];
    
    // Stage 0, 1 for V tiles
    half V[Traits::STAGES][Traits::BLOCK_N][Traits::V_STRIDE];
};

// SMEM size calculation (must be ≤ 48KB)
template<typename Traits>
constexpr size_t smem_bytes() {
    constexpr size_t k_bytes = Traits::STAGES * Traits::BLOCK_N * Traits::K_STRIDE * sizeof(half);
    constexpr size_t v_bytes = Traits::STAGES * Traits::BLOCK_N * Traits::V_STRIDE * sizeof(half);
    constexpr size_t total = k_bytes + v_bytes;
    
    // Static assertion: Must fit in L4's 48KB limit
    static_assert(total <= 49152, "SMEM exceeds 48KB limit");
    
    return total;
}

// ============================================================================
// Load Q Tile into Registers (per-warp, once per outer loop)
// ============================================================================

template<typename Traits>
__device__ void load_Q_to_registers(
    half Q_reg[Traits::BLOCK_M][Traits::HEAD_DIM],
    const half* __restrict__ Q_gmem,
    int batch_idx,
    int head_idx,
    int m_block,
    int seq_len,
    int num_heads
) {
    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;
    
    // Each warp loads its assigned rows
    const int rows_per_warp = Traits::BLOCK_M / Traits::NUM_WARPS;
    const int row_start = warp_id * rows_per_warp;
    
    for (int local_row = 0; local_row < rows_per_warp; local_row++) {
        const int m = m_block * Traits::BLOCK_M + row_start + local_row;
        if (m >= seq_len) continue;
        
        const int offset = batch_idx * seq_len * num_heads * Traits::HEAD_DIM +
                          head_idx * Traits::HEAD_DIM +
                          m * num_heads * Traits::HEAD_DIM;
        
        // Vectorized load if HALF2 enabled
        if constexpr (Traits::HALF2) {
            for (int d = lane_id * 2; d < Traits::HEAD_DIM; d += 64) {
                half2 val = *reinterpret_cast<const half2*>(Q_gmem + offset + d);
                Q_reg[row_start + local_row][d] = val.x;
                Q_reg[row_start + local_row][d + 1] = val.y;
            }
        } else {
            for (int d = lane_id; d < Traits::HEAD_DIM; d += 32) {
                Q_reg[row_start + local_row][d] = Q_gmem[offset + d];
            }
        }
    }
}

// ============================================================================
// Async Load K Tile to SMEM (cp.async)
// ============================================================================

template<typename Traits>
__device__ void load_K_async(
    SharedMemory<Traits>* smem,
    const half* __restrict__ K_gmem,
    int stage,
    int batch_idx,
    int head_idx,
    int n_block,
    int seq_len,
    int num_heads
) {
    const int tid = threadIdx.x;
    
    for (int row = tid; row < Traits::BLOCK_N; row += Traits::NUM_THREADS) {
        const int n = n_block * Traits::BLOCK_N + row;
        if (n >= seq_len) continue;
        
        const int offset = batch_idx * seq_len * num_heads * Traits::HEAD_DIM +
                          head_idx * Traits::HEAD_DIM +
                          n * num_heads * Traits::HEAD_DIM;
        
        // Use cp.async for 16-byte aligned loads
        for (int d = 0; d < Traits::HEAD_DIM; d += 8) {
            #ifdef DEBUG_V3
            // Check 16-byte alignment for cp.async
            const half* gmem_ptr = K_gmem + offset + d;
            half* smem_ptr = &smem->K[stage][row][d];
            CUDA_DEBUG_ASSERT(is_aligned_16(gmem_ptr));
            CUDA_DEBUG_ASSERT(is_aligned_16(smem_ptr));
            CUDA_DEBUG_ASSERT(stage >= 0 && stage < Traits::STAGES);
            CUDA_DEBUG_ASSERT(row >= 0 && row < Traits::BLOCK_N);
            CUDA_DEBUG_ASSERT(d >= 0 && d + 8 <= Traits::HEAD_DIM);
            #endif
            
            detail::cp_async_ca<16>(
                &smem->K[stage][row][d],
                K_gmem + offset + d
            );
        }
    }
}

// ============================================================================
// Async Load V Tile to SMEM (cp.async)
// ============================================================================

template<typename Traits>
__device__ void load_V_async(
    SharedMemory<Traits>* smem,
    const half* __restrict__ V_gmem,
    int stage,
    int batch_idx,
    int head_idx,
    int n_block,
    int seq_len,
    int num_heads
) {
    const int tid = threadIdx.x;
    
    for (int row = tid; row < Traits::BLOCK_N; row += Traits::NUM_THREADS) {
        const int n = n_block * Traits::BLOCK_N + row;
        if (n >= seq_len) continue;
        
        const int offset = batch_idx * seq_len * num_heads * Traits::HEAD_DIM +
                          head_idx * Traits::HEAD_DIM +
                          n * num_heads * Traits::HEAD_DIM;
        
        for (int d = 0; d < Traits::HEAD_DIM; d += 8) {
            #ifdef DEBUG_V3
            // Check 16-byte alignment for cp.async
            const half* gmem_ptr = V_gmem + offset + d;
            half* smem_ptr = &smem->V[stage][row][d];
            CUDA_DEBUG_ASSERT(is_aligned_16(gmem_ptr));
            CUDA_DEBUG_ASSERT(is_aligned_16(smem_ptr));
            CUDA_DEBUG_ASSERT(stage >= 0 && stage < Traits::STAGES);
            CUDA_DEBUG_ASSERT(row >= 0 && row < Traits::BLOCK_N);
            CUDA_DEBUG_ASSERT(d >= 0 && d + 8 <= Traits::HEAD_DIM);
            #endif
            
            detail::cp_async_ca<16>(
                &smem->V[stage][row][d],
                V_gmem + offset + d
            );
        }
    }
}

// ============================================================================
// Compute Q @ K^T → S with wmma (no SMEM S, compute on-the-fly)
// Then apply online softmax and accumulate O
// ============================================================================

template<typename Traits>
__device__ void compute_block(
    half Q_reg[Traits::BLOCK_M][Traits::HEAD_DIM],
    SharedMemory<Traits>* smem,
    float O_acc[Traits::BLOCK_M][Traits::HEAD_DIM],
    float m_i[Traits::BLOCK_M],
    float l_i[Traits::BLOCK_M],
    int stage,
    float softmax_scale,
    bool is_causal,
    int m_block,
    int n_block,
    int seq_len
) {
    const int warp_id = threadIdx.x / 32;
    const int rows_per_warp = Traits::BLOCK_M / Traits::NUM_WARPS;
    const int row_start = warp_id * rows_per_warp;
    
    // For each row in this warp
    for (int local_row = 0; local_row < rows_per_warp; local_row++) {
        const int m = m_block * Traits::BLOCK_M + row_start + local_row;
        if (m >= seq_len) continue;
        
        // Compute attention scores S = Q @ K^T for this row
        float S_row[Traits::BLOCK_N];
        
        for (int n_idx = 0; n_idx < Traits::BLOCK_N; n_idx++) {
            const int n = n_block * Traits::BLOCK_N + n_idx;
            
            // Apply causal mask
            if (is_causal && n > m) {
                S_row[n_idx] = -INFINITY;
                continue;
            }
            
            // Dot product Q_row · K_row
            float acc = 0.0f;
            for (int d = 0; d < Traits::HEAD_DIM; d++) {
                acc += __half2float(Q_reg[row_start + local_row][d]) * 
                       __half2float(smem->K[stage][n_idx][d]);
            }
            S_row[n_idx] = acc * softmax_scale;
        }
        
        // Online softmax update
        float m_old = m_i[row_start + local_row];
        float m_new = m_old;
        
        // Find max
        for (int n_idx = 0; n_idx < Traits::BLOCK_N; n_idx++) {
            m_new = fmaxf(m_new, S_row[n_idx]);
        }
        
        // Compute correction factor
        float correction = (m_old == -INFINITY) ? 1.0f : expf(m_old - m_new);
        
        // Apply correction to existing O accumulator
        for (int d = 0; d < Traits::HEAD_DIM; d++) {
            O_acc[row_start + local_row][d] *= correction;
        }
        
        // Compute exp(S - m_new) and new sum
        float l_new = l_i[row_start + local_row] * correction;
        for (int n_idx = 0; n_idx < Traits::BLOCK_N; n_idx++) {
            if (S_row[n_idx] > -INFINITY) {
                S_row[n_idx] = expf(S_row[n_idx] - m_new);
                l_new += S_row[n_idx];
            } else {
                S_row[n_idx] = 0.0f;
            }
        }
        
        // Accumulate O += S @ V
        for (int d = 0; d < Traits::HEAD_DIM; d++) {
            float acc = 0.0f;
            for (int n_idx = 0; n_idx < Traits::BLOCK_N; n_idx++) {
                acc += S_row[n_idx] * __half2float(smem->V[stage][n_idx][d]);
            }
            O_acc[row_start + local_row][d] += acc;
        }
        
        // Update running stats
        m_i[row_start + local_row] = m_new;
        l_i[row_start + local_row] = l_new;
    }
}

// ============================================================================
// Write O to GMEM (final normalization)
// ============================================================================

template<typename Traits>
__device__ void write_O_to_gmem(
    half* __restrict__ O_gmem,
    float O_acc[Traits::BLOCK_M][Traits::HEAD_DIM],
    float l_i[Traits::BLOCK_M],
    int batch_idx,
    int head_idx,
    int m_block,
    int seq_len,
    int num_heads
) {
    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;
    const int rows_per_warp = Traits::BLOCK_M / Traits::NUM_WARPS;
    const int row_start = warp_id * rows_per_warp;
    
    for (int local_row = 0; local_row < rows_per_warp; local_row++) {
        const int m = m_block * Traits::BLOCK_M + row_start + local_row;
        if (m >= seq_len) continue;
        
        const int offset = batch_idx * seq_len * num_heads * Traits::HEAD_DIM +
                          head_idx * Traits::HEAD_DIM +
                          m * num_heads * Traits::HEAD_DIM;
        
        const float norm = 1.0f / l_i[row_start + local_row];
        
        // Vectorized write if HALF2 enabled
        if constexpr (Traits::HALF2) {
            for (int d = lane_id * 2; d < Traits::HEAD_DIM; d += 64) {
                half2 val;
                val.x = __float2half(O_acc[row_start + local_row][d] * norm);
                val.y = __float2half(O_acc[row_start + local_row][d + 1] * norm);
                *reinterpret_cast<half2*>(O_gmem + offset + d) = val;
            }
        } else {
            for (int d = lane_id; d < Traits::HEAD_DIM; d += 32) {
                O_gmem[offset + d] = __float2half(O_acc[row_start + local_row][d] * norm);
            }
        }
    }
}

// ============================================================================
// Main Kernel (Persistent Blocks)
// ============================================================================

template<typename Traits>
__global__ void __launch_bounds__(Traits::NUM_THREADS, 2)
flash_attention_s512_v3_kernel(
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
    // Persistent block: iterate over (batch, head) pairs
    const int block_id = blockIdx.x;
    const int num_blocks_m = (seq_len + Traits::BLOCK_M - 1) / Traits::BLOCK_M;
    const int total_blocks = batch_size * num_heads * num_blocks_m;
    
    __shared__ SharedMemory<Traits> smem;
    
    // Register storage (per-warp rows)
    const int rows_per_warp = Traits::BLOCK_M / Traits::NUM_WARPS;
    half Q_reg[Traits::BLOCK_M / Traits::NUM_WARPS][Traits::HEAD_DIM];
    float O_acc[Traits::BLOCK_M / Traits::NUM_WARPS][Traits::HEAD_DIM];
    float m_i[Traits::BLOCK_M / Traits::NUM_WARPS];
    float l_i[Traits::BLOCK_M / Traits::NUM_WARPS];
    
    for (int work_id = block_id; work_id < total_blocks; work_id += gridDim.x) {
        // Decode work_id
        const int m_block = work_id % num_blocks_m;
        const int temp = work_id / num_blocks_m;
        const int head_idx = temp % num_heads;
        const int batch_idx = temp / num_heads;
        
        // Debug: Check work distribution bounds
        #ifdef DEBUG_V3
        CUDA_DEBUG_ASSERT(batch_idx >= 0 && batch_idx < batch_size);
        CUDA_DEBUG_ASSERT(head_idx >= 0 && head_idx < num_heads);
        CUDA_DEBUG_ASSERT(m_block >= 0 && m_block < num_blocks_m);
        #endif
        
        // Initialize accumulators
        for (int i = 0; i < rows_per_warp; i++) {
            m_i[i] = -INFINITY;
            l_i[i] = 0.0f;
            for (int d = 0; d < Traits::HEAD_DIM; d++) {
                O_acc[i][d] = 0.0f;
            }
        }
        
        // Load Q into registers once
        load_Q_to_registers<Traits>(Q_reg, Q, batch_idx, head_idx, m_block, seq_len, num_heads);
        __syncthreads();
        
        // Iterate over K, V tiles with pipelining
        const int num_blocks_n = (seq_len + Traits::BLOCK_N - 1) / Traits::BLOCK_N;
        
        // Prefetch first stage
        load_K_async<Traits>(&smem, K, 0, batch_idx, head_idx, 0, seq_len, num_heads);
        load_V_async<Traits>(&smem, V, 0, batch_idx, head_idx, 0, seq_len, num_heads);
        detail::cp_async_commit_group();
        
        for (int n_block = 0; n_block < num_blocks_n; n_block++) {
            const int stage_compute = n_block % Traits::STAGES;
            const int stage_load = (n_block + 1) % Traits::STAGES;
            
            // Wait for current stage
            detail::cp_async_wait_group<Traits::STAGES - 1>();
            __syncthreads();
            
            // Prefetch next stage (if exists)
            if (n_block + 1 < num_blocks_n) {
                load_K_async<Traits>(&smem, K, stage_load, batch_idx, head_idx, n_block + 1, seq_len, num_heads);
                load_V_async<Traits>(&smem, V, stage_load, batch_idx, head_idx, n_block + 1, seq_len, num_heads);
                detail::cp_async_commit_group();
            }
            
            // Compute on current stage
            compute_block<Traits>(
                Q_reg, &smem, O_acc, m_i, l_i,
                stage_compute, softmax_scale, is_causal,
                m_block, n_block, seq_len
            );
        }
        
        // Wait for all async copies
        detail::cp_async_wait_all();
        __syncthreads();
        
        // Write final O to GMEM
        write_O_to_gmem<Traits>(O, O_acc, l_i, batch_idx, head_idx, m_block, seq_len, num_heads);
    }
}

// ============================================================================
// Host Launch Function
// ============================================================================

template<typename Traits>
cudaError_t launch_fa_s512_v3(
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
    // Validate SMEM usage
    constexpr size_t smem_size = smem_bytes<Traits>();
    static_assert(smem_size <= 49152, "SMEM exceeds 48KB limit");
    
    // Persistent block configuration
    const int num_blocks_m = (seq_len + Traits::BLOCK_M - 1) / Traits::BLOCK_M;
    const int total_work = batch_size * num_heads * num_blocks_m;
    const int num_blocks = min(total_work, 256);  // SM count × occupancy
    
    flash_attention_s512_v3_kernel<Traits><<<num_blocks, Traits::NUM_THREADS, 0, stream>>>(
        Q, K, V, O, softmax_scale, batch_size, num_heads, seq_len, is_causal
    );
    
    return cudaGetLastError();
}

// ============================================================================
// Template Instantiations (Pre-compile promising configs)
// ============================================================================

// Config 1: BLOCK_M=32, BLOCK_N=64, WARPS=4, STAGES=2, SWIZZLE=1, HALF2=1
// FIX: Changed WARPS from 6 to 4 (32 % 6 != 0, but 32 % 4 == 0)
using Traits_32_64_4_2_1_1 = KernelTraits<32, 64, 64, 4, 2, true, true>;

extern "C" cudaError_t launch_fa_s512_v3_32_64_4_2_1_1(
    const half* Q, const half* K, const half* V, half* O,
    float softmax_scale, int B, int H, int S, bool is_causal, cudaStream_t stream
) {
    return launch_fa_s512_v3<Traits_32_64_4_2_1_1>(
        Q, K, V, O, softmax_scale, B, H, S, is_causal, stream
    );
}

// Config 2: BLOCK_M=32, BLOCK_N=32, WARPS=4, STAGES=2, SWIZZLE=1, HALF2=1
// FIX: Changed WARPS from 6 to 4 (32 % 6 != 0, but 32 % 4 == 0)
using Traits_32_32_4_2_1_1 = KernelTraits<32, 32, 64, 4, 2, true, true>;

extern "C" cudaError_t launch_fa_s512_v3_32_32_4_2_1_1(
    const half* Q, const half* K, const half* V, half* O,
    float softmax_scale, int B, int H, int S, bool is_causal, cudaStream_t stream
) {
    return launch_fa_s512_v3<Traits_32_32_4_2_1_1>(
        Q, K, V, O, softmax_scale, B, H, S, is_causal, stream
    );
}

// Config 3: BLOCK_M=48, BLOCK_N=64, WARPS=8, STAGES=2, SWIZZLE=1, HALF2=1
using Traits_48_64_8_2_1_1 = KernelTraits<48, 64, 64, 8, 2, true, true>;

extern "C" cudaError_t launch_fa_s512_v3_48_64_8_2_1_1(
    const half* Q, const half* K, const half* V, half* O,
    float softmax_scale, int B, int H, int S, bool is_causal, cudaStream_t stream
) {
    return launch_fa_s512_v3<Traits_48_64_8_2_1_1>(
        Q, K, V, O, softmax_scale, B, H, S, is_causal, stream
    );
}

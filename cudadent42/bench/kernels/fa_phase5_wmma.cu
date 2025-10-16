// ============================================================================
// PHASE 5: TENSOR CORE (WMMA) IMPLEMENTATION
// ============================================================================
// Target: 200-300 μs (5-10× speedup from Phase 4)
//
// Key Optimizations:
// 1. WMMA for Q@K^T (16x16x16 matrix multiply on Tensor Cores)
// 2. WMMA for P@V (16x16x16 matrix multiply on Tensor Cores)
// 3. FP16 WMMA operations (2× throughput on Ada/L4 via FP16 accumulation)
// 4. Proper tile sizes for Tensor Core utilization
// 5. Retains Phase 4 light-barrier path (4 syncs/tile)
//
// Design:
// - Each warp handles 16x16 tile using WMMA
// - BLOCK_M=32, BLOCK_N=64, 4 warps per block
// - 16x16x16 WMMA tiles for Q@K^T and P@V
// - Online softmax between WMMA operations
//
// Ada (sm_89) specific:
// - FP16 accumulation for 2× throughput
// - Requires: -gencode=arch=compute_89,code=sm_89
//
// Correctness: MUST pass torch.allclose(atol=1e-3) with Tensor Cores
// ============================================================================

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <float.h>
#include <stdio.h>

// Only include WMMA headers if USE_WMMA is enabled
#ifndef USE_WMMA
#define USE_WMMA 0  // Default: scalar fallback
#endif

#if USE_WMMA
#include <mma.h>
using namespace nv::wmma;
#endif

constexpr int HEAD_DIM = 64;

// Tunable parameters (can be overridden via -D flags)
#ifndef BLOCK_M
constexpr int BLOCK_M = 32;      // Query rows per block
#endif
#ifndef NUM_WARPS
constexpr int NUM_WARPS_DEFAULT = 4;
#endif

constexpr int BLOCK_N = 64;      // KV tile size
constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;

#ifdef NUM_WARPS
constexpr int THREADS = NUM_WARPS * 32;
#else
constexpr int THREADS = 128;
#endif

// ============================================================================
// PHASE 4: LIGHT-BARRIER PATH (2 syncs/tile instead of 5)
// ============================================================================

#ifndef SYNC_POLICY
// 0: dev (no extra barriers), 2: target (2/tile), 5: legacy (heavy)
#define SYNC_POLICY 2
#endif

// Barrier helper
__device__ __forceinline__ void cta_barrier() { 
    __syncthreads(); 
}

// Warp-level reductions (deterministic within a warp)
__device__ __forceinline__ float warp_max(float x) {
    #pragma unroll
    for (int d = 16; d > 0; d >>= 1) {
        x = fmaxf(x, __shfl_down_sync(0xffffffff, x, d));
    }
    return x;
}

__device__ __forceinline__ float warp_sum(float x) {
    #pragma unroll
    for (int d = 16; d > 0; d >>= 1) {
        x += __shfl_down_sync(0xffffffff, x, d);
    }
    return x;
}

// XOR swizzle helper for SMEM bank conflict reduction
#ifndef SWIZZLE_XOR
#define SWIZZLE_XOR 0
#endif

__device__ __forceinline__ int swz(int col) {
    #if SWIZZLE_XOR
    // 32-bank friendly pattern for D=64
    return (col ^ ((col >> 5) & 0x1)) & 63;
    #else
    return col;
    #endif
}

// ============================================================================
// PHASE 5: WMMA (Tensor Core) Infrastructure
// ============================================================================

#if USE_WMMA

// WMMA Fragment Types for Ada (sm_89)
// For Q@K^T: Q (MxK) @ K^T (KxN) = S (MxN)
// Q: row_major, K: col_major (for K^T)
using QFragment = fragment<matrix_a, 16, 16, 16, half, row_major>;
using KFragment = fragment<matrix_b, 16, 16, 16, half, col_major>;

// For P@V: P (MxK) @ V (KxN) = O (MxN)
// P: row_major (scores after softmax), V: row_major
using PFragment = fragment<matrix_a, 16, 16, 16, half, row_major>;
using VFragment = fragment<matrix_b, 16, 16, 16, half, row_major>;

// Accumulator fragments
// FP32 for numerical stability during reduction
using AccumFragment = fragment<accumulator, 16, 16, 16, float>;

// FP16 accumulation for Ada (2× throughput) - use for final O accumulation
#ifdef USE_FP16_ACCUM
using OAccumFragment = fragment<accumulator, 16, 16, 16, half>;
#else
using OAccumFragment = fragment<accumulator, 16, 16, 16, float>;
#endif

// Helper: Compute Q@K^T using WMMA (16x16x16 tiles)
// Each warp computes one 16x16 output tile
__device__ __forceinline__ void wmma_qk_transpose(
    const half* Q_tile,   // [BLOCK_M][HEAD_DIM] in SMEM
    const half* K_tile,   // [BLOCK_N][HEAD_DIM] in SMEM
    float* S_tile,        // [BLOCK_M][BLOCK_N] output in SMEM
    int m_start,          // Starting row in Q (0 or 16 for BLOCK_M=32)
    int n_start,          // Starting col in K (0, 16, 32, 48 for BLOCK_N=64)
    float scale
) {
    QFragment q_frag;
    KFragment k_frag;
    AccumFragment acc_frag;
    
    fill_fragment(acc_frag, 0.0f);
    
    // Accumulate over HEAD_DIM in 16-element chunks
    #pragma unroll
    for (int k = 0; k < HEAD_DIM; k += 16) {
        // Load Q tile (16x16 from Q[m_start:m_start+16, k:k+16])
        load_matrix_sync(q_frag, Q_tile + m_start * HEAD_DIM + k, HEAD_DIM);
        
        // Load K tile for K^T (16x16 from K[n_start:n_start+16, k:k+16])
        // Note: using col_major layout interprets this as K^T
        load_matrix_sync(k_frag, K_tile + n_start * HEAD_DIM + k, HEAD_DIM);
        
        // Accumulate: acc += Q @ K^T
        mma_sync(acc_frag, q_frag, k_frag, acc_frag);
    }
    
    // Scale and store to SMEM
    #pragma unroll
    for (int i = 0; i < acc_frag.num_elements; i++) {
        acc_frag.x[i] *= scale;
    }
    
    store_matrix_sync(S_tile + m_start * BLOCK_N + n_start, acc_frag, BLOCK_N, mem_row_major);
}

// Helper: Compute P@V using WMMA (16x16x16 tiles)
__device__ __forceinline__ void wmma_pv(
    const float* P_tile,  // [BLOCK_M][BLOCK_N] attention weights in SMEM
    const half* V_tile,   // [BLOCK_N][HEAD_DIM] in SMEM
    float* O_accum,       // [BLOCK_M][HEAD_DIM] accumulator in SMEM
    int m_start,          // Starting row (0 or 16)
    int d_start           // Starting col in HEAD_DIM (0, 16, 32, 48)
) {
    PFragment p_frag;
    VFragment v_frag;
    OAccumFragment o_frag;
    
    fill_fragment(o_frag, 0.0f);
    
    // Accumulate over BLOCK_N in 16-element chunks
    #pragma unroll
    for (int n = 0; n < BLOCK_N; n += 16) {
        // Load P tile (16x16 from P[m_start:m_start+16, n:n+16])
        // Need to convert float P to half for WMMA
        half P_half[16*16];
        #pragma unroll
        for (int i = 0; i < 16; i++) {
            #pragma unroll
            for (int j = 0; j < 16; j++) {
                P_half[i*16 + j] = __float2half(P_tile[(m_start + i) * BLOCK_N + (n + j)]);
            }
        }
        load_matrix_sync(p_frag, P_half, 16);
        
        // Load V tile (16x16 from V[n:n+16, d_start:d_start+16])
        load_matrix_sync(v_frag, V_tile + n * HEAD_DIM + d_start, HEAD_DIM);
        
        // Accumulate: o += P @ V
        mma_sync(o_frag, p_frag, v_frag, o_frag);
    }
    
    // Convert accumulator to float and add to O_accum
    #ifdef USE_FP16_ACCUM
    // Convert FP16 accumulator to FP32 for addition
    half o_half[16*16];
    store_matrix_sync(o_half, o_frag, 16, mem_row_major);
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        #pragma unroll
        for (int j = 0; j < 16; j++) {
            O_accum[(m_start + i) * HEAD_DIM + (d_start + j)] += __half2float(o_half[i*16 + j]);
        }
    }
    #else
    // Store FP32 accumulator directly
    float o_float[16*16];
    store_matrix_sync(o_float, o_frag, 16, mem_row_major);
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        #pragma unroll
        for (int j = 0; j < 16; j++) {
            O_accum[(m_start + i) * HEAD_DIM + (d_start + j)] += o_float[i*16 + j];
        }
    }
    #endif
}

#endif  // USE_WMMA

// ============================================================================
// PHASE 5 KERNEL: Tensor Cores with WMMA
// ============================================================================

__global__ void flash_attention_phase5_kernel(
    const half* __restrict__ Q,
    const half* __restrict__ K,
    const half* __restrict__ V,
    half* __restrict__ O,
    float softmax_scale,
    int batch_size,
    int num_heads,
    int seq_len
) {
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
    
    // Shared memory for tiles
    __shared__ half Q_tile[BLOCK_M][HEAD_DIM];
    __shared__ half K_tile[BLOCK_N][HEAD_DIM];
    __shared__ half V_tile[BLOCK_N][HEAD_DIM];
    __shared__ float S_tile[BLOCK_M][BLOCK_N];
    __shared__ float O_accum[BLOCK_M][HEAD_DIM];
    
    // Load Q tile
    for (int row = warp_id; row < rows_this_block; row += (THREADS / 32)) {
        for (int d = lane_id; d < HEAD_DIM; d += 32) {
            const int q_idx = q_start + row;
            const int offset = batch_idx * num_heads * seq_len * HEAD_DIM +
                              head_idx * seq_len * HEAD_DIM +
                              q_idx * HEAD_DIM + d;
            Q_tile[row][d] = Q[offset];
        }
    }
    
    // Initialize O_accum
    for (int row = warp_id; row < BLOCK_M; row += (THREADS / 32)) {
        for (int d = lane_id; d < HEAD_DIM; d += 32) {
            O_accum[row][d] = 0.0f;
        }
    }
    __syncthreads();
    
    // Softmax state (per-row, per-warp)
    float m_i[BLOCK_M / (THREADS / 32) + 1];
    float l_i[BLOCK_M / (THREADS / 32) + 1];
    for (int i = 0; i < BLOCK_M / (THREADS / 32) + 1; i++) {
        m_i[i] = -FLT_MAX;
        l_i[i] = 0.0f;
    }
    
    const int num_blocks_n = (seq_len + BLOCK_N - 1) / BLOCK_N;
    
    // Loop over KV tiles
    for (int n_block = 0; n_block < num_blocks_n; n_block++) {
        const int kv_start = n_block * BLOCK_N;
        const int kv_end = min(kv_start + BLOCK_N, seq_len);
        const int kv_size = kv_end - kv_start;
        
        // Load K and V tiles
        for (int row = warp_id; row < kv_size; row += (THREADS / 32)) {
#if defined(VEC_WIDTH) && (VEC_WIDTH >= 4)
            // VECTORIZED LOADS (Priority 1 optimization)
            // Use uint4 for 8×FP16 = 16 bytes per load
            for (int d = lane_id * 8; d + 8 <= HEAD_DIM; d += 32 * 8) {
                const int kv_idx = kv_start + row;
                const int k_offset = batch_idx * num_heads * seq_len * HEAD_DIM +
                                    head_idx * seq_len * HEAD_DIM +
                                    kv_idx * HEAD_DIM + d;
                
                // Load 8 FP16 values (16 bytes) in one instruction
                uint4 k_vec = *reinterpret_cast<const uint4*>(&K[k_offset]);
                uint4 v_vec = *reinterpret_cast<const uint4*>(&V[k_offset]);
                
                // Store to shared memory
                *reinterpret_cast<uint4*>(&K_tile[row][d]) = k_vec;
                *reinterpret_cast<uint4*>(&V_tile[row][d]) = v_vec;
            }
            // Handle remainder (if HEAD_DIM not divisible by 8)
            for (int d = lane_id + (HEAD_DIM / 8) * 8; d < HEAD_DIM; d += 32) {
                const int kv_idx = kv_start + row;
                const int k_offset = batch_idx * num_heads * seq_len * HEAD_DIM +
                                    head_idx * seq_len * HEAD_DIM +
                                    kv_idx * HEAD_DIM + d;
                K_tile[row][d] = K[k_offset];
                V_tile[row][d] = V[k_offset];
            }
#else
            // SCALAR LOADS (fallback - proven correct)
            for (int d = lane_id; d < HEAD_DIM; d += 32) {
                const int kv_idx = kv_start + row;
                const int k_offset = batch_idx * num_heads * seq_len * HEAD_DIM +
                                    head_idx * seq_len * HEAD_DIM +
                                    kv_idx * HEAD_DIM + d;
                K_tile[row][d] = K[k_offset];
                V_tile[row][d] = V[k_offset];
            }
#endif
        }
        
        // Barrier 1: After K/V load (required for shared memory correctness)
        #if SYNC_POLICY >= 1
        cta_barrier();
        #endif
        
        // Compute S = Q @ K^T 
        #if USE_WMMA
        // ================================================================
        // WMMA PATH: Tensor Core acceleration for Q@K^T
        // ================================================================
        // BLOCK_M=32, BLOCK_N=64 → 2×4 = 8 tiles of 16×16
        // 4 warps → each warp computes 2 tiles sequentially
        //
        // Target: 500 μs (scalar) → 100 μs (5× speedup)
        // ================================================================
        
        // Each warp computes 2 tiles of the output
        // Warp layout:
        //   Warp 0: tiles (0,0), (0,16)
        //   Warp 1: tiles (0,32), (0,48)
        //   Warp 2: tiles (16,0), (16,16)
        //   Warp 3: tiles (16,32), (16,48)
        
        if (warp_id < 4) {
            const int m_base = (warp_id / 2) * 16;  // 0 or 16
            const int n_base = (warp_id % 2) * 32;  // 0 or 32
            
            // Compute 2 tiles for this warp
            for (int n_offset = 0; n_offset < 2; n_offset++) {
                const int m_start = m_base;
                const int n_start = n_base + (n_offset * 16);
                
                // Only compute if within bounds
                if (m_start < rows_this_block && n_start < kv_size) {
                    wmma_qk_transpose(
                        (const half*)Q_tile[0], 
                        (const half*)K_tile[0], 
                        (float*)S_tile[0], 
                        m_start, 
                        n_start, 
                        softmax_scale
                    );
                }
            }
        }
        
        // Sync after all warps finish Q@K^T computation
        cta_barrier();
        
        #else
        // ================================================================
        // SCALAR PATH: Proven-correct fallback
        // ================================================================
        for (int row = tid; row < rows_this_block; row += THREADS) {
            for (int col = 0; col < kv_size; col++) {
                float score = 0.0f;
                for (int d = 0; d < HEAD_DIM; d++) {
                    score += __half2float(Q_tile[row][d]) * __half2float(K_tile[col][d]);
                }
                S_tile[row][col] = score * softmax_scale;
            }
        }
        #endif
        
        // Light-barrier path: No sync here (warp-synchronous softmax below)
        #if SYNC_POLICY >= 5
        cta_barrier();  // Legacy: sync after S computation
        #endif
        
        // Online softmax (simple version for correctness)
        for (int row = warp_id; row < rows_this_block; row += (THREADS / 32)) {
            const int local_row = row / (THREADS / 32);
            
            // Find max
            __shared__ float m_new_shared[THREADS / 32];
            float m_new;
            
#if defined(REDUCE_WARP) && (REDUCE_WARP == 1)
            // WARP-LEVEL REDUCTION (Priority 1 optimization)
            m_new = m_i[local_row];
            for (int col = lane_id; col < kv_size; col += 32) {
                m_new = fmaxf(m_new, S_tile[row][col]);
            }
            // Warp reduce
            for (int offset = 16; offset > 0; offset /= 2) {
                m_new = fmaxf(m_new, __shfl_down_sync(0xffffffff, m_new, offset));
            }
            if (lane_id == 0) {
                m_new_shared[warp_id] = m_new;
            }
#else
            // SERIAL REDUCTION (fallback - proven correct)
            if (lane_id == 0) {
                m_new = m_i[local_row];
                for (int col = 0; col < kv_size; col++) {
                    m_new = fmaxf(m_new, S_tile[row][col]);
                }
                m_new_shared[warp_id] = m_new;
            }
#endif
            // Need sync: shared memory write/read dependency
            #if SYNC_POLICY >= 2
            cta_barrier();  // Required: m_new_shared is shared across warps
            #endif
            m_new = m_new_shared[warp_id];
            
            // Correction
            float correction = expf(m_i[local_row] - m_new);
            
            // Apply to O_accum
            for (int d = lane_id; d < HEAD_DIM; d += 32) {
                O_accum[row][d] *= correction;
            }
            
            // Compute new l_i
            __shared__ float l_new_shared[THREADS / 32];
            float l_new;
            
#if defined(REDUCE_WARP) && (REDUCE_WARP == 1)
            // WARP-LEVEL REDUCTION (Priority 1 optimization)
            l_new = (lane_id == 0) ? (l_i[local_row] * correction) : 0.0f;
            for (int col = lane_id; col < kv_size; col += 32) {
                l_new += expf(S_tile[row][col] - m_new);
            }
            // Warp reduce (sum)
            for (int offset = 16; offset > 0; offset /= 2) {
                l_new += __shfl_down_sync(0xffffffff, l_new, offset);
            }
            if (lane_id == 0) {
                l_new_shared[warp_id] = l_new;
            }
#else
            // SERIAL REDUCTION (fallback - proven correct)
            if (lane_id == 0) {
                l_new = l_i[local_row] * correction;
                for (int col = 0; col < kv_size; col++) {
                    l_new += expf(S_tile[row][col] - m_new);
                }
                l_new_shared[warp_id] = l_new;
            }
#endif
            // Need sync: shared memory write/read dependency
            #if SYNC_POLICY >= 2
            cta_barrier();  // Required: l_new_shared is shared across warps
            #endif
            l_new = l_new_shared[warp_id];
            
            // Accumulate O += P @ V
            for (int d = lane_id; d < HEAD_DIM; d += 32) {
                float acc = 0.0f;
                for (int col = 0; col < kv_size; col++) {
                    float p = expf(S_tile[row][col] - m_new);
                    acc += p * __half2float(V_tile[col][d]);
                }
                O_accum[row][d] += acc;
            }
            
            // Update state (warp-local, no sync needed)
            m_i[local_row] = m_new;
            l_i[local_row] = l_new;
        }
        
        // Barrier 2: Before next tile (required for shared memory reuse)
        #if SYNC_POLICY >= 2
        cta_barrier();
        #endif
    }
    
    // Write output
    for (int row = warp_id; row < rows_this_block; row += (THREADS / 32)) {
        const int local_row = row / (THREADS / 32);
        const float norm = (l_i[local_row] > 0.0f) ? (1.0f / l_i[local_row]) : 0.0f;
        
        for (int d = lane_id; d < HEAD_DIM; d += 32) {
            const int q_idx = q_start + row;
            const int offset = batch_idx * num_heads * seq_len * HEAD_DIM +
                              head_idx * seq_len * HEAD_DIM +
                              q_idx * HEAD_DIM + d;
            O[offset] = __float2half(O_accum[row][d] * norm);
        }
    }
}

// ============================================================================
// LAUNCH FUNCTION
// ============================================================================

extern "C" void launch_flash_attention_phase5(
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
    
    flash_attention_phase5_kernel<<<grid, block, 0, stream>>>(
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


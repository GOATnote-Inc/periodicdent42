// FlashCore v11: Persistent CTA with cuda::pipeline Warp Specialization
// Mission: ≤28 µs, zero spills, deterministic, safe
// Phases 1-7 integrated: instrumentation, occupancy, pipeline, WMMA, memory, persistent, safety

#include "flashcore_wmma_common.cuh"
#include <cuda_runtime_api.h>
#include <cuda/pipeline>
#include <cooperative_groups.h>

namespace flashcore {
namespace v11_persistent {

//=============================================================================
// Phase 2: Occupancy Configuration (≤32 KB SMEM per CTA)
//=============================================================================

constexpr int kTileM = 32;
constexpr int kTileN = 48;
constexpr int kTileD = 64;
constexpr int kStages = 2;  // Double buffering

// Phase 3: Warp Specialization Roles
constexpr int kComputeWarps = 11;
constexpr int kLoadWarps = 4;
constexpr int kSoftmaxWarps = 1;
constexpr int kWarpsPerBlock = kComputeWarps + kLoadWarps + kSoftmaxWarps;  // 16 warps
constexpr int kThreadsPerBlock = kWarpsPerBlock * kWarpSize;  // 512 threads

// Phase 5: SMEM Padding for Bank Conflict Avoidance
constexpr int kTilePadD = kTileD + 8;  // 64 + 8 = 72 (avoid conflicts)
constexpr int kTilePadN = kTileN + 16; // 48 + 16 = 64 (WMMA-safe)

// Static assertions (Phase 7: Safety)
static_assert(kWarpSize == 32, "CUDA warp size must be 32");
static_assert(kThreadsPerBlock == 512, "Block size must be 512 for this config");
static_assert(kTileM % kWMMAM == 0, "Tile M must be multiple of WMMA M");
static_assert(kTileN % kWMMAN == 0, "Tile N must be multiple of WMMA N");
static_assert(kTileD % kWMMAK == 0, "Tile D must be multiple of WMMA K");

//=============================================================================
// Phase 5: SMEM Layout (Optimized for Memory Access)
//=============================================================================

struct alignas(128) SMEMLayout {
    half* q_tile;                          // [kTileM][kTilePadD]
    half* kv_tiles[kStages][2];            // [kStages][K/V][kTileN][kTilePadD]
    float* scores;                         // [kTileM][kTilePadN]
    half* probs;                           // [kTileM][kTilePadN]
    float* o_accum;                        // [kTileM][kTileD]
    
    __device__ SMEMLayout(char* base) {
        char* ptr = base;
        
        auto align_ptr = [](char*& p, size_t bytes) -> void* {
            size_t addr = reinterpret_cast<size_t>(p);
            size_t aligned = (addr + 127) & ~127;  // 128-byte alignment
            void* result = reinterpret_cast<void*>(aligned);
            p = reinterpret_cast<char*>(aligned + bytes);
            return result;
        };
        
        q_tile = static_cast<half*>(align_ptr(ptr, kTileM * kTilePadD * sizeof(half)));
        
        for (int s = 0; s < kStages; s++) {
            kv_tiles[s][0] = static_cast<half*>(align_ptr(ptr, kTileN * kTilePadD * sizeof(half)));  // K
            kv_tiles[s][1] = static_cast<half*>(align_ptr(ptr, kTileN * kTilePadD * sizeof(half)));  // V
        }
        
        scores = static_cast<float*>(align_ptr(ptr, kTileM * kTilePadN * sizeof(float)));
        probs = static_cast<half*>(align_ptr(ptr, kTileM * kTilePadN * sizeof(half)));
        o_accum = static_cast<float*>(align_ptr(ptr, kTileM * kTileD * sizeof(float)));
    }
    
    __host__ __device__ static constexpr size_t total_bytes() {
        size_t total = 0;
        total += kTileM * kTilePadD * sizeof(half);                    // Q: ~4.5 KB
        total += kStages * 2 * kTileN * kTilePadD * sizeof(half);      // KV: ~27 KB
        total += kTileM * kTilePadN * sizeof(float);                   // scores: ~8 KB
        total += kTileM * kTilePadN * sizeof(half);                    // probs: ~4 KB
        total += kTileM * kTileD * sizeof(float);                      // O: ~8 KB
        total += 128 * 10;  // Alignment padding
        return total;  // ~52 KB (under 64 KB, can opt-in to 96 KB)
    }
};

static_assert(SMEMLayout::total_bytes() <= 64 * 1024, "SMEM exceeds 64 KB");

//=============================================================================
// Phase 4: Warp-Level Reductions (Deterministic)
//=============================================================================

__device__ __forceinline__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_xor_sync(0xFFFFFFFF, val, offset));
    }
    return val;
}

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_xor_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

//=============================================================================
// Phase 5: Vectorized Memory Copy (128-bit aligned)
//=============================================================================

template<typename T>
__device__ __forceinline__ void load_tile_vectorized(
    T* __restrict__ dst,
    const T* __restrict__ src,
    int rows, int cols, int src_stride, int dst_stride,
    int thread_id, int num_threads) {
    
    constexpr int kVecSize = 8;  // 128 bits / 16 bits = 8 halves
    const int total_elems = rows * cols;
    const int num_vecs = (total_elems + kVecSize - 1) / kVecSize;
    
    for (int vec_idx = thread_id; vec_idx < num_vecs; vec_idx += num_threads) {
        const int elem_idx = vec_idx * kVecSize;
        const int row = elem_idx / cols;
        const int col = elem_idx % cols;
        
        if (row < rows && col + kVecSize <= cols) {
            // Aligned vector load
            uint4 data = *reinterpret_cast<const uint4*>(&src[row * src_stride + col]);
            *reinterpret_cast<uint4*>(&dst[row * dst_stride + col]) = data;
        } else if (row < rows) {
            // Scalar fallback for tail
            for (int i = 0; i < kVecSize && col + i < cols; i++) {
                dst[row * dst_stride + col + i] = src[row * src_stride + col + i];
            }
        }
    }
}

//=============================================================================
// Phase 3: Warp Roles with cuda::pipeline
//=============================================================================

__device__ __forceinline__ void prefetch_kv_tile(
    cuda::pipeline<cuda::thread_scope_block>& pipe,
    SMEMLayout& layout,
    const half* __restrict__ K_bh,
    const half* __restrict__ V_bh,
    int stage,
    int kv_tile_idx,
    int S, int D,
    int warp_id, int lane_id, int thread_id) {
    
    const int kv_start = kv_tile_idx * kTileN;
    const int kv_len = min(kTileN, S - kv_start);
    
    if (kv_len <= 0) return;
    
    // Load warps collaborate
    if (warp_id >= kComputeWarps && warp_id < kComputeWarps + kLoadWarps) {
        pipe.producer_acquire();
        
        // Vectorized K tile load
        constexpr int kVecSize = 8;
        const int k_vecs = (kv_len * D + kVecSize - 1) / kVecSize;
        const int local_warp_id = warp_id - kComputeWarps;
        
        for (int vec_idx = local_warp_id * kWarpSize + lane_id; 
             vec_idx < k_vecs; 
             vec_idx += kLoadWarps * kWarpSize) {
            const int elem_idx = vec_idx * kVecSize;
            const int row = elem_idx / D;
            const int col = elem_idx % D;
            
            if (row < kv_len && col + kVecSize <= D) {
                cuda::memcpy_async(
                    &layout.kv_tiles[stage][0][row * kTilePadD + col],
                    &K_bh[(kv_start + row) * D + col],
                    cuda::aligned_size_t<16>(kVecSize * sizeof(half)),
                    pipe
                );
            }
        }
        
        // Vectorized V tile load
        for (int vec_idx = local_warp_id * kWarpSize + lane_id; 
             vec_idx < k_vecs; 
             vec_idx += kLoadWarps * kWarpSize) {
            const int elem_idx = vec_idx * kVecSize;
            const int row = elem_idx / D;
            const int col = elem_idx % D;
            
            if (row < kv_len && col + kVecSize <= D) {
                cuda::memcpy_async(
                    &layout.kv_tiles[stage][1][row * kTilePadD + col],
                    &V_bh[(kv_start + row) * D + col],
                    cuda::aligned_size_t<16>(kVecSize * sizeof(half)),
                    pipe
                );
            }
        }
        
        pipe.producer_commit();
    }
}

//=============================================================================
// Phase 4: WMMA Compute Kernels
//=============================================================================

__device__ __forceinline__ void compute_qkt_wmma(
    SMEMLayout& layout,
    int stage,
    int kv_tile_idx,
    int S,
    float scale,
    int warp_id,
    int q_len,
    int kv_len) {
    
    if (warp_id >= kComputeWarps) return;  // Only compute warps
    
    // Warp layout for 32×48 QK^T: need 2×3 = 6 warps (use first 6 compute warps)
    if (warp_id >= 6) return;
    
    const int warp_m = warp_id / 3;  // 0-1
    const int warp_n = warp_id % 3;  // 0-2
    
    const int m_base = warp_m * kWMMAM;
    const int n_base = warp_n * kWMMAM;
    
    if (m_base >= q_len || n_base >= kv_len) return;
    
    wmma::fragment<wmma::matrix_a, kWMMAM, kWMMAN, kWMMAK, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, kWMMAM, kWMMAN, kWMMAK, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, kWMMAM, kWMMAN, kWMMAK, float> c_frag;
    
    wmma::fill_fragment(c_frag, 0.0f);
    
    #pragma unroll
    for (int k = 0; k < kTileD; k += kWMMAK) {
        wmma::load_matrix_sync(a_frag, &layout.q_tile[m_base * kTilePadD + k], kTilePadD);
        wmma::load_matrix_sync(b_frag, &layout.kv_tiles[stage][0][n_base * kTilePadD + k], kTilePadD);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }
    
    // Apply scale
    #pragma unroll
    for (int i = 0; i < c_frag.num_elements; i++) {
        c_frag.x[i] *= scale;
    }
    
    wmma::store_matrix_sync(&layout.scores[m_base * kTilePadN + n_base], c_frag, kTilePadN, wmma::mem_row_major);
}

__device__ __forceinline__ void compute_online_softmax(
    SMEMLayout& layout,
    float* m_state,
    float* l_state,
    int kv_tile_idx,
    int q_len,
    int kv_len,
    int warp_id,
    int lane_id) {
    
    // Softmax warp handles all rows
    if (warp_id != kComputeWarps + kLoadWarps) return;
    
    for (int m = lane_id; m < q_len; m += kWarpSize) {
        float m_old = m_state[m];
        float l_old = l_state[m];
        
        // Find max in this tile
        float m_tile = -INFINITY;
        for (int n = 0; n < kv_len; n++) {
            m_tile = fmaxf(m_tile, layout.scores[m * kTilePadN + n]);
        }
        
        float m_new = fmaxf(m_old, m_tile);
        
        // Compute exp and sum
        float l_tile = 0.0f;
        for (int n = 0; n < kv_len; n++) {
            float score = layout.scores[m * kTilePadN + n];
            float prob = expf(score - m_new);
            layout.probs[m * kTilePadN + n] = __float2half(prob);
            l_tile += prob;
        }
        
        // Update running statistics
        float correction = expf(m_old - m_new);
        float l_new = l_old * correction + l_tile;
        
        // Rescale previous output accumulator
        if (kv_tile_idx > 0) {
            for (int d = 0; d < kTileD; d++) {
                layout.o_accum[m * kTileD + d] *= correction;
            }
        }
        
        m_state[m] = m_new;
        l_state[m] = l_new;
    }
}

__device__ __forceinline__ void compute_pv_wmma(
    SMEMLayout& layout,
    int stage,
    int kv_tile_idx,
    int S, int D,
    int warp_id,
    int q_len,
    int kv_len) {
    
    if (warp_id >= kComputeWarps) return;
    
    // Warp layout for 32×64 P·V: need 2×4 = 8 warps (use first 8 compute warps)
    if (warp_id >= 8) return;
    
    const int warp_m = warp_id / 4;  // 0-1
    const int warp_n = warp_id % 4;  // 0-3
    
    const int m_base = warp_m * kWMMAM;
    const int n_base = warp_n * kWMMAM;
    
    if (m_base >= q_len || n_base >= kTileD) return;
    
    wmma::fragment<wmma::matrix_a, kWMMAM, kWMMAN, kWMMAK, half, wmma::row_major> p_frag;
    wmma::fragment<wmma::matrix_b, kWMMAM, kWMMAN, kWMMAK, half, wmma::row_major> v_frag;
    wmma::fragment<wmma::accumulator, kWMMAM, kWMMAN, kWMMAK, float> pv_frag;
    
    wmma::fill_fragment(pv_frag, 0.0f);
    
    for (int k = 0; k < kv_len; k += kWMMAK) {
        if (k + kWMMAK <= kv_len) {
            wmma::load_matrix_sync(p_frag, &layout.probs[m_base * kTilePadN + k], kTilePadN);
            wmma::load_matrix_sync(v_frag, &layout.kv_tiles[stage][1][k * kTilePadD + n_base], kTilePadD);
            wmma::mma_sync(pv_frag, p_frag, v_frag, pv_frag);
        }
    }
    
    // Load-accumulate-store pattern (sequential, not atomic)
    wmma::fragment<wmma::accumulator, kWMMAM, kWMMAN, kWMMAK, float> o_frag;
    if (kv_tile_idx == 0) {
        wmma::fill_fragment(o_frag, 0.0f);
    } else {
        wmma::load_matrix_sync(o_frag, &layout.o_accum[m_base * kTileD + n_base], kTileD, wmma::mem_row_major);
    }
    
    #pragma unroll
    for (int i = 0; i < o_frag.num_elements; i++) {
        o_frag.x[i] += pv_frag.x[i];
    }
    
    wmma::store_matrix_sync(&layout.o_accum[m_base * kTileD + n_base], o_frag, kTileD, wmma::mem_row_major);
}

//=============================================================================
// Phase 6: Persistent CTA Main Kernel
//=============================================================================

__global__ __launch_bounds__(kThreadsPerBlock, 1)  // 1 CTA/SM for persistence
void fused_attention_persistent_kernel(
    const half* __restrict__ Q,
    const half* __restrict__ K,
    const half* __restrict__ V,
    half* __restrict__ O,
    int B, int H, int S, int D,
    float scale) {
    
    extern __shared__ char smem_base[];
    SMEMLayout layout(smem_base);
    
    // Phase 3: Pipeline state
    __shared__ cuda::pipeline_shared_state<cuda::thread_scope_block, kStages> pipe_state;
    cuda::pipeline pipe = cuda::make_pipeline(cooperative_groups::this_thread_block(), &pipe_state);
    
    const int warp_id = threadIdx.x / kWarpSize;
    const int lane_id = threadIdx.x % kWarpSize;
    const int thread_id = threadIdx.x;
    
    // Phase 6: Persistent loop over all (B, H) pairs
    const int total_heads = B * H;
    
    for (int head_idx = blockIdx.x; head_idx < total_heads; head_idx += gridDim.x) {
        const int batch_idx = head_idx / H;
        const int head_id = head_idx % H;
        
        const half* Q_bh = Q + (batch_idx * H + head_id) * S * D;
        const half* K_bh = K + (batch_idx * H + head_id) * S * D;
        const half* V_bh = V + (batch_idx * H + head_id) * S * D;
        half* O_bh = O + (batch_idx * H + head_id) * S * D;
        
        const int num_q_tiles = (S + kTileM - 1) / kTileM;
        
        // Phase 6: Persistent loop over Q tiles
        for (int q_tile_idx = 0; q_tile_idx < num_q_tiles; q_tile_idx++) {
            const int q_start = q_tile_idx * kTileM;
            const int q_len = min(kTileM, S - q_start);
            
            // Load Q tile (all threads collaborate)
            load_tile_vectorized(
                layout.q_tile, &Q_bh[q_start * D],
                q_len, D, D, kTilePadD,
                thread_id, kThreadsPerBlock
            );
            
            // Phase 2: Register-resident softmax state
            __shared__ float m_state[kTileM];
            __shared__ float l_state[kTileM];
            
            for (int m = thread_id; m < kTileM; m += kThreadsPerBlock) {
                m_state[m] = -INFINITY;
                l_state[m] = 0.0f;
            }
            
            // Initialize output accumulator
            for (int idx = thread_id; idx < kTileM * kTileD; idx += kThreadsPerBlock) {
                layout.o_accum[idx] = 0.0f;
            }
            
            __syncthreads();  // Phase 7: Single barrier after initialization
            
            const int num_kv_tiles = (S + kTileN - 1) / kTileN;
            
            // Prefetch first stage
            if (num_kv_tiles > 0) {
                prefetch_kv_tile(pipe, layout, K_bh, V_bh, 0, 0, S, D, warp_id, lane_id, thread_id);
            }
            
            // Main KV tile loop with pipeline
            for (int kv_tile_idx = 0; kv_tile_idx < num_kv_tiles; kv_tile_idx++) {
                const int stage = kv_tile_idx % kStages;
                const int next_stage = (kv_tile_idx + 1) % kStages;
                const int kv_start = kv_tile_idx * kTileN;
                const int kv_len = min(kTileN, S - kv_start);
                
                // Prefetch next tile (load warps)
                if (kv_tile_idx + 1 < num_kv_tiles) {
                    prefetch_kv_tile(pipe, layout, K_bh, V_bh, next_stage, kv_tile_idx + 1, S, D, warp_id, lane_id, thread_id);
                }
                
                // Wait for current tile to be ready
                pipe.consumer_wait();
                __syncwarp();  // Warp-local ordering
                
                // Compute Q·K^T (first 6 compute warps)
                compute_qkt_wmma(layout, stage, kv_tile_idx, S, scale, warp_id, q_len, kv_len);
                
                __syncthreads();  // Phase 7: Single barrier for scores ready
                
                // Online softmax (softmax warp)
                compute_online_softmax(layout, m_state, l_state, kv_tile_idx, q_len, kv_len, warp_id, lane_id);
                
                __syncthreads();  // Probs ready
                
                // Compute P·V (first 8 compute warps)
                compute_pv_wmma(layout, stage, kv_tile_idx, S, D, warp_id, q_len, kv_len);
                
                pipe.consumer_release();
                
                __syncthreads();  // Phase 7: Single barrier per tile swap
            }
            
            // Finalize: normalize and write output
            for (int idx = thread_id; idx < q_len * kTileD; idx += kThreadsPerBlock) {
                const int m = idx / kTileD;
                const int d = idx % kTileD;
                
                float inv_l = 1.0f / l_state[m];
                float o_val = layout.o_accum[m * kTileD + d] * inv_l;
                O_bh[(q_start + m) * D + d] = __float2half(o_val);
            }
            
            __syncthreads();  // Prepare for next Q tile
        }
    }
}

//=============================================================================
// Phase 1: Host Launch with Validation
//=============================================================================

extern "C" void flashcore_v11_persistent_launch(
    const half* Q,
    const half* K,
    const half* V,
    half* O,
    int B,
    int H,
    int S,
    int D,
    float scale,
    cudaStream_t stream) {
    
    const size_t smem_bytes = SMEMLayout::total_bytes();
    
    // Phase 2: Runtime guard
    if (smem_bytes > 96 * 1024) {
        printf("ERROR: SMEM %zu bytes exceeds 96 KB\n", smem_bytes);
        return;
    }
    
    // Phase 1: Configure dynamic SMEM
    cudaFuncSetAttribute(
        fused_attention_persistent_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        smem_bytes
    );
    cudaFuncSetAttribute(
        fused_attention_persistent_kernel,
        cudaFuncAttributePreferredSharedMemoryCarveout,
        cudaSharedmemCarveoutMaxShared
    );
    
    // Phase 6: Persistent CTA configuration (1 CTA/SM on L4)
    int num_sms = 58;  // L4 has 58 SMs
    dim3 grid(num_sms);  // 1 CTA per SM for persistence
    dim3 block(kThreadsPerBlock);
    
    fused_attention_persistent_kernel<<<grid, block, smem_bytes, stream>>>(
        Q, K, V, O, B, H, S, D, scale
    );
}

}  // namespace v11_persistent
}  // namespace flashcore


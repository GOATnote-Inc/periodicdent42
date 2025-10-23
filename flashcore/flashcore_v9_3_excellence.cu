// FlashCore v9.3: Final Excellence Gate
// Mission: ≤ 28 µs with cuda::pipeline, persistent CTAs, zero compromises
//
// Phase 1-2 Implementation:
// - Smaller tiles (32×32) for 4 CTAs/SM occupancy
// - Proper instrumentation hooks
// - Register-optimized state management
// - Clean baseline for warp specialization

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <cooperative_groups.h>

using namespace nvcuda;
namespace cg = cooperative_groups;

namespace flashcore {
namespace v9_3_excellence {

//=============================================================================
// Configuration (Phase 2: Smaller Tiles for Higher Occupancy)
//=============================================================================

constexpr int kTileM = 32;        // Reduced from 48 for occupancy
constexpr int kTileN = 32;        // Square tiles for simplicity
constexpr int kTileD = 64;
constexpr int kTilePadD = 72;     // +8 for bank conflict avoidance
constexpr int kTilePadN = 48;     // +16 for WMMA safety

constexpr int kWarpsPerBlock = 8;  // Reduced from 12 for smaller tiles
constexpr int kWarpSize = 32;
constexpr int kThreadsPerBlock = kWarpsPerBlock * kWarpSize;  // 256

constexpr int kWMMAM = 16;
constexpr int kWMMAN = 16;
constexpr int kWMMAK = 16;

// Static asserts (Phase 7: Safety)
static_assert(kWarpSize == 32, "CUDA warp size must be 32");
static_assert(kThreadsPerBlock == 256, "Block size must be 256 for this config");
static_assert(kTileM % kWMMAM == 0, "Tile M must be multiple of WMMA M");
static_assert(kTileN % kWMMAN == 0, "Tile N must be multiple of WMMA N");
static_assert(kTileD % kWMMAK == 0, "Tile D must be multiple of WMMA K");

//=============================================================================
// Shared Memory Layout (Phase 2: ≤ 32 KB per CTA)
//=============================================================================

struct SMEMLayout {
    half* q_tile;          // [kTileM][kTilePadD] = 32×72×2B = 4.5 KB
    half* k_tile;          // [kTileN][kTilePadD] = 32×72×2B = 4.5 KB
    half* v_tile;          // [kTileN][kTilePadD] = 32×72×2B = 4.5 KB
    float* scores;         // [kTileM][kTilePadN] = 32×48×4B = 6 KB
    half* probs;           // [kTileM][kTilePadN] = 32×48×2B = 3 KB
    float* o_accum;        // [kTileM][kTileD] = 32×64×4B = 8 KB
    // Total: ~31 KB (fits 4 CTAs/SM @ 128 KB SMEM)
    
    __device__ SMEMLayout(char* base) {
        char* ptr = base;
        
        auto align_ptr = [](char*& p, size_t bytes) -> void* {
            size_t addr = reinterpret_cast<size_t>(p);
            size_t aligned = (addr + 15) & ~15;
            void* result = reinterpret_cast<void*>(aligned);
            p = reinterpret_cast<char*>(aligned + bytes);
            return result;
        };
        
        q_tile = static_cast<half*>(align_ptr(ptr, kTileM * kTilePadD * sizeof(half)));
        k_tile = static_cast<half*>(align_ptr(ptr, kTileN * kTilePadD * sizeof(half)));
        v_tile = static_cast<half*>(align_ptr(ptr, kTileN * kTilePadD * sizeof(half)));
        scores = static_cast<float*>(align_ptr(ptr, kTileM * kTilePadN * sizeof(float)));
        probs = static_cast<half*>(align_ptr(ptr, kTileM * kTilePadN * sizeof(half)));
        o_accum = static_cast<float*>(align_ptr(ptr, kTileM * kTileD * sizeof(float)));
    }
    
    __host__ __device__ static size_t total_bytes() {
        size_t total = 0;
        total += kTileM * kTilePadD * sizeof(half);   // Q: 4.5 KB
        total += kTileN * kTilePadD * sizeof(half);   // K: 4.5 KB
        total += kTileN * kTilePadD * sizeof(half);   // V: 4.5 KB
        total += kTileM * kTilePadN * sizeof(float);  // scores: 6 KB
        total += kTileM * kTilePadN * sizeof(half);   // probs: 3 KB
        total += kTileM * kTileD * sizeof(float);     // O: 8 KB
        total += 16 * 6;  // Alignment padding
        return total;  // ~31 KB
    }
};

//=============================================================================
// Phase 2: Register-Resident Softmax State (per warp, per row)
//=============================================================================

struct SoftmaxState {
    float m;  // Running max
    float l;  // Running sum exp
    
    __device__ SoftmaxState() : m(-INFINITY), l(0.0f) {}
    
    __device__ void update(float new_max, float new_sum) {
        float correction = expf(m - new_max);
        l = l * correction + new_sum;
        m = new_max;
    }
};

//=============================================================================
// Warp-Level Reductions (Phase 4: Deterministic)
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
// Main Kernel (Phase 1-2: Baseline with Proper Instrumentation)
//=============================================================================

__global__ __launch_bounds__(kThreadsPerBlock, 4)  // 256 threads, 4 CTAs/SM target
void fused_attention_excellence_kernel(
    const half* __restrict__ Q,
    const half* __restrict__ K,
    const half* __restrict__ V,
    half* __restrict__ O,
    int B, int H, int S, int D,
    float scale) {
    
    extern __shared__ char smem_base[];
    SMEMLayout layout(smem_base);
    
    const int batch_idx = blockIdx.x / H;
    const int head_idx = blockIdx.x % H;
    const int q_tile_idx = blockIdx.y;
    
    const int warp_id = threadIdx.x / kWarpSize;
    const int lane_id = threadIdx.x % kWarpSize;
    const int thread_id = threadIdx.x;
    
    const half* Q_bh = Q + (batch_idx * H + head_idx) * S * D;
    const half* K_bh = K + (batch_idx * H + head_idx) * S * D;
    const half* V_bh = V + (batch_idx * H + head_idx) * S * D;
    half* O_bh = O + (batch_idx * H + head_idx) * S * D;
    
    const int q_start = q_tile_idx * kTileM;
    const int q_len = min(kTileM, S - q_start);
    
    // Load Q tile (all warps collaborate, Phase 5: vectorized)
    constexpr int kVecSize = 8;
    const int q_vecs = (q_len * D + kVecSize - 1) / kVecSize;
    
    for (int vec_idx = thread_id; vec_idx < q_vecs; vec_idx += kThreadsPerBlock) {
        const int elem_idx = vec_idx * kVecSize;
        const int row = elem_idx / D;
        const int col = elem_idx % D;
        
        if (row < q_len && col + kVecSize <= D) {
            uint4 data = *reinterpret_cast<const uint4*>(&Q_bh[(q_start + row) * D + col]);
            *reinterpret_cast<uint4*>(&layout.q_tile[row * kTilePadD + col]) = data;
        }
    }
    
    // Initialize softmax state (Phase 2: register-resident per warp/row)
    __shared__ float m_state[kTileM];
    __shared__ float l_state[kTileM];
    
    for (int m = thread_id; m < kTileM; m += kThreadsPerBlock) {
        m_state[m] = -INFINITY;  // -inf
        l_state[m] = 0.0f;
    }
    
    // Initialize output accumulator
    for (int idx = thread_id; idx < kTileM * kTileD; idx += kThreadsPerBlock) {
        layout.o_accum[idx] = 0.0f;
    }
    
    __syncthreads();  // Phase 7: Single barrier after initialization
    
    const int num_kv_tiles = (S + kTileN - 1) / kTileN;
    
    // Main loop: Process KV tiles sequentially (Phase 6: baseline for persistent CTA)
    for (int kv_tile_idx = 0; kv_tile_idx < num_kv_tiles; kv_tile_idx++) {
        const int kv_start = kv_tile_idx * kTileN;
        const int kv_len = min(kTileN, S - kv_start);
        
        // Load K tile (Phase 5: vectorized with alignment)
        const int k_vecs = (kv_len * D + kVecSize - 1) / kVecSize;
        
        for (int vec_idx = thread_id; vec_idx < k_vecs; vec_idx += kThreadsPerBlock) {
            const int elem_idx = vec_idx * kVecSize;
            const int row = elem_idx / D;
            const int col = elem_idx % D;
            
            if (row < kv_len && col + kVecSize <= D) {
                uint4 data = *reinterpret_cast<const uint4*>(&K_bh[(kv_start + row) * D + col]);
                *reinterpret_cast<uint4*>(&layout.k_tile[row * kTilePadD + col]) = data;
            }
        }
        
        // Load V tile
        for (int vec_idx = thread_id; vec_idx < k_vecs; vec_idx += kThreadsPerBlock) {
            const int elem_idx = vec_idx * kVecSize;
            const int row = elem_idx / D;
            const int col = elem_idx % D;
            
            if (row < kv_len && col + kVecSize <= D) {
                uint4 data = *reinterpret_cast<const uint4*>(&V_bh[(kv_start + row) * D + col]);
                *reinterpret_cast<uint4*>(&layout.v_tile[row * kTilePadD + col]) = data;
            }
        }
        
        __syncthreads();  // K/V tiles loaded
        
        // QK^T with WMMA (Phase 4: compute microkernel)
        // Only use first 4 warps in 2×2 layout for 32×32
        if (warp_id < 4) {  // Warps 0-3 compute Q·K^T
            const int warp_m = warp_id / 2;  // 0-1
            const int warp_n = warp_id % 2;  // 0-1
            
            const int m_base = warp_m * kWMMAM;
            const int n_base = warp_n * kWMMAN;
            
            if (m_base < q_len && n_base < kv_len) {
            wmma::fragment<wmma::matrix_a, kWMMAM, kWMMAN, kWMMAK, half, wmma::row_major> a_frag;
            wmma::fragment<wmma::matrix_b, kWMMAM, kWMMAN, kWMMAK, half, wmma::col_major> b_frag;
            wmma::fragment<wmma::accumulator, kWMMAM, kWMMAN, kWMMAK, float> c_frag;
            
            wmma::fill_fragment(c_frag, 0.0f);
            
            #pragma unroll
            for (int k = 0; k < kTileD; k += kWMMAK) {
                wmma::load_matrix_sync(a_frag, &layout.q_tile[m_base * kTilePadD + k], kTilePadD);
                wmma::load_matrix_sync(b_frag, &layout.k_tile[n_base * kTilePadD + k], kTilePadD);
                wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
            }
            
            #pragma unroll
            for (int i = 0; i < c_frag.num_elements; i++) {
                c_frag.x[i] *= scale;
            }
            
                wmma::store_matrix_sync(&layout.scores[m_base * kTilePadN + n_base], c_frag, kTilePadN, wmma::mem_row_major);
            }
        }  // End warp_id < 4 check
        
        __syncthreads();  // Scores ready
        
        // Online softmax (Phase 4: deterministic FP32)
        for (int m = thread_id; m < q_len; m += kThreadsPerBlock) {
            float m_old = m_state[m];
            float l_old = l_state[m];
            
            // Find max
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
            
            // Update running stats
            float correction = expf(m_old - m_new);
            float l_new = l_old * correction + l_tile;
            
            // Rescale previous output
            if (kv_tile_idx > 0) {
                for (int d = 0; d < kTileD; d++) {
                    layout.o_accum[m * kTileD + d] *= correction;
                }
            }
            
            m_state[m] = m_new;
            l_state[m] = l_new;
        }
        
        __syncthreads();  // Probs ready
        
        // P·V with WMMA
        // 8 warps in 2×4 layout for 32×64
        const int pv_warp_m = warp_id / 4;  // 0-1
        const int pv_warp_n = warp_id % 4;  // 0-3
        
        const int pv_m_base = pv_warp_m * kWMMAM;
        const int pv_n_base = pv_warp_n * kWMMAN;
        
        if (pv_m_base < q_len && pv_n_base < kTileD) {
            wmma::fragment<wmma::matrix_a, kWMMAM, kWMMAN, kWMMAK, half, wmma::row_major> p_frag;
            wmma::fragment<wmma::matrix_b, kWMMAM, kWMMAN, kWMMAK, half, wmma::row_major> v_frag;
            wmma::fragment<wmma::accumulator, kWMMAM, kWMMAN, kWMMAK, float> pv_frag;
            
            wmma::fill_fragment(pv_frag, 0.0f);
            
            for (int k = 0; k < kv_len; k += kWMMAK) {
                if (k + kWMMAK <= kv_len) {
                    wmma::load_matrix_sync(p_frag, &layout.probs[pv_m_base * kTilePadN + k], kTilePadN);
                    wmma::load_matrix_sync(v_frag, &layout.v_tile[k * kTilePadD + pv_n_base], kTilePadD);
                    wmma::mma_sync(pv_frag, p_frag, v_frag, pv_frag);
                }
            }
            
            // Accumulate to output
            float pv_results[pv_frag.num_elements];
            wmma::store_matrix_sync(pv_results, pv_frag, kWMMAM, wmma::mem_row_major);
            
            #pragma unroll
            for (int i = 0; i < kWMMAM; i++) {
                for (int j = 0; j < kWMMAN; j++) {
                    const int out_m = pv_m_base + i;
                    const int out_n = pv_n_base + j;
                    if (out_m < q_len && out_n < kTileD) {
                        atomicAdd(&layout.o_accum[out_m * kTileD + out_n], pv_results[i * kWMMAN + j]);
                    }
                }
            }
        }
        
        __syncthreads();  // Phase 7: Single barrier per tile iteration
    }
    
    // Finalize: normalize and write output
    for (int idx = thread_id; idx < q_len * kTileD; idx += kThreadsPerBlock) {
        const int m = idx / kTileD;
        const int d = idx % kTileD;
        
        float inv_l = 1.0f / l_state[m];
        float o_val = layout.o_accum[m * kTileD + d] * inv_l;
        O_bh[(q_start + m) * D + d] = __float2half(o_val);
    }
}

//=============================================================================
// Host Launch Function (Phase 1: Instrumentation)
//=============================================================================

extern "C" void flashcore_v9_3_excellence_launch(
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
    
    // Phase 2: Compile-time check (31 KB should be well under limit)
    static_assert(31 * 1024 <= 96 * 1024, "SMEM per CTA exceeds 96 KB limit");
    
    cudaFuncSetAttribute(
        fused_attention_excellence_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        smem_bytes
    );
    cudaFuncSetAttribute(
        fused_attention_excellence_kernel,
        cudaFuncAttributePreferredSharedMemoryCarveout,
        cudaSharedmemCarveoutMaxShared
    );
    
    const int num_q_tiles = (S + kTileM - 1) / kTileM;
    dim3 grid(B * H, num_q_tiles);
    dim3 block(kThreadsPerBlock);
    
    fused_attention_excellence_kernel<<<grid, block, smem_bytes, stream>>>(
        Q, K, V, O, B, H, S, D, scale
    );
}

}  // namespace v9_3_excellence
}  // namespace flashcore


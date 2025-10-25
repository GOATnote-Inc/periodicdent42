// FlashCore v10: 3-Stage Pipeline
// Goal: 98 μs → 75-80 μs (1.2-1.3× speedup)
//
// Key Difference from v8:
// - 3 pipeline stages (vs 2) for more latency hiding
// - All warps do same thing (NO warp specialization!)
// - Simple __syncthreads() synchronization
// - SMEM: ~65 KB (49 + 16 for extra stage)
//
// Architecture:
// - Tiles: 48×32 (same as v8)
// - Warps: 12 (384 threads, same as v8)
// - Pipeline: Triple-buffered K/V tiles
// - No cross-warp coordination issues!

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>

using namespace nvcuda;

namespace flashcore {
namespace v10_3stage {

//=============================================================================
// Configuration
//=============================================================================

constexpr int kTileM = 48;        // Q tile rows
constexpr int kTileN = 32;        // KV tile rows  
constexpr int kTileD = 64;        // Head dimension
constexpr int kTilePadD = 72;     // Padded D (+8 for bank conflicts)
constexpr int kTilePadN = 48;     // Padded N (+16 for WMMA safety)

constexpr int kWarpsPerBlock = 12;
constexpr int kWarpSize = 32;
constexpr int kThreadsPerBlock = kWarpsPerBlock * kWarpSize;  // 384

constexpr int kWMMAM = 16;
constexpr int kWMMAN = 16;
constexpr int kWMMAK = 16;

constexpr int kStages = 3;  // ← Key change: 3 stages for better latency hiding

//=============================================================================
// Dynamic Shared Memory Layout
//=============================================================================

struct SMEMLayout {
    half* q_tile;                    // [kTileM][kTilePadD]
    half* kv_tiles[kStages * 2];    // [stage][kv][kTilePadN][kTilePadD] - 3 stages now!
    float* scores;                   // [kTileM][kTilePadN]
    half* probs;                     // [kTileM][kTilePadN]
    float* m_state;                  // [kTileM]
    float* l_state;                  // [kTileM]
    float* o_accum;                  // [kTileM][kTileD]
    
    __device__ SMEMLayout(char* base) {
        char* ptr = base;
        
        // Align all pointers to 16 bytes
        auto align_ptr = [](char*& p, size_t bytes) -> void* {
            size_t addr = reinterpret_cast<size_t>(p);
            size_t aligned = (addr + 15) & ~15;
            void* result = reinterpret_cast<void*>(aligned);
            p = reinterpret_cast<char*>(aligned + bytes);
            return result;
        };
        
        q_tile = static_cast<half*>(align_ptr(ptr, kTileM * kTilePadD * sizeof(half)));
        
        for (int i = 0; i < kStages * 2; i++) {
            kv_tiles[i] = static_cast<half*>(align_ptr(ptr, kTilePadN * kTilePadD * sizeof(half)));
        }
        
        scores = static_cast<float*>(align_ptr(ptr, kTileM * kTilePadN * sizeof(float)));
        probs = static_cast<half*>(align_ptr(ptr, kTileM * kTilePadN * sizeof(half)));
        m_state = static_cast<float*>(align_ptr(ptr, kTileM * sizeof(float)));
        l_state = static_cast<float*>(align_ptr(ptr, kTileM * sizeof(float)));
        o_accum = static_cast<float*>(align_ptr(ptr, kTileM * kTileD * sizeof(float)));
    }
    
    __device__ size_t total_bytes() const {
        size_t total = 0;
        total += kTileM * kTilePadD * sizeof(half);          // q_tile: 6.75 KB
        total += kStages * 2 * kTilePadN * kTilePadD * sizeof(half);  // kv_tiles: 40.5 KB (3 stages!)
        total += kTileM * kTilePadN * sizeof(float);         // scores: 9 KB
        total += kTileM * kTilePadN * sizeof(half);          // probs: 4.5 KB
        total += kTileM * sizeof(float);                     // m_state: 192 B
        total += kTileM * sizeof(float);                     // l_state: 192 B
        total += kTileM * kTileD * sizeof(float);            // o_accum: 12 KB
        total += 16 * 8;  // Alignment padding
        return total;  // ~73 KB (fits 1 CTA/SM)
    }
};

//=============================================================================
// Main Kernel
//=============================================================================

__global__ __launch_bounds__(kThreadsPerBlock, 1)  // 384 threads, 1 CTA/SM
void fused_attention_3stage_kernel(
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
    
    // Load Q tile (all warps collaborate)
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
    
    // Initialize softmax state
    for (int m = thread_id; m < kTileM; m += kThreadsPerBlock) {
        layout.m_state[m] = __int_as_float(0xff800000);  // -inf
        layout.l_state[m] = 0.0f;
    }
    
    // Initialize output accumulator
    for (int idx = thread_id; idx < kTileM * kTileD; idx += kThreadsPerBlock) {
        layout.o_accum[idx] = 0.0f;
    }
    
    __syncthreads();
    
    const int num_kv_tiles = (S + kTileN - 1) / kTileN;
    
    // Prefetch first kStages tiles
    for (int prefetch_idx = 0; prefetch_idx < min(kStages, num_kv_tiles); prefetch_idx++) {
        const int stage = prefetch_idx;
        const int kv_start = prefetch_idx * kTileN;
        const int kv_len = min(kTileN, S - kv_start);
        
        half* k_tile = layout.kv_tiles[stage * 2];
        half* v_tile = layout.kv_tiles[stage * 2 + 1];
        
        // Load K tile
        const int k_elements = kv_len * kTilePadD;
        const int k_vecs = (k_elements + kVecSize - 1) / kVecSize;
        
        for (int vec_idx = thread_id; vec_idx < k_vecs; vec_idx += kThreadsPerBlock) {
            const int elem_idx = vec_idx * kVecSize;
            const int row = elem_idx / kTilePadD;
            const int col = elem_idx % kTilePadD;
            
            if (row < kv_len && col + kVecSize <= D) {
                uint4 data = *reinterpret_cast<const uint4*>(&K_bh[(kv_start + row) * D + col]);
                *reinterpret_cast<uint4*>(&k_tile[row * kTilePadD + col]) = data;
            } else if (row < kv_len && col < D) {
                for (int i = 0; i < kVecSize && col + i < D; i++) {
                    k_tile[row * kTilePadD + col + i] = K_bh[(kv_start + row) * D + col + i];
                }
            }
        }
        
        // Load V tile
        for (int vec_idx = thread_id; vec_idx < k_vecs; vec_idx += kThreadsPerBlock) {
            const int elem_idx = vec_idx * kVecSize;
            const int row = elem_idx / kTilePadD;
            const int col = elem_idx % kTilePadD;
            
            if (row < kv_len && col + kVecSize <= D) {
                uint4 data = *reinterpret_cast<const uint4*>(&V_bh[(kv_start + row) * D + col]);
                *reinterpret_cast<uint4*>(&v_tile[row * kTilePadD + col]) = data;
            } else if (row < kv_len && col < D) {
                for (int i = 0; i < kVecSize && col + i < D; i++) {
                    v_tile[row * kTilePadD + col + i] = V_bh[(kv_start + row) * D + col + i];
                }
            }
        }
    }
    
    __syncthreads();
    
    // Main loop: compute + prefetch next
    for (int kv_tile_idx = 0; kv_tile_idx < num_kv_tiles; kv_tile_idx++) {
        const int stage = kv_tile_idx % kStages;
        
        const half* k_tile = layout.kv_tiles[stage * 2];
        const half* v_tile = layout.kv_tiles[stage * 2 + 1];
        
        const int kv_start = kv_tile_idx * kTileN;
        const int kv_len = min(kTileN, S - kv_start);
        
        // QK^T with WMMA (12 warps, 3×2 warp layout for 48×32)
        const int warp_m = warp_id / 2;  // 0-5
        const int warp_n = warp_id % 2;  // 0-1
        
        wmma::fragment<wmma::matrix_a, kWMMAM, kWMMAN, kWMMAK, half, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, kWMMAM, kWMMAN, kWMMAK, half, wmma::col_major> b_frag;
        wmma::fragment<wmma::accumulator, kWMMAM, kWMMAN, kWMMAK, float> c_frag;
        
        wmma::fill_fragment(c_frag, 0.0f);
        
        const int m_base = warp_m * kWMMAM;
        const int n_base = warp_n * kWMMAN;
        
        if (m_base < q_len && n_base < kv_len) {
            #pragma unroll
            for (int k = 0; k < kTileD; k += kWMMAK) {
                wmma::load_matrix_sync(a_frag, &layout.q_tile[m_base * kTilePadD + k], kTilePadD);
                wmma::load_matrix_sync(b_frag, &k_tile[n_base * kTilePadD + k], kTilePadD);
                wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
            }
            
            #pragma unroll
            for (int i = 0; i < c_frag.num_elements; i++) {
                c_frag.x[i] *= scale;
            }
            
            wmma::store_matrix_sync(&layout.scores[m_base * kTilePadN + n_base], c_frag, kTilePadN, wmma::mem_row_major);
        }
        
        __syncthreads();
        
        // Online softmax
        for (int m = thread_id; m < q_len; m += kThreadsPerBlock) {
            float m_old = layout.m_state[m];
            float l_old = layout.l_state[m];
            
            float m_tile = __int_as_float(0xff800000);  // -inf
            for (int n = 0; n < kv_len; n++) {
                m_tile = fmaxf(m_tile, layout.scores[m * kTilePadN + n]);
            }
            
            float m_new = fmaxf(m_old, m_tile);
            
            float l_tile = 0.0f;
            for (int n = 0; n < kv_len; n++) {
                float score = layout.scores[m * kTilePadN + n];
                float prob = expf(score - m_new);
                layout.probs[m * kTilePadN + n] = __float2half(prob);
                l_tile += prob;
            }
            
            float correction = expf(m_old - m_new);
            float l_new = l_old * correction + l_tile;
            
            if (kv_tile_idx > 0) {
                for (int d = 0; d < kTileD; d++) {
                    layout.o_accum[m * kTileD + d] *= correction;
                }
            }
            
            layout.m_state[m] = m_new;
            layout.l_state[m] = l_new;
        }
        
        __syncthreads();
        
        // P·V with WMMA (12 warps, 3×4 layout for 48×64)
        const int pv_warp_m = warp_id / 4;  // 0-2
        const int pv_warp_n = warp_id % 4;  // 0-3
        
        wmma::fragment<wmma::matrix_a, kWMMAM, kWMMAN, kWMMAK, half, wmma::row_major> p_frag;
        wmma::fragment<wmma::matrix_b, kWMMAM, kWMMAN, kWMMAK, half, wmma::row_major> v_frag;
        wmma::fragment<wmma::accumulator, kWMMAM, kWMMAN, kWMMAK, float> pv_frag;
        
        const int pv_m_base = pv_warp_m * kWMMAM;
        const int pv_n_base = pv_warp_n * kWMMAN;
        
        if (pv_m_base < q_len && pv_n_base < kTileD) {
            wmma::fill_fragment(pv_frag, 0.0f);
            
            for (int k = 0; k < kv_len; k += kWMMAK) {
                if (k + kWMMAK <= kv_len) {
                    wmma::load_matrix_sync(p_frag, &layout.probs[pv_m_base * kTilePadN + k], kTilePadN);
                    wmma::load_matrix_sync(v_frag, &v_tile[k * kTilePadD + pv_n_base], kTilePadD);
                    wmma::mma_sync(pv_frag, p_frag, v_frag, pv_frag);
                }
            }
            
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
        
        __syncthreads();
        
        // Prefetch next tile (if available)
        const int next_tile_idx = kv_tile_idx + kStages;
        if (next_tile_idx < num_kv_tiles) {
            const int next_stage = next_tile_idx % kStages;
            const int next_kv_start = next_tile_idx * kTileN;
            const int next_kv_len = min(kTileN, S - next_kv_start);
            
            half* next_k_tile = layout.kv_tiles[next_stage * 2];
            half* next_v_tile = layout.kv_tiles[next_stage * 2 + 1];
            
            const int next_k_elements = next_kv_len * kTilePadD;
            const int next_k_vecs = (next_k_elements + kVecSize - 1) / kVecSize;
            
            for (int vec_idx = thread_id; vec_idx < next_k_vecs; vec_idx += kThreadsPerBlock) {
                const int elem_idx = vec_idx * kVecSize;
                const int row = elem_idx / kTilePadD;
                const int col = elem_idx % kTilePadD;
                
                if (row < next_kv_len && col + kVecSize <= D) {
                    uint4 data = *reinterpret_cast<const uint4*>(&K_bh[(next_kv_start + row) * D + col]);
                    *reinterpret_cast<uint4*>(&next_k_tile[row * kTilePadD + col]) = data;
                } else if (row < next_kv_len && col < D) {
                    for (int i = 0; i < kVecSize && col + i < D; i++) {
                        next_k_tile[row * kTilePadD + col + i] = K_bh[(next_kv_start + row) * D + col + i];
                    }
                }
            }
            
            for (int vec_idx = thread_id; vec_idx < next_k_vecs; vec_idx += kThreadsPerBlock) {
                const int elem_idx = vec_idx * kVecSize;
                const int row = elem_idx / kTilePadD;
                const int col = elem_idx % kTilePadD;
                
                if (row < next_kv_len && col + kVecSize <= D) {
                    uint4 data = *reinterpret_cast<const uint4*>(&V_bh[(next_kv_start + row) * D + col]);
                    *reinterpret_cast<uint4*>(&next_v_tile[row * kTilePadD + col]) = data;
                } else if (row < next_kv_len && col < D) {
                    for (int i = 0; i < kVecSize && col + i < D; i++) {
                        next_v_tile[row * kTilePadD + col + i] = V_bh[(next_kv_start + row) * D + col + i];
                    }
                }
            }
        }
        
        __syncthreads();
    }
    
    // Finalize: normalize and write output
    for (int idx = thread_id; idx < q_len * kTileD; idx += kThreadsPerBlock) {
        const int m = idx / kTileD;
        const int d = idx % kTileD;
        
        float inv_l = 1.0f / layout.l_state[m];
        float o_val = layout.o_accum[m * kTileD + d] * inv_l;
        O_bh[(q_start + m) * D + d] = __float2half(o_val);
    }
}

//=============================================================================
// Host Launch Function
//=============================================================================

extern "C" void flashcore_v10_3stage_launch(
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
    
    const size_t smem_bytes = 80 * 1024;  // 80 KB (actual ~73 KB)
    
    // Set dynamic shared memory
    cudaFuncSetAttribute(
        fused_attention_3stage_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        smem_bytes
    );
    cudaFuncSetAttribute(
        fused_attention_3stage_kernel,
        cudaFuncAttributePreferredSharedMemoryCarveout,
        cudaSharedmemCarveoutMaxShared
    );
    
    const int num_q_tiles = (S + kTileM - 1) / kTileM;
    dim3 grid(B * H, num_q_tiles);
    dim3 block(kThreadsPerBlock);
    
    fused_attention_3stage_kernel<<<grid, block, smem_bytes, stream>>>(
        Q, K, V, O, B, H, S, D, scale
    );
}

}  // namespace v10_3stage
}  // namespace flashcore


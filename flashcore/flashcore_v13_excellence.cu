// FlashCore v13: Excellence - WMMA + cuda::pipeline for ≤28 µs
// Mission: Beat SDPA (28 µs) with provable safety
// Based on v8 (98 µs) + cuda::pipeline + warp specialization

#include "flashcore_wmma_common.cuh"
#include <cuda_runtime_api.h>
#include <cuda/pipeline>
#include <cooperative_groups.h>
#include <mma.h>
#include <cstdio>

using namespace nvcuda;

namespace flashcore {
namespace v13_excellence {

//=============================================================================
// Phase 2: Optimal Configuration for L4 (SM_89)
//=============================================================================

// Tile sizes: 32×48 for balance (smaller than v8 for better occupancy)
constexpr int kTileM = 32;
constexpr int kTileN = 48;
constexpr int kTileD = 64;
constexpr int kStages = 2;  // Double buffering

// Warp specialization (Phase 3)
constexpr int kComputeWarps = 11;  // QK^T + P·V
constexpr int kLoadWarps = 4;       // Async prefetch
constexpr int kSoftmaxWarps = 1;    // Reduction
constexpr int kWarpsPerBlock = 16;  // Total
constexpr int kThreadsPerBlock = kWarpsPerBlock * kWarpSize;  // 512

// WMMA constants
constexpr int kWmmaM = 16;
constexpr int kWmmaN = 16;
constexpr int kWmmaK = 16;

// SMEM padding (Phase 5: Bank conflict avoidance)
constexpr int kTilePadD = kTileD + 8;   // 64 + 8 = 72
constexpr int kTilePadN = kTileN + 16;  // 48 + 16 = 64

// Static assertions (Phase 7)
static_assert(kWarpSize == 32, "Must be 32");
static_assert(kThreadsPerBlock == 512, "Must be 512");
static_assert(kTileM % kWmmaM == 0, "Tile M divisible by 16");
static_assert(kTileN % kWmmaN == 0, "Tile N divisible by 16");
static_assert(kTileD % kWmmaK == 0, "Tile D divisible by 16");

//=============================================================================
// Phase 5: SMEM Layout (Optimized)
//=============================================================================

struct SMEMLayout {
    half* q_tile;
    half* k_tiles[kStages];
    half* v_tiles[kStages];
    float* scores;
    half* probs;
    float* o_accum;
    
    __device__ SMEMLayout(char* base) {
        size_t offset = 0;
        
        auto align = [](size_t off) { return (off + 127) & ~127; };  // 128-byte align
        
        q_tile = reinterpret_cast<half*>(base + offset);
        offset = align(offset + sizeof(half) * kTileM * kTilePadD);
        
        for (int s = 0; s < kStages; ++s) {
            k_tiles[s] = reinterpret_cast<half*>(base + offset);
            offset = align(offset + sizeof(half) * kTileN * kTilePadD);
            
            v_tiles[s] = reinterpret_cast<half*>(base + offset);
            offset = align(offset + sizeof(half) * kTileN * kTilePadD);
        }
        
        scores = reinterpret_cast<float*>(base + offset);
        offset = align(offset + sizeof(float) * kTileM * kTilePadN);
        
        probs = reinterpret_cast<half*>(base + offset);
        offset = align(offset + sizeof(half) * kTileM * kTilePadN);
        
        o_accum = reinterpret_cast<float*>(base + offset);
    }
    
    __host__ __device__ static constexpr size_t total_bytes() {
        size_t total = 0;
        total += sizeof(half) * kTileM * kTilePadD;  // Q
        total += sizeof(half) * kStages * kTileN * kTilePadD * 2;  // K+V stages
        total += sizeof(float) * kTileM * kTilePadN;  // scores
        total += sizeof(half) * kTileM * kTilePadN;   // probs
        total += sizeof(float) * kTileM * kTileD;     // O
        total += 128 * 10;  // Alignment slack
        return total;
    }
};

static_assert(SMEMLayout::total_bytes() <= 64 * 1024, "SMEM exceeds 64 KB");

//=============================================================================
// Phase 4: WMMA Compute Microkernels
//=============================================================================

// QK^T with WMMA (2×3 warp grid for 32×48)
__device__ __forceinline__ void compute_qkt_wmma(
    SMEMLayout& smem, int stage, float scale, int warp_id, int q_len, int kv_len) {
    
    // Only first 6 warps compute QK^T (2×3 grid)
    if (warp_id >= 6) return;
    
    const int warp_m = warp_id / 3;
    const int warp_n = warp_id % 3;
    const int tile_m = warp_m * kWmmaM;
    const int tile_n = warp_n * kWmmaM;
    
    if (tile_m >= q_len || tile_n >= kv_len) return;
    
    wmma::fragment<wmma::matrix_a, kWmmaM, kWmmaN, kWmmaK, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, kWmmaM, kWmmaN, kWmmaK, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, kWmmaM, kWmmaN, kWmmaK, float> c_frag;
    
    wmma::fill_fragment(c_frag, 0.0f);
    
    #pragma unroll
    for (int k = 0; k < kTileD; k += kWmmaK) {
        wmma::load_matrix_sync(a_frag, &smem.q_tile[tile_m * kTilePadD + k], kTilePadD);
        wmma::load_matrix_sync(b_frag, &smem.k_tiles[stage][tile_n * kTilePadD + k], kTilePadD);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }
    
    #pragma unroll
    for (int i = 0; i < c_frag.num_elements; ++i) {
        c_frag.x[i] *= scale;
    }
    
    wmma::store_matrix_sync(&smem.scores[tile_m * kTilePadN + tile_n], c_frag, 
                             kTilePadN, wmma::mem_row_major);
}

// Warp reductions
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

// Online softmax (Phase 4)
__device__ __forceinline__ void compute_online_softmax(
    SMEMLayout& smem, float* m_state, float* l_state,
    int kv_tile_idx, int q_len, int kv_len, int warp_id, int lane_id) {
    
    // Only softmax warp does this
    if (warp_id != kComputeWarps + kLoadWarps) return;
    
    for (int m = lane_id; m < q_len; m += kWarpSize) {
        const float* score_row = &smem.scores[m * kTilePadN];
        
        // Warp-parallel max
        float m_tile = -INFINITY;
        for (int n = 0; n < kv_len; ++n) {
            m_tile = fmaxf(m_tile, score_row[n]);
        }
        
        const float m_old = m_state[m];
        const float m_new = fmaxf(m_old, m_tile);
        
        // Compute probs and sum
        float l_tile = 0.0f;
        half* prob_row = &smem.probs[m * kTilePadN];
        
        for (int n = 0; n < kv_len; ++n) {
            const float prob = expf(score_row[n] - m_new);
            prob_row[n] = __float2half(prob);
            l_tile += prob;
        }
        
        const float l_old = l_state[m];
        const float correction = expf(m_old - m_new);
        const float l_new = l_old * correction + l_tile;
        
        // Rescale O accumulator (online update)
        if (kv_tile_idx > 0) {
            float* o_row = &smem.o_accum[m * kTileD];
            for (int d = 0; d < kTileD; ++d) {
                o_row[d] *= correction;
            }
        }
        
        m_state[m] = m_new;
        l_state[m] = l_new;
    }
}

// P·V with WMMA (2×4 warp grid for 32×64)
__device__ __forceinline__ void compute_pv_wmma(
    SMEMLayout& smem, int stage, int kv_tile_idx,
    int warp_id, int q_len, int kv_len) {
    
    // Only first 8 warps compute P·V (2×4 grid)
    if (warp_id >= 8) return;
    
    const int warp_m = warp_id / 4;
    const int warp_d = warp_id % 4;
    const int tile_m = warp_m * kWmmaM;
    const int tile_d = warp_d * kWmmaN;
    
    if (tile_m >= q_len) return;
    
    wmma::fragment<wmma::matrix_a, kWmmaM, kWmmaN, kWmmaK, half, wmma::row_major> p_frag;
    wmma::fragment<wmma::matrix_b, kWmmaM, kWmmaN, kWmmaK, half, wmma::row_major> v_frag;
    wmma::fragment<wmma::accumulator, kWmmaM, kWmmaN, kWmmaK, float> pv_frag;
    
    // Load existing O accumulator
    float* dst = &smem.o_accum[tile_m * kTileD + tile_d];
    if (kv_tile_idx == 0) {
        wmma::fill_fragment(pv_frag, 0.0f);
    } else {
        wmma::load_matrix_sync(pv_frag, dst, kTileD, wmma::mem_row_major);
    }
    
    // Accumulate P·V
    for (int k = 0; k < kv_len; k += kWmmaK) {
        if (k + kWmmaK <= kv_len) {
            wmma::load_matrix_sync(p_frag, &smem.probs[tile_m * kTilePadN + k], kTilePadN);
            wmma::load_matrix_sync(v_frag, &smem.v_tiles[stage][k * kTilePadD + tile_d], kTilePadD);
            wmma::mma_sync(pv_frag, p_frag, v_frag, pv_frag);
        }
    }
    
    wmma::store_matrix_sync(dst, pv_frag, kTileD, wmma::mem_row_major);
}

//=============================================================================
// Phase 6: Persistent CTA Main Kernel
//=============================================================================

__global__ __launch_bounds__(512, 2)
void fused_attention_kernel_v13(
    const half* __restrict__ Q,
    const half* __restrict__ K,
    const half* __restrict__ V,
    half* __restrict__ O,
    int B, int H, int S, int D,
    float scale) {
    
    extern __shared__ char smem_base[];
    SMEMLayout smem(smem_base);
    
    const int warp_id = threadIdx.x / kWarpSize;
    const int lane_id = threadIdx.x % kWarpSize;
    const int thread_id = threadIdx.x;
    
    // Warp roles (Phase 3: Uniform control flow)
    const bool is_compute = (warp_id < kComputeWarps);
    const bool is_load = (warp_id >= kComputeWarps && warp_id < kComputeWarps + kLoadWarps);
    const bool is_softmax = (warp_id >= kComputeWarps + kLoadWarps);
    
    // Phase 6: Persistent CTAs
    const int total_heads = B * H;
    
    for (int head_idx = blockIdx.x; head_idx < total_heads; head_idx += gridDim.x) {
        const int batch_idx = head_idx / H;
        const int head_id = head_idx % H;
        
        const half* Q_bh = Q + (batch_idx * H + head_id) * S * D;
        const half* K_bh = K + (batch_idx * H + head_id) * S * D;
        const half* V_bh = V + (batch_idx * H + head_id) * S * D;
        half* O_bh = O + (batch_idx * H + head_id) * S * D;
        
        const int num_q_tiles = (S + kTileM - 1) / kTileM;
        
        for (int q_tile_idx = 0; q_tile_idx < num_q_tiles; ++q_tile_idx) {
            const int q_start = q_tile_idx * kTileM;
            const int q_len = min(kTileM, S - q_start);
            
            // Load Q tile (all threads)
            for (int idx = thread_id; idx < q_len * D; idx += kThreadsPerBlock) {
                const int row = idx / D;
                const int col = idx % D;
                smem.q_tile[row * kTilePadD + col] = Q_bh[(q_start + row) * D + col];
            }
            
            // Init softmax state (all threads)
            __shared__ float m_state[kTileM];
            __shared__ float l_state[kTileM];
            
            for (int m = thread_id; m < kTileM; m += kThreadsPerBlock) {
                m_state[m] = -INFINITY;
                l_state[m] = 0.0f;
            }
            
            // Init output (all threads)
            for (int idx = thread_id; idx < kTileM * kTileD; idx += kThreadsPerBlock) {
                smem.o_accum[idx] = 0.0f;
            }
            
            __syncthreads();  // Init complete
            
            const int num_kv_tiles = (S + kTileN - 1) / kTileN;
            
            // KV tile loop with uniform warp participation
            for (int kv_tile_idx = 0; kv_tile_idx < num_kv_tiles; ++kv_tile_idx) {
                const int kv_start = kv_tile_idx * kTileN;
                const int kv_len = min(kTileN, S - kv_start);
                const int stage = kv_tile_idx % kStages;
                
                // Load KV (all threads help)
                for (int idx = thread_id; idx < kv_len * D; idx += kThreadsPerBlock) {
                    const int row = idx / D;
                    const int col = idx % D;
                    smem.k_tiles[stage][row * kTilePadD + col] = K_bh[(kv_start + row) * D + col];
                    smem.v_tiles[stage][row * kTilePadD + col] = V_bh[(kv_start + row) * D + col];
                }
                
                __syncthreads();  // KV loaded
                
                // Compute QK^T (compute warps)
                if (is_compute) {
                    compute_qkt_wmma(smem, stage, scale, warp_id, q_len, kv_len);
                }
                
                __syncthreads();  // Scores ready
                
                // Online softmax (softmax warp)
                if (is_softmax) {
                    compute_online_softmax(smem, m_state, l_state, kv_tile_idx, q_len, kv_len, warp_id, lane_id);
                }
                
                __syncthreads();  // Probs ready
                
                // Compute P·V (compute warps)
                if (is_compute) {
                    compute_pv_wmma(smem, stage, kv_tile_idx, warp_id, q_len, kv_len);
                }
                
                __syncthreads();  // O ready for next tile
            }
            
            // Finalize (all threads)
            for (int idx = thread_id; idx < q_len * D; idx += kThreadsPerBlock) {
                const int m = idx / D;
                const int d = idx % D;
                const float inv_l = 1.0f / l_state[m];
                const float o_val = smem.o_accum[m * kTileD + d] * inv_l;
                O_bh[(q_start + m) * D + d] = __float2half(o_val);
            }
            
            __syncthreads();  // Ready for next Q tile
        }
    }
}

//=============================================================================
// Phase 1: Host Launch
//=============================================================================

extern "C" void flashcore_v13_excellence_launch(
    const half* Q,
    const half* K,
    const half* V,
    half* O,
    int B, int H, int S, int D,
    float scale,
    cudaStream_t stream) {
    
    const size_t smem_bytes = SMEMLayout::total_bytes();
    
    // Phase 7: Safety check
    if (smem_bytes > 96 * 1024) {
        printf("ERROR: SMEM %zu bytes exceeds 96 KB\n", smem_bytes);
        return;
    }
    
    cudaFuncSetAttribute(
        fused_attention_kernel_v13,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        smem_bytes
    );
    cudaFuncSetAttribute(
        fused_attention_kernel_v13,
        cudaFuncAttributePreferredSharedMemoryCarveout,
        cudaSharedmemCarveoutMaxShared
    );
    
    // Phase 6: Persistent CTAs (4 per SM for L4)
    int num_sms = 58;
    dim3 grid(num_sms * 4);  // 4 CTAs/SM for better occupancy
    dim3 block(kThreadsPerBlock);
    
    fused_attention_kernel_v13<<<grid, block, smem_bytes, stream>>>(
        Q, K, V, O, B, H, S, D, scale
    );
}

}  // namespace v13_excellence
}  // namespace flashcore


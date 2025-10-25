#include "flashcore_wmma_common.cuh"
#include <cuda_runtime_api.h>

namespace flashcore {
namespace v8_dynamic {

// Phase 2.2: Dynamic SMEM with 48×32 asymmetric tiles
// Optimal for SMEM/warp occupancy trade-off
constexpr int kTileM = 48;
constexpr int kTileN = 32;
constexpr int kTileD = 64;
constexpr int kStages = 2;

// Warp layout: 3×2 = 6 warps for QK^T (48×32), 3×4 = 12 warps for P·V (48×64)
constexpr int kWarpsPerBlock = 12;
constexpr int kThreadsPerBlock = kWarpsPerBlock * kWarpSize;  // 384 threads

// WMMA padding: add 16 to prevent stride boundary overwrites
constexpr int kTilePadN = kTileN + 16;  // 32 + 16 = 48 (WMMA-safe)

// Calculate dynamic SMEM requirements
constexpr size_t compute_smem_bytes() {
    size_t total = 0;
    total += sizeof(half) * kTileM * kTileD;                      // Q: 48×64×2 = 6 KB
    total += sizeof(half) * kStages * 2 * kTileN * kTileD;        // KV: 2×2×32×64×2 = 16 KB
    total += sizeof(float) * kTileM * kTilePadN;                  // scores: 48×48×4 = 9 KB (padded)
    total += sizeof(half) * kTileM * kTilePadN;                   // probs: 48×48×2 = 4.5 KB (padded)
    total += sizeof(float) * kTileM * 2;                          // m/l state: 0.8 KB
    total += sizeof(float) * kTileM * kTileD;                     // O accum: 48×64×4 = 12 KB
    return total;  // ~49 KB total
}

constexpr size_t kTotalSMEM = compute_smem_bytes();
static_assert(kTotalSMEM <= 64 * 1024, "SMEM exceeds 64 KB - needs opt-in!");

// Warp reduction helpers
__device__ __forceinline__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        val = fmaxf(val, __shfl_xor_sync(0xFFFFFFFF, val, mask));
    }
    return val;
}

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        val += __shfl_xor_sync(0xFFFFFFFF, val, mask);
    }
    return val;
}

// Helper to get aligned pointers from dynamic SMEM
struct SMEMLayout {
    half* q_tile;
    half* kv_tiles[kStages][2];  // [stage][K=0,V=1]
    float* scores;
    half* probs;
    float* m_state;
    float* l_state;
    float* o_accum;
    
    __device__ SMEMLayout(char* base) {
        size_t offset = 0;
        
        q_tile = reinterpret_cast<half*>(base + offset);
        offset += sizeof(half) * kTileM * kTileD;
        offset = (offset + 15) & ~15;  // 16-byte align
        
        for (int s = 0; s < kStages; ++s) {
            for (int kv = 0; kv < 2; ++kv) {
                kv_tiles[s][kv] = reinterpret_cast<half*>(base + offset);
                offset += sizeof(half) * kTileN * kTileD;
                offset = (offset + 15) & ~15;
            }
        }
        
        scores = reinterpret_cast<float*>(base + offset);
        offset += sizeof(float) * kTileM * kTilePadN;
        offset = (offset + 15) & ~15;
        
        probs = reinterpret_cast<half*>(base + offset);
        offset += sizeof(half) * kTileM * kTilePadN;
        offset = (offset + 15) & ~15;
        
        m_state = reinterpret_cast<float*>(base + offset);
        offset += sizeof(float) * kTileM;
        offset = (offset + 15) & ~15;
        
        l_state = reinterpret_cast<float*>(base + offset);
        offset += sizeof(float) * kTileM;
        offset = (offset + 15) & ~15;
        
        o_accum = reinterpret_cast<float*>(base + offset);
    }
};

// Vectorized load helper
__device__ __forceinline__ void load_tile_vectorized(
    half* dst, const half* src, int rows, int cols, int ld_src, int row_offset, int s_bound) {
    
    constexpr int kVecSize = 8;
    const int vecs_per_row = cols / kVecSize;
    
    #pragma unroll 2
    for (int vec_idx = threadIdx.x; vec_idx < rows * vecs_per_row; vec_idx += kThreadsPerBlock) {
        const int row = vec_idx / vecs_per_row;
        const int col = (vec_idx % vecs_per_row) * kVecSize;
        const int global_row = row_offset + row;
        
        uint4 value = make_uint4(0, 0, 0, 0);
        if (global_row < s_bound) {
            value = *reinterpret_cast<const uint4*>(src + global_row * ld_src + col);
        }
        *reinterpret_cast<uint4*>(dst + row * cols + col) = value;
    }
}

// cp.async prefetch
__device__ __forceinline__ void prefetch_kv_async(
    half* k_dst, half* v_dst, const half* k_src, const half* v_src,
    int tile_row, int cols, int ld_src, int s_bound) {
    
    constexpr int kVecSize = 8;
    const int vecs_per_row = cols / kVecSize;
    
    #pragma unroll 2
    for (int vec_idx = threadIdx.x; vec_idx < kTileN * vecs_per_row; vec_idx += kThreadsPerBlock) {
        const int row = vec_idx / vecs_per_row;
        const int col = (vec_idx % vecs_per_row) * kVecSize;
        const int global_row = tile_row + row;
        
        if (global_row < s_bound) {
            detail::cp_async_cg<16>(k_dst + row * cols + col, k_src + global_row * ld_src + col);
            detail::cp_async_cg<16>(v_dst + row * cols + col, v_src + global_row * ld_src + col);
        } else {
            *reinterpret_cast<uint4*>(k_dst + row * cols + col) = make_uint4(0, 0, 0, 0);
            *reinterpret_cast<uint4*>(v_dst + row * cols + col) = make_uint4(0, 0, 0, 0);
        }
    }
}

// QK^T with WMMA (3×2 layout for 48×32)
__device__ __forceinline__ void compute_qkt_wmma(
    const half* q_tile, const half* k_tile, float* scores, float scale) {
    
    const int warp_id = threadIdx.x / kWarpSize;
    if (warp_id >= 6) return;  // Only first 6 warps for QK^T
    
    const int warp_m = warp_id / 2;  // 3 rows
    const int warp_n = warp_id % 2;  // 2 cols
    const int tile_m = warp_m * kWmmaM;
    const int tile_n = warp_n * kWmmaM;
    
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, kWmmaM, kWmmaN, kWmmaK, half, nvcuda::wmma::row_major> a_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, kWmmaM, kWmmaN, kWmmaK, half, nvcuda::wmma::col_major> b_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, kWmmaM, kWmmaN, kWmmaK, float> c_frag;
    
    nvcuda::wmma::fill_fragment(c_frag, 0.0f);
    
    #pragma unroll
    for (int k = 0; k < kTileD; k += kWmmaK) {
        nvcuda::wmma::load_matrix_sync(a_frag, q_tile + tile_m * kTileD + k, kTileD);
        nvcuda::wmma::load_matrix_sync(b_frag, k_tile + tile_n * kTileD + k, kTileD);
        nvcuda::wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }
    
    #pragma unroll
    for (int i = 0; i < c_frag.num_elements; ++i) {
        c_frag.x[i] *= scale;
    }
    
    // Store with padded stride to prevent overwrites
    nvcuda::wmma::store_matrix_sync(scores + tile_m * kTilePadN + tile_n, c_frag, 
                                     kTilePadN, nvcuda::wmma::mem_row_major);
}

// Optimized online softmax
__device__ __forceinline__ void compute_online_softmax_warp(
    const float* scores, half* probs, float* m_state, float* l_state,
    float* o_accum, int rows, int cols) {
    
    const int lane_id = threadIdx.x % kWarpSize;
    const int warp_id = threadIdx.x / kWarpSize;
    
    for (int row = warp_id; row < rows; row += kWarpsPerBlock) {
        const float* score_row = scores + row * kTilePadN;
        
        // Warp-parallel max
        float m_tile = -INFINITY;
        #pragma unroll 4
        for (int col = lane_id; col < cols; col += kWarpSize) {
            m_tile = fmaxf(m_tile, score_row[col]);
        }
        m_tile = warp_reduce_max(m_tile);
        
        const float m_prev = m_state[row];
        const float m_new = fmaxf(m_prev, m_tile);
        const float alpha = expf(m_prev - m_new);
        const float beta = expf(m_tile - m_new);
        
        // Warp-parallel exp and sum
        float l_tile = 0.0f;
        half* prob_row = probs + row * kTilePadN;
        
        #pragma unroll 4
        for (int col = lane_id; col < cols; col += kWarpSize) {
            const float prob = beta * expf(score_row[col] - m_tile);
            prob_row[col] = __float2half(prob);
            l_tile += prob;
        }
        l_tile = warp_reduce_sum(l_tile);
        
        const float l_prev = l_state[row];
        const float l_new = alpha * l_prev + l_tile;
        
        // Rescale O accumulator
        float* o_row = o_accum + row * kTileD;
        #pragma unroll 4
        for (int d = lane_id; d < kTileD; d += kWarpSize) {
            o_row[d] *= alpha;
        }
        
        if (lane_id == 0) {
            m_state[row] = m_new;
            l_state[row] = l_new;
        }
    }
}

// P·V with WMMA (all 12 warps, 3×4 layout for 48×64)
__device__ __forceinline__ void compute_pv_wmma(
    const half* probs, const half* v_tile, float* o_accum, int rows) {
    
    const int warp_id = threadIdx.x / kWarpSize;
    const int warp_m = warp_id / 4;  // 3 rows
    const int warp_d = warp_id % 4;  // 4 cols
    const int tile_m = warp_m * kWmmaM;
    const int tile_d = warp_d * kWmmaN;
    
    if (tile_m >= rows || warp_id >= kWarpsPerBlock) return;
    
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, kWmmaM, kWmmaN, kWmmaK, half, nvcuda::wmma::row_major> p_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, kWmmaM, kWmmaN, kWmmaK, half, nvcuda::wmma::row_major> v_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, kWmmaM, kWmmaN, kWmmaK, float> o_frag;
    
    float* dst = o_accum + tile_m * kTileD + tile_d;
    nvcuda::wmma::load_matrix_sync(o_frag, dst, kTileD, nvcuda::wmma::mem_row_major);
    
    #pragma unroll
    for (int k = 0; k < kTileN; k += kWmmaK) {
        nvcuda::wmma::load_matrix_sync(p_frag, probs + tile_m * kTilePadN + k, kTilePadN);
        nvcuda::wmma::load_matrix_sync(v_frag, v_tile + k * kTileD + tile_d, kTileD);
        nvcuda::wmma::mma_sync(o_frag, p_frag, v_frag, o_frag);
    }
    
    nvcuda::wmma::store_matrix_sync(dst, o_frag, kTileD, nvcuda::wmma::mem_row_major);
}

__global__ __launch_bounds__(384, 2) void fused_attention_kernel_v8(
    const half* __restrict__ Q,
    const half* __restrict__ K,
    const half* __restrict__ V,
    half* __restrict__ O,
    int B, int H, int S, int D, float scale) {
    
    // Dynamic SMEM allocation
    extern __shared__ char smem_base[];
    SMEMLayout smem(smem_base);
    
    const int batch_idx = blockIdx.z;
    const int head_idx = blockIdx.y;
    const int q_tile_idx = blockIdx.x;
    const int q_start = q_tile_idx * kTileM;
    
    const size_t bhs_offset = ((batch_idx * H + head_idx) * S) * D;
    const half* q_base = Q + bhs_offset;
    const half* k_base = K + bhs_offset;
    const half* v_base = V + bhs_offset;
    half* o_base = O + bhs_offset + q_start * D;
    
    // Initialize softmax state
    for (int i = threadIdx.x; i < kTileM; i += kThreadsPerBlock) {
        smem.m_state[i] = -INFINITY;
        smem.l_state[i] = 0.0f;
    }
    for (int i = threadIdx.x; i < kTileM * kTileD; i += kThreadsPerBlock) {
        smem.o_accum[i] = 0.0f;
    }
    __syncthreads();
    
    // Load Q tile
    load_tile_vectorized(smem.q_tile, q_base, kTileM, kTileD, D, q_start, S);
    __syncthreads();
    
    const int kv_tiles_count = (S + kTileN - 1) / kTileN;
    
    // Prefetch first KV tile
    if (kv_tiles_count > 0) {
        prefetch_kv_async(smem.kv_tiles[0][0], smem.kv_tiles[0][1], k_base, v_base, 0, kTileD, D, S);
        cp_async_commit();
    }
    
    // Main loop with pipelined prefetch
    for (int tile = 0; tile < kv_tiles_count; ++tile) {
        const int stage = tile % kStages;
        const int kv_start = tile * kTileN;
        const int kv_len = min(kTileN, S - kv_start);
        
        // Wait for current tile
        cp_async_wait<0>();
        __syncthreads();
        
        // Prefetch next tile while computing
        if (tile + 1 < kv_tiles_count) {
            const int next_stage = (tile + 1) % kStages;
            const int next_start = (tile + 1) * kTileN;
            prefetch_kv_async(smem.kv_tiles[next_stage][0], smem.kv_tiles[next_stage][1],
                            k_base, v_base, next_start, kTileD, D, S);
            cp_async_commit();
        }
        
        // Compute QK^T
        compute_qkt_wmma(smem.q_tile, smem.kv_tiles[stage][0], smem.scores, scale);
        __syncthreads();
        
        // Online softmax
        const int q_len = min(kTileM, S - q_start);
        compute_online_softmax_warp(smem.scores, smem.probs, smem.m_state, 
                                   smem.l_state, smem.o_accum, q_len, kv_len);
        __syncthreads();
        
        // P @ V
        compute_pv_wmma(smem.probs, smem.kv_tiles[stage][1], smem.o_accum, q_len);
        __syncthreads();
    }
    
    // Vectorized output write
    const int q_len = min(kTileM, S - q_start);
    constexpr int kVecSize = 8;
    const int vecs_total = q_len * (kTileD / kVecSize);
    
    for (int vec_idx = threadIdx.x; vec_idx < vecs_total; vec_idx += kThreadsPerBlock) {
        const int row = vec_idx / (kTileD / kVecSize);
        const int d_start = (vec_idx % (kTileD / kVecSize)) * kVecSize;
        
        const float inv_l = 1.0f / smem.l_state[row];
        const float* o_row = smem.o_accum + row * kTileD;
        half* out_row = o_base + row * D;
        
        half vec_data[kVecSize];
        #pragma unroll
        for (int i = 0; i < kVecSize; ++i) {
            vec_data[i] = __float2half(o_row[d_start + i] * inv_l);
        }
        
        *reinterpret_cast<uint4*>(out_row + d_start) = *reinterpret_cast<const uint4*>(vec_data);
    }
}

void launch_v8_dynamic(
    const half* Q, const half* K, const half* V, half* O,
    int B, int H, int S, int D, float scale, cudaStream_t stream) {
    
    // Set dynamic SMEM limit (one-time setup)
    static bool smem_configured = false;
    if (!smem_configured) {
        cudaFuncSetAttribute(
            fused_attention_kernel_v8,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            static_cast<int>(kTotalSMEM));
        
        cudaFuncSetAttribute(
            fused_attention_kernel_v8,
            cudaFuncAttributePreferredSharedMemoryCarveout,
            cudaSharedmemCarveoutMaxShared);
        
        smem_configured = true;
    }
    
    dim3 grid((S + kTileM - 1) / kTileM, H, B);
    dim3 block(kThreadsPerBlock);
    
    fused_attention_kernel_v8<<<grid, block, kTotalSMEM, stream>>>(
        Q, K, V, O, B, H, S, D, scale);
}

}  // namespace v8_dynamic
}  // namespace flashcore

extern "C" void flashcore_v8_dynamic_launch(
    const half* Q, const half* K, const half* V, half* O,
    int B, int H, int S, int D, float scale, cudaStream_t stream) {
    flashcore::v8_dynamic::launch_v8_dynamic(Q, K, V, O, B, H, S, D, scale, stream);
}



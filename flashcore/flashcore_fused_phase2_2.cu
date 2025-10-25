#include "flashcore_wmma_common.cuh"
#include <cuda_runtime_api.h>

namespace flashcore {
namespace fused_phase2_2 {

// Phase 2.2: Aggressive optimization for <40 μs
// - 40×40 tiles (fit in 48 KB SMEM limit)
// - Improved cp.async usage
// - Reduced synchronization
// - Optimized softmax
constexpr int kTileM = 40;
constexpr int kTileN = 40;
constexpr int kTileD = 64;
constexpr int kStages = 2;
constexpr int kWarpsPerBlock = 10;  // 10 warps = 320 threads
constexpr int kThreadsPerBlock = kWarpsPerBlock * kWarpSize;

// SMEM calculation (~43 KB - fits in 48 KB static limit!)
constexpr size_t kQTileBytes = sizeof(half) * kTileM * kTileD;  // 5 KB
constexpr size_t kKVTilesBytes = sizeof(half) * kStages * 2 * kTileN * kTileD;  // 20 KB
constexpr size_t kScoresBytes = sizeof(float) * kTileM * kTileN;  // 6.4 KB
constexpr size_t kProbsBytes = sizeof(half) * kTileM * kTileN;  // 3.2 KB
constexpr size_t kStateBytes = sizeof(float) * kTileM * 2;  // 0.3 KB
constexpr size_t kOAccumBytes = sizeof(float) * kTileM * kTileD;  // 10 KB

constexpr size_t kTotalSMEM = kQTileBytes + kKVTilesBytes + kScoresBytes + 
                              kProbsBytes + kStateBytes + kOAccumBytes;  // ~45 KB

static_assert(kTotalSMEM <= 48 * 1024, "SMEM exceeds 48 KB static limit!");

// Warp reduction helpers (optimized)
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

// Pad arrays to 48×48 for WMMA alignment (even though we only use 40×40)
constexpr int kTilePadM = 48;  // Round up to WMMA-friendly size
constexpr int kTilePadN = 48;

struct SharedStorage {
    __align__(16) half q_tile[kTilePadM * kTileD];
    __align__(16) half kv_tiles[kStages][2][kTilePadN * kTileD];
    __align__(16) float scores[kTilePadM * kTilePadN];  // Padded for WMMA stores
    __align__(16) half probs[kTilePadM * kTilePadN];    // Padded for WMMA stores
    __align__(16) float m_state[kTilePadM];
    __align__(16) float l_state[kTilePadM];
    __align__(16) float o_accum[kTilePadM * kTileD];
};

// Optimized vectorized load
__device__ __forceinline__ void load_tile_vectorized(
    half* dst, const half* src, int rows, int cols, int ld_src, 
    int row_offset, int s_bound) {
    
    constexpr int kVecSize = 8;
    const int vecs_per_row = cols / kVecSize;
    const int total_vecs = rows * vecs_per_row;
    
    #pragma unroll 2
    for (int vec_idx = threadIdx.x; vec_idx < total_vecs; vec_idx += kThreadsPerBlock) {
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

// Aggressive cp.async prefetch
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

// QK^T with WMMA (3×3 warp layout for 40×40, produces 48×48 with bounds check)
__device__ __forceinline__ void compute_qkt_wmma(
    const half* q_tile, const half* k_tile, float* scores, float scale) {
    
    const int warp_id = threadIdx.x / kWarpSize;
    
    // First 9 warps compute QK^T (3×3 layout covers 48×48, mask to 40×40)
    if (warp_id >= 9) return;
    
    const int warp_m = warp_id / 3;
    const int warp_n = warp_id % 3;
    const int tile_m = warp_m * kWmmaM;
    const int tile_n = warp_n * kWmmaN;
    
    // Bounds check for 40×40 tiles
    if (tile_m >= kTileM || tile_n >= kTileN) return;
    
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
    
    // Store with padded stride
    nvcuda::wmma::store_matrix_sync(scores + tile_m * kTilePadN + tile_n, c_frag, kTilePadN, nvcuda::wmma::mem_row_major);
}

// Optimized online softmax with warp-level reductions
__device__ __forceinline__ void compute_online_softmax_optimized(
    const float* scores, half* probs, float* m_state, float* l_state,
    float* o_accum, int rows, int cols) {
    
    const int lane_id = threadIdx.x % kWarpSize;
    const int warp_id = threadIdx.x / kWarpSize;
    
    for (int row = warp_id; row < rows; row += kWarpsPerBlock) {
        const float* score_row = scores + row * kTilePadN;  // Use padded stride
        
        // Warp-parallel max
        float m_tile = -INFINITY;
        #pragma unroll 4
        for (int col = lane_id; col < cols; col += kWarpSize) {
            m_tile = fmaxf(m_tile, score_row[col]);
        }
        m_tile = warp_reduce_max(m_tile);
        
        // Update global max and compute correction
        const float m_prev = m_state[row];
        const float m_new = fmaxf(m_prev, m_tile);
        const float alpha = expf(m_prev - m_new);
        const float beta = expf(m_tile - m_new);
        
        // Warp-parallel exp and sum
        float l_tile = 0.0f;
        half* prob_row = probs + row * kTilePadN;  // Use padded stride
        
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
        
        // Update state (lane 0 only)
        if (lane_id == 0) {
            m_state[row] = m_new;
            l_state[row] = l_new;
        }
    }
}

// P·V with WMMA (3×4 warp layout for 40×64 output, produces 48×64 with bounds check)
__device__ __forceinline__ void compute_pv_wmma(
    const half* probs, const half* v_tile, float* o_accum, int rows) {
    
    const int warp_id = threadIdx.x / kWarpSize;
    
    // First 10 warps (all available warps)
    if (warp_id >= kWarpsPerBlock) return;
    
    const int warp_m = warp_id / 4;  // 3 rows (covers up to 48)
    const int warp_d = warp_id % 4;   // 4 cols (covers 64)
    const int tile_m = warp_m * kWmmaM;
    const int tile_d = warp_d * kWmmaN;
    
    // Bounds check for 40×64 output
    if (tile_m >= rows || tile_d >= kTileD) return;
    
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, kWmmaM, kWmmaN, kWmmaK, half, nvcuda::wmma::row_major> p_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, kWmmaM, kWmmaN, kWmmaK, half, nvcuda::wmma::row_major> v_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, kWmmaM, kWmmaN, kWmmaK, float> o_frag;
    
    float* dst = o_accum + tile_m * kTileD + tile_d;
    nvcuda::wmma::load_matrix_sync(o_frag, dst, kTileD, nvcuda::wmma::mem_row_major);
    
    #pragma unroll
    for (int k = 0; k < kTileN; k += kWmmaK) {
        // Load from padded probs array
        nvcuda::wmma::load_matrix_sync(p_frag, probs + tile_m * kTilePadN + k, kTilePadN);
        nvcuda::wmma::load_matrix_sync(v_frag, v_tile + k * kTileD + tile_d, kTileD);
        nvcuda::wmma::mma_sync(o_frag, p_frag, v_frag, o_frag);
    }
    
    nvcuda::wmma::store_matrix_sync(dst, o_frag, kTileD, nvcuda::wmma::mem_row_major);
}

__global__ __launch_bounds__(320, 2) void fused_attention_kernel_phase2_2(
    const half* __restrict__ Q,
    const half* __restrict__ K,
    const half* __restrict__ V,
    half* __restrict__ O,
    int B, int H, int S, int D, float scale) {
    
    __shared__ SharedStorage shared;
    
    const int batch_idx = blockIdx.z;
    const int head_idx = blockIdx.y;
    const int q_tile_idx = blockIdx.x;
    const int q_start = q_tile_idx * kTileM;
    
    const size_t bhs_offset = ((batch_idx * H + head_idx) * S) * D;
    const half* q_base = Q + bhs_offset;
    const half* k_base = K + bhs_offset;
    const half* v_base = V + bhs_offset;
    half* o_base = O + bhs_offset + q_start * D;
    
    // Initialize state
    for (int i = threadIdx.x; i < kTileM; i += kThreadsPerBlock) {
        shared.m_state[i] = -INFINITY;
        shared.l_state[i] = 0.0f;
    }
    for (int i = threadIdx.x; i < kTileM * kTileD; i += kThreadsPerBlock) {
        shared.o_accum[i] = 0.0f;
    }
    __syncthreads();
    
    // Load Q tile
    load_tile_vectorized(shared.q_tile, q_base, kTileM, kTileD, D, q_start, S);
    __syncthreads();
    
    const int kv_tiles_count = (S + kTileN - 1) / kTileN;
    
    // Prefetch first KV tile
    if (kv_tiles_count > 0) {
        prefetch_kv_async(shared.kv_tiles[0][0], shared.kv_tiles[0][1], k_base, v_base, 0, kTileD, D, S);
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
        
        // Prefetch next tile while computing current
        if (tile + 1 < kv_tiles_count) {
            const int next_stage = (tile + 1) % kStages;
            const int next_start = (tile + 1) * kTileN;
            prefetch_kv_async(shared.kv_tiles[next_stage][0], shared.kv_tiles[next_stage][1],
                            k_base, v_base, next_start, kTileD, D, S);
            cp_async_commit();
        }
        
        // Compute QK^T
        compute_qkt_wmma(shared.q_tile, shared.kv_tiles[stage][0], shared.scores, scale);
        __syncthreads();
        
        // Online softmax
        const int q_len = min(kTileM, S - q_start);
        compute_online_softmax_optimized(shared.scores, shared.probs, shared.m_state, 
                                        shared.l_state, shared.o_accum, q_len, kv_len);
        __syncthreads();
        
        // P @ V
        compute_pv_wmma(shared.probs, shared.kv_tiles[stage][1], shared.o_accum, q_len);
        __syncthreads();
    }
    
    // Vectorized output write
    const int q_len = min(kTileM, S - q_start);
    constexpr int kVecSize = 8;
    const int vecs_total = q_len * (kTileD / kVecSize);
    
    for (int vec_idx = threadIdx.x; vec_idx < vecs_total; vec_idx += kThreadsPerBlock) {
        const int row = vec_idx / (kTileD / kVecSize);
        const int d_start = (vec_idx % (kTileD / kVecSize)) * kVecSize;
        
        const float inv_l = 1.0f / shared.l_state[row];
        const float* o_row = shared.o_accum + row * kTileD;
        half* out_row = o_base + row * D;
        
        half vec_data[kVecSize];
        #pragma unroll
        for (int i = 0; i < kVecSize; ++i) {
            vec_data[i] = __float2half(o_row[d_start + i] * inv_l);
        }
        
        *reinterpret_cast<uint4*>(out_row + d_start) = *reinterpret_cast<const uint4*>(vec_data);
    }
}

void launch_fused_phase2_2(
    const half* Q, const half* K, const half* V, half* O,
    int B, int H, int S, int D, float scale, cudaStream_t stream) {
    
    dim3 grid((S + kTileM - 1) / kTileM, H, B);
    dim3 block(kThreadsPerBlock);
    
    fused_attention_kernel_phase2_2<<<grid, block, 0, stream>>>(
        Q, K, V, O, B, H, S, D, scale);
}

}  // namespace fused_phase2_2
}  // namespace flashcore

extern "C" void flashcore_fused_phase2_2_launch(
    const half* Q, const half* K, const half* V, half* O,
    int B, int H, int S, int D, float scale, cudaStream_t stream) {
    flashcore::fused_phase2_2::launch_fused_phase2_2(Q, K, V, O, B, H, S, D, scale, stream);
}


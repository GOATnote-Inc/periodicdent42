#include "flashcore_wmma_common.cuh"
#include <cuda_runtime_api.h>

namespace flashcore {
namespace fused_phase2 {

// Phase 2.0: 64×64 tiles with dynamic SMEM (<40 μs target)
// Uses extern __shared__ to bypass 48 KB static SMEM limit
constexpr int kTileM = 64;
constexpr int kTileN = 64;
constexpr int kTileD = 64;
constexpr int kStages = 2;
constexpr int kWarpsPerBlock = 16;  // 16 warps for 64×64 tiles (4×4 layout)
constexpr int kThreadsPerBlock = kWarpsPerBlock * kWarpSize;  // 512 threads

// Calculate required dynamic SMEM (bytes)
constexpr size_t align_to_16(size_t size) {
    return (size + 15) & ~15;  // Round up to 16-byte boundary
}

constexpr size_t kQTileBytes = align_to_16(sizeof(half) * kTileM * kTileD);
constexpr size_t kKVTilesBytes = align_to_16(sizeof(half) * kStages * 2 * kTileN * kTileD);
constexpr size_t kScoresBytes = align_to_16(sizeof(float) * kTileM * kTileN);
constexpr size_t kProbsBytes = align_to_16(sizeof(half) * kTileM * kTileN);
constexpr size_t kMStateBytes = align_to_16(sizeof(float) * kTileM);
constexpr size_t kLStateBytes = align_to_16(sizeof(float) * kTileM);
constexpr size_t kOAccumBytes = align_to_16(sizeof(float) * kTileM * kTileD);

constexpr size_t kTotalSMEM = 
    kQTileBytes +      // Q tile
    kKVTilesBytes +    // K/V tiles (double-buffered)
    kScoresBytes +     // QK^T scores
    kProbsBytes +      // Softmax probs
    kMStateBytes +     // Max state
    kLStateBytes +     // Sum state
    kOAccumBytes;      // Output accumulator

// Static assert to catch SMEM overflow at compile time
static_assert(kTotalSMEM <= 100 * 1024, "SMEM usage exceeds 100 KB limit on L4!");

// Warp-level reduction helpers (from Phase 1.3)
__device__ __forceinline__ float warp_reduce_max(float val) {
    unsigned mask = __activemask();
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_xor_sync(mask, val, offset));
    }
    return val;
}

__device__ __forceinline__ float warp_reduce_sum(float val) {
    unsigned mask = __activemask();
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_xor_sync(mask, val, offset);
    }
    return val;
}

// Vectorized Q load (8-element half2x4 = 16 bytes)
__device__ __forceinline__ void load_q_tile(
    half* q_smem,
    const half* q_gmem,
    int rows,
    int cols,
    int ld_gmem,
    int row_offset,
    int s_bound) {
    
    constexpr int kVecSize = 8;  // Load 8 halfs = 16 bytes per thread
    const int vecs_per_row = cols / kVecSize;
    
    for (int vec_idx = threadIdx.x; vec_idx < rows * vecs_per_row; vec_idx += kThreadsPerBlock) {
        const int row = vec_idx / vecs_per_row;
        const int col_vec = vec_idx % vecs_per_row;
        const int col = col_vec * kVecSize;
        const int global_row = row_offset + row;
        
        uint4 data = make_uint4(0, 0, 0, 0);
        if (global_row < s_bound) {
            const uint4* src = reinterpret_cast<const uint4*>(q_gmem + global_row * ld_gmem + col);
            data = *src;
        }
        
        uint4* dst = reinterpret_cast<uint4*>(q_smem + row * cols + col);
        *dst = data;
    }
}

// Vectorized K/V load
__device__ __forceinline__ void load_kv_tile(
    half* kv_smem,
    const half* kv_gmem,
    int tile_row,
    int cols,
    int ld_gmem,
    int s_bound,
    int d_bound) {
    
    constexpr int kVecSize = 8;
    const int vecs_per_row = cols / kVecSize;
    
    for (int vec_idx = threadIdx.x; vec_idx < kTileN * vecs_per_row; vec_idx += kThreadsPerBlock) {
        const int row = vec_idx / vecs_per_row;
        const int col_vec = vec_idx % vecs_per_row;
        const int col = col_vec * kVecSize;
        const int global_row = tile_row + row;
        
        uint4 data = make_uint4(0, 0, 0, 0);
        const bool row_valid = global_row < s_bound;
        const bool col_valid = (col + kVecSize) <= d_bound;
        
        if (row_valid && col_valid) {
            const uint4* src = reinterpret_cast<const uint4*>(kv_gmem + global_row * ld_gmem + col);
            data = *src;
        }
        
        uint4* dst = reinterpret_cast<uint4*>(kv_smem + row * cols + col);
        *dst = data;
    }
}

// QK^T with WMMA (4×4 warp layout for 64×64)
__device__ __forceinline__ void compute_qkt_wmma(
    const half* q_tile,
    const half* k_tile,
    float* scores,
    float scale) {
    
    const int warp_id = threadIdx.x / kWarpSize;
    const int warp_m = warp_id / 4;  // 4×4 layout
    const int warp_n = warp_id % 4;
    
    const int tile_m = warp_m * kWmmaM;
    const int tile_n = warp_n * kWmmaN;
    
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, kWmmaM, kWmmaN, kWmmaK, half, nvcuda::wmma::row_major> a_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, kWmmaM, kWmmaN, kWmmaK, half, nvcuda::wmma::col_major> b_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, kWmmaM, kWmmaN, kWmmaK, float> c_frag;
    
    nvcuda::wmma::fill_fragment(c_frag, 0.0f);
    
    #pragma unroll
    for (int k = 0; k < kTileD; k += kWmmaK) {
        const half* a_ptr = q_tile + tile_m * kTileD + k;
        const half* b_ptr = k_tile + tile_n * kTileD + k;
        
        nvcuda::wmma::load_matrix_sync(a_frag, a_ptr, kTileD);
        nvcuda::wmma::load_matrix_sync(b_frag, b_ptr, kTileD);
        nvcuda::wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }
    
    // Apply scale
    #pragma unroll
    for (int i = 0; i < c_frag.num_elements; ++i) {
        c_frag.x[i] *= scale;
    }
    
    float* dst = scores + tile_m * kTileN + tile_n;
    nvcuda::wmma::store_matrix_sync(dst, c_frag, kTileN, nvcuda::wmma::mem_row_major);
}

// Warp-synchronous online softmax (from Phase 1.3)
__device__ __forceinline__ void compute_online_softmax_warp(
    const float* scores,
    half* probs,
    float* m_state,
    float* l_state,
    float* o_accum,
    int rows,
    int cols) {
    
    const int lane_id = threadIdx.x % kWarpSize;
    const int warp_id = threadIdx.x / kWarpSize;
    
    for (int row = warp_id; row < rows; row += kWarpsPerBlock) {
        const float* score_row = scores + row * kTileN;
        
        // 1. Warp-parallel max
        float m_tile = -INFINITY;
        for (int col = lane_id; col < cols; col += kWarpSize) {
            m_tile = fmaxf(m_tile, score_row[col]);
        }
        m_tile = warp_reduce_max(m_tile);
        
        // 2. Update global max
        float m_prev = m_state[row];
        float m_new = fmaxf(m_prev, m_tile);
        
        // 3. Correction factors
        float alpha = expf(m_prev - m_new);
        float beta = expf(m_tile - m_new);
        
        // 4. Warp-parallel exp and sum
        float l_prev = l_state[row];
        float l_tile = 0.0f;
        
        half* prob_row = probs + row * kTileN;
        for (int col = lane_id; col < cols; col += kWarpSize) {
            float prob = beta * expf(score_row[col] - m_tile);
            prob_row[col] = __float2half(prob);
            l_tile += prob;
        }
        l_tile = warp_reduce_sum(l_tile);
        
        float l_new = alpha * l_prev + l_tile;
        
        // 5. Rescale O accumulator
        float rescale = alpha;
        float* o_row = o_accum + row * kTileD;
        for (int d = lane_id; d < kTileD; d += kWarpSize) {
            o_row[d] *= rescale;
        }
        
        // 6. Update state (lane 0 writes)
        if (lane_id == 0) {
            m_state[row] = m_new;
            l_state[row] = l_new;
        }
    }
}

// P·V with WMMA (4×4 warp layout for 64×64)
__device__ __forceinline__ void compute_pv_wmma(
    const half* probs,
    const half* v_tile,
    float* o_accum,
    int rows,
    int cols) {
    
    const int warp_id = threadIdx.x / kWarpSize;
    const int warp_m = warp_id / 4;  // 4×4 layout
    const int warp_d = warp_id % 4;
    
    const int tile_m = warp_m * kWmmaM;
    const int tile_d = warp_d * kWmmaN;
    
    if (tile_m >= rows || tile_d >= kTileD) return;
    
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, kWmmaM, kWmmaN, kWmmaK, half, nvcuda::wmma::row_major> p_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, kWmmaM, kWmmaN, kWmmaK, half, nvcuda::wmma::row_major> v_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, kWmmaM, kWmmaN, kWmmaK, float> o_frag;
    
    // Load existing accumulator
    float* dst = o_accum + tile_m * kTileD + tile_d;
    nvcuda::wmma::load_matrix_sync(o_frag, dst, kTileD, nvcuda::wmma::mem_row_major);
    
    #pragma unroll
    for (int k = 0; k < cols; k += kWmmaK) {
        const half* p_ptr = probs + tile_m * kTileN + k;
        const half* v_ptr = v_tile + k * kTileD + tile_d;
        
        nvcuda::wmma::load_matrix_sync(p_frag, p_ptr, kTileN);
        nvcuda::wmma::load_matrix_sync(v_frag, v_ptr, kTileD);
        nvcuda::wmma::mma_sync(o_frag, p_frag, v_frag, o_frag);
    }
    
    nvcuda::wmma::store_matrix_sync(dst, o_frag, kTileD, nvcuda::wmma::mem_row_major);
}

__global__ __launch_bounds__(512, 1) void fused_attention_kernel_phase2(
    const half* __restrict__ Q,
    const half* __restrict__ K,
    const half* __restrict__ V,
    half* __restrict__ O,
    int B,
    int H,
    int S,
    int D,
    float scale) {
    
    // Dynamic SMEM allocation
    extern __shared__ char smem[];
    
    // Calculate buffer pointers with 16-byte alignment
    size_t offset = 0;
    
    half* q_tile = reinterpret_cast<half*>(smem + offset);
    offset += kQTileBytes;
    
    half* kv_tiles = reinterpret_cast<half*>(smem + offset);
    offset += kKVTilesBytes;
    
    float* scores = reinterpret_cast<float*>(smem + offset);
    offset += kScoresBytes;
    
    half* probs = reinterpret_cast<half*>(smem + offset);
    offset += kProbsBytes;
    
    float* m_state = reinterpret_cast<float*>(smem + offset);
    offset += kMStateBytes;
    
    float* l_state = reinterpret_cast<float*>(smem + offset);
    offset += kLStateBytes;
    
    float* o_accum = reinterpret_cast<float*>(smem + offset);
    
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
    for (int row = threadIdx.x; row < kTileM; row += kThreadsPerBlock) {
        m_state[row] = -INFINITY;
        l_state[row] = 0.0f;
    }
    
    // Initialize output accumulator
    for (int idx = threadIdx.x; idx < kTileM * kTileD; idx += kThreadsPerBlock) {
        o_accum[idx] = 0.0f;
    }
    
    // Load Q tile
    load_q_tile(q_tile, q_base, kTileM, kTileD, D, q_start, S);
    __syncthreads();
    
    // Process K/V tiles
    const int kv_tiles_count = (S + kTileN - 1) / kTileN;
    
    for (int kv_tile_idx = 0; kv_tile_idx < kv_tiles_count; ++kv_tile_idx) {
        const int kv_start = kv_tile_idx * kTileN;
        const int kv_len = min(kTileN, S - kv_start);
        
        // Load K and V tiles
        half* k_tile = kv_tiles;
        half* v_tile = kv_tiles + kTileN * kTileD;
        
        load_kv_tile(k_tile, k_base, kv_start, kTileD, D, S, D);
        load_kv_tile(v_tile, v_base, kv_start, kTileD, D, S, D);
        __syncthreads();
        
        // Compute QK^T
        compute_qkt_wmma(q_tile, k_tile, scores, scale);
        __syncthreads();
        
        // Online softmax
        const int rows = min(kTileM, S - q_start);
        compute_online_softmax_warp(scores, probs, m_state, l_state, o_accum, rows, kv_len);
        __syncthreads();
        
        // Compute P·V
        compute_pv_wmma(probs, v_tile, o_accum, rows, kv_len);
        __syncthreads();
    }
    
    // Final normalization and output
    const int q_len = min(kTileM, S - q_start);
    for (int row = threadIdx.x; row < q_len; row += kThreadsPerBlock) {
        float l_final = l_state[row];
        const float* o_row = o_accum + row * kTileD;
        half* out_row = o_base + row * D;
        
        for (int d = 0; d < kTileD; ++d) {
            out_row[d] = __float2half(o_row[d] / l_final);
        }
    }
}

void launch_fused_phase2(
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
    
    // Set up dynamic SMEM attributes (one-time setup)
    static bool attributes_set = false;
    if (!attributes_set) {
        cudaFuncSetAttribute(
            fused_attention_kernel_phase2,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            static_cast<int>(kTotalSMEM));
        
        cudaFuncSetAttribute(
            fused_attention_kernel_phase2,
            cudaFuncAttributePreferredSharedMemoryCarveout,
            cudaSharedmemCarveoutMaxShared);
        
        attributes_set = true;
    }
    
    dim3 grid((S + kTileM - 1) / kTileM, H, B);
    dim3 block(kThreadsPerBlock);
    
    fused_attention_kernel_phase2<<<grid, block, kTotalSMEM, stream>>>(
        Q, K, V, O, B, H, S, D, scale);
}

}  // namespace fused_phase2
}  // namespace flashcore

extern "C" void flashcore_fused_phase2_launch(
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
    flashcore::fused_phase2::launch_fused_phase2(Q, K, V, O, B, H, S, D, scale, stream);
}

extern "C" size_t flashcore_fused_phase2_get_smem_bytes() {
    return flashcore::fused_phase2::kTotalSMEM;
}


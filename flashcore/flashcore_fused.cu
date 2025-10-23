#include "flashcore_wmma_common.cuh"
#include <cuda_runtime_api.h>

namespace flashcore {
namespace fused {

// Reduced tile sizes to fit in 48 KB shared memory limit
constexpr int kTileM = 32;
constexpr int kTileN = 32;
constexpr int kTileD = 64;
constexpr int kStages = 2;
constexpr int kWarpsPerBlock = 4;  // 4 warps for 32×32 tiles (2×2 warp layout)
constexpr int kThreadsPerBlock = kWarpsPerBlock * kWarpSize;

struct SharedStorage {
    __align__(16) half q_tile[kTileM * kTileD];
    __align__(16) half kv_tiles[kStages][2][kTileN * kTileD];  // [stage][K=0,V=1][data]
    float scores[kTileM * kTileN];    // QK^T scores (F32)
    half probs[kTileM * kTileN];      // Softmax output (F16)
    float m_state[kTileM];            // Max per row
    float l_state[kTileM];            // Sum per row
    float o_accum[kTileM * kTileD];   // Output accumulator (F32)
};

// Warp reduce max
__device__ __forceinline__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, offset));
    }
    return val;
}

// Warp reduce sum
__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_xor_sync(0xffffffff, val, offset);
    }
    return val;
}

// Vectorized Q load (reuse from v6)
__device__ __forceinline__ void load_q_tile(
    half* dst, const half* src, int rows, int cols, int ld_src, int row_offset, int s_bound) {
    constexpr int elems_per_vec = 8;
    const int vecs_per_row = cols / elems_per_vec;
    for (int vec_idx = threadIdx.x; vec_idx < rows * vecs_per_row; vec_idx += kThreadsPerBlock) {
        const int row = vec_idx / vecs_per_row;
        const int col_vec = vec_idx % vecs_per_row;
        const int col = col_vec * elems_per_vec;
        const int global_row = row_offset + row;
        uint4 value = make_uint4(0, 0, 0, 0);
        if (global_row < s_bound) {
            const uint4* gmem_ptr = reinterpret_cast<const uint4*>(src + global_row * ld_src + col);
            value = *gmem_ptr;
        }
        uint4* smem_ptr = reinterpret_cast<uint4*>(dst + row * cols + col);
        *smem_ptr = value;
    }
}

// Prefetch K or V tile with cp.async
__device__ __forceinline__ void prefetch_kv_tile(
    half* dst, const half* src, int tile_row, int cols, int ld_src, int s_bound, int d_bound) {
    constexpr int elems_per_vec = 8;
    const int vecs_per_row = cols / elems_per_vec;
    for (int vec_idx = threadIdx.x; vec_idx < kTileN * vecs_per_row; vec_idx += kThreadsPerBlock) {
        const int row = vec_idx / vecs_per_row;
        const int col_vec = vec_idx % vecs_per_row;
        const int col = col_vec * elems_per_vec;
        const int global_row = tile_row + row;
        half* smem_ptr = dst + row * cols + col;
        const half* gmem_ptr = src + global_row * ld_src + col;
        const bool row_valid = global_row < s_bound;
        const bool col_valid = (col + elems_per_vec) <= d_bound;
        if (row_valid && col_valid) {
            cp_async_cg(smem_ptr, gmem_ptr);
        } else {
            *reinterpret_cast<uint4*>(smem_ptr) = make_uint4(0, 0, 0, 0);
        }
    }
}

// Compute QK^T with WMMA (output F32 scores)
__device__ __forceinline__ void compute_qkt_wmma(
    const half* q_tile, const half* k_tile, float* scores, float scale) {
    const int warp_id = threadIdx.x / kWarpSize;
    const int warp_m = warp_id / 2;  // 2×4 warp layout for 32×32 tiles
    const int warp_n = warp_id % 2;

    const int tile_m = warp_m * kWmmaM;
    const int tile_n = warp_n * kWmmaN;

    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, kWmmaM, kWmmaN, kWmmaK, half, nvcuda::wmma::row_major> a_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, kWmmaM, kWmmaN, kWmmaK, half, nvcuda::wmma::col_major> b_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, kWmmaM, kWmmaN, kWmmaK, float> c_frag;

    nvcuda::wmma::fill_fragment(c_frag, 0.0f);

    #pragma unroll
    for (int k = 0; k < kTileD; k += kWmmaK) {
        const half* q_ptr = q_tile + tile_m * kTileD + k;
        const half* k_ptr = k_tile + tile_n * kTileD + k;
        nvcuda::wmma::load_matrix_sync(a_frag, q_ptr, kTileD);
        nvcuda::wmma::load_matrix_sync(b_frag, k_ptr, kTileD);
        nvcuda::wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    // Scale and store
    #pragma unroll
    for (int i = 0; i < c_frag.num_elements; ++i) {
        c_frag.x[i] *= scale;
    }

    float* dst = scores + tile_m * kTileN + tile_n;
    nvcuda::wmma::store_matrix_sync(dst, c_frag, kTileN, nvcuda::wmma::mem_row_major);
}

// Step 1: Compute softmax and rescale O
__device__ __forceinline__ void compute_online_softmax(
    const float* scores,    // [kTileM, kTileN]
    half* probs,           // [kTileM, kTileN] output
    float* m_state,        // [kTileM] max
    float* l_state,        // [kTileM] sum
    float* o_accum,        // [kTileM, kTileD] accumulator
    int rows,
    int cols) {
    
    // Each thread processes multiple rows
    for (int row = threadIdx.x; row < rows; row += kThreadsPerBlock) {
        const float* score_row = scores + row * kTileN;
        
        // 1. Find max in this tile
        float m_tile = -INFINITY;
        for (int col = 0; col < cols; ++col) {
            m_tile = fmaxf(m_tile, score_row[col]);
        }
        
        // 2. Update global max
        float m_prev = m_state[row];
        float m_new = fmaxf(m_prev, m_tile);
        
        // 3. Compute correction factors
        float alpha = expf(m_prev - m_new);  // Correction for previous tiles
        float beta = expf(m_tile - m_new);   // Scale for current tile
        
        // 4. Update sum and compute probabilities
        float l_prev = l_state[row];
        float l_new = alpha * l_prev;
        
        half* prob_row = probs + row * kTileN;
        for (int col = 0; col < cols; ++col) {
            float prob = beta * expf(score_row[col] - m_tile);
            prob_row[col] = __float2half(prob);
            l_new += prob;
        }
        
        // 5. Rescale previous O accumulator
        float rescale = alpha;
        float* o_row = o_accum + row * kTileD;
        for (int d = 0; d < kTileD; ++d) {
            o_row[d] *= rescale;
        }
        
        // 6. Update state
        m_state[row] = m_new;
        l_state[row] = l_new;
    }
}

// Step 2: WMMA for P @ V (Tensor Core acceleration!)
__device__ __forceinline__ void compute_pv_wmma(
    const half* probs,     // [kTileM, kTileN] probabilities
    const half* v_tile,    // [kTileN, kTileD] values
    float* o_accum) {      // [kTileM, kTileD] output accumulator
    
    const int warp_id = threadIdx.x / kWarpSize;
    const int warp_m = warp_id / 2;  // 2×2 warp layout for 32×32 tiles
    const int warp_n = warp_id % 2;

    const int tile_m = warp_m * kWmmaM;
    const int tile_d = warp_n * kWmmaN;

    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, kWmmaM, kWmmaN, kWmmaK, half, nvcuda::wmma::row_major> p_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, kWmmaM, kWmmaN, kWmmaK, half, nvcuda::wmma::row_major> v_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, kWmmaM, kWmmaN, kWmmaK, float> o_frag;

    float* dst = o_accum + tile_m * kTileD + tile_d;
    
    // Load existing accumulator
    nvcuda::wmma::load_matrix_sync(o_frag, dst, kTileD, nvcuda::wmma::mem_row_major);

    // Compute P @ V and accumulate: O += P @ V
    #pragma unroll
    for (int k = 0; k < kTileN; k += kWmmaK) {
        const half* p_ptr = probs + tile_m * kTileN + k;
        const half* v_ptr = v_tile + k * kTileD + tile_d;
        nvcuda::wmma::load_matrix_sync(p_frag, p_ptr, kTileN);
        nvcuda::wmma::load_matrix_sync(v_frag, v_ptr, kTileD);
        nvcuda::wmma::mma_sync(o_frag, p_frag, v_frag, o_frag);
    }

    // Store accumulated result back
    nvcuda::wmma::store_matrix_sync(dst, o_frag, kTileD, nvcuda::wmma::mem_row_major);
}

__global__ __launch_bounds__(128, 2) void fused_attention_kernel(
    const half* __restrict__ Q,
    const half* __restrict__ K,
    const half* __restrict__ V,
    half* __restrict__ O,
    int B,
    int H,
    int S,
    int D,
    float scale) {
    
    __shared__ SharedStorage shared;

    const int batch_idx = blockIdx.z;
    const int head_idx = blockIdx.y;
    const int q_tile_idx = blockIdx.x;

    const int q_start = q_tile_idx * kTileM;
    const int kv_tiles = (S + kTileN - 1) / kTileN;

    const int bhs_offset = ((batch_idx * H + head_idx) * S) * D;
    const half* q_base = Q + bhs_offset;
    const half* k_base = K + bhs_offset;
    const half* v_base = V + bhs_offset;
    half* o_base = O + ((batch_idx * H + head_idx) * S + q_start) * D;

    // Load Q tile (stays resident for all K/V tiles)
    load_q_tile(shared.q_tile, q_base, kTileM, kTileD, D, q_start, S);
    
    // Initialize softmax state
    for (int idx = threadIdx.x; idx < kTileM; idx += kThreadsPerBlock) {
        shared.m_state[idx] = -INFINITY;
        shared.l_state[idx] = 0.0f;
    }
    
    // Initialize output accumulator
    for (int idx = threadIdx.x; idx < kTileM * kTileD; idx += kThreadsPerBlock) {
        shared.o_accum[idx] = 0.0f;
    }
    __syncthreads();

    // Preload first K/V tiles
    const int preload = kv_tiles < kStages ? kv_tiles : kStages;
    for (int t = 0; t < preload; ++t) {
        const int kv_start = t * kTileN;
        half* k_dst = &shared.kv_tiles[t % kStages][0][0];
        half* v_dst = &shared.kv_tiles[t % kStages][1][0];
        prefetch_kv_tile(k_dst, k_base, kv_start, kTileD, D, S, D);
        prefetch_kv_tile(v_dst, v_base, kv_start, kTileD, D, S, D);
        cp_async_commit();
    }
    cp_async_wait<0>();
    cp_async_fence();
    __syncthreads();

    // Process all K/V tiles
    for (int tile = 0; tile < kv_tiles; ++tile) {
        const int stage = tile % kStages;
        const int kv_start = tile * kTileN;
        const int kv_len = min(kTileN, S - kv_start);
        
        half* k_tile = &shared.kv_tiles[stage][0][0];
        half* v_tile = &shared.kv_tiles[stage][1][0];

        // Compute QK^T for this tile
        compute_qkt_wmma(shared.q_tile, k_tile, shared.scores, scale);
        __syncthreads();

        // Step 1: Online softmax (compute probabilities and rescale O)
        const int q_len = min(kTileM, S - q_start);
        compute_online_softmax(
            shared.scores,
            shared.probs,
            shared.m_state,
            shared.l_state,
            shared.o_accum,
            q_len,
            kv_len);
        __syncthreads();

        // Step 2: P @ V using WMMA (Tensor Cores!)
        compute_pv_wmma(
            shared.probs,
            v_tile,
            shared.o_accum);
        __syncthreads();

        // Prefetch next K/V tiles
        const int next_tile = tile + kStages;
        if (next_tile < kv_tiles) {
            const int next_start = next_tile * kTileN;
            half* k_dst = &shared.kv_tiles[stage][0][0];
            half* v_dst = &shared.kv_tiles[stage][1][0];
            prefetch_kv_tile(k_dst, k_base, next_start, kTileD, D, S, D);
            prefetch_kv_tile(v_dst, v_base, next_start, kTileD, D, S, D);
            cp_async_commit();
            cp_async_wait<0>();
            cp_async_fence();
            __syncthreads();
        }
    }

    // Final normalization and write output
    const int q_len = min(kTileM, S - q_start);
    for (int row = threadIdx.x; row < q_len; row += kThreadsPerBlock) {
        float l_final = shared.l_state[row];
        float* o_row = shared.o_accum + row * kTileD;
        half* out_row = o_base + row * D;
        
        for (int d = 0; d < kTileD; ++d) {
            out_row[d] = __float2half(o_row[d] / l_final);
        }
    }
}

void launch_fused(
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
    
    // Opt-in for 100% shared memory carveout (L4 allows up to 99 KB per SM)
    cudaFuncSetAttribute(
        fused_attention_kernel,
        cudaFuncAttributePreferredSharedMemoryCarveout,
        cudaSharedmemCarveoutMaxShared);
    
    dim3 grid((S + kTileM - 1) / kTileM, H, B);
    dim3 block(kThreadsPerBlock);
    
    fused_attention_kernel<<<grid, block, 0, stream>>>(
        Q, K, V, O, B, H, S, D, scale);
}

}  // namespace fused
}  // namespace flashcore

extern "C" void flashcore_fused_launch(
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
    flashcore::fused::launch_fused(Q, K, V, O, B, H, S, D, scale, stream);
}


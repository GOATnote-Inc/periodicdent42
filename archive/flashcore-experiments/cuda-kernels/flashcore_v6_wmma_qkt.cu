#include "flashcore_wmma_common.cuh"

#include <cuda_runtime_api.h>

namespace flashcore {
namespace v6 {

constexpr int kTileM = 64;
constexpr int kTileN = 64;
constexpr int kTileK = 64;
constexpr int kStages = 2;
constexpr int kWarpsPerBlock = 8;  // 8 warps × 32 threads = 256 threads
constexpr int kThreadsPerBlock = kWarpsPerBlock * kWarpSize;

struct SharedStorage {
    __align__(16) half q_tile[kTileM * kTileK];
    __align__(16) half k_tiles[kStages][kTileN * kTileK];
    float accum[kTileM * kTileN];
};

__device__ __forceinline__ void vectorized_q_load(
    half* dst, const half* src, int rows, int cols, int ld_src, int ld_dst, int row_offset, int s_bound) {
    constexpr int elems_per_vec = 8;  // 8 × half = 16 bytes
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
        uint4* smem_ptr = reinterpret_cast<uint4*>(dst + row * ld_dst + col);
        *smem_ptr = value;
    }
}

__device__ __forceinline__ void prefetch_k_tile(
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
#if FLASHCORE_CP_ASYNC_SUPPORTED
        if (row_valid && col_valid) {
            cp_async_cg(smem_ptr, gmem_ptr);
        } else if (row_valid) {
            *reinterpret_cast<uint4*>(smem_ptr) = make_uint4(0, 0, 0, 0);
        } else {
            *reinterpret_cast<uint4*>(smem_ptr) = make_uint4(0, 0, 0, 0);
        }
#else
        if (row_valid && col_valid) {
            *reinterpret_cast<uint4*>(smem_ptr) = *reinterpret_cast<const uint4*>(gmem_ptr);
        } else {
            *reinterpret_cast<uint4*>(smem_ptr) = make_uint4(0, 0, 0, 0);
        }
#endif
    }
}

__device__ __forceinline__ void mma_qk(const half* q_tile, const half* k_tile, float* accum, float scale) {
    const int warp_id = threadIdx.x / kWarpSize;
    const int warp_m = warp_id / 4;
    const int warp_n = warp_id % 4;

    const int tile_m = warp_m * kWmmaM;
    const int tile_n = warp_n * kWmmaN;

    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, kWmmaM, kWmmaN, kWmmaK, half, nvcuda::wmma::row_major> a_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, kWmmaM, kWmmaN, kWmmaK, half, nvcuda::wmma::col_major> b_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, kWmmaM, kWmmaN, kWmmaK, float> c_frag;

    nvcuda::wmma::fill_fragment(c_frag, 0.0f);

    #pragma unroll
    for (int k = 0; k < kTileK; k += kWmmaK) {
        const half* q_ptr = q_tile + tile_m * kTileK + k;
        // K stored as [N][K], access as K[tile_n][k] for col-major K^T
        const half* k_ptr = k_tile + tile_n * kTileK + k;
        nvcuda::wmma::load_matrix_sync(a_frag, q_ptr, kTileK);
        nvcuda::wmma::load_matrix_sync(b_frag, k_ptr, kTileK);
        nvcuda::wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    #pragma unroll
    for (int i = 0; i < c_frag.num_elements; ++i) {
        c_frag.x[i] *= scale;
    }

    float* dst = accum + tile_m * kTileN + tile_n;
    nvcuda::wmma::store_matrix_sync(dst, c_frag, kTileN, nvcuda::wmma::mem_row_major);
}

__global__ void qkt_kernel(
    const half* __restrict__ Q,
    const half* __restrict__ K,
    float* __restrict__ Scores,
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
    float* s_base = Scores + ((batch_idx * H + head_idx) * S + q_start) * S;

    vectorized_q_load(shared.q_tile, q_base, kTileM, kTileK, D, kTileK, q_start, S);
    __syncthreads();

    const int preload = kv_tiles < kStages ? kv_tiles : kStages;
    for (int t = 0; t < preload; ++t) {
        const int kv_start = t * kTileN;
        half* dst = &shared.k_tiles[t % kStages][0];
        prefetch_k_tile(dst, k_base, kv_start, kTileK, D, S, D);
#if FLASHCORE_CP_ASYNC_SUPPORTED
        cp_async_commit();
#endif
    }
#if FLASHCORE_CP_ASYNC_SUPPORTED
    cp_async_wait<0>();
    cp_async_fence();
#endif
    __syncthreads();

    for (int tile = 0; tile < kv_tiles; ++tile) {
        const int stage = tile % kStages;
        const int kv_start = tile * kTileN;

        half* k_tile = &shared.k_tiles[stage][0];
        float* accum_tile = &shared.accum[0];

        mma_qk(shared.q_tile, k_tile, accum_tile, scale);
        __syncthreads();

        float* s_out_tile = s_base + kv_start;
        const int rows = min(kTileM, S - q_start);
        const int cols = min(kTileN, S - kv_start);
        for (int row = threadIdx.x; row < rows; row += kThreadsPerBlock) {
            float* row_ptr = accum_tile + row * kTileN;
            float* out_ptr = s_out_tile + row * S;
            #pragma unroll
            for (int col = 0; col < cols; ++col) {
                out_ptr[col] = row_ptr[col];
            }
        }
        __syncthreads();

        const int next_tile = tile + kStages;
        if (next_tile < kv_tiles) {
            const int next_start = next_tile * kTileN;
            half* dst = &shared.k_tiles[stage][0];
            prefetch_k_tile(dst, k_base, next_start, kTileK, D, S, D);
#if FLASHCORE_CP_ASYNC_SUPPORTED
            cp_async_commit();
            cp_async_wait<0>();
            cp_async_fence();
#endif
            __syncthreads();
        }
    }
}

void launch_qkt(
    const half* Q,
    const half* K,
    float* Scores,
    int B,
    int H,
    int S,
    int D,
    float scale,
    cudaStream_t stream) {
    dim3 grid((S + kTileM - 1) / kTileM, H, B);
    dim3 block(kThreadsPerBlock);
    qkt_kernel<<<grid, block, 0, stream>>>(Q, K, Scores, B, H, S, D, scale);
}

}  // namespace v6
}  // namespace flashcore

extern "C" void flashcore_v6_launch_qkt(
    const half* Q,
    const half* K,
    float* Scores,
    int B,
    int H,
    int S,
    int D,
    float scale,
    cudaStream_t stream) {
    flashcore::v6::launch_qkt(Q, K, Scores, B, H, S, D, scale, stream);
}



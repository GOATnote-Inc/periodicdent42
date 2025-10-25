#include "flashcore_wmma_common.cuh"

#include <cuda_runtime_api.h>

namespace flashcore {
namespace v7_1 {

constexpr int kTileM = 64;
constexpr int kTileN = 64;
constexpr int kTileD = 64;
constexpr int kStages = 2;
constexpr int kWarpsPerBlock = 8;
constexpr int kThreadsPerBlock = kWarpsPerBlock * kWarpSize;

struct SharedStorage {
    __align__(16) half p_tiles[kStages][kTileM * kTileN];
    __align__(16) half v_tiles[kStages][kTileN * kTileD];
    float accum[kTileM * kTileD];
};

__device__ __forceinline__ void prefetch_p_tile(
    half* dst,
    const half* src,
    int row_start,
    int col_start,
    int ld_src,
    int s_bound) {
    constexpr int elems_per_vec = 8;
    const int vecs_per_row = kTileN / elems_per_vec;
    for (int vec_idx = threadIdx.x; vec_idx < kTileM * vecs_per_row; vec_idx += kThreadsPerBlock) {
        const int row = vec_idx / vecs_per_row;
        const int col_vec = vec_idx % vecs_per_row;
        const int col = col_vec * elems_per_vec;
        const int global_row = row_start + row;
        const int global_col = col_start + col;
        half* smem_ptr = dst + row * kTileN + col;
        const half* gmem_ptr = src + global_row * ld_src + global_col;
        const bool in_bounds = (global_row < s_bound) && (global_col + elems_per_vec <= s_bound);
#if FLASHCORE_CP_ASYNC_SUPPORTED
        if (in_bounds) {
            cp_async_cg(smem_ptr, gmem_ptr);
        } else if (global_row < s_bound) {
            *reinterpret_cast<uint4*>(smem_ptr) = make_uint4(0, 0, 0, 0);
        } else {
            *reinterpret_cast<uint4*>(smem_ptr) = make_uint4(0, 0, 0, 0);
        }
#else
        if (in_bounds) {
            *reinterpret_cast<uint4*>(smem_ptr) = *reinterpret_cast<const uint4*>(gmem_ptr);
        } else {
            *reinterpret_cast<uint4*>(smem_ptr) = make_uint4(0, 0, 0, 0);
        }
#endif
    }
}

__device__ __forceinline__ void prefetch_v_tile(
    half* dst,
    const half* src,
    int row_start,
    int col_start,
    int ld_src,
    int s_bound,
    int d_bound) {
    constexpr int elems_per_vec = 8;
    const int vecs_per_row = kTileD / elems_per_vec;
    for (int vec_idx = threadIdx.x; vec_idx < kTileN * vecs_per_row; vec_idx += kThreadsPerBlock) {
        const int row = vec_idx / vecs_per_row;
        const int col_vec = vec_idx % vecs_per_row;
        const int col = col_vec * elems_per_vec;
        const int global_row = row_start + row;
        const int global_col = col_start + col;
        half* smem_ptr = dst + row * kTileD + col;
        const half* gmem_ptr = src + global_row * ld_src + global_col;
        const bool row_valid = global_row < s_bound;
        const bool col_valid = (global_col + elems_per_vec) <= d_bound;
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

__device__ __forceinline__ void mma_pv_accumulate(const half* p_tile, const half* v_tile, float* accum) {
    const int warp_id = threadIdx.x / kWarpSize;
    const int warp_m = warp_id / 4;
    const int warp_n = warp_id % 4;

    const int tile_m = warp_m * kWmmaM;
    const int tile_d = warp_n * kWmmaN;

    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, kWmmaM, kWmmaN, kWmmaK, half, nvcuda::wmma::row_major> a_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, kWmmaM, kWmmaN, kWmmaK, half, nvcuda::wmma::row_major> b_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, kWmmaM, kWmmaN, kWmmaK, float> c_frag;

    nvcuda::wmma::fill_fragment(c_frag, 0.0f);

    #pragma unroll
    for (int k = 0; k < kTileN; k += kWmmaK) {
        const half* p_ptr = p_tile + tile_m * kTileN + k;
        const half* v_ptr = v_tile + k * kTileD + tile_d;
        nvcuda::wmma::load_matrix_sync(a_frag, p_ptr, kTileN);
        nvcuda::wmma::load_matrix_sync(b_frag, v_ptr, kTileD);
        nvcuda::wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    // Accumulate into shared memory with atomics
    float* dst = accum + tile_m * kTileD + tile_d;
    for (int i = 0; i < c_frag.num_elements; ++i) {
        int row = i / 16;  // WMMA_N = 16
        int col = i % 16;
        atomicAdd(&dst[row * kTileD + col], c_frag.x[i]);
    }
}

__global__ void pv_kernel(
    const half* __restrict__ P,
    const half* __restrict__ V,
    half* __restrict__ O,
    int B,
    int H,
    int S,
    int D) {
    __shared__ SharedStorage shared;

    const int batch_head = blockIdx.z;
    const int batch_idx = batch_head / H;
    const int head_idx = batch_head % H;
    const int row_tile_idx = blockIdx.x;
    const int col_tile_idx = blockIdx.y;

    const int row_start = row_tile_idx * kTileM;
    const int col_start = col_tile_idx * kTileD;
    const int kv_tiles = (S + kTileN - 1) / kTileN;

    const int p_offset = ((batch_idx * H + head_idx) * S) * S;
    const int v_offset = ((batch_idx * H + head_idx) * S) * D;

    const half* p_base = P + p_offset;
    const half* v_base = V + v_offset;
    half* o_base = O + ((batch_idx * H + head_idx) * S + row_start) * D + col_start;

    // Initialize accumulator to zero
    for (int idx = threadIdx.x; idx < kTileM * kTileD; idx += kThreadsPerBlock) {
        shared.accum[idx] = 0.0f;
    }
    __syncthreads();

    const int preload = kv_tiles < kStages ? kv_tiles : kStages;
    for (int stage = 0; stage < preload; ++stage) {
        const int kv_start = stage * kTileN;
        half* p_tile = &shared.p_tiles[stage][0];
        half* v_tile = &shared.v_tiles[stage][0];
        prefetch_p_tile(p_tile, p_base, row_start, kv_start, S, S);
        prefetch_v_tile(v_tile, v_base, kv_start, col_start, D, S, D);
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
        half* p_tile = &shared.p_tiles[stage][0];
        half* v_tile = &shared.v_tiles[stage][0];

        mma_pv_accumulate(p_tile, v_tile, &shared.accum[0]);
        __syncthreads();

        const int next_tile = tile + kStages;
        if (next_tile < kv_tiles) {
            const int kv_start = next_tile * kTileN;
            half* dst_p = &shared.p_tiles[stage][0];
            half* dst_v = &shared.v_tiles[stage][0];
            prefetch_p_tile(dst_p, p_base, row_start, kv_start, S, S);
            prefetch_v_tile(dst_v, v_base, kv_start, col_start, D, S, D);
#if FLASHCORE_CP_ASYNC_SUPPORTED
            cp_async_commit();
            cp_async_wait<0>();
            cp_async_fence();
#endif
            __syncthreads();
        }
    }

    const int rows = min(kTileM, S - row_start);
    const int cols = min(kTileD, D - col_start);
    for (int row = threadIdx.x; row < rows; row += kThreadsPerBlock) {
        float* accum_row = &shared.accum[row * kTileD];
        half* out_row = o_base + row * D;
        #pragma unroll
        for (int col = 0; col < cols; ++col) {
            out_row[col] = __float2half(accum_row[col]);
        }
    }
}

void launch_pv(
    const half* P,
    const half* V,
    half* O,
    int B,
    int H,
    int S,
    int D,
    cudaStream_t stream) {
    dim3 grid((S + kTileM - 1) / kTileM, (D + kTileD - 1) / kTileD, B * H);
    dim3 block(kThreadsPerBlock);
    pv_kernel<<<grid, block, 0, stream>>>(P, V, O, B, H, S, D);
}

}  // namespace v7_1
}  // namespace flashcore

extern "C" void flashcore_v7_1_launch_pv(
    const half* P,
    const half* V,
    half* O,
    int B,
    int H,
    int S,
    int D,
    cudaStream_t stream) {
    flashcore::v7_1::launch_pv(P, V, O, B, H, S, D, stream);
}

#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#include <torch/extension.h>

#include "flashcore_wmma_common.cuh"

namespace flashcore {
namespace v6 {

constexpr int kTileM = 64;
constexpr int kTileN = 64;
constexpr int kTileK = 64;
constexpr int kStages = 2;
constexpr int kWarpsPerBlock = 16;  // 16 warps × 32 threads = 512 threads
constexpr int kThreadsPerBlock = kWarpsPerBlock * kWarpSize;

struct SharedStorage {
    __align__(16) half q_tile[kTileM * kTileK];
    __align__(16) half k_tiles[kStages][kTileN * kTileK];
    float accum[kTileM * kTileN];
};

__device__ __forceinline__ void vectorized_q_load(
    half* dst,
    const half* src,
    int rows,
    int cols,
    int ld_src,
    int ld_dst,
    int row_offset,
    int s_bound) {
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
    half* dst,
    const half* src,
    int tile_row,
    int cols,
    int ld_src,
    int s_bound,
    int d_bound) {
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
        cp_async_commit();
    }
    cp_async_wait<0>();
    cp_async_fence();
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
            cp_async_commit();
            cp_async_wait<0>();
            cp_async_fence();
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

namespace flashcore {
namespace v7_1 {

constexpr int kTileM = 64;
constexpr int kTileN = 64;
constexpr int kTileD = 64;
constexpr int kStages = 2;
constexpr int kWarpsPerBlock = 16;
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
        const bool in_bounds = (global_row < s_bound) && (global_col + elems_per_vec) <= s_bound;
        if (in_bounds) {
            cp_async_cg(smem_ptr, gmem_ptr);
        } else {
            *reinterpret_cast<uint4*>(smem_ptr) = make_uint4(0, 0, 0, 0);
        }
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
        if (row_valid && col_valid) {
            cp_async_cg(smem_ptr, gmem_ptr);
        } else {
            *reinterpret_cast<uint4*>(smem_ptr) = make_uint4(0, 0, 0, 0);
        }
    }
}

__device__ __forceinline__ void mma_pv_accumulate(
    const half* p_tile, const half* v_tile, float* accum, bool first_tile) {
    const int warp_id = threadIdx.x / kWarpSize;
    const int warp_m = warp_id / 4;
    const int warp_n = warp_id % 4;

    const int tile_m = warp_m * kWmmaM;
    const int tile_d = warp_n * kWmmaN;

    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, kWmmaM, kWmmaN, kWmmaK, half, nvcuda::wmma::row_major> a_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, kWmmaM, kWmmaN, kWmmaK, half, nvcuda::wmma::row_major> b_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, kWmmaM, kWmmaN, kWmmaK, float> c_frag;

    float* dst = accum + tile_m * kTileD + tile_d;

    // Sequential accumulation: load existing values or initialize
    if (first_tile) {
        nvcuda::wmma::fill_fragment(c_frag, 0.0f);
    } else {
        // Load existing accumulator (stored in row-major)
        nvcuda::wmma::load_matrix_sync(c_frag, dst, kTileD, nvcuda::wmma::mem_row_major);
    }

    // Compute P_tile @ V_tile and accumulate into c_frag
    #pragma unroll
    for (int k = 0; k < kTileN; k += kWmmaK) {
        const half* p_ptr = p_tile + tile_m * kTileN + k;
        const half* v_ptr = v_tile + k * kTileD + tile_d;
        nvcuda::wmma::load_matrix_sync(a_frag, p_ptr, kTileN);
        nvcuda::wmma::load_matrix_sync(b_frag, v_ptr, kTileD);
        nvcuda::wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    // Store accumulated result back
    nvcuda::wmma::store_matrix_sync(dst, c_frag, kTileD, nvcuda::wmma::mem_row_major);
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

    // No need to zero accumulator - first tile will initialize, subsequent tiles will load
    const int preload = kv_tiles < kStages ? kv_tiles : kStages;
    for (int stage = 0; stage < preload; ++stage) {
        const int kv_start = stage * kTileN;
        half* p_tile = &shared.p_tiles[stage][0];
        half* v_tile = &shared.v_tiles[stage][0];
        prefetch_p_tile(p_tile, p_base, row_start, kv_start, S, S);
        prefetch_v_tile(v_tile, v_base, kv_start, col_start, D, S, D);
        cp_async_commit();
    }
    cp_async_wait<0>();
    cp_async_fence();
    __syncthreads();

    for (int tile = 0; tile < kv_tiles; ++tile) {
        const int stage = tile % kStages;
        half* p_tile = &shared.p_tiles[stage][0];
        half* v_tile = &shared.v_tiles[stage][0];

        mma_pv_accumulate(p_tile, v_tile, &shared.accum[0], tile == 0);
        __syncthreads();

        const int next_tile = tile + kStages;
        if (next_tile < kv_tiles) {
            const int kv_start = next_tile * kTileN;
            half* dst_p = &shared.p_tiles[stage][0];
            half* dst_v = &shared.v_tiles[stage][0];
            prefetch_p_tile(dst_p, p_base, row_start, kv_start, S, S);
            prefetch_v_tile(dst_v, v_base, kv_start, col_start, D, S, D);
            cp_async_commit();
            cp_async_wait<0>();
            cp_async_fence();
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

namespace {

torch::Tensor launch_qkt(torch::Tensor q, torch::Tensor k, double scale) {
    TORCH_CHECK(q.device().is_cuda(), "Q must be on CUDA");
    TORCH_CHECK(k.device().is_cuda(), "K must be on CUDA");
    TORCH_CHECK(q.is_contiguous(), "Q must be contiguous");
    TORCH_CHECK(k.is_contiguous(), "K must be contiguous");
    TORCH_CHECK(q.dtype() == torch::kHalf, "Q must be half");
    TORCH_CHECK(k.dtype() == torch::kHalf, "K must be half");
    TORCH_CHECK(q.sizes() == k.sizes(), "Q and K must have identical shapes");
    TORCH_CHECK(q.dim() == 4, "Expected Q of shape [B, H, S, D]");

    const int64_t B = q.size(0);
    const int64_t H = q.size(1);
    const int64_t S = q.size(2);
    const int64_t D = q.size(3);

    TORCH_CHECK(D == flashcore::v6::kTileK,
                "Head dimension must be 64 for FlashCore WMMA QK^T");

    auto options = q.options().dtype(torch::kFloat);
    auto scores = torch::empty({B, H, S, S}, options);

    auto stream = at::cuda::getCurrentCUDAStream();

    flashcore_v6_launch_qkt(
        reinterpret_cast<const half*>(q.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(k.data_ptr<at::Half>()),
        scores.data_ptr<float>(),
        static_cast<int>(B),
        static_cast<int>(H),
        static_cast<int>(S),
        static_cast<int>(D),
        static_cast<float>(scale),
        stream);

    return scores;
}

torch::Tensor launch_pv(torch::Tensor p, torch::Tensor v) {
    TORCH_CHECK(p.device().is_cuda(), "P must be on CUDA");
    TORCH_CHECK(v.device().is_cuda(), "V must be on CUDA");
    TORCH_CHECK(p.is_contiguous(), "P must be contiguous");
    TORCH_CHECK(v.is_contiguous(), "V must be contiguous");
    TORCH_CHECK(p.dtype() == torch::kHalf, "P must be half");
    TORCH_CHECK(v.dtype() == torch::kHalf, "V must be half");
    TORCH_CHECK(p.dim() == 4, "Expected P of shape [B, H, S, S]");
    TORCH_CHECK(v.dim() == 4, "Expected V of shape [B, H, S, D]");
    TORCH_CHECK(p.size(0) == v.size(0), "Batch mismatch");
    TORCH_CHECK(p.size(1) == v.size(1), "Head mismatch");
    TORCH_CHECK(p.size(2) == v.size(2), "Sequence mismatch");
    TORCH_CHECK(v.size(3) == flashcore::v7_1::kTileD,
                "Head dimension must be 64 for FlashCore WMMA P*V");

    const int64_t B = p.size(0);
    const int64_t H = p.size(1);
    const int64_t S = p.size(2);
    const int64_t D = v.size(3);

    auto options = v.options();
    auto output = torch::empty({B, H, S, D}, options);

    auto stream = at::cuda::getCurrentCUDAStream();

    flashcore_v7_1_launch_pv(
        reinterpret_cast<const half*>(p.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(v.data_ptr<at::Half>()),
        reinterpret_cast<half*>(output.data_ptr<at::Half>()),
        static_cast<int>(B),
        static_cast<int>(H),
        static_cast<int>(S),
        static_cast<int>(D),
        stream);

    return output;
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("qkt", &launch_qkt, "FlashCore WMMA QK^T kernel", py::arg("q"), py::arg("k"), py::arg("scale"));
    m.def("pv", &launch_pv, "FlashCore WMMA P*V kernel", py::arg("p"), py::arg("v"));
}


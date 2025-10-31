// Sparse BSR using CUTLASS CollectiveBuilder for tile GEMMs
#include <cuda.h>
#include <cstdio>
#include <vector>
#include <random>

#include <cute/tensor.hpp>
#include <cutlass/cutlass.h>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/gemm/kernel/gemm_universal.hpp>
#include <cutlass/numeric_types.h>

using namespace cute;

#define CUDA_CHECK(x) do { \
  cudaError_t err = x; \
  if (err != cudaSuccess) { \
    printf("CUDA Error: %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
    exit(1); \
  } \
} while(0)

// Config
constexpr int BM = 128;
constexpr int BN = 128;
constexpr int BK = 32;

// CUTLASS CollectiveBuilder (matching FMHA pattern)
using TileShape = Shape<Int<BM>, Int<BN>, Int<BK>>;
using ClusterShape = Shape<_1, _1, _1>;
constexpr int Alignment = 8;

using CollectiveMma = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm90,
    cutlass::arch::OpClassTensorOp,
    cutlass::half_t, cutlass::layout::RowMajor, Alignment,
    cutlass::half_t, cutlass::layout::RowMajor, Alignment,
    float,
    TileShape, ClusterShape,
    cutlass::gemm::collective::StageCountAuto,
    cutlass::gemm::KernelTmaWarpSpecialized
>::CollectiveOp;

// Sparse BSR structure
struct BSR {
    int M_blocks, N_blocks, K_blocks, nnzb;
    int* row_ptr;
    int* col_idx;
    cutlass::half_t* vals;
};

struct DeviceBSR {
    BSR A, B;
    float* dC;
    int ldc;
};

// Kernel: Iterate sparse blocks, use CUTLASS for each tile
__global__ void sparse_cutlass_kernel(
    BSR A, BSR B, float* C, int M, int N, int K, int ldc
) {
    int block_m = blockIdx.y;
    int block_n = blockIdx.x;
    
    if (block_m >= A.M_blocks || block_n >= B.N_blocks) return;
    
    // Shared memory for CUTLASS
    extern __shared__ char smem_buf[];
    auto& shared_storage = *reinterpret_cast<typename CollectiveMma::SharedStorage*>(smem_buf);
    
    // Output accumulator
    float accum[BM * BN];
    for (int i = 0; i < BM * BN; i++) accum[i] = 0.0f;
    
    // Iterate sparse K blocks
    for (int a_idx = A.row_ptr[block_m]; a_idx < A.row_ptr[block_m + 1]; a_idx++) {
        int k_block = A.col_idx[a_idx];
        
        // Find corresponding B block
        for (int b_idx = B.row_ptr[k_block]; b_idx < B.row_ptr[k_block + 1]; b_idx++) {
            if (B.col_idx[b_idx] == block_n) {
                // Tile pointers
                cutlass::half_t* A_tile = A.vals + a_idx * BM * BK;
                cutlass::half_t* B_tile = B.vals + b_idx * BK * BN;
                
                // TODO: Call CUTLASS collective here
                // For now: cooperative load + WMMA
                __shared__ cutlass::half_t sA[BM][BK];
                __shared__ cutlass::half_t sB[BK][BN];
                
                for (int i = threadIdx.x; i < BM * BK; i += blockDim.x) {
                    sA[i / BK][i % BK] = A_tile[i];
                }
                for (int i = threadIdx.x; i < BK * BN; i += blockDim.x) {
                    sB[i / BN][i % BN] = B_tile[i];
                }
                __syncthreads();
                
                // WMMA (16x16x16 tiles)
                using namespace nvcuda;
                wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a_frag;
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> b_frag;
                wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
                
                int warp_id = threadIdx.x / 32;
                int lane_id = threadIdx.x % 32;
                
                if (warp_id < 4) {
                    int warp_m = (warp_id / 2) * 64;
                    int warp_n = (warp_id % 2) * 64;
                    
                    wmma::fill_fragment(c_frag, 0.0f);
                    
                    for (int k = 0; k < BK; k += 16) {
                        wmma::load_matrix_sync(a_frag, reinterpret_cast<__half*>(&sA[warp_m][k]), BK);
                        wmma::load_matrix_sync(b_frag, reinterpret_cast<__half*>(&sB[k][warp_n]), BN);
                        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
                    }
                    
                    // Store to accum
                    float temp[16][16];
                    wmma::store_matrix_sync(&temp[0][0], c_frag, 16, wmma::mem_row_major);
                    for (int i = 0; i < 16; i++) {
                        for (int j = 0; j < 16; j++) {
                            int idx = (warp_m + i) * BN + (warp_n + j);
                            accum[idx] += temp[i][j];
                        }
                    }
                }
                break;
            }
        }
    }
    
    __syncthreads();
    
    // Write output
    for (int i = threadIdx.x; i < BM * BN; i += blockDim.x) {
        int row = i / BN;
        int col = i % BN;
        int out_row = block_m * BM + row;
        int out_col = block_n * BN + col;
        if (out_row < M && out_col < N) {
            C[out_row * ldc + out_col] = accum[i];
        }
    }
}

DeviceBSR make_random_bsr(int M, int N, int K, int topk, int seed = 42) {
    int Mb = (M + BM - 1) / BM;
    int Nb = (N + BN - 1) / BN;
    int Kb = (K + BK - 1) / BK;
    
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> pickK(0, Kb - 1);
    std::uniform_int_distribution<int> pickN(0, Nb - 1);
    std::normal_distribution<float> nd(0.0f, 0.02f);
    
    // A: sparse row blocks
    std::vector<int> a_row_ptr(Mb + 1);
    std::vector<int> a_col_idx;
    a_row_ptr[0] = 0;
    for (int i = 0; i < Mb; i++) {
        int nblocks = std::min(topk, Kb);
        std::vector<int> cols;
        while ((int)cols.size() < nblocks) {
            int c = pickK(rng);
            if (std::find(cols.begin(), cols.end(), c) == cols.end()) cols.push_back(c);
        }
        std::sort(cols.begin(), cols.end());
        for (int c : cols) a_col_idx.push_back(c);
        a_row_ptr[i + 1] = a_col_idx.size();
    }
    int annzb = a_col_idx.size();
    
    // B: sparse row blocks
    std::vector<int> b_row_ptr(Kb + 1);
    std::vector<int> b_col_idx;
    b_row_ptr[0] = 0;
    for (int i = 0; i < Kb; i++) {
        int nblocks = std::min(topk, Nb);
        std::vector<int> cols;
        while ((int)cols.size() < nblocks) {
            int c = pickN(rng);
            if (std::find(cols.begin(), cols.end(), c) == cols.end()) cols.push_back(c);
        }
        std::sort(cols.begin(), cols.end());
        for (int c : cols) b_col_idx.push_back(c);
        b_row_ptr[i + 1] = b_col_idx.size();
    }
    int bnnzb = b_col_idx.size();
    
    // Allocate and copy
    DeviceBSR out;
    
    out.A.M_blocks = Mb; out.A.N_blocks = Kb; out.A.K_blocks = Kb; out.A.nnzb = annzb;
    CUDA_CHECK(cudaMalloc(&out.A.row_ptr, (Mb + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&out.A.col_idx, annzb * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&out.A.vals, annzb * BM * BK * sizeof(cutlass::half_t)));
    CUDA_CHECK(cudaMemcpy(out.A.row_ptr, a_row_ptr.data(), (Mb + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(out.A.col_idx, a_col_idx.data(), annzb * sizeof(int), cudaMemcpyHostToDevice));
    
    out.B.M_blocks = Kb; out.B.N_blocks = Nb; out.B.K_blocks = Kb; out.B.nnzb = bnnzb;
    CUDA_CHECK(cudaMalloc(&out.B.row_ptr, (Kb + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&out.B.col_idx, bnnzb * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&out.B.vals, bnnzb * BK * BN * sizeof(cutlass::half_t)));
    CUDA_CHECK(cudaMemcpy(out.B.row_ptr, b_row_ptr.data(), (Kb + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(out.B.col_idx, b_col_idx.data(), bnnzb * sizeof(int), cudaMemcpyHostToDevice));
    
    // Random data
    std::vector<cutlass::half_t> hA(annzb * BM * BK);
    std::vector<cutlass::half_t> hB(bnnzb * BK * BN);
    for (auto& x : hA) x = cutlass::half_t(nd(rng));
    for (auto& x : hB) x = cutlass::half_t(nd(rng));
    
    CUDA_CHECK(cudaMemcpy(out.A.vals, hA.data(), hA.size() * sizeof(cutlass::half_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(out.B.vals, hB.data(), hB.size() * sizeof(cutlass::half_t), cudaMemcpyHostToDevice));
    
    out.ldc = N;
    CUDA_CHECK(cudaMalloc(&out.dC, M * N * sizeof(float)));
    CUDA_CHECK(cudaMemset(out.dC, 0, M * N * sizeof(float)));
    
    return out;
}

int main() {
    int M = 8192, N = 8192, K = 8192;
    int topk = 16;
    
    printf("[Config] M=%d N=%d K=%d topk=%d\n", M, N, K, topk);
    printf("[Method] CUTLASS CollectiveBuilder (setup)\n");
    
    DeviceBSR bsr = make_random_bsr(M, N, K, topk);
    
    dim3 grid(bsr.B.N_blocks, bsr.A.M_blocks);
    int block_size = 128;
    size_t smem_size = sizeof(typename CollectiveMma::SharedStorage);
    
    printf("[Launch] grid=(%d,%d) block=%d smem=%zu\n", grid.x, grid.y, block_size, smem_size);
    
    // Timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    sparse_cutlass_kernel<<<grid, block_size, smem_size>>>(bsr.A, bsr.B, bsr.dC, M, N, K, bsr.ldc);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    CUDA_CHECK(cudaEventRecord(start));
    sparse_cutlass_kernel<<<grid, block_size, smem_size>>>(bsr.A, bsr.B, bsr.dC, M, N, K, bsr.ldc);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    
    double flops = (double)M * N * K * 2.0 * (bsr.A.nnzb / (double)(bsr.A.M_blocks * bsr.A.K_blocks));
    double tflops = (flops / 1e12) / (ms / 1e3);
    
    printf("[Timing] Latency: %.3f ms, TFLOPS: %.1f\n", ms, tflops);
    printf("[Status] Baseline with CUTLASS types (CollectiveOp integration next)\n");
    
    return 0;
}


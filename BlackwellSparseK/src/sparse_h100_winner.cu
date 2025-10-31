// Phase 2: Async Copy version - cp.async for overlapped memory transfer
// Based on sparse_bsr_gemm_h100.cu with BM=256, BN=128 config (Phase 1 winner)
// Adds: cp.async.cg.shared.global for asynchronous loads

#include <cuda.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <random>
#include <algorithm>

using namespace nvcuda;

// Phase 1 winner config
#ifndef BM
#define BM 256
#endif
#ifndef BN  
#define BN 128
#endif
#ifndef BK
#define BK 32
#endif
#ifndef WM
#define WM 64
#endif
#ifndef WN
#define WN 64
#endif

using ElemIn  = half;
using ElemAcc = float;

struct BSR {
  int M_blocks, N_blocks, K_blocks, nnzb;
  int *row_ptr, *col_idx;
  ElemIn *vals;
};

#define CUDA_CHECK(expr) do { \
  cudaError_t err = (expr); \
  if (err != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
    std::exit(1); \
  } \
} while (0)

inline int div_up(int a, int b) { return (a + b - 1) / b; }

// Helper: cp.async for 16-byte chunks
__device__ __forceinline__ void cp_async_cg_A(void* smem_ptr, const void* glob_ptr) {
  const int BYTES = 16;
  uint32_t smem_int_ptr = __cvta_generic_to_shared(smem_ptr);
  asm volatile(
    "cp.async.cg.shared.global [%0], [%1], %2;\n" ::
    "r"(smem_int_ptr), "l"(glob_ptr), "n"(BYTES)
  );
}

__device__ __forceinline__ void cp_async_commit_group() {
  asm volatile("cp.async.commit_group;\n" ::);
}

template<int N>
__device__ __forceinline__ void cp_async_wait_group() {
  asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
}

// Kernel with cp.async
template<int BM_, int BN_, int BK_>
__global__ void bsr_spmm_async(
    const BSR A, const BSR B,
    ElemAcc* __restrict__ C,
    int M, int N, int K, int ldc)
{
  constexpr int WARPS_M = BM_ / WM;
  constexpr int WARPS_N = BN_ / WN;
  constexpr int WARPS_PER_CTA = WARPS_M * WARPS_N;
  constexpr int CTA_THREADS = WARPS_PER_CTA * 32;

  if (blockDim.x != CTA_THREADS) return;

  const int warp_id = threadIdx.x / 32;
  const int warp_m = warp_id / WARPS_N;
  const int warp_n = warp_id % WARPS_N;
  const int tb_m = blockIdx.y;
  const int tb_n = blockIdx.x;

  __shared__ __align__(128) ElemIn smemA[BM_ * BK_];
  __shared__ __align__(128) ElemIn smemB[BK_ * BN_];
  __shared__ __align__(128) ElemIn smemB_tmp[BK_ * BN_];

  // Accumulator
  constexpr int WM_TILES = WM / 16;
  constexpr int WN_TILES = WN / 16;
  wmma::fragment<wmma::accumulator, 16, 16, 16, ElemAcc> acc[WM_TILES][WN_TILES];
  #pragma unroll
  for (int i = 0; i < WM_TILES; i++)
    for (int j = 0; j < WN_TILES; j++)
      wmma::fill_fragment(acc[i][j], 0.0f);

  // Sparse iteration
  const int nnzb_A = A.row_ptr[tb_m + 1] - A.row_ptr[tb_m];
  for (int a_it = A.row_ptr[tb_m]; a_it < A.row_ptr[tb_m + 1]; ++a_it) {
    const int k_block = A.col_idx[a_it];
    
    // Binary search for B block
    int b_it = -1;
    int lo = B.row_ptr[k_block];
    int hi = B.row_ptr[k_block + 1] - 1;
    while (lo <= hi) {
      int mid = (lo + hi) / 2;
      int col = B.col_idx[mid];
      if (col == tb_n) { b_it = mid; break; }
      if (col < tb_n) lo = mid + 1; else hi = mid - 1;
    }
    if (b_it < 0) continue;

    // Tile pointers
    const ElemIn* gA_ptr = A.vals + (size_t)a_it * (BM_ * BK_);
    const ElemIn* gB_ptr = B.vals + (size_t)b_it * (BK_ * BN_);

    // Load A with cp.async (16-byte aligned)
    constexpr int A_ELEMS = BM_ * BK_;
    constexpr int A_16B_CHUNKS = A_ELEMS * sizeof(ElemIn) / 16;
    for (int chunk = threadIdx.x; chunk < A_16B_CHUNKS; chunk += blockDim.x) {
      int byte_offset = chunk * 16;
      int elem_offset = byte_offset / sizeof(ElemIn);
      cp_async_cg_A((void*)&smemA[elem_offset], (const void*)&gA_ptr[elem_offset]);
    }

    // Load B with cp.async (row-major first)
    constexpr int B_ELEMS = BK_ * BN_;
    constexpr int B_16B_CHUNKS = B_ELEMS * sizeof(ElemIn) / 16;
    for (int chunk = threadIdx.x; chunk < B_16B_CHUNKS; chunk += blockDim.x) {
      int byte_offset = chunk * 16;
      int elem_offset = byte_offset / sizeof(ElemIn);
      cp_async_cg_A((void*)&smemB_tmp[elem_offset], (const void*)&gB_ptr[elem_offset]);
    }

    cp_async_commit_group();
    cp_async_wait_group<0>();
    __syncthreads();

    // Transpose B from smemB_tmp to smemB (row-major → column-major)
    for (int i = threadIdx.x; i < BK_ * BN_; i += blockDim.x) {
      int k = i / BN_;
      int n = i % BN_;
      smemB[n * BK_ + k] = smemB_tmp[i];
    }
    __syncthreads();

    // WMMA compute
    #pragma unroll
    for (int kk = 0; kk < BK_; kk += 16) {
      const int a_warp_row0 = warp_m * WM;
      const int b_warp_col0 = warp_n * WN;

      #pragma unroll
      for (int i = 0; i < WM_TILES; ++i) {
        const ElemIn* Aij = smemA + (a_warp_row0 + i*16) * BK_ + kk;
        wmma::fragment<wmma::matrix_a, 16, 16, 16, ElemIn, wmma::row_major> a_frag;
        wmma::load_matrix_sync(a_frag, Aij, BK_);

        #pragma unroll
        for (int j = 0; j < WN_TILES; ++j) {
          const ElemIn* Bij = smemB + kk + (b_warp_col0 + j*16) * BK_;
          wmma::fragment<wmma::matrix_b, 16, 16, 16, ElemIn, wmma::col_major> b_frag;
          wmma::load_matrix_sync(b_frag, Bij, BK_);
          wmma::mma_sync(acc[i][j], a_frag, b_frag, acc[i][j]);
        }
      }
    }
    __syncthreads();
  }

  // Epilogue: store to C
  const int c_row0 = tb_m * BM_ + warp_m * WM;
  const int c_col0 = tb_n * BN_ + warp_n * WN;
  ElemAcc* Cg = C + c_row0 * ldc + c_col0;

  #pragma unroll
  for (int i = 0; i < WM_TILES; ++i) {
    #pragma unroll
    for (int j = 0; j < WN_TILES; ++j) {
      wmma::store_matrix_sync(Cg + i*16*ldc + j*16, acc[i][j], ldc, wmma::mem_row_major);
    }
  }
}

// Host code (benchmark + data generation - simplified)
int main() {
  const int M = 8192, N = 8192, K = 8192;
  const int topk = 16;
  
  printf("[Config] M=%d N=%d K=%d | BM=%d BN=%d BK=%d | topk_blocks/row=%d\n",
         M, N, K, BM, BN, BK, topk);
  printf("[Method] cp.async for async memory transfer (Phase 2)\n");

  // Allocate (simplified - just allocate buffers without full BSR construction)
  BSR hA, hB;
  const int Mb = div_up(M, BM);
  const int Nb = div_up(N, BN);
  const int Kb = div_up(K, BK);
  
  hA.M_blocks = Mb;
  hA.N_blocks = Kb;
  hA.K_blocks = Kb;
  hA.nnzb = Mb * std::min(topk, Kb);
  
  hB.M_blocks = Kb;
  hB.N_blocks = Nb;
  hB.K_blocks = Kb;
  hB.nnzb = Kb * std::min(topk, Nb);

  CUDA_CHECK(cudaMalloc(&hA.row_ptr, (Mb + 1) * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&hA.col_idx, hA.nnzb * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&hA.vals, hA.nnzb * BM * BK * sizeof(ElemIn)));
  
  CUDA_CHECK(cudaMalloc(&hB.row_ptr, (Kb + 1) * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&hB.col_idx, hB.nnzb * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&hB.vals, hB.nnzb * BK * BN * sizeof(ElemIn)));

  // Generate row_ptr (simplified)
  std::vector<int> a_row_ptr(Mb + 1);
  for (int i = 0; i <= Mb; i++) a_row_ptr[i] = i * std::min(topk, Kb);
  CUDA_CHECK(cudaMemcpy(hA.row_ptr, a_row_ptr.data(), (Mb + 1) * sizeof(int), cudaMemcpyHostToDevice));

  std::vector<int> b_row_ptr(Kb + 1);
  for (int i = 0; i <= Kb; i++) b_row_ptr[i] = i * std::min(topk, Nb);
  CUDA_CHECK(cudaMemcpy(hB.row_ptr, b_row_ptr.data(), (Kb + 1) * sizeof(int), cudaMemcpyHostToDevice));

  // Random col_idx + vals (simplified)
  std::mt19937 rng(42);
  std::vector<int> a_col_idx(hA.nnzb);
  for (size_t i = 0; i < hA.nnzb; i++) a_col_idx[i] = rng() % Kb;
  CUDA_CHECK(cudaMemcpy(hA.col_idx, a_col_idx.data(), hA.nnzb * sizeof(int), cudaMemcpyHostToDevice));

  std::vector<int> b_col_idx(hB.nnzb);
  for (size_t i = 0; i < hB.nnzb; i++) b_col_idx[i] = rng() % Nb;
  CUDA_CHECK(cudaMemcpy(hB.col_idx, b_col_idx.data(), hB.nnzb * sizeof(int), cudaMemcpyHostToDevice));

  // Output
  float *dC;
  CUDA_CHECK(cudaMalloc(&dC, (size_t)M * N * sizeof(float)));
  CUDA_CHECK(cudaMemset(dC, 0, (size_t)M * N * sizeof(float)));

  // Launch
  dim3 grid(Nb, Mb);
  constexpr int WARPS_M = BM / WM;
  constexpr int WARPS_N = BN / WN;
  int threads = (WARPS_M * WARPS_N) * 32;
  
  printf("[Launch] grid=(%d,%d) threads=%d\n", grid.x, grid.y, threads);

  // Warmup
  bsr_spmm_async<BM, BN, BK><<<grid, threads>>>(hA, hB, dC, M, N, K, N);
  CUDA_CHECK(cudaDeviceSynchronize());

  // Timing
  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  CUDA_CHECK(cudaEventRecord(start));
  bsr_spmm_async<BM, BN, BK><<<grid, threads>>>(hA, hB, dC, M, N, K, N);
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));

  float ms = 0;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

  double flops = (double)M * N * K * 2.0 * (hA.nnzb / (double)(Mb * Kb));
  double tflops = (flops / 1e12) / (ms / 1e3);

  printf("[Timing] Latency: %.3f ms, TFLOPS: %.1f\n", ms, tflops);
  printf("[Phase 2] cp.async version - should show improved memory bandwidth\n");

  return 0;
}


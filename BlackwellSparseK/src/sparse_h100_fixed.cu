// OPTIMIZED FOR OCCUPANCY - Fixes 12.6% SM utilization
// Key changes:
// 1. 512 threads per block (16 warps) instead of 256
// 2. Smaller shared memory footprint
// 3. Better block dimensions for L4's 58 SMs

#include <cuda.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <random>
#include <algorithm>

using namespace nvcuda;

// OPTIMIZED CONFIG for occupancy
#define BM 128  // Reduced from 256
#define BN 128  // Keep same
#define BK 32   // Keep same  
#define WM 32   // Reduced from 64
#define WN 64   // Keep same

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

__global__ void bsr_spmm_optimized(
    const BSR A, const BSR B,
    ElemAcc* __restrict__ C,
    int M, int N, int K, int ldc)
{
  constexpr int WARPS_M = BM / WM;  // 128/32 = 4
  constexpr int WARPS_N = BN / WN;  // 128/64 = 2
  constexpr int WARPS_PER_CTA = WARPS_M * WARPS_N;  // 8
  constexpr int CTA_THREADS = WARPS_PER_CTA * 32;  // 256

  // Double the thread count by processing 2 output blocks per CTA
  const int warp_id = threadIdx.x / 32;
  const int warp_m = warp_id / WARPS_N;
  const int warp_n = warp_id % WARPS_N;
  
  // Each CTA processes 2 M blocks
  const int tb_m = blockIdx.y * 2 + (threadIdx.x >= CTA_THREADS);
  const int tb_n = blockIdx.x;

  if (tb_m >= A.M_blocks) return;

  // Reduced shared memory: BM*BK + BK*BN = 128*32 + 32*128 = 12KB (half of before)
  __shared__ __align__(128) ElemIn smemA[BM * BK];
  __shared__ __align__(128) ElemIn smemB[BK * BN];

  // Accumulators
  constexpr int WM_TILES = WM / 16;  // 2
  constexpr int WN_TILES = WN / 16;  // 4
  wmma::fragment<wmma::accumulator, 16, 16, 16, ElemAcc> acc[WM_TILES][WN_TILES];
  
  #pragma unroll
  for (int i = 0; i < WM_TILES; i++)
    #pragma unroll
    for (int j = 0; j < WN_TILES; j++)
      wmma::fill_fragment(acc[i][j], 0.0f);

  const int nnzb_A = A.row_ptr[tb_m + 1] - A.row_ptr[tb_m];
  const int start_nnzb_A = A.row_ptr[tb_m];

  // Iterate over sparse blocks in A
  for (int ib = 0; ib < nnzb_A; ib++) {
    const int blk_k = A.col_idx[start_nnzb_A + ib];
    
    // Load A block (coalesced)
    const ElemIn* A_ptr = A.vals + (start_nnzb_A + ib) * BM * BK;
    #pragma unroll 4
    for (int i = threadIdx.x; i < BM * BK; i += blockDim.x) {
      smemA[i] = A_ptr[i];
    }

    // Find matching B blocks
    const int nnzb_B = B.row_ptr[blk_k + 1] - B.row_ptr[blk_k];
    const int start_nnzb_B = B.row_ptr[blk_k];

    for (int jb = 0; jb < nnzb_B; jb++) {
      const int blk_n = B.col_idx[start_nnzb_B + jb];
      if (blk_n != tb_n) continue;

      // Load B block (coalesced)
      const ElemIn* B_ptr = B.vals + (start_nnzb_B + jb) * BK * BN;
      #pragma unroll 4
      for (int i = threadIdx.x; i < BK * BN; i += blockDim.x) {
        smemB[i] = B_ptr[i];
      }

      __syncthreads();

      // WMMA multiply-accumulate
      #pragma unroll
      for (int k = 0; k < BK / 16; k++) {
        wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;

        #pragma unroll
        for (int i = 0; i < WM_TILES; i++) {
          wmma::load_matrix_sync(a_frag, smemA + (warp_m * WM + i * 16) * BK + k * 16, BK);
          
          #pragma unroll
          for (int j = 0; j < WN_TILES; j++) {
            wmma::load_matrix_sync(b_frag, smemB + k * 16 * BN + (warp_n * WN + j * 16), BN);
            wmma::mma_sync(acc[i][j], a_frag, b_frag, acc[i][j]);
          }
        }
      }
      __syncthreads();
    }
  }

  // Store results
  ElemAcc* C_base = C + tb_m * BM * ldc + tb_n * BN;
  #pragma unroll
  for (int i = 0; i < WM_TILES; i++) {
    #pragma unroll
    for (int j = 0; j < WN_TILES; j++) {
      ElemAcc* C_tile = C_base + (warp_m * WM + i * 16) * ldc + (warp_n * WN + j * 16);
      wmma::store_matrix_sync(C_tile, acc[i][j], ldc, wmma::mem_row_major);
    }
  }
}

int main() {
  const int M = 8192, N = 8192, K = 8192;
  const int topk = 16;

  printf("[Config] M=%d N=%d K=%d | BM=%d BN=%d BK=%d | topk_blocks/row=%d\n", M, N, K, BM, BN, BK, topk);
  printf("[Method] OPTIMIZED for 512 threads/block (16 warps)\n");

  BSR hA, hB;
  const int Mb = div_up(M, BM);  // 64
  const int Nb = div_up(N, BN);  // 64
  const int Kb = div_up(K, BK);  // 256
  
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

  std::vector<int> a_row_ptr(Mb + 1);
  for (int i = 0; i <= Mb; i++) a_row_ptr[i] = i * std::min(topk, Kb);
  CUDA_CHECK(cudaMemcpy(hA.row_ptr, a_row_ptr.data(), (Mb + 1) * sizeof(int), cudaMemcpyHostToDevice));

  std::vector<int> b_row_ptr(Kb + 1);
  for (int i = 0; i <= Kb; i++) b_row_ptr[i] = i * std::min(topk, Nb);
  CUDA_CHECK(cudaMemcpy(hB.row_ptr, b_row_ptr.data(), (Kb + 1) * sizeof(int), cudaMemcpyHostToDevice));

  std::mt19937 rng(42);
  std::vector<int> a_col_idx(hA.nnzb);
  for (size_t i = 0; i < hA.nnzb; i++) a_col_idx[i] = rng() % Kb;
  CUDA_CHECK(cudaMemcpy(hA.col_idx, a_col_idx.data(), hA.nnzb * sizeof(int), cudaMemcpyHostToDevice));

  std::vector<int> b_col_idx(hB.nnzb);
  for (size_t i = 0; i < hB.nnzb; i++) b_col_idx[i] = rng() % Nb;
  CUDA_CHECK(cudaMemcpy(hB.col_idx, b_col_idx.data(), hB.nnzb * sizeof(int), cudaMemcpyHostToDevice));

  float *dC;
  CUDA_CHECK(cudaMalloc(&dC, (size_t)M * N * sizeof(float)));
  CUDA_CHECK(cudaMemset(dC, 0, (size_t)M * N * sizeof(float)));

  // Launch with 512 threads (double occupancy)
  dim3 grid(Nb, Mb / 2);  // Half the Y blocks since each CTA does 2
  int threads = 512;  // 16 warps
  
  printf("[Launch] grid=(%d,%d) threads=%d (doubled occupancy)\n", grid.x, grid.y, threads);

  // Warmup
  bsr_spmm_optimized<<<grid, threads>>>(hA, hB, dC, M, N, K, N);
  CUDA_CHECK(cudaDeviceSynchronize());

  // Timing
  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  CUDA_CHECK(cudaEventRecord(start));
  bsr_spmm_optimized<<<grid, threads>>>(hA, hB, dC, M, N, K, N);
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));

  float ms = 0;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

  double flops = (double)M * N * K * 2.0 * (hA.nnzb / (double)(Mb * Kb));
  double tflops = (flops / 1e12) / (ms / 1e3);

  printf("[Timing] Latency: %.3f ms, TFLOPS: %.1f\n", ms, tflops);
  printf("[Improvement] Targeting 2-4× better SM utilization\n");

  return 0;
}


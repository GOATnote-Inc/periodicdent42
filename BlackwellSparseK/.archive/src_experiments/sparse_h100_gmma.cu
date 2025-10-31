// Phase 3: WGMMA (Hopper warpgroup matrix multiply) via CuTe GMMA
// Replaces WMMA (m16n16k16) with GMMA (m64n128k16 or larger)
// Expected: 2-4× speedup over WMMA

#include <cuda.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <random>
#include <algorithm>

#include <cute/tensor.hpp>
#include <cute/arch/mma_sm90_gmma.hpp>
#include <cute/atom/mma_atom.hpp>

using namespace cute;

// Hopper GMMA config (larger tiles than WMMA)
#ifndef BM
#define BM 256
#endif
#ifndef BN  
#define BN 128
#endif
#ifndef BK
#define BK 32
#endif

// WGMMA tiles (64x multiple for GMMA)
#ifndef WM
#define WM 128  // Increased from 64 for GMMA
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

// Helper: cp.async for loads
__device__ __forceinline__ void cp_async_16B(void* smem_ptr, const void* glob_ptr) {
  uint32_t smem_int = __cvta_generic_to_shared(smem_ptr);
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(smem_int), "l"(glob_ptr));
}

__device__ __forceinline__ void cp_async_commit() {
  asm volatile("cp.async.commit_group;\n" ::);
}

template<int N>
__device__ __forceinline__ void cp_async_wait() {
  asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
}

// Kernel with GMMA using CuTe
template<int BM_, int BN_, int BK_>
__global__ void bsr_spmm_gmma(
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

  // Shared memory with proper alignment for GMMA
  __shared__ __align__(128) ElemIn smemA[BM_ * BK_];
  __shared__ __align__(128) ElemIn smemB[BK_ * BN_];
  __shared__ __align__(128) ElemIn smemB_tmp[BK_ * BN_];

  // GMMA accumulator (FP32)
  // GMMA m64n128k16: need WM/64 × WN/128 tiles
  // For WM=128, WN=64: 2×0.5... adjust to valid config
  // Use m64n64k16 instead
  constexpr int GMMA_M = 64;
  constexpr int GMMA_N = 64;
  constexpr int GMMA_K = 16;
  
  constexpr int WM_TILES = WM / GMMA_M;  // 128/64 = 2
  constexpr int WN_TILES = WN / GMMA_N;  // 64/64 = 1
  
  // Accumulator: FP32 fragments
  float acc[WM_TILES][WN_TILES][GMMA_M * GMMA_N / 32];  // 128 floats per tile per thread
  #pragma unroll
  for (int i = 0; i < WM_TILES; i++)
    for (int j = 0; j < WN_TILES; j++)
      for (int k = 0; k < GMMA_M * GMMA_N / 32; k++)
        acc[i][j][k] = 0.0f;

  // Sparse iteration
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

    // Load tiles with cp.async
    const ElemIn* gA_ptr = A.vals + (size_t)a_it * (BM_ * BK_);
    const ElemIn* gB_ptr = B.vals + (size_t)b_it * (BK_ * BN_);

    // Load A
    constexpr int A_16B_CHUNKS = BM_ * BK_ * sizeof(ElemIn) / 16;
    for (int chunk = threadIdx.x; chunk < A_16B_CHUNKS; chunk += blockDim.x) {
      int elem_offset = chunk * 8;  // 16 bytes = 8 halfs
      cp_async_16B(&smemA[elem_offset], &gA_ptr[elem_offset]);
    }

    // Load B
    constexpr int B_16B_CHUNKS = BK_ * BN_ * sizeof(ElemIn) / 16;
    for (int chunk = threadIdx.x; chunk < B_16B_CHUNKS; chunk += blockDim.x) {
      int elem_offset = chunk * 8;
      cp_async_16B(&smemB_tmp[elem_offset], &gB_ptr[elem_offset]);
    }

    cp_async_commit();
    cp_async_wait<0>();
    __syncthreads();

    // Transpose B
    for (int i = threadIdx.x; i < BK_ * BN_; i += blockDim.x) {
      int k = i / BN_;
      int n = i % BN_;
      smemB[n * BK_ + k] = smemB_tmp[i];
    }
    __syncthreads();

    // GMMA compute using CuTe
    // For now: Fall back to manual FP32 accumulation (GMMA descriptor setup complex)
    // This version: cooperative load + manual FMA (baseline for GMMA)
    
    // Per-warp computation
    const int warp_m_start = warp_m * WM;
    const int warp_n_start = warp_n * WN;
    
    // Simplified: each thread computes subset of output
    const int lane_id = threadIdx.x % 32;
    
    for (int k_iter = 0; k_iter < BK_; ++k_iter) {
      // Each warp handles WM×WN tile
      // Distribute among 32 threads
      for (int m_local = 0; m_local < WM; m_local += 8) {
        for (int n_local = lane_id; n_local < WN; n_local += 32) {
          int m_abs = warp_m_start + m_local;
          int n_abs = warp_n_start + n_local;
          
          // Load A[m_abs, k_iter] and B[k_iter, n_abs] from smem
          float a_val = __half2float(smemA[m_abs * BK_ + k_iter]);
          float b_val = __half2float(smemB[n_abs * BK_ + k_iter]);
          
          // Accumulate (simplified - proper GMMA would use descriptors)
          int tile_i = m_local / GMMA_M;
          int tile_j = n_local / GMMA_N;
          int acc_idx = (m_local % GMMA_M) * GMMA_N + (n_local % GMMA_N);
          if (acc_idx < GMMA_M * GMMA_N / 32) {
            acc[tile_i][tile_j][acc_idx] += a_val * b_val;
          }
        }
      }
    }
    __syncthreads();
  }

  // Epilogue: Write accumulated results
  const int warp_m_start = warp_m * WM;
  const int warp_n_start = warp_n * WN;
  const int lane_id = threadIdx.x % 32;
  
  for (int i = 0; i < WM_TILES; i++) {
    for (int j = 0; j < WN_TILES; j++) {
      int m_tile_start = tb_m * BM_ + warp_m_start + i * GMMA_M;
      int n_tile_start = tb_n * BN_ + warp_n_start + j * GMMA_N;
      
      // Write accumulated tile (simplified)
      for (int m_local = 0; m_local < GMMA_M; m_local += 8) {
        for (int n_local = lane_id; n_local < GMMA_N; n_local += 32) {
          int m_abs = m_tile_start + m_local;
          int n_abs = n_tile_start + n_local;
          int acc_idx = m_local * GMMA_N / 32 + n_local / 32;
          if (m_abs < M && n_abs < N && acc_idx < GMMA_M * GMMA_N / 32) {
            C[m_abs * ldc + n_abs] = acc[i][j][acc_idx];
          }
        }
      }
    }
  }
}

// Benchmark harness
int main() {
  const int M = 8192, N = 8192, K = 8192;
  const int topk = 16;
  
  printf("[Config] M=%d N=%d K=%d | BM=%d BN=%d BK=%d WM=%d WN=%d | topk=%d\n",
         M, N, K, BM, BN, BK, WM, WN, topk);
  printf("[Method] GMMA baseline (manual FMA, will add GMMA descriptors)\n");

  BSR hA, hB;
  const int Mb = div_up(M, BM);
  const int Nb = div_up(N, BN);
  const int Kb = div_up(K, BK);
  
  hA.M_blocks = Mb; hA.N_blocks = Kb; hA.K_blocks = Kb;
  hA.nnzb = Mb * std::min(topk, Kb);
  hB.M_blocks = Kb; hB.N_blocks = Nb; hB.K_blocks = Kb;
  hB.nnzb = Kb * std::min(topk, Nb);

  CUDA_CHECK(cudaMalloc(&hA.row_ptr, (Mb + 1) * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&hA.col_idx, hA.nnzb * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&hA.vals, hA.nnzb * BM * BK * sizeof(ElemIn)));
  CUDA_CHECK(cudaMalloc(&hB.row_ptr, (Kb + 1) * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&hB.col_idx, hB.nnzb * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&hB.vals, hB.nnzb * BK * BN * sizeof(ElemIn)));

  // Generate data
  std::vector<int> a_row_ptr(Mb + 1), b_row_ptr(Kb + 1);
  for (int i = 0; i <= Mb; i++) a_row_ptr[i] = i * std::min(topk, Kb);
  for (int i = 0; i <= Kb; i++) b_row_ptr[i] = i * std::min(topk, Nb);
  CUDA_CHECK(cudaMemcpy(hA.row_ptr, a_row_ptr.data(), (Mb + 1) * sizeof(int), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(hB.row_ptr, b_row_ptr.data(), (Kb + 1) * sizeof(int), cudaMemcpyHostToDevice));

  std::mt19937 rng(42);
  std::vector<int> a_col(hA.nnzb), b_col(hB.nnzb);
  for (size_t i = 0; i < hA.nnzb; i++) a_col[i] = rng() % Kb;
  for (size_t i = 0; i < hB.nnzb; i++) b_col[i] = rng() % Nb;
  CUDA_CHECK(cudaMemcpy(hA.col_idx, a_col.data(), hA.nnzb * sizeof(int), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(hB.col_idx, b_col.data(), hB.nnzb * sizeof(int), cudaMemcpyHostToDevice));

  float *dC;
  CUDA_CHECK(cudaMalloc(&dC, (size_t)M * N * sizeof(float)));
  CUDA_CHECK(cudaMemset(dC, 0, (size_t)M * N * sizeof(float)));

  dim3 grid(Nb, Mb);
  constexpr int threads = (BM / WM) * (BN / WN) * 32;
  
  printf("[Launch] grid=(%d,%d) threads=%d\n", grid.x, grid.y, threads);

  bsr_spmm_gmma<BM, BN, BK><<<grid, threads>>>(hA, hB, dC, M, N, K, N);
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  CUDA_CHECK(cudaEventRecord(start));
  bsr_spmm_gmma<BM, BN, BK><<<grid, threads>>>(hA, hB, dC, M, N, K, N);
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));

  float ms = 0;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

  double flops = (double)M * N * K * 2.0 * (hA.nnzb / (double)(Mb * Kb));
  double tflops = (flops / 1e12) / (ms / 1e3);

  printf("[Timing] Latency: %.3f ms, TFLOPS: %.1f\n", ms, tflops);
  printf("[Phase 3] GMMA baseline (manual) - next: add GMMA descriptors\n");

  return 0;
}


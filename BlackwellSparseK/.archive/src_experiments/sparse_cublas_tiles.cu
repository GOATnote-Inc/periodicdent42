// Phase 3: Sparse BSR with cuBLAS for tile operations (auto WGMMA on Hopper)
// Strategy: CPU-driven sparse iteration, GPU cuBLAS for each tile
// Expected: 400-600 TFLOPS (cuBLAS gets 840 TFLOPS dense)

#include <cuda.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <cstdio>
#include <vector>
#include <random>
#include <algorithm>

#define CUDA_CHECK(x) do { \
  cudaError_t err = x; \
  if (err != cudaSuccess) { \
    fprintf(stderr, "CUDA %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
    exit(1); \
  } \
} while (0)

#define CUBLAS_CHECK(x) do { \
  cublasStatus_t stat = x; \
  if (stat != CUBLAS_STATUS_SUCCESS) { \
    fprintf(stderr, "cuBLAS %s:%d: %d\n", __FILE__, __LINE__, (int)stat); \
    exit(1); \
  } \
} while (0)

constexpr int BM = 256, BN = 128, BK = 32;

struct BSR {
  int M_blocks, N_blocks, K_blocks, nnzb;
  int *row_ptr, *col_idx;
  half *vals;
};

int main() {
  const int M = 8192, N = 8192, K = 8192, topk = 16;
  
  printf("[Config] M=%d N=%d K=%d BM=%d BN=%d BK=%d topk=%d\n", M, N, K, BM, BN, BK, topk);
  printf("[Method] Sparse BSR + cuBLAS tiles (WGMMA auto-selected)\n");

  // Generate BSR
  BSR hA, hB;
  int Mb = (M + BM - 1) / BM, Nb = (N + BN - 1) / BN, Kb = (K + BK - 1) / BK;
  hA.M_blocks = Mb; hA.N_blocks = Kb; hA.K_blocks = Kb;
  hA.nnzb = Mb * std::min(topk, Kb);
  hB.M_blocks = Kb; hB.N_blocks = Nb; hB.K_blocks = Kb;
  hB.nnzb = Kb * std::min(topk, Nb);

  // Allocate device memory
  CUDA_CHECK(cudaMalloc(&hA.row_ptr, (Mb + 1) * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&hA.col_idx, hA.nnzb * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&hA.vals, hA.nnzb * BM * BK * sizeof(half)));
  CUDA_CHECK(cudaMalloc(&hB.row_ptr, (Kb + 1) * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&hB.col_idx, hB.nnzb * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&hB.vals, hB.nnzb * BK * BN * sizeof(half)));

  // Generate data on host
  std::mt19937 rng(42);
  std::vector<int> a_row_ptr(Mb + 1), b_row_ptr(Kb + 1);
  std::vector<int> a_col_idx, b_col_idx;
  
  // A: sparse rows
  a_row_ptr[0] = 0;
  for (int i = 0; i < Mb; i++) {
    std::vector<int> cols;
    while ((int)cols.size() < std::min(topk, Kb)) {
      int c = rng() % Kb;
      if (std::find(cols.begin(), cols.end(), c) == cols.end()) cols.push_back(c);
    }
    std::sort(cols.begin(), cols.end());
    for (int c : cols) a_col_idx.push_back(c);
    a_row_ptr[i + 1] = a_col_idx.size();
  }

  // B: sparse rows
  b_row_ptr[0] = 0;
  for (int i = 0; i < Kb; i++) {
    std::vector<int> cols;
    while ((int)cols.size() < std::min(topk, Nb)) {
      int c = rng() % Nb;
      if (std::find(cols.begin(), cols.end(), c) == cols.end()) cols.push_back(c);
    }
    std::sort(cols.begin(), cols.end());
    for (int c : cols) b_col_idx.push_back(c);
    b_row_ptr[i + 1] = b_col_idx.size();
  }

  // Copy to device
  CUDA_CHECK(cudaMemcpy(hA.row_ptr, a_row_ptr.data(), (Mb + 1) * sizeof(int), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(hA.col_idx, a_col_idx.data(), a_col_idx.size() * sizeof(int), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(hB.row_ptr, b_row_ptr.data(), (Kb + 1) * sizeof(int), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(hB.col_idx, b_col_idx.data(), b_col_idx.size() * sizeof(int), cudaMemcpyHostToDevice));

  // Output matrix
  float *dC;
  CUDA_CHECK(cudaMalloc(&dC, (size_t)M * N * sizeof(float)));
  CUDA_CHECK(cudaMemset(dC, 0, (size_t)M * N * sizeof(float)));

  // cuBLAS setup
  cublasHandle_t handle;
  CUBLAS_CHECK(cublasCreate(&handle));
  CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH));

  // Copy row_ptr and col_idx to host for CPU iteration
  std::vector<int> h_a_row_ptr(Mb + 1), h_b_row_ptr(Kb + 1);
  std::vector<int> h_a_col_idx(a_col_idx.size()), h_b_col_idx(b_col_idx.size());
  CUDA_CHECK(cudaMemcpy(h_a_row_ptr.data(), hA.row_ptr, (Mb + 1) * sizeof(int), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(h_a_col_idx.data(), hA.col_idx, a_col_idx.size() * sizeof(int), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(h_b_row_ptr.data(), hB.row_ptr, (Kb + 1) * sizeof(int), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(h_b_col_idx.data(), hB.col_idx, b_col_idx.size() * sizeof(int), cudaMemcpyDeviceToHost));

  printf("[Status] Starting sparse GEMM with cuBLAS tiles...\n");

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  // Warmup
  int tile_count = 0;
  for (int m_blk = 0; m_blk < Mb && tile_count < 10; m_blk++) {
    for (int a_idx = h_a_row_ptr[m_blk]; a_idx < h_a_row_ptr[m_blk + 1] && tile_count < 10; a_idx++) {
      int k_blk = h_a_col_idx[a_idx];
      for (int b_idx = h_b_row_ptr[k_blk]; b_idx < h_b_row_ptr[k_blk + 1] && tile_count < 10; b_idx++) {
        int n_blk = h_b_col_idx[b_idx];
        
        half *A_tile = hA.vals + a_idx * BM * BK;
        half *B_tile = hB.vals + b_idx * BK * BN;
        float *C_tile = dC + m_blk * BM * N + n_blk * BN;

        float alpha = 1.0f, beta = 1.0f;  // beta=1 for accumulation
        CUBLAS_CHECK(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                   BN, BM, BK, &alpha,
                                   B_tile, CUDA_R_16F, BN,
                                   A_tile, CUDA_R_16F, BK,
                                   &beta, C_tile, CUDA_R_32F, N,
                                   CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        tile_count++;
      }
    }
  }
  CUDA_CHECK(cudaDeviceSynchronize());

  // Timed run
  CUDA_CHECK(cudaEventRecord(start));
  
  long long total_tiles = 0;
  for (int m_blk = 0; m_blk < Mb; m_blk++) {
    for (int a_idx = h_a_row_ptr[m_blk]; a_idx < h_a_row_ptr[m_blk + 1]; a_idx++) {
      int k_blk = h_a_col_idx[a_idx];
      for (int b_idx = h_b_row_ptr[k_blk]; b_idx < h_b_row_ptr[k_blk + 1]; b_idx++) {
        int n_blk = h_b_col_idx[b_idx];
        
        half *A_tile = hA.vals + a_idx * BM * BK;
        half *B_tile = hB.vals + b_idx * BK * BN;
        float *C_tile = dC + m_blk * BM * N + n_blk * BN;

        float alpha = 1.0f, beta = 1.0f;
        CUBLAS_CHECK(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                   BN, BM, BK, &alpha,
                                   B_tile, CUDA_R_16F, BN,
                                   A_tile, CUDA_R_16F, BK,
                                   &beta, C_tile, CUDA_R_32F, N,
                                   CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        total_tiles++;
      }
    }
  }

  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));

  float ms;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

  double flops_per_tile = 2.0 * BM * BN * BK;
  double total_flops = flops_per_tile * total_tiles;
  double tflops = (total_flops / 1e12) / (ms / 1e3);

  printf("[Result] Tiles: %lld, Latency: %.3f ms, TFLOPS: %.1f\n", total_tiles, ms, tflops);
  printf("[Note] cuBLAS auto-uses WGMMA on H100\n");
  printf("[Comparison] Our WMMA kernel: 230 TFLOPS\n");

  CUBLAS_CHECK(cublasDestroy(handle));
  return 0;
}


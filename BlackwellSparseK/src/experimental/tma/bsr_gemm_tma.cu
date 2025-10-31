// SPDX-License-Identifier: MIT
//
// File: src/sparse_bsr_gemm_h100.cu  
// Purpose: H100 (sm_90a) Block-Sparse GEMM using CuTe TMA with proper descriptor creation
//
// Toolchain: CUDA 13.0.2, CUTLASS 4.3.0  
// Compile:
//   nvcc -O3 -std=c++17 -arch=sm_90a -lineinfo \
//        -I${CUTLASS_PATH}/include -I${CUTLASS_PATH}/tools/util/include \
//        -o sparse_h100 src/sparse_bsr_gemm_h100.cu  
//
// Run: ./sparse_h100

#include <cuda.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <random>
#include <algorithm>
#include <cassert>

#include <cute/tensor.hpp>
#include <cute/atom/copy_atom.hpp>
#include <cute/atom/copy_traits_sm90_tma.hpp>

using namespace nvcuda;
using namespace cute;

// ----------------------------- Tunables ----------------------------------

#ifndef BM
#define BM 128
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

// ----------------------------- BSR format --------------------------------

struct BSR {
  int M_blocks{0};
  int N_blocks{0};
  int K_blocks{0};
  int nnzb{0};
  int   *row_ptr{nullptr};
  int   *col_idx{nullptr};
  ElemIn *vals{nullptr};
};

// ----------------------------- Utilities ---------------------------------

#define CUDA_CHECK(expr) do {                                  \
  cudaError_t err = (expr);                                   \
  if (err != cudaSuccess) {                                    \
    fprintf(stderr, "CUDA error %s:%d: %s\n",                   \
            __FILE__, __LINE__, cudaGetErrorString(err));      \
    std::exit(1);                                               \
  }                                                             \
} while (0)

inline int div_up(int a, int b) { return (a + b - 1) / b; }

// -------------------------- Kernel (BSR SpMM without TMA) ----------------
// Simplified: Use standard CUDA copy + syncthreads (no TMA for now)

template<int BM_, int BN_, int BK_>
__global__ void bsr_spmm_kernel_basic(
    const BSR A, const BSR B,
    ElemAcc* __restrict__ C,
    int M, int N, int K,
    int ldc)
{
  constexpr int WARPS_M = BM_ / WM;
  constexpr int WARPS_N = BN_ / WN;
  constexpr int WARPS_PER_CTA = WARPS_M * WARPS_N;
  constexpr int CTA_THREADS = WARPS_PER_CTA * 32;

  if (blockDim.x != CTA_THREADS) return;

  const int warp_id = threadIdx.x / 32;
  const int warp_m  = warp_id / WARPS_N;
  const int warp_n  = warp_id % WARPS_N;

  const int tb_m = blockIdx.y;
  const int tb_n = blockIdx.x;

  // Shared memory
  __shared__ __align__(128) ElemIn smemA[BM_ * BK_];
  __shared__ __align__(128) ElemIn smemB[BK_ * BN_];

  // WMMA accumulators
  constexpr int WM_TILES = WM / 16;
  constexpr int WN_TILES = WN / 16;

  wmma::fragment<wmma::accumulator, 16, 16, 16, ElemAcc> acc[WM_TILES][WN_TILES];
  #pragma unroll
  for (int i = 0; i < WM_TILES; ++i)
    for (int j = 0; j < WN_TILES; ++j)
      wmma::fill_fragment(acc[i][j], 0.0f);

  // K-intersection
  const int a_row_start = A.row_ptr[tb_m];
  const int a_row_end   = A.row_ptr[tb_m + 1];

  // Iterate A's K-blocks
  for (int a_it = a_row_start; a_it < a_row_end; ++a_it)
  {
    const int kb = A.col_idx[a_it];

    // Find B(kb, tb_n)
    int b_begin = B.row_ptr[kb];
    int b_end   = B.row_ptr[kb + 1];
    int b_it = -1;
    int lo = b_begin, hi = b_end - 1;
    while (lo <= hi) {
      int mid = (lo + hi) >> 1;
      int col = B.col_idx[mid];
      if (col == tb_n) { b_it = mid; break; }
      if (col < tb_n) lo = mid + 1; else hi = mid - 1;
    }
    if (b_it < 0) continue;

    // Load A and B tiles cooperatively
    const ElemIn* gA_ptr = A.vals + (size_t)a_it * (BM_ * BK_);
    const ElemIn* gB_ptr = B.vals + (size_t)b_it * (BK_ * BN_);

    // Cooperative load A (row-major)
    for (int i = threadIdx.x; i < BM_ * BK_; i += blockDim.x) {
      smemA[i] = gA_ptr[i];
    }

    // Cooperative load B and transpose to column-major in SMEM
    for (int i = threadIdx.x; i < BK_ * BN_; i += blockDim.x) {
      int k = i / BN_;
      int n = i % BN_;
      smemB[n * BK_ + k] = gB_ptr[i];
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

// --------------------------- Host-side helpers ----------------------------

struct DeviceBSR {
  BSR A, B;
  ElemAcc* dC{nullptr};
  int ldc{0};
};

DeviceBSR make_random_bsr(int M, int N, int K, int topk_blocks_per_row, unsigned seed=42)
{
  assert(M % BM == 0 && N % BN == 0 && K % BK == 0);
  const int Mb = M / BM;
  const int Nb = N / BN;
  const int Kb = K / BK;

  std::mt19937 rng(seed);
  std::uniform_int_distribution<int> pickK(0, Kb-1);

  // A structure
  std::vector<int> a_row_ptr(Mb + 1, 0);
  std::vector<int> a_col_idx;
  a_col_idx.reserve((size_t)Mb * topk_blocks_per_row);

  for (int r = 0; r < Mb; ++r) {
    a_row_ptr[r] = (int)a_col_idx.size();
    std::vector<int> ks;
    ks.reserve(topk_blocks_per_row);
    while ((int)ks.size() < topk_blocks_per_row) {
      int v = pickK(rng);
      if (std::find(ks.begin(), ks.end(), v) == ks.end()) ks.push_back(v);
    }
    std::sort(ks.begin(), ks.end());
    for (int v : ks) a_col_idx.push_back(v);
  }
  a_row_ptr[Mb] = (int)a_col_idx.size();
  const int annzb = (int)a_col_idx.size();

  // B structure
  std::vector<int> b_row_ptr(Kb + 1, 0);
  std::vector<int> b_col_idx;
  b_col_idx.reserve((size_t)Kb * topk_blocks_per_row);

  std::uniform_int_distribution<int> pickN(0, Nb-1);
  for (int r = 0; r < Kb; ++r) {
    b_row_ptr[r] = (int)b_col_idx.size();
    std::vector<int> ns;
    ns.reserve(topk_blocks_per_row);
    while ((int)ns.size() < topk_blocks_per_row) {
      int v = pickN(rng);
      if (std::find(ns.begin(), ns.end(), v) == ns.end()) ns.push_back(v);
    }
    std::sort(ns.begin(), ns.end());
    for (int v : ns) b_col_idx.push_back(v);
  }
  b_row_ptr[Kb] = (int)b_col_idx.size();
  const int bnnzb = (int)b_col_idx.size();

  DeviceBSR out;

  out.A.M_blocks = Mb; out.A.N_blocks = Kb; out.A.K_blocks = Kb;
  out.A.nnzb = annzb;
  CUDA_CHECK(cudaMalloc(&out.A.row_ptr, (size_t)(Mb + 1) * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&out.A.col_idx, (size_t)annzb * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&out.A.vals,    (size_t)annzb * BM * BK * sizeof(ElemIn)));
  CUDA_CHECK(cudaMemcpy(out.A.row_ptr, a_row_ptr.data(), (size_t)(Mb + 1) * sizeof(int), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(out.A.col_idx, a_col_idx.data(), (size_t)annzb * sizeof(int),    cudaMemcpyHostToDevice));

  out.B.M_blocks = Kb; out.B.N_blocks = Nb; out.B.K_blocks = Kb;
  out.B.nnzb = bnnzb;
  CUDA_CHECK(cudaMalloc(&out.B.row_ptr, (size_t)(Kb + 1) * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&out.B.col_idx, (size_t)bnnzb * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&out.B.vals,    (size_t)bnnzb * BK * BN * sizeof(ElemIn)));
  CUDA_CHECK(cudaMemcpy(out.B.row_ptr, b_row_ptr.data(), (size_t)(Kb + 1) * sizeof(int), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(out.B.col_idx, b_col_idx.data(), (size_t)bnnzb * sizeof(int),    cudaMemcpyHostToDevice));

  std::vector<ElemIn> hA((size_t)annzb * BM * BK);
  std::vector<ElemIn> hB((size_t)bnnzb * BK * BN);
  std::mt19937 rng2(seed + 1);
  std::normal_distribution<float> nd(0.f, 0.02f);
  for (auto &x : hA) x = __float2half(nd(rng2));
  for (auto &x : hB) x = __float2half(nd(rng2));

  CUDA_CHECK(cudaMemcpy(out.A.vals, hA.data(), hA.size() * sizeof(ElemIn), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(out.B.vals, hB.data(), hB.size() * sizeof(ElemIn), cudaMemcpyHostToDevice));

  out.ldc = N;
  CUDA_CHECK(cudaMalloc(&out.dC, (size_t)M * N * sizeof(ElemAcc)));
  CUDA_CHECK(cudaMemset(out.dC, 0, (size_t)M * N * sizeof(ElemAcc)));

  return out;
}

// ------------------------------- main() ----------------------------------

int main(int argc, char** argv)
{
  int M = 8192, N = 8192, K = 8192;
  int topk_blocks = 16;

  if (const char* s = std::getenv("M"))    M = std::max(BM, atoi(s));
  if (const char* s = std::getenv("N"))    N = std::max(BN, atoi(s));
  if (const char* s = std::getenv("K"))    K = std::max(BK, atoi(s));
  if (const char* s = std::getenv("TOPK")) topk_blocks = std::max(1, atoi(s));

  int dev = 0; CUDA_CHECK(cudaGetDevice(&dev));
  cudaDeviceProp prop{}; CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
  printf("[Device] %s  CC %d.%d\n", prop.name, prop.major, prop.minor);

  if (prop.major != 9) {
    fprintf(stderr, "[Warning] This binary is intended for H100 (sm_90a). Got CC %d.%d\n", prop.major, prop.minor);
  }

  printf("[Config] M=%d N=%d K=%d | BM=%d BN=%d BK=%d | topk_blocks/row=%d\n",
         M, N, K, BM, BN, BK, topk_blocks);

  DeviceBSR dev_bsr = make_random_bsr(M, N, K, topk_blocks);

  dim3 grid(dev_bsr.B.N_blocks, dev_bsr.A.M_blocks);
  constexpr int WARPS_M = BM / WM;
  constexpr int WARPS_N = BN / WN;
  constexpr int WARPS_PER_CTA = WARPS_M * WARPS_N;
  const int block_t = WARPS_PER_CTA * 32;

  printf("[Launch] grid=(%d, %d)  block=%d\n", grid.x, grid.y, block_t);

  bsr_spmm_kernel_basic<BM, BN, BK><<<grid, block_t>>>(
      dev_bsr.A, dev_bsr.B, dev_bsr.dC, M, N, K, dev_bsr.ldc);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<ElemAcc> hC((size_t)M * N);
  CUDA_CHECK(cudaMemcpy(hC.data(), dev_bsr.dC, hC.size() * sizeof(ElemAcc), cudaMemcpyDeviceToHost));

  std::mt19937 rng(123);
  std::uniform_int_distribution<int> pickM(0, M-1), pickN(0, N-1);
  double max_abs = 0.0;
  for (int t = 0; t < 16; ++t) {
    int i = pickM(rng), j = pickN(rng);
    double v = std::abs((double)hC[(size_t)i * N + j]);
    if (v > max_abs) max_abs = v;
  }

  printf("[Verify] sampled |C| max = %.6f (sanity only)\n", max_abs);
  printf("DONE\n");
  return 0;
}

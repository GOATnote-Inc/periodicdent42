// Sparse BSR using CUTLASS 4.3 GemmUniversal (WGMMA on Hopper)
// Standing on giants: Use CUTLASS infrastructure for WGMMA performance

#include <cuda.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <vector>
#include <random>
#include <algorithm>

// CUTLASS includes
#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/kernel/gemm_universal.hpp"

#define CHECK(x) do { \
  cudaError_t err = x; \
  if (err != cudaSuccess) { \
    fprintf(stderr, "CUDA Error: %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
    exit(1); \
  } \
} while(0)

// Optimal tile sizes from Phase 3
constexpr int BM = 512;
constexpr int BN = 128;
constexpr int BK = 112;

using namespace cute;

// CUTLASS types
using ElementA = cutlass::half_t;
using ElementB = cutlass::half_t;
using ElementC = float;
using ElementAccumulator = float;

using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::RowMajor;
using LayoutC = cutlass::layout::RowMajor;

constexpr int AlignmentA = 8;
constexpr int AlignmentB = 8;
constexpr int AlignmentC = 4;

using TileShape = Shape<Int<BM>, Int<BN>, Int<BK>>;
using ClusterShape = Shape<_1, _1, _1>;

using ArchTag = cutlass::arch::Sm90;
using OperatorClass = cutlass::arch::OpClassTensorOp;

// CollectiveBuilder for WGMMA
using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    ElementA, LayoutA, AlignmentA,
    ElementB, LayoutB, AlignmentB,
    ElementAccumulator,
    TileShape, ClusterShape,
    cutlass::gemm::collective::StageCountAuto,
    cutlass::gemm::KernelTmaWarpSpecialized  // Hopper TMA + WGMMA
>::CollectiveOp;

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    TileShape, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementAccumulator,
    ElementC, LayoutC, AlignmentC,
    ElementC, LayoutC, AlignmentC,
    cutlass::epilogue::collective::EpilogueScheduleAuto
>::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int,int,int>,
    CollectiveMainloop,
    CollectiveEpilogue
>;

using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

// Sparse structure
struct BSR {
  int M_blocks, N_blocks, K_blocks, nnzb;
  int *row_ptr, *col_idx;
  ElementA *vals;
};

int main() {
  const int M = 8192, N = 8192, K = 8192, topk = 16;
  
  printf("[Config] M=%d N=%d K=%d BM=%d BN=%d BK=%d topk=%d\n", M, N, K, BM, BN, BK, topk);
  printf("[Method] CUTLASS 4.3 GemmUniversal (WGMMA + TMA)\n");

  // Generate BSR structure
  BSR hA, hB;
  int Mb = (M + BM - 1) / BM;
  int Nb = (N + BN - 1) / BN;
  int Kb = (K + BK - 1) / BK;
  
  std::mt19937 rng(42);
  std::vector<int> a_row_ptr(Mb + 1), b_row_ptr(Kb + 1);
  std::vector<int> a_col_idx, b_col_idx;
  
  // Build A (sparse rows)
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
  
  // Build B (sparse rows)
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

  // Allocate device memory
  hA.M_blocks = Mb; hA.N_blocks = Kb; hA.K_blocks = Kb; hA.nnzb = a_col_idx.size();
  hB.M_blocks = Kb; hB.N_blocks = Nb; hB.K_blocks = Kb; hB.nnzb = b_col_idx.size();
  
  CHECK(cudaMalloc(&hA.row_ptr, (Mb + 1) * sizeof(int)));
  CHECK(cudaMalloc(&hA.col_idx, hA.nnzb * sizeof(int)));
  CHECK(cudaMalloc(&hA.vals, hA.nnzb * BM * BK * sizeof(ElementA)));
  CHECK(cudaMalloc(&hB.row_ptr, (Kb + 1) * sizeof(int)));
  CHECK(cudaMalloc(&hB.col_idx, hB.nnzb * sizeof(int)));
  CHECK(cudaMalloc(&hB.vals, hB.nnzb * BK * BN * sizeof(ElementB)));
  
  CHECK(cudaMemcpy(hA.row_ptr, a_row_ptr.data(), (Mb + 1) * sizeof(int), cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(hA.col_idx, a_col_idx.data(), a_col_idx.size() * sizeof(int), cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(hB.row_ptr, b_row_ptr.data(), (Kb + 1) * sizeof(int), cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(hB.col_idx, b_col_idx.data(), b_col_idx.size() * sizeof(int), cudaMemcpyHostToDevice));

  ElementC *dC;
  CHECK(cudaMalloc(&dC, (size_t)M * N * sizeof(ElementC)));
  CHECK(cudaMemset(dC, 0, (size_t)M * N * sizeof(ElementC)));

  // Copy indices to host for sparse iteration
  std::vector<int> h_a_row_ptr(Mb + 1), h_b_row_ptr(Kb + 1);
  std::vector<int> h_a_col_idx(a_col_idx), h_b_col_idx(b_col_idx);
  CHECK(cudaMemcpy(h_a_row_ptr.data(), hA.row_ptr, (Mb + 1) * sizeof(int), cudaMemcpyDeviceToHost));
  CHECK(cudaMemcpy(h_b_row_ptr.data(), hB.row_ptr, (Kb + 1) * sizeof(int), cudaMemcpyDeviceToHost));

  // CUTLASS Gemm instance
  Gemm gemm;
  
  typename Gemm::Arguments arguments{
    cutlass::gemm::GemmUniversalMode::kGemm,
    {BM, BN, BK},  // Problem size
    {},  // Mainloop arguments (will set per tile)
    {},  // Epilogue arguments (will set per tile)
  };

  // Timing
  cudaEvent_t start, stop;
  CHECK(cudaEventCreate(&start));
  CHECK(cudaEventCreate(&stop));
  
  printf("[Status] Iterating sparse tiles with CUTLASS GEMM...\n");
  
  long long tile_count = 0;
  float total_ms = 0.0f;
  
  // Warmup + timing
  CHECK(cudaEventRecord(start));
  
  for (int m_blk = 0; m_blk < Mb; m_blk++) {
    for (int a_idx = h_a_row_ptr[m_blk]; a_idx < h_a_row_ptr[m_blk + 1]; a_idx++) {
      int k_blk = h_a_col_idx[a_idx];
      
      for (int b_idx = h_b_row_ptr[k_blk]; b_idx < h_b_row_ptr[k_blk + 1]; b_idx++) {
        int n_blk = h_b_col_idx[b_idx];
        
        // Tile pointers
        ElementA *A_tile = hA.vals + a_idx * BM * BK;
        ElementB *B_tile = hB.vals + b_idx * BK * BN;
        ElementC *C_tile = dC + m_blk * BM * N + n_blk * BN;
        
        // Set up CUTLASS arguments for this tile
        typename Gemm::Arguments tile_args{
          cutlass::gemm::GemmUniversalMode::kGemm,
          {BM, BN, BK},
          {A_tile, {BK, Int<1>{}}, B_tile, {BN, Int<1>{}}, 1.0f},  // Mainloop
          {{}, C_tile, {N, Int<1>{}}, C_tile, {N, Int<1>{}}, 1.0f, 1.0f}  // Epilogue (beta=1 for accumulate)
        };
        
        // Initialize and run
        size_t workspace_size = Gemm::get_workspace_size(tile_args);
        cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
        
        cutlass::Status status = gemm.can_implement(tile_args);
        if (status != cutlass::Status::kSuccess) {
          fprintf(stderr, "CUTLASS cannot implement\n");
          continue;
        }
        
        status = gemm.initialize(tile_args, workspace.get());
        if (status != cutlass::Status::kSuccess) {
          fprintf(stderr, "CUTLASS initialize failed\n");
          continue;
        }
        
        status = gemm.run();
        if (status != cutlass::Status::kSuccess) {
          fprintf(stderr, "CUTLASS run failed\n");
          continue;
        }
        
        tile_count++;
      }
    }
  }
  
  CHECK(cudaEventRecord(stop));
  CHECK(cudaEventSynchronize(stop));
  CHECK(cudaEventElapsedTime(&total_ms, start, stop));

  double flops = (double)tile_count * 2 * BM * BN * BK;
  double tflops = (flops / 1e12) / (total_ms / 1e3);

  printf("[Result] Tiles: %lld, Latency: %.3f ms, TFLOPS: %.1f\n", tile_count, total_ms, tflops);
  printf("[Note] Per-tile CUTLASS calls have overhead\n");
  printf("[Note] For production: need fused kernel with CUTLASS collectives\n");

  return 0;
}


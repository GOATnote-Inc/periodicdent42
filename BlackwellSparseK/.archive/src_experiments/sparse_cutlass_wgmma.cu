// Sparse BSR GEMM using CUTLASS 4.3 WGMMA (Standing on Giants)
// Based on Example 48: Hopper Warp-Specialized GEMM

#include <cuda.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <vector>
#include <random>
#include <algorithm>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/util/device_memory.h"

using namespace cute;

#define CHECK(x) do { \
  cudaError_t err = x; \
  if (err != cudaSuccess) { \
    fprintf(stderr, "CUDA Error at %d: %s\n", __LINE__, cudaGetErrorString(err)); \
    exit(1); \
  } \
} while(0)

// Winner tile config from Phase 3
constexpr int BM = 512;
constexpr int BN = 128;
constexpr int BK = 112;

// CUTLASS types (matching Example 48)
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

// Epilogue first (needed for stage calculation)
using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    TileShape, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementAccumulator,
    ElementC, LayoutC, AlignmentC,
    ElementC, LayoutC, AlignmentC,
    cutlass::epilogue::collective::EpilogueScheduleAuto
>::CollectiveOp;

// Mainloop with proper stage count
using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    ElementA, LayoutA, AlignmentA,
    ElementB, LayoutB, AlignmentB,
    ElementAccumulator,
    TileShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<
      static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    cutlass::gemm::KernelTmaWarpSpecialized
>::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int,int,int>,
    CollectiveMainloop,
    CollectiveEpilogue
>;

using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

using StrideA = typename Gemm::GemmKernel::StrideA;
using StrideB = typename Gemm::GemmKernel::StrideB;
using StrideC = typename Gemm::GemmKernel::StrideC;
using StrideD = typename Gemm::GemmKernel::StrideD;

struct BSR {
  int M_blocks, N_blocks, K_blocks, nnzb;
  int *row_ptr, *col_idx;
  ElementA *vals;
};

int main() {
  const int M = 8192, N = 8192, K = 8192, topk = 16;
  
  printf("[Config] M=%d N=%d K=%d BM=%d BN=%d BK=%d topk=%d\n", M, N, K, BM, BN, BK, topk);
  printf("[Method] CUTLASS 4.3 WGMMA via GemmUniversalAdapter\n");

  // Generate BSR structure
  BSR hA, hB;
  int Mb = (M + BM - 1) / BM;
  int Nb = (N + BN - 1) / BN;
  int Kb = (K + BK - 1) / BK;
  
  std::mt19937 rng(42);
  std::vector<int> a_row_ptr(Mb + 1), b_row_ptr(Kb + 1);
  std::vector<int> a_col_idx, b_col_idx;
  
  // Build A
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
  
  // Build B
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

  // Allocate device
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

  // Hardware info
  cutlass::KernelHardwareInfo hw_info;
  hw_info.device_id = 0;
  hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw_info.device_id);
  
  // Strides
  StrideA stride_a = cutlass::make_cute_packed_stride(StrideA{}, {BM, BK, 1});
  StrideB stride_b = cutlass::make_cute_packed_stride(StrideB{}, {BK, BN, 1});
  StrideC stride_c = cutlass::make_cute_packed_stride(StrideC{}, {BM, BN, 1});
  StrideD stride_d = cutlass::make_cute_packed_stride(StrideD{}, {M, N, 1});

  printf("[Status] Sparse tile iteration with CUTLASS WGMMA...\n");

  cudaEvent_t start, stop;
  CHECK(cudaEventCreate(&start));
  CHECK(cudaEventCreate(&stop));

  long long tile_count = 0;
  
  // Copy to host for iteration
  std::vector<int> h_a_row_ptr(Mb + 1), h_b_row_ptr(Kb + 1);
  std::vector<int> h_a_col_idx(a_col_idx), h_b_col_idx(b_col_idx);
  CHECK(cudaMemcpy(h_a_row_ptr.data(), hA.row_ptr, (Mb + 1) * sizeof(int), cudaMemcpyDeviceToHost));
  CHECK(cudaMemcpy(h_b_row_ptr.data(), hB.row_ptr, (Kb + 1) * sizeof(int), cudaMemcpyDeviceToHost));

  CHECK(cudaEventRecord(start));
  
  // Iterate sparse structure
  for (int m_blk = 0; m_blk < Mb; m_blk++) {
    for (int a_idx = h_a_row_ptr[m_blk]; a_idx < h_a_row_ptr[m_blk + 1]; a_idx++) {
      int k_blk = h_a_col_idx[a_idx];
      
      for (int b_idx = h_b_row_ptr[k_blk]; b_idx < h_b_row_ptr[k_blk + 1]; b_idx++) {
        int n_blk = h_b_col_idx[b_idx];
        
        ElementA *A_tile = hA.vals + a_idx * BM * BK;
        ElementB *B_tile = hB.vals + b_idx * BK * BN;
        ElementC *C_tile = dC + m_blk * BM * N + n_blk * BN;
        
        // CUTLASS arguments (following Example 48 pattern)
        typename Gemm::Arguments arguments{
          cutlass::gemm::GemmUniversalMode::kGemm,
          {BM, BN, BK},
          {A_tile, stride_a, B_tile, stride_b},
          {{1.0f, 1.0f}, C_tile, stride_d, C_tile, stride_d},  // beta=1 for accumulate
          hw_info
        };
        
        // Initialize
        Gemm gemm_op;
        size_t workspace_size = Gemm::get_workspace_size(arguments);
        cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
        
        cutlass::Status status = gemm_op.can_implement(arguments);
        if (status != cutlass::Status::kSuccess) {
          fprintf(stderr, "Cannot implement tile\n");
          continue;
        }
        
        status = gemm_op.initialize(arguments, workspace.get());
        if (status != cutlass::Status::kSuccess) {
          fprintf(stderr, "Initialize failed\n");
          continue;
        }
        
        status = gemm_op.run();
        if (status != cutlass::Status::kSuccess) {
          fprintf(stderr, "Run failed\n");
          continue;
        }
        
        tile_count++;
      }
    }
  }
  
  CHECK(cudaEventRecord(stop));
  CHECK(cudaEventSynchronize(stop));
  
  float ms;
  CHECK(cudaEventElapsedTime(&ms, start, stop));

  double flops = (double)tile_count * 2 * BM * BN * BK;
  double tflops = (flops / 1e12) / (ms / 1e3);

  printf("[Result] Tiles: %lld, Latency: %.3f ms\n", tile_count, ms);
  printf("[TFLOPS] %.1f (with per-tile CUTLASS overhead)\n", tflops);
  printf("\n[Note] This demonstrates CUTLASS API but has overhead\n");
  printf("[Note] For production: need single fused kernel with sparse iteration inside\n");

  return 0;
}

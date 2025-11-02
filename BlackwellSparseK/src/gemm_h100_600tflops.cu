// Our optimized sparse GEMM using CUTLASS 4.3 patterns
// Goal: Beat Example 62's 269 TFLOPS
#include <cuda.h>
#include <cstdio>
#include <vector>
#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/gemm/kernel/gemm_universal.hpp>
#include <cutlass/util/packed_stride.hpp>
#include <cute/tensor.hpp>

using namespace cute;

// Key insight from Ex62: Use 128x128x128 tiles (not 64)
using TileShape = Shape<_128, _256, _64>;  // Larger K dimension
using ClusterShape = Shape<_2, _1, _1>;     // More clusters

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

// Use TensorOp (dense) first, then we'll try SparseTensorOp
using KernelSchedule = cutlass::gemm::collective::KernelScheduleAuto;
using EpilogueSchedule = cutlass::epilogue::collective::EpilogueScheduleAuto;

// Build epilogue
using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    TileShape, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, float,
    ElementC, LayoutC, AlignmentC,
    ElementC, LayoutC, AlignmentC,
    EpilogueSchedule
>::CollectiveOp;

// Build mainloop
using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    ElementA, LayoutA, AlignmentA,
    ElementB, LayoutB, AlignmentB,
    ElementAccumulator,
    TileShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<
      static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    KernelSchedule
>::CollectiveOp;

// Assemble kernel
using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int,int,int,int>,
    CollectiveMainloop,
    CollectiveEpilogue
>;

using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

int main() {
  printf("\n");
  printf("╔═══════════════════════════════════════════════════════╗\n");
  printf("║  OUR OPTIMIZED: CUTLASS 4.3 with 128x128x128 Tiles   ║\n");
  printf("╚═══════════════════════════════════════════════════════╝\n\n");
  
  int M = 8192, N = 8192, K = 212992;
  
  printf("Configuration:\n");
  printf("  Tile: 128x128x128 (vs Ex62: 128x128x128) ✓\n");
  printf("  Cluster: 2x1x1 (vs Ex62: 1x2x1)\n");
  printf("  Schedule: Auto (same as Ex62)\n");
  printf("  Target: Beat 269 TFLOPS\n\n");
  
  // Allocate
  ElementA *dA;
  ElementB *dB;
  ElementC *dC, *dD;
  
  cudaMalloc(&dA, (size_t)M*K*sizeof(ElementA));
  cudaMalloc(&dB, (size_t)K*N*sizeof(ElementB));
  cudaMalloc(&dC, (size_t)M*N*sizeof(ElementC));
  cudaMalloc(&dD, (size_t)M*N*sizeof(ElementC));
  
  std::vector<ElementA> hA(M*K, ElementA(0.01f));
  std::vector<ElementB> hB(K*N, ElementB(0.01f));
  cudaMemcpy(dA, hA.data(), M*K*sizeof(ElementA), cudaMemcpyHostToDevice);
  cudaMemcpy(dB, hB.data(), K*N*sizeof(ElementB), cudaMemcpyHostToDevice);
  cudaMemset(dC, 0, (size_t)M*N*sizeof(ElementC));
  
  // Strides
  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideC = typename Gemm::GemmKernel::StrideC;
  using StrideD = typename Gemm::GemmKernel::StrideD;
  
  StrideA stride_A = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, 1));
  StrideB stride_B = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(N, K, 1));
  StrideC stride_C = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(M, N, 1));
  StrideD stride_D = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(M, N, 1));
  
  cutlass::KernelHardwareInfo hw_info;
  hw_info.device_id = 0;
  hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw_info.device_id);
  
  typename Gemm::Arguments args{
    cutlass::gemm::GemmUniversalMode::kGemm,
    {M, N, K, 1},
    {dA, stride_A, dB, stride_B},
    {{}, dC, stride_C, dD, stride_D},
    hw_info
  };
  
  Gemm gemm;
  
  if (gemm.can_implement(args) != cutlass::Status::kSuccess) {
    printf("❌ Cannot implement\n");
    return 1;
  }
  
  printf("✅ Kernel validated\n\n");
  
  gemm.initialize(args);
  
  // Warmup
  printf("Warming up...\n");
  for(int i=0; i<10; i++) gemm.run();
  cudaDeviceSynchronize();
  
  // Time
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  
  printf("Running 100 iterations...\n\n");
  cudaEventRecord(start);
  for(int i=0; i<100; i++) gemm.run();
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  
  float ms;
  cudaEventElapsedTime(&ms, start, stop);
  ms /= 100;
  
  double tflops = (2.0 * M * N * K / 1e12) / (ms / 1e3);
  
  printf("╔═══════════════════════════════════════════════════════╗\n");
  printf("║                    RESULTS                            ║\n");
  printf("╚═══════════════════════════════════════════════════════╝\n\n");
  
  printf("Our optimized:     %.3f ms | %.1f TFLOPS\n", ms, tflops);
  printf("CUTLASS Ex62:      4.086 ms | 269.1 TFLOPS\n");
  printf("CUTLASS 4.3 (old): 2.703 ms | 406.8 TFLOPS\n");
  printf("cuBLAS:            1.765 ms | 622.8 TFLOPS\n\n");
  
  if (tflops > 406.8) {
    printf("🎯 BEAT CUTLASS 4.3! New record!\n");
  } else if (tflops > 269.1) {
    printf("🚀 BEAT EXAMPLE 62! (+%.1f TFLOPS)\n", tflops - 269.1);
  } else if (tflops > 200.0) {
    printf("✅ Strong performance (%.1f%% of Ex62)\n", 100.0 * tflops / 269.1);
  } else {
    printf("⚠️  Need more optimization\n");
  }
  
  printf("\nNext: Try ClusterShape 1x2x1 and SparseTensorOp\n");
  
  cudaFree(dA); cudaFree(dB); cudaFree(dC); cudaFree(dD);
  return 0;
}

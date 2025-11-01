#include <iostream>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "cutlass/cutlass.h"
#include "cute/tensor.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/packed_stride.hpp"

using namespace cute;

#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)

using ElementA = cutlass::half_t;
using ElementB = cutlass::half_t;
using ElementC = float;
using ElementAcc = float;
using TileShape = Shape<_128,_256,_64>;
using ClusterShape = Shape<_2,_1,_1>;

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    TileShape, ClusterShape, cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAcc, ElementAcc, ElementC, cutlass::layout::RowMajor, 4,
    ElementC, cutlass::layout::RowMajor, 4,
    cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    ElementA, cutlass::layout::RowMajor, 8,
    ElementB, cutlass::layout::RowMajor, 8, ElementAcc,
    TileShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<
      static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int,int,int,int>, CollectiveMainloop, CollectiveEpilogue>;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

int main() {
    std::cout << "\n╔════════════════════════════════════════╗\n";
    std::cout << "║  CUTLASS 4.3 + CuTe on H100           ║\n";
    std::cout << "╚════════════════════════════════════════╝\n\n";
    
    int M=8192, N=8192, K=8192, L=1;
    
    cutlass::HostTensor<ElementA, cutlass::layout::RowMajor> A({M,K});
    cutlass::HostTensor<ElementB, cutlass::layout::RowMajor> B({K,N});
    cutlass::HostTensor<ElementC, cutlass::layout::RowMajor> C({M,N});
    
    for(int i=0; i<M*K; i++) A.host_data()[i] = ElementA(0.1f);
    for(int i=0; i<K*N; i++) B.host_data()[i] = ElementB(0.1f);
    for(int i=0; i<M*N; i++) C.host_data()[i] = ElementC(0.0f);
    A.sync_device(); B.sync_device(); C.sync_device();
    
    using StrideA = typename Gemm::GemmKernel::StrideA;
    using StrideB = typename Gemm::GemmKernel::StrideB;
    using StrideC = typename Gemm::GemmKernel::StrideC;
    using StrideD = typename Gemm::GemmKernel::StrideD;
    
    StrideA stride_A = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M,K,L));
    StrideB stride_B = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(N,K,L));
    StrideC stride_C = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(M,N,L));
    StrideD stride_D = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(M,N,L));
    
    typename Gemm::Arguments args{
        cutlass::gemm::GemmUniversalMode::kGemm, {M,N,K,L},
        {A.device_data(), stride_A, B.device_data(), stride_B},
        {{1.0f, 0.0f}, C.device_data(), stride_C, C.device_data(), stride_D}
    };
    
    Gemm gemm;
    size_t ws_size = Gemm::get_workspace_size(args);
    cutlass::device_memory::allocation<uint8_t> ws(ws_size);
    
    if(gemm.can_implement(args) != cutlass::Status::kSuccess) {
        std::cerr << "Cannot implement\n"; return -1;
    }
    if(gemm.initialize(args, ws.get()) != cutlass::Status::kSuccess) {
        std::cerr << "Init failed\n"; return -1;
    }
    
    for(int i=0; i<5; i++) gemm.run();
    cudaDeviceSynchronize();
    
    cudaEvent_t s,t;
    cudaEventCreate(&s); cudaEventCreate(&t);
    cudaEventRecord(s);
    for(int i=0; i<100; i++) gemm.run();
    cudaEventRecord(t);
    cudaEventSynchronize(t);
    
    float ms; cudaEventElapsedTime(&ms, s, t);
    float avg=ms/100, tflops=(2.0*M*N*K/1e12)/(avg/1000);
    std::cout << "CUTLASS+CuTe: " << avg << " ms | " << tflops << " TFLOPS\n\n";
    
    // cuBLAS
    cublasHandle_t h; cublasCreate(&h); cublasSetMathMode(h, CUBLAS_TENSOR_OP_MATH);
    half *dA, *dB; float *dC;
    cudaMalloc(&dA, M*K*sizeof(half)); cudaMalloc(&dB, K*N*sizeof(half)); cudaMalloc(&dC, M*N*sizeof(float));
    cudaMemset(dA, 0, M*K*sizeof(half)); cudaMemset(dB, 0, K*N*sizeof(half));
    
    float alpha=1.0f, beta=0.0f;
    for(int i=0; i<5; i++)
        cublasGemmEx(h, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, dB, CUDA_R_16F, N, dA, CUDA_R_16F, K,
                    &beta, dC, CUDA_R_32F, N, CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    cudaDeviceSynchronize();
    
    cudaEventRecord(s);
    for(int i=0; i<100; i++)
        cublasGemmEx(h, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, dB, CUDA_R_16F, N, dA, CUDA_R_16F, K,
                    &beta, dC, CUDA_R_32F, N, CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    cudaEventRecord(t);
    cudaEventSynchronize(t);
    
    cudaEventElapsedTime(&ms, s, t);
    avg=ms/100; tflops=(2.0*M*N*K/1e12)/(avg/1000);
    std::cout << "cuBLAS:       " << avg << " ms | " << tflops << " TFLOPS\n\n";
    std::cout << "✅ Using CUTLASS 4.3.0 CollectiveBuilder + CuTe DSL\n";
    
    cudaFree(dA); cudaFree(dB); cudaFree(dC); cublasDestroy(h);
    return 0;
}
#else
int main() { std::cout << "SM90 not supported\n"; return 0; }
#endif

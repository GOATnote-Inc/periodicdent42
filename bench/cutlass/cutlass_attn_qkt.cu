// bench/cutlass/cutlass_attn_qkt.cu
// CUTLASS-based Q@K^T for attention (Tensor Core baseline)

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdio.h>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/reference/host/tensor_compare.h"

using ElementA = cutlass::half_t;
using ElementB = cutlass::half_t;
using ElementC = float;
using ElementAccumulator = float;

using LayoutA = cutlass::layout::RowMajor;  // Q: [M x K]
using LayoutB = cutlass::layout::ColumnMajor; // K^T: [K x N] stored as [N x K] col-major
using LayoutC = cutlass::layout::RowMajor;  // S: [M x N]

using Gemm = cutlass::gemm::device::Gemm<
    ElementA, LayoutA,
    ElementB, LayoutB,
    ElementC, LayoutC,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80  // Use Sm80 config, compile for sm_89
>;

cudaError_t cutlass_qkt(
    const half* Q,
    const half* K,
    float* S,
    int M,
    int N,
    int K_dim,
    float alpha,
    float beta
) {
    typename Gemm::Arguments args{
        {M, N, K_dim},
        {reinterpret_cast<const ElementA*>(Q), K_dim},
        {reinterpret_cast<const ElementB*>(K), K_dim},
        {S, N},
        {S, N},
        {alpha, beta}
    };
    
    Gemm gemm_op;
    cutlass::Status status = gemm_op(args);
    
    if (status != cutlass::Status::kSuccess) {
        return cudaErrorUnknown;
    }
    
    return cudaSuccess;
}

int main() {
    const int M = 32, N = 32, K_dim = 64;
    const float alpha = 1.0f / sqrtf(K_dim);
    const float beta = 0.0f;
    
    // Allocate
    cutlass::HostTensor<ElementA, LayoutA> Q({M, K_dim});
    cutlass::HostTensor<ElementB, LayoutB> K({N, K_dim});
    cutlass::HostTensor<ElementC, LayoutC> S({M, N});
    
    // Fill with test data
    cutlass::reference::host::TensorFillRandomUniform(Q.host_view(), 1, ElementA(1.0), ElementA(-1.0), 0);
    cutlass::reference::host::TensorFillRandomUniform(K.host_view(), 1, ElementB(1.0), ElementB(-1.0), 1);
    cutlass::reference::host::TensorFill(S.host_view(), ElementC(0));
    
    Q.sync_device();
    K.sync_device();
    S.sync_device();
    
    // Run CUTLASS
    cudaError_t err = cutlass_qkt(
        reinterpret_cast<const half*>(Q.device_data()),
        reinterpret_cast<const half*>(K.device_data()),
        S.device_data(),
        M, N, K_dim,
        alpha, beta
    );
    
    if (err != cudaSuccess) {
        printf("❌ CUTLASS GEMM failed\n");
        return 1;
    }
    
    S.sync_host();
    
    // Verify non-zero
    bool has_nonzero = false;
    for (int i = 0; i < M * N; i++) {
        if (S.host_data()[i] != 0.0f) {
            has_nonzero = true;
            break;
        }
    }
    
    printf("%s CUTLASS Q@K^T: M=%d, N=%d, K=%d\n", 
           has_nonzero ? "✅" : "❌", M, N, K_dim);
    printf("Sample S[0,0]=%.6f\n", S.host_data()[0]);
    
    // Benchmark
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    const int iters = 100;
    cudaEventRecord(start);
    for (int i = 0; i < iters; i++) {
        cutlass_qkt(
            reinterpret_cast<const half*>(Q.device_data()),
            reinterpret_cast<const half*>(K.device_data()),
            S.device_data(),
            M, N, K_dim,
            alpha, beta
        );
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    printf("CUTLASS Q@K^T: %.2f μs/iter\n", ms * 1000.0f / iters);
    
    return 0;
}


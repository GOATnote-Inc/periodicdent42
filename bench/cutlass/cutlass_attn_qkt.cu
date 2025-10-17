// bench/cutlass/cutlass_attn_qkt.cu
// CUTLASS-based Q@K^T for attention (Tensor Core baseline)

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdio.h>
#include <stdlib.h>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"

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
        cutlass::gemm::GemmCoord{M, N, K_dim},
        {reinterpret_cast<const ElementA*>(Q), K_dim},
        {reinterpret_cast<const ElementB*>(K), K_dim},
        {S, N},
        {S, N},
        {alpha, beta}
    };
    
    Gemm gemm_op;
    
    // Check workspace size
    size_t workspace_size = gemm_op.get_workspace_size(args);
    void* workspace = nullptr;
    if (workspace_size > 0) {
        cudaMalloc(&workspace, workspace_size);
    }
    
    cutlass::Status status = gemm_op.initialize(args, workspace);
    if (status != cutlass::Status::kSuccess) {
        if (workspace) cudaFree(workspace);
        return cudaErrorUnknown;
    }
    
    status = gemm_op();
    
    if (workspace) cudaFree(workspace);
    
    if (status != cutlass::Status::kSuccess) {
        return cudaErrorUnknown;
    }
    
    return cudaSuccess;
}

int main() {
    const int M = 32, N = 32, K_dim = 64;
    const float alpha = 1.0f / sqrtf(K_dim);
    const float beta = 0.0f;
    
    // Allocate host
    half *Q_h = (half*)malloc(M * K_dim * sizeof(half));
    half *K_h = (half*)malloc(N * K_dim * sizeof(half));
    float *S_h = (float*)malloc(M * N * sizeof(float));
    
    // Initialize
    for (int i = 0; i < M * K_dim; i++) Q_h[i] = __float2half(0.1f);
    for (int i = 0; i < N * K_dim; i++) K_h[i] = __float2half(0.1f);
    for (int i = 0; i < M * N; i++) S_h[i] = 0.0f;
    
    // Allocate device
    half *Q_d, *K_d;
    float *S_d;
    cudaMalloc(&Q_d, M * K_dim * sizeof(half));
    cudaMalloc(&K_d, N * K_dim * sizeof(half));
    cudaMalloc(&S_d, M * N * sizeof(float));
    
    cudaMemcpy(Q_d, Q_h, M * K_dim * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(K_d, K_h, N * K_dim * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(S_d, S_h, M * N * sizeof(float), cudaMemcpyHostToDevice);
    
    // Run CUTLASS
    cudaError_t err = cutlass_qkt(Q_d, K_d, S_d, M, N, K_dim, alpha, beta);
    
    if (err != cudaSuccess) {
        printf("❌ CUTLASS GEMM failed\n");
        return 1;
    }
    
    cudaMemcpy(S_h, S_d, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Verify non-zero
    bool has_nonzero = false;
    for (int i = 0; i < M * N; i++) {
        if (S_h[i] != 0.0f) {
            has_nonzero = true;
            break;
        }
    }
    
    printf("%s CUTLASS Q@K^T: M=%d, N=%d, K=%d\n", 
           has_nonzero ? "✅" : "❌", M, N, K_dim);
    printf("Sample S[0,0]=%.6f\n", S_h[0]);
    
    // Benchmark
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    const int iters = 100;
    cudaEventRecord(start);
    for (int i = 0; i < iters; i++) {
        cutlass_qkt(Q_d, K_d, S_d, M, N, K_dim, alpha, beta);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    printf("CUTLASS Q@K^T: %.2f μs/iter\n", ms * 1000.0f / iters);
    
    free(Q_h);
    free(K_h);
    free(S_h);
    cudaFree(Q_d);
    cudaFree(K_d);
    cudaFree(S_d);
    
    return 0;
}


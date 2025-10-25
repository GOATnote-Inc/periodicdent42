// Minimal CUTLASS TensorOp using defaults
#include <cuda_runtime.h>
#include <stdio.h>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"

// Let CUTLASS pick sensible defaults for Sm80
using ElementA = cutlass::half_t;
using ElementB = cutlass::half_t;
using ElementC = float;
using ElementAccumulator = float;

using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutC = cutlass::layout::RowMajor;

// Minimal template args - let CUTLASS fill in the rest
using Gemm = cutlass::gemm::device::Gemm<
    ElementA, LayoutA,
    ElementB, LayoutB,
    ElementC, LayoutC,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80
>;

int main() {
    const int M = 32, N = 32, K = 64;
    
    printf("CUTLASS TensorOp (defaults)\n");
    printf("M=%d, N=%d, K=%d\n", M, N, K);
    
    // Allocate
    ElementA *h_A = (ElementA*)malloc(M * K * sizeof(ElementA));
    ElementB *h_B = (ElementB*)malloc(K * N * sizeof(ElementB));
    ElementC *h_C = (ElementC*)malloc(M * N * sizeof(ElementC));
    
    for (int i = 0; i < M * K; i++) h_A[i] = ElementA(0.1f);
    for (int i = 0; i < K * N; i++) h_B[i] = ElementB(0.1f);
    for (int i = 0; i < M * N; i++) h_C[i] = 0.0f;
    
    ElementA *d_A;
    ElementB *d_B;
    ElementC *d_C;
    cudaMalloc(&d_A, M * K * sizeof(ElementA));
    cudaMalloc(&d_B, K * N * sizeof(ElementB));
    cudaMalloc(&d_C, M * N * sizeof(ElementC));
    
    cudaMemcpy(d_A, h_A, M * K * sizeof(ElementA), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, K * N * sizeof(ElementB), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C, M * N * sizeof(ElementC), cudaMemcpyHostToDevice);
    
    // Setup
    typename Gemm::Arguments args(
        {M, N, K},
        {d_A, K},
        {d_B, K},
        {d_C, N},
        {d_C, N},
        {1.0f / sqrtf(K), 0.0f}
    );
    
    Gemm gemm_op;
    
    cutlass::Status status = gemm_op.can_implement(args);
    if (status != cutlass::Status::kSuccess) {
        fprintf(stderr, "❌ can_implement: %s\n", cutlassGetStatusString(status));
        return 1;
    }
    printf("✅ can_implement\n");
    
    size_t workspace_size = gemm_op.get_workspace_size(args);
    void* workspace = nullptr;
    if (workspace_size > 0) {
        cudaMalloc(&workspace, workspace_size);
    }
    printf("Workspace: %zu bytes\n", workspace_size);
    
    status = gemm_op.initialize(args, workspace);
    if (status != cutlass::Status::kSuccess) {
        fprintf(stderr, "❌ initialize: %s\n", cutlassGetStatusString(status));
        if (workspace) cudaFree(workspace);
        return 1;
    }
    printf("✅ initialize\n");
    
    printf("Launching...\n");
    status = gemm_op();
    
    cudaError_t cuda_err = cudaGetLastError();
    cudaDeviceSynchronize();
    cudaError_t sync_err = cudaGetLastError();
    
    if (status != cutlass::Status::kSuccess) {
        fprintf(stderr, "❌ launch: %s\n", cutlassGetStatusString(status));
        fprintf(stderr, "   CUDA: %s / %s\n", 
                cudaGetErrorString(cuda_err), cudaGetErrorString(sync_err));
        if (workspace) cudaFree(workspace);
        return 1;
    }
    
    if (sync_err != cudaSuccess) {
        fprintf(stderr, "❌ sync: %s\n", cudaGetErrorString(sync_err));
        if (workspace) cudaFree(workspace);
        return 1;
    }
    
    printf("✅ launch\n");
    
    // Verify
    cudaMemcpy(h_C, d_C, M * N * sizeof(ElementC), cudaMemcpyDeviceToHost);
    
    float expected = 0.1f * 0.1f * K / sqrtf(K);
    printf("C[0,0]=%.6f (expect ~%.6f)\n", h_C[0], expected);
    
    bool ok = (h_C[0] > 0.0f && h_C[0] < 2.0f * expected);
    printf("%s\n", ok ? "✅ PASS" : "❌ FAIL");
    
    // Benchmark if OK
    if (ok) {
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        
        cudaEventRecord(start);
        for (int i = 0; i < 100; i++) {
            gemm_op();
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float ms = 0;
        cudaEventElapsedTime(&ms, start, stop);
        printf("Perf: %.2f μs/iter\n", ms * 10.0f);
        
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
    
    // Cleanup
    free(h_A); free(h_B); free(h_C);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    if (workspace) cudaFree(workspace);
    
    return ok ? 0 : 1;
}


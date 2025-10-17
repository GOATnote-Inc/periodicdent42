// Simplest possible CUTLASS GEMM for sm_89 debugging
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdio.h>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"

// Use simplest config: no TensorOp, just basic threading
using Gemm = cutlass::gemm::device::Gemm<
    half,                           // ElementA
    cutlass::layout::RowMajor,     // LayoutA
    half,                           // ElementB
    cutlass::layout::ColumnMajor,   // LayoutB (for K^T)
    float,                          // ElementC
    cutlass::layout::RowMajor,     // LayoutC
    float                           // ElementAccumulator
    // Default threading model (no TensorOp)
>;

int main() {
    const int M = 16, N = 16, K = 16;  // Tiny test
    
    // Host allocations
    half *h_A = (half*)malloc(M * K * sizeof(half));
    half *h_B = (half*)malloc(K * N * sizeof(half));
    float *h_C = (float*)malloc(M * N * sizeof(float));
    
    // Initialize
    for (int i = 0; i < M * K; i++) h_A[i] = __float2half(0.1f);
    for (int i = 0; i < K * N; i++) h_B[i] = __float2half(0.1f);
    for (int i = 0; i < M * N; i++) h_C[i] = 0.0f;
    
    // Device allocations
    half *d_A, *d_B;
    float *d_C;
    cudaMalloc(&d_A, M * K * sizeof(half));
    cudaMalloc(&d_B, K * N * sizeof(half));
    cudaMalloc(&d_C, M * N * sizeof(float));
    
    cudaMemcpy(d_A, h_A, M * K * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, K * N * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C, M * N * sizeof(float), cudaMemcpyHostToDevice);
    
    // CUTLASS arguments
    typename Gemm::Arguments args{
        {M, N, K},
        {d_A, K},
        {d_B, K},
        {d_C, N},
        {d_C, N},
        {1.0f, 0.0f}
    };
    
    Gemm gemm_op;
    
    // Check support
    cutlass::Status status = gemm_op.can_implement(args);
    if (status != cutlass::Status::kSuccess) {
        fprintf(stderr, "❌ can_implement failed: %s\n", cutlassGetStatusString(status));
        return 1;
    }
    printf("✅ can_implement: Success\n");
    
    // Get workspace
    size_t workspace_size = gemm_op.get_workspace_size(args);
    printf("Workspace size: %zu bytes\n", workspace_size);
    
    void* workspace = nullptr;
    if (workspace_size > 0) {
        cudaMalloc(&workspace, workspace_size);
    }
    
    // Initialize
    status = gemm_op.initialize(args, workspace);
    if (status != cutlass::Status::kSuccess) {
        fprintf(stderr, "❌ initialize failed: %s\n", cutlassGetStatusString(status));
        if (workspace) cudaFree(workspace);
        return 1;
    }
    printf("✅ initialize: Success\n");
    
    // Launch
    printf("Launching kernel...\n");
    status = gemm_op();
    
    cudaError_t cuda_err = cudaGetLastError();
    if (cuda_err != cudaSuccess) {
        fprintf(stderr, "❌ CUDA error after launch: %s\n", cudaGetErrorString(cuda_err));
    }
    
    cudaDeviceSynchronize();
    cuda_err = cudaGetLastError();
    
    if (status != cutlass::Status::kSuccess) {
        fprintf(stderr, "❌ launch failed: %s\n", cutlassGetStatusString(status));
        fprintf(stderr, "   CUDA error: %s\n", cudaGetErrorString(cuda_err));
        if (workspace) cudaFree(workspace);
        return 1;
    }
    
    if (cuda_err != cudaSuccess) {
        fprintf(stderr, "❌ CUDA error after sync: %s\n", cudaGetErrorString(cuda_err));
        if (workspace) cudaFree(workspace);
        return 1;
    }
    
    printf("✅ launch: Success\n");
    
    // Copy back and verify
    cudaMemcpy(h_C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    
    bool has_nonzero = false;
    for (int i = 0; i < M * N; i++) {
        if (h_C[i] != 0.0f) {
            has_nonzero = true;
            break;
        }
    }
    
    printf("Result: %s (C[0,0]=%.6f)\n", 
           has_nonzero ? "✅ NON-ZERO" : "❌ ALL-ZERO", h_C[0]);
    
    // Cleanup
    free(h_A); free(h_B); free(h_C);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    if (workspace) cudaFree(workspace);
    
    return has_nonzero ? 0 : 1;
}


// CUTLASS TensorOp GEMM for sm_89 (Ada) - properly configured
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdio.h>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"

// Use Sm89 directly instead of Sm80
using Gemm = cutlass::gemm::device::Gemm<
    cutlass::half_t,                      // ElementA
    cutlass::layout::RowMajor,            // LayoutA
    cutlass::half_t,                      // ElementB  
    cutlass::layout::ColumnMajor,         // LayoutB (for K^T)
    float,                                // ElementC
    cutlass::layout::RowMajor,            // LayoutC
    float,                                // ElementAccumulator
    cutlass::arch::OpClassTensorOp,       // TensorOp
    cutlass::arch::Sm89,                  // Use Sm89 directly
    cutlass::gemm::GemmShape<32, 32, 16>, // ThreadblockShape (M, N, K)
    cutlass::gemm::GemmShape<16, 16, 16>, // WarpShape
    cutlass::gemm::GemmShape<16, 8, 16>,  // InstructionShape (Ada MMA)
    cutlass::epilogue::thread::LinearCombination<
        float,                            // ElementOutput
        128 / cutlass::sizeof_bits<float>::value, // ElementsPerAccess
        float,                            // ElementAccumulator
        float                             // ElementCompute
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    2                                     // Stages
>;

int main() {
    const int M = 32, N = 32, K = 64;  // Q@K^T dimensions
    
    printf("Testing CUTLASS TensorOp on sm_89\n");
    printf("Problem: M=%d, N=%d, K=%d\n", M, N, K);
    
    // Host allocations
    cutlass::half_t *h_A = (cutlass::half_t*)malloc(M * K * sizeof(cutlass::half_t));
    cutlass::half_t *h_B = (cutlass::half_t*)malloc(K * N * sizeof(cutlass::half_t));
    float *h_C = (float*)malloc(M * N * sizeof(float));
    
    // Initialize with small values
    for (int i = 0; i < M * K; i++) h_A[i] = cutlass::half_t(0.1f);
    for (int i = 0; i < K * N; i++) h_B[i] = cutlass::half_t(0.1f);
    for (int i = 0; i < M * N; i++) h_C[i] = 0.0f;
    
    // Device allocations
    cutlass::half_t *d_A, *d_B;
    float *d_C;
    cudaMalloc(&d_A, M * K * sizeof(cutlass::half_t));
    cudaMalloc(&d_B, K * N * sizeof(cutlass::half_t));
    cudaMalloc(&d_C, M * N * sizeof(float));
    
    cudaMemcpy(d_A, h_A, M * K * sizeof(cutlass::half_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, K * N * sizeof(cutlass::half_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C, M * N * sizeof(float), cudaMemcpyHostToDevice);
    
    // CUTLASS arguments
    float alpha = 1.0f / sqrtf(K);  // Softmax scale for attention
    float beta = 0.0f;
    
    typename Gemm::Arguments args{
        cutlass::gemm::GemmCoord{M, N, K},
        {d_A, K},  // A: M×K, lda=K (row-major)
        {d_B, K},  // B: K×N (stored col-major), ldb=K
        {d_C, N},  // C: M×N, ldc=N
        {d_C, N},
        {alpha, beta}
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
        if (cudaMalloc(&workspace, workspace_size) != cudaSuccess) {
            fprintf(stderr, "❌ cudaMalloc workspace failed\n");
            return 1;
        }
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
    printf("Launching TensorOp kernel...\n");
    status = gemm_op();
    
    cudaError_t cuda_err = cudaGetLastError();
    if (cuda_err != cudaSuccess) {
        fprintf(stderr, "❌ CUDA error after launch: %s\n", cudaGetErrorString(cuda_err));
    }
    
    cudaDeviceSynchronize();
    cuda_err = cudaGetLastError();
    
    if (status != cutlass::Status::kSuccess) {
        fprintf(stderr, "❌ launch failed: %s\n", cutlassGetStatusString(status));
        fprintf(stderr, "   CUDA: %s\n", cudaGetErrorString(cuda_err));
        if (workspace) cudaFree(workspace);
        return 1;
    }
    
    if (cuda_err != cudaSuccess) {
        fprintf(stderr, "❌ CUDA error: %s\n", cudaGetErrorString(cuda_err));
        if (workspace) cudaFree(workspace);
        return 1;
    }
    
    printf("✅ launch: Success\n");
    
    // Copy back and verify
    cudaMemcpy(h_C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    
    float expected = 0.1f * 0.1f * K * alpha;  // Expected value
    bool correct = (h_C[0] > 0.0f) && (h_C[0] < 2.0f * expected);
    
    printf("Result: C[0,0]=%.6f (expected ~%.6f)\n", h_C[0], expected);
    printf("%s\n", correct ? "✅ PASS" : "❌ FAIL");
    
    // Benchmark
    if (correct) {
        const int iters = 100;
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        
        cudaEventRecord(start);
        for (int i = 0; i < iters; i++) {
            gemm_op();
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float ms = 0;
        cudaEventElapsedTime(&ms, start, stop);
        
        printf("Performance: %.2f μs/iteration\n", ms * 1000.0f / iters);
        
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
    
    // Cleanup
    free(h_A); free(h_B); free(h_C);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    if (workspace) cudaFree(workspace);
    
    return correct ? 0 : 1;
}


// cuBLAS Tensor Core baseline for Q@K^T
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <stdio.h>

int main() {
    const int M = 32, N = 32, K = 64;
    
    printf("cuBLAS TensorCore Baseline\n");
    printf("M=%d, N=%d, K=%d\n", M, N, K);
    
    // Allocate
    half *h_A = (half*)malloc(M * K * sizeof(half));
    half *h_B = (half*)malloc(K * N * sizeof(half));
    float *h_C = (float*)malloc(M * N * sizeof(float));
    
    for (int i = 0; i < M * K; i++) h_A[i] = __float2half(0.1f);
    for (int i = 0; i < K * N; i++) h_B[i] = __float2half(0.1f);
    for (int i = 0; i < M * N; i++) h_C[i] = 0.0f;
    
    half *d_A, *d_B;
    float *d_C;
    cudaMalloc(&d_A, M * K * sizeof(half));
    cudaMalloc(&d_B, K * N * sizeof(half));
    cudaMalloc(&d_C, M * N * sizeof(float));
    
    cudaMemcpy(d_A, h_A, M * K * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, K * N * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C, M * N * sizeof(float), cudaMemcpyHostToDevice);
    
    // cuBLAS setup
    cublasHandle_t handle;
    cublasCreate(&handle);
    
    // Enable TensorCore (default on Ada)
    cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH);
    
    // C = alpha * A @ B^T + beta * C
    // A: M×K row-major → K×M col-major (transposed)
    // B: K×N row-major → N×K col-major (transposed)
    // Result: M×N row-major → N×M col-major
    
    float alpha = 1.0f / sqrtf(K);
    float beta = 0.0f;
    
    // GemmEx: FP16 input, FP32 output, FP32 compute
    cublasStatus_t status = cublasGemmEx(
        handle,
        CUBLAS_OP_T,  // B transposed
        CUBLAS_OP_T,  // A transposed
        N, M, K,      // Swapped for col-major
        &alpha,
        d_B, CUDA_R_16F, K,  // B
        d_A, CUDA_R_16F, K,  // A
        &beta,
        d_C, CUDA_R_32F, N,  // C
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP
    );
    
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "❌ cublasGemmEx failed: %d\n", status);
        return 1;
    }
    
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "❌ CUDA error: %s\n", cudaGetErrorString(err));
        return 1;
    }
    
    printf("✅ cuBLAS launch success\n");
    
    // Verify
    cudaMemcpy(h_C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    
    float expected = 0.1f * 0.1f * K * alpha;
    printf("C[0,0]=%.6f (expect ~%.6f)\n", h_C[0], expected);
    
    bool ok = (h_C[0] > 0.0f && h_C[0] < 2.0f * expected);
    printf("%s\n", ok ? "✅ CORRECT" : "❌ WRONG");
    
    // Benchmark
    if (ok) {
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        
        const int warmup = 10, iters = 100;
        for (int i = 0; i < warmup; i++) {
            cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_T, N, M, K,
                         &alpha, d_B, CUDA_R_16F, K, d_A, CUDA_R_16F, K,
                         &beta, d_C, CUDA_R_32F, N,
                         CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        }
        cudaDeviceSynchronize();
        
        cudaEventRecord(start);
        for (int i = 0; i < iters; i++) {
            cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_T, N, M, K,
                         &alpha, d_B, CUDA_R_16F, K, d_A, CUDA_R_16F, K,
                         &beta, d_C, CUDA_R_32F, N,
                         CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float ms = 0;
        cudaEventElapsedTime(&ms, start, stop);
        printf("\n✅ cuBLAS TC Baseline: %.2f μs/iter\n", ms * 1000.0f / iters);
        printf("   (This is reference speed for Q@K^T with Tensor Cores)\n");
        
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
    
    // Cleanup
    cublasDestroy(handle);
    free(h_A); free(h_B); free(h_C);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    
    return ok ? 0 : 1;
}


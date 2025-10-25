/*
 * Phase B.1 Test 1: Minimal cuBLAS GEMM
 * 
 * Goal: Verify cuBLAS setup and Tensor Core operation
 * Test: 4×4×4 FP16 GEMM with Tensor Core math mode
 * Expected: All outputs = 4.0 (sum of 4 products of 1.0)
 */

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define CHECK_CUDA(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

#define CHECK_CUBLAS(call) do { \
    cublasStatus_t status = call; \
    if (status != CUBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "cuBLAS error at %s:%d: status=%d\n", __FILE__, __LINE__, status); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

int main() {
    printf("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
    printf("Phase B.1 Test 1: Minimal cuBLAS GEMM (4×4×4)\n");
    printf("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n");
    
    // Test parameters
    const int M = 4;
    const int N = 4;
    const int K = 4;
    
    printf("Test Configuration:\n");
    printf("  Matrix A: %d×%d (FP16)\n", M, K);
    printf("  Matrix B: %d×%d (FP16)\n", K, N);
    printf("  Matrix C: %d×%d (FP32)\n", M, N);
    printf("  Operation: C = A @ B\n");
    printf("  Compute: FP32 with FP16 Tensor Cores\n\n");
    
    // Initialize cuBLAS
    printf("Step 1: Initialize cuBLAS...\n");
    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));
    
    // Enable Tensor Core math mode
    CHECK_CUBLAS(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH));
    printf("  ✅ cuBLAS initialized with TENSOR_OP_MATH\n\n");
    
    // Allocate host memory
    printf("Step 2: Allocate and initialize host memory...\n");
    half *h_A = (half*)malloc(M * K * sizeof(half));
    half *h_B = (half*)malloc(K * N * sizeof(half));
    float *h_C = (float*)malloc(M * N * sizeof(float));
    
    // Initialize A and B to 1.0 (FP16)
    for (int i = 0; i < M * K; i++) {
        h_A[i] = __float2half(1.0f);
    }
    for (int i = 0; i < K * N; i++) {
        h_B[i] = __float2half(1.0f);
    }
    printf("  ✅ A[%d×%d] = 1.0 (all elements)\n", M, K);
    printf("  ✅ B[%d×%d] = 1.0 (all elements)\n\n", K, N);
    
    // Allocate device memory
    printf("Step 3: Allocate device memory...\n");
    half *d_A, *d_B;
    float *d_C;
    CHECK_CUDA(cudaMalloc(&d_A, M * K * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&d_B, K * N * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&d_C, M * N * sizeof(float)));
    printf("  ✅ Device memory allocated\n\n");
    
    // Copy to device
    printf("Step 4: Copy data to device...\n");
    CHECK_CUDA(cudaMemcpy(d_A, h_A, M * K * sizeof(half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, K * N * sizeof(half), cudaMemcpyHostToDevice));
    printf("  ✅ Data transferred\n\n");
    
    // Launch cuBLAS GEMM
    printf("Step 5: Launch cuBLAS GEMM...\n");
    float alpha = 1.0f;
    float beta = 0.0f;
    
    // C = alpha * A @ B + beta * C
    // cuBLAS uses column-major, but we'll use row-major interpretation
    CHECK_CUBLAS(cublasGemmEx(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        N, M, K,                           // cuBLAS dimensions (column-major)
        &alpha,
        d_B, CUDA_R_16F, N,                // B: leading dimension
        d_A, CUDA_R_16F, K,                // A: leading dimension
        &beta,
        d_C, CUDA_R_32F, N,                // C: leading dimension
        CUBLAS_COMPUTE_32F_FAST_16F,       // FP32 compute with FP16 Tensor Cores
        CUBLAS_GEMM_DEFAULT_TENSOR_OP      // Enable Tensor Core usage
    ));
    CHECK_CUDA(cudaDeviceSynchronize());
    printf("  ✅ GEMM completed\n\n");
    
    // Copy result back
    printf("Step 6: Copy result back to host...\n");
    CHECK_CUDA(cudaMemcpy(h_C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));
    printf("  ✅ Result transferred\n\n");
    
    // Verify result
    printf("Step 7: Verify correctness...\n");
    float expected = (float)K;  // Each element should be sum of K products (1.0 * 1.0)
    float max_diff = 0.0f;
    int num_correct = 0;
    int num_total = M * N;
    
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            int idx = i * N + j;
            float val = h_C[idx];
            float diff = fabs(val - expected);
            max_diff = fmax(max_diff, diff);
            
            if (diff < 1e-3) {
                num_correct++;
            }
            
            if (i == 0 && j < 4) {  // Print first row
                printf("  C[%d,%d] = %.6f (expected: %.6f, diff: %.6f)\n", 
                       i, j, val, expected, diff);
            }
        }
    }
    
    printf("\n");
    printf("Correctness Results:\n");
    printf("  Expected value: %.1f\n", expected);
    printf("  Max difference: %.6f\n", max_diff);
    printf("  Correct elements: %d/%d\n", num_correct, num_total);
    printf("  Tolerance: 1e-3\n\n");
    
    // Clean up
    printf("Step 8: Clean up...\n");
    free(h_A);
    free(h_B);
    free(h_C);
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));
    CHECK_CUBLAS(cublasDestroy(handle));
    printf("  ✅ Resources freed\n\n");
    
    // Final verdict
    printf("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
    if (num_correct == num_total && max_diff < 1e-3) {
        printf("✅ TEST PASSED: cuBLAS minimal GEMM works correctly\n");
        printf("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
        return 0;
    } else {
        printf("❌ TEST FAILED: Correctness issues detected\n");
        printf("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
        return 1;
    }
}


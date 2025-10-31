// flashcore/fast/attention_cublas_tc.cu
// cuBLAS (non-Lt) approach - proven Tensor Core support
// Simpler API, better H100 compatibility

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <cmath>
#include <iostream>

// Test if cuBLAS provides better Tensor Core algorithms
extern "C" void test_cublas_vs_cublaslt(int M, int N, int K) {
    // Allocate test matrices
    __half *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, M * K * sizeof(__half));
    cudaMalloc(&d_B, K * N * sizeof(__half));
    cudaMalloc(&d_C, M * N * sizeof(__half));
    
    // Fill with random data
    // (skipped for brevity)
    
    // cuBLAS approach
    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);  // Force Tensor Cores!
    
    __half alpha = __float2half(1.0f);
    __half beta = __float2half(0.0f);
    
    // Warmup
    for (int i = 0; i < 10; i++) {
        cublasHgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T,
                    M, N, K,
                    &alpha,
                    d_A, M,
                    d_B, N,
                    &beta,
                    d_C, M);
    }
    
    // Benchmark
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    for (int i = 0; i < 100; i++) {
        cublasHgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T,
                    M, N, K,
                    &alpha,
                    d_A, M,
                    d_B, N,
                    &beta,
                    d_C, M);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    ms /= 100.0f;  // Average
    
    // Compute TFLOPS
    double flops = 2.0 * M * N * K;
    double tflops = (flops / (ms * 1e-3)) / 1e12;
    
    std::cout << "cuBLAS Hgemm: " << M << "×" << K << " @ " << K << "×" << N
              << " = " << ms << " ms (" << tflops << " TFLOPS)\n";
    
    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cublasDestroy(handle);
}


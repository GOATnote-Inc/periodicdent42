#include "roofline_analysis.h"
#include <cuda_runtime.h>

// Dummy GEMM kernel for demonstration
__global__ void dummy_gemm(const half* A, const half* B, half* C,
                           int M, int N, int K) {
    // Simplified for demonstration
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += __half2float(A[row * K + k]) * __half2float(B[k * N + col]);
        }
        C[row * N + col] = __float2half(sum);
    }
}

int main() {
    const int M = 4096, N = 4096, K = 4096;
    
    // Allocate
    half *A, *B, *C;
    cudaMalloc(&A, M * K * sizeof(half));
    cudaMalloc(&B, K * N * sizeof(half));
    cudaMalloc(&C, M * N * sizeof(half));
    
    // Warm up
    dim3 block(16, 16);
    dim3 grid((N + 15) / 16, (M + 15) / 16);
    for (int i = 0; i < 10; i++) {
        dummy_gemm<<<grid, block>>>(A, B, C, M, N, K);
    }
    cudaDeviceSynchronize();
    
    // Timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    
    for (int i = 0; i < 100; i++) {
        dummy_gemm<<<grid, block>>>(A, B, C, M, N, K);
    }
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    ms /= 100.0f;
    
    // Roofline analysis
    RooflineMetrics metrics;
    metrics.compute(
        gemm_flops(M, N, K),
        gemm_bytes_fp16(M, N, K),
        ms
    );
    metrics.print("Dummy GEMM");
    
    cudaFree(A); cudaFree(B); cudaFree(C);
    return 0;
}

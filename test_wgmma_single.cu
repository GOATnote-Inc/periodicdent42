// ============================================================================
// Test Program for Single WGMMA Operation
// ============================================================================
// Target: Validate 64×64×16 WGMMA and measure 2-3 TFLOPS
// ============================================================================

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <iostream>
#include <cmath>
#include <chrono>

// Include the WGMMA kernel
#include "flashcore/fast/attention_phase6_wgmma_native.cu"

#define CHECK_CUDA(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) \
                  << " at " << __FILE__ << ":" << __LINE__ << "\n"; \
        exit(1); \
    } \
} while(0)

// Reference CPU implementation for validation
void reference_matmul_fp32(
    const __half* A, const __half* B, float* C,
    int M, int N, int K
) {
    // C = A @ B^T
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                float a_val = __half2float(A[m * K + k]);
                float b_val = __half2float(B[n * K + k]);  // B is stored row-major
                sum += a_val * b_val;
            }
            C[m * N + n] = sum;
        }
    }
}

int main() {
    std::cout << "==================================================\n";
    std::cout << "  WGMMA Single Operation Test (Phase 6A Step 1)  \n";
    std::cout << "==================================================\n\n";
    
    // Test configuration
    constexpr int M = 64;
    constexpr int K = 16;
    constexpr int N = 64;
    constexpr int WARMUP = 10;
    constexpr int ITERS = 100;
    
    std::cout << "Configuration:\n";
    std::cout << "  Operation: C[64,64] = A[64,16] @ B[64,16]^T\n";
    std::cout << "  WGMMA: m64n64k16.f32.f16.f16\n";
    std::cout << "  Warmup: " << WARMUP << " iterations\n";
    std::cout << "  Benchmark: " << ITERS << " iterations\n\n";
    
    // Allocate host memory
    __half* h_A = new __half[M * K];
    __half* h_B = new __half[N * K];
    float* h_C = new float[M * N];
    float* h_C_ref = new float[M * N];
    
    // Initialize with random values
    std::cout << "Initializing matrices...\n";
    for (int i = 0; i < M * K; i++) {
        h_A[i] = __float2half((float)rand() / RAND_MAX - 0.5f);
    }
    for (int i = 0; i < N * K; i++) {
        h_B[i] = __float2half((float)rand() / RAND_MAX - 0.5f);
    }
    
    // Allocate device memory
    __half *d_A, *d_B;
    float *d_C;
    CHECK_CUDA(cudaMalloc(&d_A, M * K * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&d_B, N * K * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&d_C, M * N * sizeof(float)));
    
    // Copy to device
    CHECK_CUDA(cudaMemcpy(d_A, h_A, M * K * sizeof(__half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, N * K * sizeof(__half), cudaMemcpyHostToDevice));
    
    // Launch configuration
    dim3 grid(1, 1, 1);   // Single block
    dim3 block(256, 1, 1); // 256 threads (2 warp groups)
    
    std::cout << "Launching WGMMA kernel...\n";
    std::cout << "  Grid: " << grid.x << " blocks\n";
    std::cout << "  Block: " << block.x << " threads (2 warp groups)\n\n";
    
    // Warmup
    for (int i = 0; i < WARMUP; i++) {
        test_wgmma_single<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaGetLastError());
    
    // Benchmark
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < ITERS; i++) {
        test_wgmma_single<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();
    
    CHECK_CUDA(cudaGetLastError());
    
    // Copy result back
    CHECK_CUDA(cudaMemcpy(h_C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));
    
    // Calculate performance
    double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
    double avg_time_ms = elapsed_ms / ITERS;
    double avg_time_s = avg_time_ms / 1000.0;
    
    // FLOPs: C[M,N] = A[M,K] @ B[N,K]^T requires M*N*K*2 operations
    double flops = (double)M * N * K * 2.0;
    double tflops = (flops / avg_time_s) / 1e12;
    
    std::cout << "==================================================\n";
    std::cout << "  PERFORMANCE RESULTS\n";
    std::cout << "==================================================\n";
    std::cout << "  Average Time: " << avg_time_ms << " ms\n";
    std::cout << "  Throughput:   " << tflops << " TFLOPS\n";
    std::cout << "  Target:       2-3 TFLOPS\n";
    
    if (tflops >= 2.0 && tflops <= 5.0) {
        std::cout << "  Status:       ✅ PASS (within expected range)\n";
    } else if (tflops > 5.0) {
        std::cout << "  Status:       ✅✅ EXCELLENT (exceeds expectations)\n";
    } else {
        std::cout << "  Status:       ⚠️  BELOW TARGET (needs investigation)\n";
    }
    std::cout << "==================================================\n\n";
    
    // Validate correctness
    std::cout << "Validating correctness...\n";
    reference_matmul_fp32(h_A, h_B, h_C_ref, M, N, K);
    
    float max_diff = 0.0f;
    float avg_diff = 0.0f;
    int num_errors = 0;
    
    for (int i = 0; i < M * N; i++) {
        float diff = std::abs(h_C[i] - h_C_ref[i]);
        avg_diff += diff;
        max_diff = std::max(max_diff, diff);
        if (diff > 1e-2f) {  // Tolerance for FP16 accumulation
            num_errors++;
            if (num_errors < 10) {  // Print first few errors
                int row = i / N;
                int col = i % N;
                std::cout << "  [" << row << "," << col << "] "
                          << "GPU: " << h_C[i] << " vs CPU: " << h_C_ref[i]
                          << " (diff: " << diff << ")\n";
            }
        }
    }
    avg_diff /= (M * N);
    
    std::cout << "\n==================================================\n";
    std::cout << "  CORRECTNESS RESULTS\n";
    std::cout << "==================================================\n";
    std::cout << "  Max Error:  " << max_diff << "\n";
    std::cout << "  Avg Error:  " << avg_diff << "\n";
    std::cout << "  Num Errors: " << num_errors << " / " << (M*N) << "\n";
    
    if (max_diff < 1e-2f) {
        std::cout << "  Status:     ✅ CORRECT\n";
    } else {
        std::cout << "  Status:     ❌ ERRORS DETECTED\n";
    }
    std::cout << "==================================================\n\n";
    
    // Cleanup
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    delete[] h_C_ref;
    
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));
    
    std::cout << "Test complete!\n";
    
    // Return success if performance and correctness are good
    bool perf_pass = (tflops >= 2.0);
    bool correctness_pass = (max_diff < 1e-2f);
    
    if (perf_pass && correctness_pass) {
        std::cout << "\n🎉 SUCCESS: WGMMA single operation validated!\n";
        return 0;
    } else {
        std::cout << "\n❌ FAILURE: Issues detected, see above\n";
        return 1;
    }
}


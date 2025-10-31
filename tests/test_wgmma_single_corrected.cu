// ============================================================================
// Corrected Test Program for Single WGMMA Operation
// ============================================================================
// Target: Validate 64×64×16 WGMMA with expected 2.8-3.5 TFLOPS
// All critical fixes applied from expert review
// ============================================================================

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <iostream>
#include <cmath>
#include <chrono>
#include <algorithm>
#include <vector>

// Include the corrected WGMMA kernel
#include "flashcore/fast/attention_phase6_wgmma_corrected.cu"

#define CHECK_CUDA(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) \
                  << " at " << __FILE__ << ":" << __LINE__ << "\n"; \
        exit(1); \
    } \
} while(0)

// Reference CPU implementation for validation (A @ B^T)
void reference_matmul_fp32(
    const __half* A, const __half* B, float* C,
    int M, int N, int K
) {
    // C = A @ B^T (B is transposed)
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                float a_val = __half2float(A[m * K + k]);
                float b_val = __half2float(B[n * K + k]);  // B stored row-major
                sum += a_val * b_val;
            }
            C[m * N + n] = sum;
        }
    }
}

int main() {
    std::cout << "==================================================\n";
    std::cout << "  CORRECTED WGMMA Test (All Fixes Applied)\n";
    std::cout << "==================================================\n\n";
    
    // Configuration
    constexpr int M = 64;
    constexpr int K = 16;
    constexpr int N = 64;
    constexpr int WARMUP = 20;   // More warmup for accurate timing
    constexpr int ITERS = 200;   // More iterations for stable results
    
    std::cout << "Configuration:\n";
    std::cout << "  Operation: C[64,64] = A[64,16] @ B[64,16]^T\n";
    std::cout << "  WGMMA: m64n64k16.f32.f16.f16\n";
    std::cout << "  Fixes Applied:\n";
    std::cout << "    ✅ Correct thread-to-output mapping\n";
    std::cout << "    ✅ Padding: 32 elements (bank conflict-free)\n";
    std::cout << "    ✅ Swizzle mode: 3 (128B)\n";
    std::cout << "    ✅ B matrix transposed\n";
    std::cout << "    ✅ Fence ordering corrected\n";
    std::cout << "  Warmup: " << WARMUP << " iterations\n";
    std::cout << "  Benchmark: " << ITERS << " iterations\n\n";
    
    // Allocate host memory
    __half* h_A = new __half[M * K];
    __half* h_B = new __half[N * K];
    float* h_C = new float[M * N];
    float* h_C_ref = new float[M * N];
    
    // Initialize with controlled random values for better validation
    std::cout << "Initializing matrices...\n";
    srand(42);  // Fixed seed for reproducibility
    for (int i = 0; i < M * K; i++) {
        h_A[i] = __float2half((float)rand() / RAND_MAX * 2.0f - 1.0f);
    }
    for (int i = 0; i < N * K; i++) {
        h_B[i] = __float2half((float)rand() / RAND_MAX * 2.0f - 1.0f);
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
    
    std::cout << "Launching corrected WGMMA kernel...\n\n";
    
    // Warmup
    for (int i = 0; i < WARMUP; i++) {
        launch_test_wgmma_single_corrected(d_A, d_B, d_C, M, N, K, 0);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaGetLastError());
    
    // Benchmark with precise timing
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    
    std::vector<float> times;
    times.reserve(ITERS);
    
    for (int i = 0; i < ITERS; i++) {
        CHECK_CUDA(cudaEventRecord(start));
        launch_test_wgmma_single_corrected(d_A, d_B, d_C, M, N, K, 0);
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));
        
        float ms;
        CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
        times.push_back(ms);
    }
    
    CHECK_CUDA(cudaGetLastError());
    
    // Calculate statistics (use median for robustness)
    std::sort(times.begin(), times.end());
    float median_time_ms = times[ITERS / 2];
    float min_time_ms = times[0];
    float max_time_ms = times[ITERS - 1];
    
    // Calculate average (excluding outliers)
    float sum = 0.0f;
    int count = 0;
    for (float t : times) {
        if (t < median_time_ms * 1.5f) {  // Exclude large outliers
            sum += t;
            count++;
        }
    }
    float avg_time_ms = sum / count;
    
    // Copy result back
    CHECK_CUDA(cudaMemcpy(h_C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));
    
    // Calculate performance metrics
    double flops = (double)M * N * K * 2.0;  // M*N*K*2 for A @ B^T
    double median_tflops = (flops / (median_time_ms / 1000.0)) / 1e12;
    double avg_tflops = (flops / (avg_time_ms / 1000.0)) / 1e12;
    double max_tflops = (flops / (min_time_ms / 1000.0)) / 1e12;
    
    std::cout << "==================================================\n";
    std::cout << "  PERFORMANCE RESULTS\n";
    std::cout << "==================================================\n";
    std::cout << "  Median Time:  " << median_time_ms << " ms\n";
    std::cout << "  Average Time: " << avg_time_ms << " ms\n";
    std::cout << "  Min Time:     " << min_time_ms << " ms\n";
    std::cout << "  Max Time:     " << max_time_ms << " ms\n";
    std::cout << "  ---\n";
    std::cout << "  Median:       " << median_tflops << " TFLOPS\n";
    std::cout << "  Average:      " << avg_tflops << " TFLOPS\n";
    std::cout << "  Peak:         " << max_tflops << " TFLOPS\n";
    std::cout << "  ---\n";
    std::cout << "  Target:       2.8-3.5 TFLOPS\n";
    
    bool perf_pass = (median_tflops >= 2.8 && median_tflops <= 5.0);
    bool perf_excellent = (median_tflops >= 3.0);
    
    if (perf_excellent) {
        std::cout << "  Status:       ✅✅ EXCELLENT (exceeds 3.0 TFLOPS)\n";
    } else if (perf_pass) {
        std::cout << "  Status:       ✅ PASS (within expected range)\n";
    } else if (median_tflops > 5.0) {
        std::cout << "  Status:       ✅✅✅ EXCEPTIONAL (>5 TFLOPS!)\n";
    } else {
        std::cout << "  Status:       ⚠️  BELOW TARGET (" << median_tflops << " TFLOPS)\n";
    }
    std::cout << "==================================================\n\n";
    
    // Validate correctness
    std::cout << "Validating correctness against CPU reference...\n";
    reference_matmul_fp32(h_A, h_B, h_C_ref, M, N, K);
    
    float max_diff = 0.0f;
    float avg_diff = 0.0f;
    int num_errors = 0;
    float threshold = 1e-2f;  // FP16 tolerance
    
    int error_details_shown = 0;
    constexpr int MAX_ERROR_DETAILS = 5;
    
    for (int i = 0; i < M * N; i++) {
        float diff = std::abs(h_C[i] - h_C_ref[i]);
        avg_diff += diff;
        max_diff = std::max(max_diff, diff);
        
        if (diff > threshold) {
            num_errors++;
            if (error_details_shown < MAX_ERROR_DETAILS) {
                int row = i / N;
                int col = i % N;
                std::cout << "  Error [" << row << "," << col << "]: "
                          << "GPU=" << h_C[i] << " vs CPU=" << h_C_ref[i]
                          << " (diff=" << diff << ")\n";
                error_details_shown++;
            }
        }
    }
    avg_diff /= (M * N);
    
    std::cout << "\n==================================================\n";
    std::cout << "  CORRECTNESS RESULTS\n";
    std::cout << "==================================================\n";
    std::cout << "  Max Error:    " << max_diff << "\n";
    std::cout << "  Avg Error:    " << avg_diff << "\n";
    std::cout << "  Num Errors:   " << num_errors << " / " << (M*N) 
              << " (threshold=" << threshold << ")\n";
    
    bool correctness_pass = (max_diff < threshold);
    bool correctness_excellent = (max_diff < 5e-3f);  // Tighter tolerance
    
    if (correctness_excellent) {
        std::cout << "  Status:       ✅✅ EXCELLENT (max error < 0.005)\n";
    } else if (correctness_pass) {
        std::cout << "  Status:       ✅ PASS (max error < 0.01)\n";
    } else {
        std::cout << "  Status:       ❌ FAIL (errors detected)\n";
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
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    
    // Final status
    if (perf_pass && correctness_pass) {
        std::cout << "🎉 SUCCESS: All tests passed!\n\n";
        std::cout << "Next Steps:\n";
        std::cout << "  1. ✅ Step 1 validated (" << median_tflops << " TFLOPS)\n";
        std::cout << "  2. 🚀 Proceed to Step 2: Multiple WGMMAs (target: 10-15 TFLOPS)\n";
        std::cout << "  3. 📊 Profile with: ncu --set full ./build/bin/test_wgmma_corrected\n";
        return 0;
    } else {
        std::cout << "❌ FAILURE: Issues detected\n\n";
        if (!correctness_pass) {
            std::cout << "  Correctness FAILED - investigate thread mapping\n";
        }
        if (!perf_pass) {
            std::cout << "  Performance BELOW TARGET - check for:\n";
            std::cout << "    - Register spills (use --ptxas-options=-v)\n";
            std::cout << "    - Bank conflicts (profile with Nsight Compute)\n";
            std::cout << "    - SM utilization (ncu --metrics sm__warps_active)\n";
        }
        return 1;
    }
}


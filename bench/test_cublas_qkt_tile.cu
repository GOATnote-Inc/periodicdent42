/*
 * Phase B.1 Test 2: FlashAttention Q@K^T Tile (32×64×64)
 * 
 * Goal: Test actual FlashAttention tile size for Q@K^T operation
 * Operation: S = Q @ K^T (with softmax scale)
 * Q: [32, 64] (BLOCK_M × HEAD_DIM)
 * K: [64, 64] (BLOCK_N × HEAD_DIM)
 * S: [32, 64] (BLOCK_M × BLOCK_N)
 */

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

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

// FlashAttention tile dimensions
const int BLOCK_M = 32;   // Q tile height
const int HEAD_DIM = 64;  // Dimension (both Q and K)
const int BLOCK_N = 64;   // K tile height

// Scalar reference implementation for validation
void reference_qkt(const half* Q, const half* K, float* S, 
                   int M, int N, int D, float scale) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float acc = 0.0f;
            for (int k = 0; k < D; k++) {
                float q_val = __half2float(Q[i * D + k]);
                float k_val = __half2float(K[j * D + k]);  // K^T: swap indices
                acc += q_val * k_val;
            }
            S[i * N + j] = acc * scale;
        }
    }
}

int main() {
    printf("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
    printf("Phase B.1 Test 2: FlashAttention Q@K^T Tile\n");
    printf("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n");
    
    printf("Test Configuration:\n");
    printf("  Q: [%d, %d] (BLOCK_M × HEAD_DIM)\n", BLOCK_M, HEAD_DIM);
    printf("  K: [%d, %d] (BLOCK_N × HEAD_DIM)\n", BLOCK_N, HEAD_DIM);
    printf("  S = Q @ K^T: [%d, %d] (BLOCK_M × BLOCK_N)\n", BLOCK_M, BLOCK_N);
    printf("  Scale: 1/sqrt(64) = 0.125\n");
    printf("  Compute: FP32 with FP16 Tensor Cores\n\n");
    
    float scale = 1.0f / sqrtf((float)HEAD_DIM);  // 0.125
    
    // Initialize cuBLAS
    printf("Step 1: Initialize cuBLAS...\n");
    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));
    CHECK_CUBLAS(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH));
    printf("  ✅ cuBLAS initialized with TENSOR_OP_MATH\n\n");
    
    // Allocate host memory
    printf("Step 2: Allocate and initialize host memory...\n");
    half *h_Q = (half*)malloc(BLOCK_M * HEAD_DIM * sizeof(half));
    half *h_K = (half*)malloc(BLOCK_N * HEAD_DIM * sizeof(half));
    float *h_S_cublas = (float*)malloc(BLOCK_M * BLOCK_N * sizeof(float));
    float *h_S_ref = (float*)malloc(BLOCK_M * BLOCK_N * sizeof(float));
    
    // Initialize Q and K with random values
    srand(42);  // Fixed seed for reproducibility
    for (int i = 0; i < BLOCK_M * HEAD_DIM; i++) {
        h_Q[i] = __float2half((float)rand() / RAND_MAX - 0.5f);  // Range: [-0.5, 0.5]
    }
    for (int i = 0; i < BLOCK_N * HEAD_DIM; i++) {
        h_K[i] = __float2half((float)rand() / RAND_MAX - 0.5f);
    }
    printf("  ✅ Q[%d×%d] initialized (random, seed=42)\n", BLOCK_M, HEAD_DIM);
    printf("  ✅ K[%d×%d] initialized (random, seed=42)\n\n", BLOCK_N, HEAD_DIM);
    
    // Allocate device memory
    printf("Step 3: Allocate device memory...\n");
    half *d_Q, *d_K;
    float *d_S;
    CHECK_CUDA(cudaMalloc(&d_Q, BLOCK_M * HEAD_DIM * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&d_K, BLOCK_N * HEAD_DIM * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&d_S, BLOCK_M * BLOCK_N * sizeof(float)));
    printf("  ✅ Device memory allocated\n\n");
    
    // Copy to device
    printf("Step 4: Copy data to device...\n");
    CHECK_CUDA(cudaMemcpy(d_Q, h_Q, BLOCK_M * HEAD_DIM * sizeof(half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_K, h_K, BLOCK_N * HEAD_DIM * sizeof(half), cudaMemcpyHostToDevice));
    printf("  ✅ Data transferred\n\n");
    
    // Launch cuBLAS GEMM for Q @ K^T
    printf("Step 5: Launch cuBLAS Q@K^T (with scale)...\n");
    float alpha = scale;  // 0.125
    float beta = 0.0f;
    
    // S = alpha * Q @ K^T + beta * S
    // Q: [BLOCK_M, HEAD_DIM] row-major
    // K^T: [HEAD_DIM, BLOCK_N] (K is [BLOCK_N, HEAD_DIM] row-major, transposed)
    // S: [BLOCK_M, BLOCK_N] row-major
    
    CHECK_CUBLAS(cublasGemmEx(
        handle,
        CUBLAS_OP_T,                      // Transpose K (for K^T)
        CUBLAS_OP_N,                       // No transpose Q
        BLOCK_N, BLOCK_M, HEAD_DIM,        // N, M, K dimensions
        &alpha,
        d_K, CUDA_R_16F, HEAD_DIM,         // K: [BLOCK_N, HEAD_DIM] row-major
        d_Q, CUDA_R_16F, HEAD_DIM,         // Q: [BLOCK_M, HEAD_DIM] row-major
        &beta,
        d_S, CUDA_R_32F, BLOCK_N,          // S: [BLOCK_M, BLOCK_N] row-major
        CUBLAS_COMPUTE_32F_FAST_16F,       // FP32 compute with FP16 Tensor Cores
        CUBLAS_GEMM_DEFAULT_TENSOR_OP      // Enable Tensor Core usage
    ));
    CHECK_CUDA(cudaDeviceSynchronize());
    printf("  ✅ cuBLAS Q@K^T completed\n\n");
    
    // Copy result back
    printf("Step 6: Copy result back to host...\n");
    CHECK_CUDA(cudaMemcpy(h_S_cublas, d_S, BLOCK_M * BLOCK_N * sizeof(float), cudaMemcpyDeviceToHost));
    printf("  ✅ Result transferred\n\n");
    
    // Compute reference on CPU
    printf("Step 7: Compute scalar reference (CPU)...\n");
    reference_qkt(h_Q, h_K, h_S_ref, BLOCK_M, BLOCK_N, HEAD_DIM, scale);
    printf("  ✅ Reference computed\n\n");
    
    // Verify correctness
    printf("Step 8: Verify correctness...\n");
    float max_diff = 0.0f;
    float sum_diff = 0.0f;
    int num_correct = 0;
    int num_total = BLOCK_M * BLOCK_N;
    
    for (int i = 0; i < BLOCK_M; i++) {
        for (int j = 0; j < BLOCK_N; j++) {
            int idx = i * BLOCK_N + j;
            float cublas_val = h_S_cublas[idx];
            float ref_val = h_S_ref[idx];
            float diff = fabs(cublas_val - ref_val);
            max_diff = fmax(max_diff, diff);
            sum_diff += diff;
            
            if (diff < 1e-3) {
                num_correct++;
            }
            
            // Print first few elements
            if (i == 0 && j < 4) {
                printf("  S[%d,%d] = %.6f (ref: %.6f, diff: %.6f)\n", 
                       i, j, cublas_val, ref_val, diff);
            }
        }
    }
    
    float avg_diff = sum_diff / num_total;
    
    printf("\n");
    printf("Correctness Results:\n");
    printf("  Max difference: %.6f\n", max_diff);
    printf("  Avg difference: %.6f\n", avg_diff);
    printf("  Correct elements: %d/%d (%.1f%%)\n", num_correct, num_total,
           100.0f * num_correct / num_total);
    printf("  Tolerance: 1e-3\n\n");
    
    // Benchmark performance
    printf("Step 9: Benchmark performance...\n");
    const int warmup = 10;
    const int iters = 1000;
    
    // Warmup
    for (int i = 0; i < warmup; i++) {
        CHECK_CUBLAS(cublasGemmEx(
            handle, CUBLAS_OP_T, CUBLAS_OP_N,
            BLOCK_N, BLOCK_M, HEAD_DIM,
            &alpha, d_K, CUDA_R_16F, HEAD_DIM,
            d_Q, CUDA_R_16F, HEAD_DIM,
            &beta, d_S, CUDA_R_32F, BLOCK_N,
            CUBLAS_COMPUTE_32F_FAST_16F,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP
        ));
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    
    // Benchmark
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    
    for (int i = 0; i < iters; i++) {
        CHECK_CUBLAS(cublasGemmEx(
            handle, CUBLAS_OP_T, CUBLAS_OP_N,
            BLOCK_N, BLOCK_M, HEAD_DIM,
            &alpha, d_K, CUDA_R_16F, HEAD_DIM,
            d_Q, CUDA_R_16F, HEAD_DIM,
            &beta, d_S, CUDA_R_32F, BLOCK_N,
            CUBLAS_COMPUTE_32F_FAST_16F,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP
        ));
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    
    clock_gettime(CLOCK_MONOTONIC, &end);
    double elapsed = (end.tv_sec - start.tv_sec) + 
                     (end.tv_nsec - start.tv_nsec) * 1e-9;
    double latency_us = (elapsed / iters) * 1e6;
    
    printf("  Latency: %.2f μs/tile\n", latency_us);
    printf("  Throughput: %.2f GFLOPS\n", 
           (2.0 * BLOCK_M * BLOCK_N * HEAD_DIM) / (latency_us * 1e3));
    printf("  Expected: 5-10 μs (vs ~30 μs scalar)\n");
    printf("  Speedup vs scalar: %.1f×\n\n", 30.0f / latency_us);
    
    // Clean up
    printf("Step 10: Clean up...\n");
    free(h_Q);
    free(h_K);
    free(h_S_cublas);
    free(h_S_ref);
    CHECK_CUDA(cudaFree(d_Q));
    CHECK_CUDA(cudaFree(d_K));
    CHECK_CUDA(cudaFree(d_S));
    CHECK_CUBLAS(cublasDestroy(handle));
    printf("  ✅ Resources freed\n\n");
    
    // Final verdict
    printf("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
    bool correct = (num_correct == num_total) && (max_diff < 1e-3);
    bool fast = (latency_us < 15.0f);  // Should be < 10 μs, allow 15 μs margin
    
    if (correct && fast) {
        printf("✅ TEST PASSED: cuBLAS Q@K^T tile works correctly and fast\n");
        printf("   Correctness: %d/%d elements (max_diff=%.6f)\n", num_correct, num_total, max_diff);
        printf("   Performance: %.2f μs/tile (%.1f× vs scalar)\n", latency_us, 30.0f/latency_us);
        printf("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
        return 0;
    } else {
        if (!correct) {
            printf("❌ TEST FAILED: Correctness issues\n");
            printf("   Correct: %d/%d, max_diff=%.6f\n", num_correct, num_total, max_diff);
        }
        if (!fast) {
            printf("❌ TEST FAILED: Performance issues\n");
            printf("   Latency: %.2f μs (expected < 15 μs)\n", latency_us);
        }
        printf("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
        return 1;
    }
}


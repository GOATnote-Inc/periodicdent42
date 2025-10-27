// Test harness for Hopper attention kernel

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <iostream>
#include <cmath>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <numeric>

// Forward declarations
extern "C" void launch_attention_hopper_minimal(
    const void* Q, const void* K, const void* V, void* O,
    int B, int H, int S, int D,
    float scale, bool is_causal, cudaStream_t stream
);

extern "C" void launch_attention_phase2_async(
    const void* Q, const void* K, const void* V, void* O,
    int B, int H, int S, int D,
    float scale, bool is_causal, cudaStream_t stream
);

extern "C" void launch_attention_phase2_aggressive(
    const void* Q, const void* K, const void* V, void* O,
    int B, int H, int S, int D,
    float scale, bool is_causal, cudaStream_t stream
);

extern "C" void launch_attention_phase3_wgmma(
    const void* Q, const void* K, const void* V, void* O,
    int B, int H, int S, int D,
    float scale, bool is_causal, cudaStream_t stream
);

extern "C" void launch_attention_cublaslt(
    const void* Q, const void* K, const void* V, void* O,
    int B, int H, int S, int D,
    float scale, bool is_causal, cudaStream_t stream
);

extern "C" void launch_attention_cublaslt_sparse(
    const void* Q, const void* K, const void* V, void* O,
    int B, int H, int S, int D,
    float scale, bool is_causal,
    const void* pager, cudaStream_t stream
);

extern "C" void launch_attention_cublaslt_splitk(
    const void* Q, const void* K, const void* V, void* O,
    int B, int H, int S, int D,
    float scale, bool is_causal, cudaStream_t stream
);

// Select which kernel to test
#ifndef KERNEL_PHASE
#define KERNEL_PHASE 5  // Default to Phase 5 Split-K (EXPERT)
#endif

#if KERNEL_PHASE == 1
#define launch_attention launch_attention_hopper_minimal
#define KERNEL_NAME "Phase 1 (Minimal Baseline)"
#elif KERNEL_PHASE == 2
#define launch_attention launch_attention_phase2_aggressive
#define KERNEL_NAME "Phase 2 (Async + Coalesced + Tiling)"
#elif KERNEL_PHASE == 3
#define launch_attention launch_attention_phase3_wgmma
#define KERNEL_NAME "Phase 3A (WGMMA Tensor Cores)"
#elif KERNEL_PHASE == 4
#define launch_attention launch_attention_cublaslt
#define KERNEL_NAME "Phase 3B (cuBLASLt Sparse GEMM - 320 TFLOPS Target)"
#elif KERNEL_PHASE == 5
#define launch_attention launch_attention_cublaslt_splitk
#define KERNEL_NAME "Phase 3C (EXPERT: Split-K + FP32 Stability)"
#endif

// Helper: Fill with random data
void fill_random(__half* data, int size) {
    for (int i = 0; i < size; ++i) {
        data[i] = __float2half((float)rand() / RAND_MAX - 0.5f);
    }
}

// Helper: Compute TFLOPS
float compute_tflops(int B, int H, int S, int D, float ms) {
    // FLOPs for attention: 4 * B * H * S * S * D
    // (2 matmuls: Q@K^T and P@V, each counts as 2*S*S*D)
    float flops = 4.0f * B * H * S * S * D;
    float tflops = flops / (ms / 1000.0f) / 1e12f;
    return tflops;
}

int main(int argc, char** argv) {
    // Configuration
    int B = 16;
    int H = 16;
    int S = 4096;  // ARCHITECT DEBUG: Test if M=4096 unlocks Tensor Cores
    int D = 64;
    
    std::cout << "========================================\n";
    std::cout << "FLASHCORE HOPPER KERNEL TEST\n";
    std::cout << "Kernel: " << KERNEL_NAME << "\n";
    std::cout << "========================================\n\n";
    
    // Print GPU info
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "GPU: " << prop.name << "\n";
    std::cout << "Compute Capability: " << prop.major << "." << prop.minor << "\n";
    std::cout << "SMs: " << prop.multiProcessorCount << "\n\n";
    
    // Configure device limits for WMMA kernel (standing on NVIDIA's shoulders!)
    // Increase per-thread stack size for WMMA fragments
    size_t stackSize = 16 * 1024;  // 16KB per thread (generous headroom)
    cudaDeviceSetLimit(cudaLimitStackSize, stackSize);
    std::cout << "Set stack size: " << (stackSize / 1024) << "KB per thread\n";
    std::cout << "(Shared memory carveout configured in kernel launch)\n\n";
    
    if (prop.major < 9) {
        std::cerr << "ERROR: This kernel requires Hopper (sm_90+)\n";
        std::cerr << "       Your GPU is sm_" << prop.major << prop.minor << "\n";
        return 1;
    }
    
    std::cout << "Test config:\n";
    std::cout << "  B (batch): " << B << "\n";
    std::cout << "  H (heads): " << H << "\n";
    std::cout << "  S (seq):   " << S << "\n";
    std::cout << "  D (dim):   " << D << "\n\n";
    
    // Allocate host memory
    int qkv_size = B * H * S * D;
    __half* h_Q = new __half[qkv_size];
    __half* h_K = new __half[qkv_size];
    __half* h_V = new __half[qkv_size];
    __half* h_O = new __half[qkv_size];
    
    // Fill with random data
    srand(42);
    fill_random(h_Q, qkv_size);
    fill_random(h_K, qkv_size);
    fill_random(h_V, qkv_size);
    
    // Allocate device memory
    __half *d_Q, *d_K, *d_V, *d_O;
    size_t bytes = qkv_size * sizeof(__half);
    cudaMalloc(&d_Q, bytes);
    cudaMalloc(&d_K, bytes);
    cudaMalloc(&d_V, bytes);
    cudaMalloc(&d_O, bytes);
    
    // Copy to device
    cudaMemcpy(d_Q, h_Q, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, h_K, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, h_V, bytes, cudaMemcpyHostToDevice);
    
    // Compute scale
    float scale = 1.0f / std::sqrt(static_cast<float>(D));
    
    // Warmup with detailed error checking
    std::cout << "[1/3] Warmup (10 iterations)...\n";
    
    // Check device limits
    std::cout << "Device limits:\n";
    std::cout << "  Max shared memory per block: " << prop.sharedMemPerBlock << " bytes\n";
    std::cout << "  Max shared memory per SM:    " << prop.sharedMemPerMultiprocessor << " bytes\n";
    std::cout << "  Max registers per block:     " << prop.regsPerBlock << "\n";
    std::cout << "  Max threads per block:       " << prop.maxThreadsPerBlock << "\n\n";
    
    cudaError_t err;
        for (int i = 0; i < 10; ++i) {
            launch_attention(d_Q, d_K, d_V, d_O, B, H, S, D, scale, true, 0);
            err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "❌ Kernel launch failed at iteration " << i << ": " << cudaGetErrorString(err) << "\n";
            std::cerr << "Grid config: (" << (B * H) << ", " << ((S + 64 - 1) / 64) << ")\n";
            std::cerr << "Block config: 256 threads\n";
            std::cerr << "Expected smem: ~66KB\n";
            return 1;
        }
        
        cudaDeviceSynchronize();
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "❌ Kernel execution failed at iteration " << i << ": " << cudaGetErrorString(err) << "\n";
            std::cerr << "This usually means:\n";
            std::cerr << "  1. Invalid memory access in kernel\n";
            std::cerr << "  2. WMMA operation failure\n";
            std::cerr << "  3. Shared memory bank conflict or overflow\n";
            return 1;
        }
        
        if (i == 0) {
            std::cout << "✅ First iteration successful\n";
        }
    }
    std::cout << "✅ Warmup complete (no errors)\n\n";
    
    // Benchmark with proper timing
    std::cout << "[2/3] Benchmarking (100 iterations)...\n";
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    const int iters = 100;
    std::vector<float> times;
    times.reserve(iters);
    
    for (int i = 0; i < iters; ++i) {
        cudaDeviceSynchronize();  // Ensure previous work is done
        
            cudaEventRecord(start, 0);
            launch_attention(d_Q, d_K, d_V, d_O, B, H, S, D, scale, true, 0);
            cudaEventRecord(stop, 0);
        
        cudaEventSynchronize(stop);
        
        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        times.push_back(ms);
        
        // Check for errors periodically
        if (i % 10 == 0) {
            err = cudaGetLastError();
            if (err != cudaSuccess) {
                std::cerr << "❌ Error at iteration " << i << ": " << cudaGetErrorString(err) << "\n";
                return 1;
            }
        }
    }
    
    // Compute statistics
    std::sort(times.begin(), times.end());
    float min_ms = times[0];
    float median_ms = times[iters / 2];
    float max_ms = times[iters - 1];
    float p95_ms = times[int(iters * 0.95)];
    float avg_ms = std::accumulate(times.begin(), times.end(), 0.0f) / iters;
    
    float median_tflops = compute_tflops(B, H, S, D, median_ms);
    float min_tflops = compute_tflops(B, H, S, D, min_ms);
    
    std::cout << "\nTiming Statistics:\n";
    std::cout << "  Min:    " << min_ms << " ms (" << min_tflops << " TFLOPS)\n";
    std::cout << "  Median: " << median_ms << " ms (" << median_tflops << " TFLOPS)\n";
    std::cout << "  P95:    " << p95_ms << " ms\n";
    std::cout << "  Max:    " << max_ms << " ms\n";
    std::cout << "  Mean:   " << avg_ms << " ms\n\n";
    
    float tflops = median_tflops;
    
    // Copy result back
    cudaMemcpy(h_O, d_O, bytes, cudaMemcpyDeviceToHost);
    
    // Print results
    std::cout << "\n========================================\n";
    std::cout << "RESULTS - ITERATION 1 (WMMA)\n";
    std::cout << "========================================\n";
    std::cout << "Performance:\n";
    std::cout << "  Median: " << median_ms << " ms → " << median_tflops << " TFLOPS\n";
    std::cout << "  Best:   " << min_ms << " ms → " << min_tflops << " TFLOPS\n\n";
    
    std::cout << "Targets:\n";
    std::cout << "  Baseline (scalar):    1.6 TFLOPS ✅\n";
    std::cout << "  Iteration 1 goal:     100-200 TFLOPS\n";
    std::cout << "  Phase 1 (Foundation): 150 TFLOPS\n";
    std::cout << "  FA3 baseline:         450 TFLOPS\n";
    std::cout << "  Final goal:           500+ TFLOPS\n\n";
    
    // Improvement from baseline
    float improvement = median_tflops / 1.6;
    std::cout << "Improvement: " << improvement << "× over baseline\n\n";
    
    if (median_tflops >= 450) {
        float speedup = (median_tflops / 450.0 - 1.0) * 100.0;
        std::cout << "✅ EXCELLENT: Beat FA3 by " << speedup << "%!\n";
    } else if (median_tflops >= 150) {
        std::cout << "✅ GOOD: Phase 1 target achieved (" << (median_tflops/450.0)*100 << "% of FA3)\n";
    } else if (median_tflops >= 50) {
        std::cout << "✅ PROGRESS: Major improvement (" << (median_tflops/450.0)*100 << "% of FA3)\n";
    } else if (median_tflops > 1.6) {
        std::cout << "⚡ WORKING: " << improvement << "× improvement, continuing optimization\n";
    } else {
        std::cout << "⚠️  ISSUE: Performance below baseline - debugging needed\n";
    }
    std::cout << "========================================\n\n";
    
    // Simple correctness check
    std::cout << "[3/3] Basic correctness check...\n";
    bool has_output = false;
    bool has_nan = false;
    bool has_inf = false;
    float max_val = 0.0f;
    
    for (size_t i = 0; i < B * H * S * D; ++i) {
        float val = __half2float(h_O[i]);
        if (std::abs(val) > 1e-6) has_output = true;
        if (std::isnan(val)) has_nan = true;
        if (std::isinf(val)) has_inf = true;
        max_val = std::max(max_val, std::abs(val));
    }
    
    std::cout << "  Output range: [0, " << max_val << "]\n";
    std::cout << "  Has non-zero: " << (has_output ? "✅" : "❌") << "\n";
    std::cout << "  Has NaN:      " << (has_nan ? "❌" : "✅") << "\n";
    std::cout << "  Has Inf:      " << (has_inf ? "❌" : "✅") << "\n";
    
    if (has_output && !has_nan && !has_inf) {
        std::cout << "\n✅ Basic sanity checks PASS\n";
        std::cout << "Next: Compare vs PyTorch SDPA for full correctness\n";
    } else {
        std::cout << "\n❌ Basic sanity checks FAIL - debugging needed\n";
    }
    std::cout << "========================================\n";
    
    // Cleanup
    delete[] h_Q;
    delete[] h_K;
    delete[] h_V;
    delete[] h_O;
    cudaFree(d_Q);
    cudaFree(d_K);
    cudaFree(d_V);
    cudaFree(d_O);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return 0;
}


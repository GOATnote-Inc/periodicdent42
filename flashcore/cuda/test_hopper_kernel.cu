// Test harness for Hopper attention kernel

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <iostream>
#include <cmath>
#include <cstdlib>

// Forward declaration
extern "C" void launch_attention_hopper(
    const void* Q,
    const void* K,
    const void* V,
    void* O,
    int B, int H, int S, int D,
    float scale,
    bool is_causal,
    cudaStream_t stream
);

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
    int S = 2048;
    int D = 64;
    
    std::cout << "========================================\n";
    std::cout << "FLASHCORE HOPPER KERNEL TEST\n";
    std::cout << "========================================\n\n";
    
    // Print GPU info
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "GPU: " << prop.name << "\n";
    std::cout << "Compute Capability: " << prop.major << "." << prop.minor << "\n";
    std::cout << "SMs: " << prop.multiProcessorCount << "\n\n";
    
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
    
    // Warmup
    std::cout << "[1/2] Warmup (10 iterations)...\n";
    for (int i = 0; i < 10; ++i) {
        launch_attention_hopper(d_Q, d_K, d_V, d_O, B, H, S, D, scale, true, 0);
    }
    cudaDeviceSynchronize();
    
    // Benchmark
    std::cout << "[2/2] Benchmarking (100 iterations)...\n";
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    const int iters = 100;
    float total_ms = 0.0f;
    
    for (int i = 0; i < iters; ++i) {
        cudaEventRecord(start);
        launch_attention_hopper(d_Q, d_K, d_V, d_O, B, H, S, D, scale, true, 0);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        total_ms += ms;
    }
    
    float avg_ms = total_ms / iters;
    float tflops = compute_tflops(B, H, S, D, avg_ms);
    
    // Copy result back
    cudaMemcpy(h_O, d_O, bytes, cudaMemcpyDeviceToHost);
    
    // Print results
    std::cout << "\n========================================\n";
    std::cout << "RESULTS\n";
    std::cout << "========================================\n";
    std::cout << "Average latency: " << avg_ms << " ms\n";
    std::cout << "TFLOPS:          " << tflops << "\n\n";
    
    std::cout << "Targets:\n";
    std::cout << "  Phase 1 (Foundation): 140 TFLOPS\n";
    std::cout << "  Phase 2 (Optimized):  210 TFLOPS\n";
    std::cout << "  FA3 baseline:         190 TFLOPS\n\n";
    
    if (tflops >= 210) {
        std::cout << "✅ EXCELLENT: Beat FA3 by " << ((tflops / 190.0f - 1) * 100) << "%!\n";
    } else if (tflops >= 190) {
        std::cout << "✅ SUCCESS: Matched/beat FA3!\n";
    } else if (tflops >= 140) {
        std::cout << "⚠️  GOOD: Phase 1 target met, continue optimization\n";
    } else {
        std::cout << "❌ BELOW TARGET: Need more optimization\n";
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


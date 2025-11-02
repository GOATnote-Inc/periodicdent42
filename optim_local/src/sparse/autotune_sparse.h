// Sparse BSR Auto-Tuning Framework
#pragma once

#include <cuda_runtime.h>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>

struct BSR_Matrix {
    float *values;
    int *row_ptr;
    int *col_indices;
    int nnzb;  // Number of non-zero blocks
    int M, N;  // Matrix dimensions
    int block_size;
};

struct SparseConfig {
    int M, N, K;
    int block_size;
    float sparsity;
    
    std::string key() const {
        char buf[128];
        sprintf(buf, "%d_%d_%d_bs%d_sp%.2f", M, N, K, block_size, sparsity);
        return std::string(buf);
    }
};

struct SparseKernelVariant {
    std::string name;
    void (*kernel_fn)(const BSR_Matrix&, const float*, float*, int, int, int);
    int priority;
};

class SparseAutoTuner {
public:
    static float benchmark_variant(
        const SparseKernelVariant& variant,
        const BSR_Matrix& A,
        const float* B,
        float* C,
        const SparseConfig& config,
        int num_runs = 20
    ) {
        // Warmup
        for (int i = 0; i < 5; i++) {
            variant.kernel_fn(A, B, C, config.M, config.N, config.K);
        }
        cudaDeviceSynchronize();
        
        // Timing
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);
        
        for (int i = 0; i < num_runs; i++) {
            variant.kernel_fn(A, B, C, config.M, config.N, config.K);
        }
        
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        return ms / num_runs;
    }
    
    static std::string select_best_variant(
        const std::vector<SparseKernelVariant>& variants,
        const BSR_Matrix& A,
        const float* B,
        float* C,
        const SparseConfig& config
    ) {
        std::string cache_file = "/tmp/sparse_cache_" + config.key() + ".txt";
        
        // Check cache
        std::ifstream cache_in(cache_file);
        if (cache_in) {
            std::string cached_best;
            std::getline(cache_in, cached_best);
            cache_in.close();
            printf("📦 Loaded from cache: %s\n", cached_best.c_str());
            return cached_best;
        }
        
        // Benchmark all variants
        float best_time = 1e9f;
        std::string best_name;
        
        printf("Auto-tuning sparse BSR for config %s:\n", config.key().c_str());
        for (const auto& variant : variants) {
            float ms = benchmark_variant(variant, A, B, C, config);
            
            // Calculate TFLOPS (sparse: only count non-zero operations)
            double flops = 2.0 * A.nnzb * config.block_size * config.block_size * config.N * config.block_size;
            double tflops = (flops / (ms / 1000.0)) / 1e12;
            
            printf("  %-20s: %6.3f ms → %6.1f TFLOPS\n", variant.name.c_str(), ms, tflops);
            
            if (ms < best_time) {
                best_time = ms;
                best_name = variant.name;
            }
        }
        
        printf("  Best: %s (%.3f ms)\n\n", best_name.c_str(), best_time);
        
        // Cache result
        std::ofstream cache_out(cache_file);
        cache_out << best_name;
        cache_out.close();
        
        return best_name;
    }
};

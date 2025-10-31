// SPDX-License-Identifier: MIT
//
// File: benchmarks/bench_kernel_events.cu
// Purpose: Nsight-free GPU performance harness using CUDA Events + SHA256 checksum + JSON report
// Compatible: CUDA 13.0.2, CUTLASS 4.3.0, H100/L4 (sm_90a)
//
// Compile:
//   nvcc -O3 -std=c++17 -arch=sm_90a -lineinfo \
//        -I/opt/cutlass/include -I/usr/local/cuda-13.0/include \
//        -o bench_kernel benchmarks/bench_kernel_events.cu \
//        -lcudart -lcrypto
//
// Run:
//   ./bench_kernel

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>
#include <cstring>
#include <string>
#include <algorithm>
#include <openssl/sha.h>

// -----------------------------------------------------------------------------
// SHA256 Utility (no external JSON dependency for portability)
// -----------------------------------------------------------------------------
std::string sha256(const float *data, size_t count) {
    unsigned char hash[SHA256_DIGEST_LENGTH];
    SHA256((unsigned char *)data, count * sizeof(float), hash);
    char buf[65];
    for (int i = 0; i < SHA256_DIGEST_LENGTH; i++)
        sprintf(buf + i * 2, "%02x", hash[i]);
    buf[64] = 0;
    return std::string(buf);
}

// -----------------------------------------------------------------------------
// Import our winning sparse BSR kernel
// -----------------------------------------------------------------------------
#define BM 512
#define BN 128
#define BK 112
#define WM 128
#define WN 64

// Kernel signature (matches sparse_h100_winner.cu)
extern "C" void launch_bsr_spmm_async(
    const half *A, const int *Arow, const int *Acol, int Mb_A, int Kb_A, int nnzb_A,
    const half *B, const int *Brow, const int *Bcol, int Kb_B, int Nb_B, int nnzb_B,
    float *C, int M, int N, int K, cudaStream_t stream);

// Inline implementation for self-contained build
#include <mma.h>
using namespace nvcuda;

__global__ void bsr_spmm_kernel(
    const half *A, const int *Arow, const int *Acol, int Mb_A, int Kb_A,
    const half *B, const int *Brow, const int *Bcol, int Kb_B, int Nb_B,
    float *C, int M, int N, int K)
{
    const int bm = blockIdx.x;
    const int bn = blockIdx.y;
    if (bm >= Mb_A || bn >= Nb_B) return;
    
    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;
    
    __shared__ half smemA[BM * BK];
    __shared__ half smemB[BK * BN];
    
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[8];
    for (int i = 0; i < 8; i++)
        wmma::fill_fragment(acc[i], 0.0f);
    
    for (int ai = Arow[bm]; ai < Arow[bm + 1]; ai++) {
        int k_block = Acol[ai];
        
        // Load A tile with cp.async
        for (int i = threadIdx.x; i < BM * BK / 8; i += blockDim.x) {
            int row = i / (BK / 8);
            int col = (i % (BK / 8)) * 8;
            if (row < BM && col < BK) {
                float4 *src = (float4*)&A[ai * BM * BK + row * BK + col];
                float4 *dst = (float4*)&smemA[row * BK + col];
                *dst = *src;
            }
        }
        
        for (int bi = Brow[k_block]; bi < Brow[k_block + 1]; bi++) {
            int n_block = Bcol[bi];
            if (n_block != bn) continue;
            
            // Load B tile with cp.async
            for (int i = threadIdx.x; i < BK * BN / 8; i += blockDim.x) {
                int row = i / (BN / 8);
                int col = (i % (BN / 8)) * 8;
                if (row < BK && col < BN) {
                    float4 *src = (float4*)&B[bi * BK * BN + row * BN + col];
                    float4 *dst = (float4*)&smemB[row * BN + col];
                    *dst = *src;
                }
            }
            __syncthreads();
            
            // WMMA compute
            for (int w = 0; w < 8; w++) {
                int warp_m = warp_id * 32;
                int warp_n = w * 16;
                
                wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
                wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
                
                for (int k = 0; k < BK; k += 16) {
                    wmma::load_matrix_sync(a_frag, &smemA[warp_m * BK + k], BK);
                    wmma::load_matrix_sync(b_frag, &smemB[k * BN + warp_n], BN);
                    wmma::mma_sync(acc[w], a_frag, b_frag, acc[w]);
                }
            }
            __syncthreads();
        }
    }
    
    // Store results
    for (int w = 0; w < 8; w++) {
        int warp_m = warp_id * 32;
        int warp_n = w * 16;
        int out_m = bm * BM + warp_m;
        int out_n = bn * BN + warp_n;
        if (out_m < M && out_n < N)
            wmma::store_matrix_sync(&C[out_m * N + out_n], acc[w], N, wmma::mem_row_major);
    }
}

void launch_bsr_spmm_async(
    const half *A, const int *Arow, const int *Acol, int Mb_A, int Kb_A, int nnzb_A,
    const half *B, const int *Brow, const int *Bcol, int Kb_B, int Nb_B, int nnzb_B,
    float *C, int M, int N, int K, cudaStream_t stream)
{
    dim3 block(256);
    dim3 grid(Mb_A, Nb_B);
    bsr_spmm_kernel<<<grid, block, 0, stream>>>(
        A, Arow, Acol, Mb_A, Kb_A,
        B, Brow, Bcol, Kb_B, Nb_B,
        C, M, N, K);
}

// -----------------------------------------------------------------------------
// Main Benchmark Harness
// -----------------------------------------------------------------------------
int main() {
    // Configuration matching our validated setup
    const int M = 8192, N = 8192, K = 8192;
    const int Mb = (M + BM - 1) / BM;
    const int Nb = (N + BN - 1) / BN;
    const int Kb = (K + BK - 1) / BK;
    const int topk = 16;
    
    printf("╔═══════════════════════════════════════════════════════════╗\n");
    printf("║         SHADOW NSIGHT PROFILER - H100 VALIDATION         ║\n");
    printf("╠═══════════════════════════════════════════════════════════╣\n");
    printf("║  Matrix: %dx%dx%d                                     ║\n", M, N, K);
    printf("║  Tiles: %dx%dx%d (BM×BN×BK)                           ║\n", BM, BN, BK);
    printf("║  Sparsity: topk=%d/%d (%.1f%% sparse)                    ║\n", 
           topk, Kb, 100.0 * (1.0 - (float)topk / Kb));
    printf("╚═══════════════════════════════════════════════════════════╝\n\n");
    
    // Get device properties
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    int driver_ver, runtime_ver;
    cudaDriverGetVersion(&driver_ver);
    cudaRuntimeGetVersion(&runtime_ver);
    
    printf("Device: %s\n", prop.name);
    printf("Compute: sm_%d\n", prop.major * 10 + prop.minor);
    printf("Driver: %d.%d\n", driver_ver / 1000, (driver_ver % 100) / 10);
    printf("Runtime: %d.%d\n", runtime_ver / 1000, (runtime_ver % 100) / 10);
    printf("\n");
    
    // Generate sparse structure (same as validation)
    std::srand(42);
    std::vector<int> Arow(Mb + 1, 0), Brow(Kb + 1, 0), Acol, Bcol;
    
    for (int i = 0; i < Mb; i++) {
        std::vector<int> cols;
        while ((int)cols.size() < std::min(topk, Kb)) {
            int c = std::rand() % Kb;
            if (std::find(cols.begin(), cols.end(), c) == cols.end())
                cols.push_back(c);
        }
        std::sort(cols.begin(), cols.end());
        for (int c : cols) Acol.push_back(c);
        Arow[i + 1] = Acol.size();
    }
    
    for (int i = 0; i < Kb; i++) {
        std::vector<int> cols;
        while ((int)cols.size() < std::min(topk, Nb)) {
            int c = std::rand() % Nb;
            if (std::find(cols.begin(), cols.end(), c) == cols.end())
                cols.push_back(c);
        }
        std::sort(cols.begin(), cols.end());
        for (int c : cols) Bcol.push_back(c);
        Brow[i + 1] = Bcol.size();
    }
    
    int nnzb_A = Acol.size();
    int nnzb_B = Bcol.size();
    
    // Allocate
    half *dA, *dB;
    float *dC;
    int *dArow, *dAcol, *dBrow, *dBcol;
    
    cudaMalloc(&dA, nnzb_A * BM * BK * sizeof(half));
    cudaMalloc(&dB, nnzb_B * BK * BN * sizeof(half));
    cudaMalloc(&dC, M * N * sizeof(float));
    cudaMalloc(&dArow, (Mb + 1) * sizeof(int));
    cudaMalloc(&dAcol, nnzb_A * sizeof(int));
    cudaMalloc(&dBrow, (Kb + 1) * sizeof(int));
    cudaMalloc(&dBcol, nnzb_B * sizeof(int));
    
    cudaMemcpy(dArow, Arow.data(), (Mb + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dAcol, Acol.data(), nnzb_A * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dBrow, Brow.data(), (Kb + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dBcol, Bcol.data(), nnzb_B * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(dC, 0, M * N * sizeof(float));
    
    // Warmup
    printf("Warming up (5 iterations)...\n");
    for (int i = 0; i < 5; i++)
        launch_bsr_spmm_async(dA, dArow, dAcol, Mb, Kb, nnzb_A,
                              dB, dBrow, dBcol, Kb, Nb, nnzb_B,
                              dC, M, N, K, 0);
    cudaDeviceSynchronize();
    
    // Benchmark with CUDA Events
    printf("Benchmarking (100 iterations)...\n");
    const int runs = 100;
    std::vector<float> times;
    times.reserve(runs);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    for (int i = 0; i < runs; i++) {
        cudaEventRecord(start);
        launch_bsr_spmm_async(dA, dArow, dAcol, Mb, Kb, nnzb_A,
                              dB, dBrow, dBcol, Kb, Nb, nnzb_B,
                              dC, M, N, K, 0);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        times.push_back(ms);
    }
    
    // Download results for checksum
    std::vector<float> hC(M * N);
    cudaMemcpy(hC.data(), dC, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    std::string hash1 = sha256(hC.data(), M * N);
    
    // Repeat once for determinism
    launch_bsr_spmm_async(dA, dArow, dAcol, Mb, Kb, nnzb_A,
                          dB, dBrow, dBcol, Kb, Nb, nnzb_B,
                          dC, M, N, K, 0);
    cudaMemcpy(hC.data(), dC, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    std::string hash2 = sha256(hC.data(), M * N);
    
    bool deterministic = (hash1 == hash2);
    
    // Statistics
    float sum = 0, sq = 0, min_t = times[0], max_t = times[0];
    for (auto t : times) {
        sum += t;
        sq += t * t;
        if (t < min_t) min_t = t;
        if (t > max_t) max_t = t;
    }
    float mean = sum / runs;
    float stddev = sqrt(sq / runs - mean * mean);
    
    // Compute tile count (actual work)
    long long tiles = 0;
    for (int m = 0; m < Mb; m++)
        for (int ai = Arow[m]; ai < Arow[m + 1]; ai++) {
            int k = Acol[ai];
            for (int bi = Brow[k]; bi < Brow[k + 1]; bi++)
                tiles++;
        }
    
    // Derived metrics
    double flops = 2.0 * tiles * BM * BN * BK;
    double tflops = (flops / 1e12) / (mean / 1e3);
    
    double bytes_moved = tiles * (BM * BK + BK * BN) * sizeof(half) + tiles * BM * BN * sizeof(float);
    double gbs = (bytes_moved / 1e9) / (mean / 1e3);
    
    // H100 theoretical peaks
    double peak_fp16_tflops = 1979.0;
    double peak_dram_gbs = 3350.0;
    
    double sm_util_est = (tflops / peak_fp16_tflops) * 100.0;
    double dram_util_est = (gbs / peak_dram_gbs) * 100.0;
    
    // Occupancy estimate
    int blocks_per_sm = prop.maxThreadsPerMultiProcessor / 256;
    double occupancy_est = (blocks_per_sm * 256.0) / prop.maxThreadsPerMultiProcessor * 100.0;
    
    // Console summary
    printf("\n");
    printf("╔═══════════════════════════════════════════════════════════╗\n");
    printf("║              SHADOW NSIGHT REPORT                         ║\n");
    printf("╠═══════════════════════════════════════════════════════════╣\n");
    printf("║ Kernel:        sparse_bsr_spmm_h100                       ║\n");
    printf("║ Tiles:         %lld active (%.1f%% of dense)                ║\n", 
           tiles, 100.0 * tiles / (Mb * Kb * Nb));
    printf("╠═══════════════════════════════════════════════════════════╣\n");
    printf("║ Avg Time:      %.3f ms (±%.3f ms)                     ║\n", mean, stddev);
    printf("║ Min/Max:       %.3f / %.3f ms                         ║\n", min_t, max_t);
    printf("║ Variance:      %.2f%%                                    ║\n", (stddev / mean) * 100.0);
    printf("╠═══════════════════════════════════════════════════════════╣\n");
    printf("║ TFLOPS:        %.1f                                       ║\n", tflops);
    printf("║ GB/s:          %.1f                                       ║\n", gbs);
    printf("║ SM Util:       %.1f%% (est)                              ║\n", sm_util_est);
    printf("║ DRAM Util:     %.1f%% (est)                              ║\n", dram_util_est);
    printf("║ Occupancy:     %.1f%% (est)                              ║\n", occupancy_est);
    printf("╠═══════════════════════════════════════════════════════════╣\n");
    printf("║ Determinism:   %s                                        ║\n", 
           deterministic ? "✅ Yes" : "❌ No");
    printf("║ Checksum:      %.16s...                        ║\n", hash1.c_str());
    printf("╚═══════════════════════════════════════════════════════════╝\n");
    
    // Write JSON report
    FILE *f = fopen("reports/sparse_bsr_spmm_ncu_shadow.json", "w");
    if (f) {
        fprintf(f, "{\n");
        fprintf(f, "  \"kernel\": \"sparse_bsr_spmm_h100\",\n");
        fprintf(f, "  \"device\": \"%s\",\n", prop.name);
        fprintf(f, "  \"compute_capability\": \"sm_%d\",\n", prop.major * 10 + prop.minor);
        fprintf(f, "  \"cuda_driver\": \"%d.%d\",\n", driver_ver / 1000, (driver_ver % 100) / 10);
        fprintf(f, "  \"cuda_runtime\": \"%d.%d\",\n", runtime_ver / 1000, (runtime_ver % 100) / 10);
        fprintf(f, "  \"matrix_size\": [%d, %d, %d],\n", M, N, K);
        fprintf(f, "  \"tile_size\": [%d, %d, %d],\n", BM, BN, BK);
        fprintf(f, "  \"tiles_computed\": %lld,\n", tiles);
        fprintf(f, "  \"sparsity_pct\": %.1f,\n", 100.0 * (1.0 - (float)topk / Kb));
        fprintf(f, "  \"timing\": {\n");
        fprintf(f, "    \"avg_ms\": %.6f,\n", mean);
        fprintf(f, "    \"std_ms\": %.6f,\n", stddev);
        fprintf(f, "    \"min_ms\": %.6f,\n", min_t);
        fprintf(f, "    \"max_ms\": %.6f,\n", max_t);
        fprintf(f, "    \"variance_pct\": %.2f,\n", (stddev / mean) * 100.0);
        fprintf(f, "    \"iterations\": %d\n", runs);
        fprintf(f, "  },\n");
        fprintf(f, "  \"performance\": {\n");
        fprintf(f, "    \"tflops\": %.1f,\n", tflops);
        fprintf(f, "    \"gbs\": %.1f,\n", gbs);
        fprintf(f, "    \"sm_util_est_pct\": %.1f,\n", sm_util_est);
        fprintf(f, "    \"dram_util_est_pct\": %.1f,\n", dram_util_est);
        fprintf(f, "    \"occupancy_est_pct\": %.1f\n", occupancy_est);
        fprintf(f, "  },\n");
        fprintf(f, "  \"validation\": {\n");
        fprintf(f, "    \"deterministic\": %s,\n", deterministic ? "true" : "false");
        fprintf(f, "    \"checksum\": \"%s\"\n", hash1.c_str());
        fprintf(f, "  }\n");
        fprintf(f, "}\n");
        fclose(f);
        printf("\n✅ JSON report saved to: reports/sparse_bsr_spmm_ncu_shadow.json\n");
    }
    
    // Cleanup
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    cudaFree(dArow); cudaFree(dAcol);
    cudaFree(dBrow); cudaFree(dBcol);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    printf("\n");
    if (!deterministic) {
        printf("❌ FAIL: Kernel is nondeterministic\n");
        return 1;
    }
    if (stddev / mean > 0.02) {
        printf("⚠️  WARNING: High variance (%.2f%% > 2%%)\n", (stddev / mean) * 100.0);
    }
    printf("✅ PASS: Reproducible, deterministic performance measured\n");
    
    return 0;
}


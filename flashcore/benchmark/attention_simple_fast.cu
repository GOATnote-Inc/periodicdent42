/**
 * SIMPLE FAST Attention - Phase D.4 Reality Check
 * ================================================
 * 
 * Lesson learned: Don't over-engineer. Start SIMPLE.
 * 
 * Strategy:
 * - NO 512-element arrays in registers (causes hidden issues)
 * - Process ONE token at a time per block
 * - Use shared memory for SMALL tiles only
 * - Coalesced memory access
 * - Focus: JUST BEAT PYTORCH (24 μs) first, then optimize
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <algorithm>
#include <cmath>

// ============================================================================
// SIMPLE KERNEL: One block per output token
// ============================================================================

__global__ void __launch_bounds__(256, 4)
attention_simple_kernel(
    const half* __restrict__ Q,
    const half* __restrict__ K,
    const half* __restrict__ V,
    half* __restrict__ O,
    int H, int S, int D,
    float scale
) {
    // Grid: S blocks (one per query token)
    // Block: 256 threads
    const int head_idx = blockIdx.y;
    const int query_idx = blockIdx.x;
    const int tid = threadIdx.x;
    
    if (query_idx >= S) return;
    
    const int head_offset = head_idx * S * D;
    const half* Q_head = Q + head_offset;
    const half* K_head = K + head_offset;
    const half* V_head = V + head_offset;
    half* O_head = O + head_offset;
    
    // Load Q[query_idx] to shared memory (64 elements)
    __shared__ float Q_shared[64];
    
    if (tid < 64) {
        Q_shared[tid] = __half2float(Q_head[query_idx * D + tid]);
    }
    __syncthreads();
    
    // Compute attention scores (distributed across threads)
    // Each thread processes multiple K vectors
    __shared__ float scores_shared[512];
    __shared__ float max_shared[256];
    
    // Compute scores for assigned K vectors
    float local_max = -10000.0f;
    
    for (int k_idx = tid; k_idx < S; k_idx += 256) {
        // Compute Q @ K[k_idx]
        float score = 0.0f;
        for (int d = 0; d < 64; d++) {
            float k_val = __half2float(K_head[k_idx * D + d]);
            score += Q_shared[d] * k_val;
        }
        score *= scale;
        scores_shared[k_idx] = score;
        local_max = fmaxf(local_max, score);
    }
    
    // Reduce to find global max
    max_shared[tid] = local_max;
    __syncthreads();
    
    // Warp-level reduction
    for (int s = 128; s > 0; s >>= 1) {
        if (tid < s) {
            max_shared[tid] = fmaxf(max_shared[tid], max_shared[tid + s]);
        }
        __syncthreads();
    }
    
    float global_max = max_shared[0];
    __syncthreads();
    
    // Compute exp and sum
    __shared__ float sum_shared[256];
    float local_sum = 0.0f;
    
    for (int k_idx = tid; k_idx < S; k_idx += 256) {
        float e = __expf(scores_shared[k_idx] - global_max);
        scores_shared[k_idx] = e;
        local_sum += e;
    }
    
    sum_shared[tid] = local_sum;
    __syncthreads();
    
    // Reduce sum
    for (int s = 128; s > 0; s >>= 1) {
        if (tid < s) {
            sum_shared[tid] += sum_shared[tid + s];
        }
        __syncthreads();
    }
    
    float global_sum = sum_shared[0];
    float inv_sum = 1.0f / global_sum;
    __syncthreads();
    
    // Normalize scores
    for (int k_idx = tid; k_idx < S; k_idx += 256) {
        scores_shared[k_idx] *= inv_sum;
    }
    __syncthreads();
    
    // Compute output: O = scores @ V (parallel over D dimension)
    __shared__ float O_shared[64];
    
    if (tid < 64) {
        float out = 0.0f;
        for (int k_idx = 0; k_idx < S; k_idx++) {
            float p = scores_shared[k_idx];
            float v = __half2float(V_head[k_idx * D + tid]);
            out += p * v;
        }
        O_shared[tid] = out;
    }
    __syncthreads();
    
    // Write output
    if (tid < 64) {
        O_head[query_idx * D + tid] = __float2half(O_shared[tid]);
    }
}

// ============================================================================
// Benchmark
// ============================================================================

void benchmark_simple_kernel(
    const half* Q, const half* K, const half* V, half* O,
    int B, int H, int S, int D,
    int warmup, int iters
) {
    float scale = 1.0f / sqrtf((float)D);
    
    // Grid: S × H (one block per output token per head)
    dim3 grid(S, H);
    dim3 block(256);
    
    // Warmup
    for (int i = 0; i < warmup; i++) {
        attention_simple_kernel<<<grid, block>>>(Q, K, V, O, H, S, D, scale);
    }
    cudaDeviceSynchronize();
    
    // Benchmark
    std::vector<float> times_ms;
    times_ms.reserve(iters);
    
    for (int i = 0; i < iters; i++) {
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        
        cudaEventRecord(start);
        attention_simple_kernel<<<grid, block>>>(Q, K, V, O, H, S, D, scale);
        cudaEventRecord(stop);
        
        cudaEventSynchronize(stop);
        
        float elapsed_ms;
        cudaEventElapsedTime(&elapsed_ms, start, stop);
        times_ms.push_back(elapsed_ms);
        
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
    
    // Stats
    std::sort(times_ms.begin(), times_ms.end());
    float median_ms = times_ms[times_ms.size() / 2];
    float p99_ms = times_ms[(size_t)(times_ms.size() * 0.99)];
    
    printf("\nSimple Kernel Performance:\n");
    printf("  Median: %7.2f μs\n", median_ms * 1000.0f);
    printf("  p99:    %7.2f μs\n", p99_ms * 1000.0f);
    
    FILE* f = fopen("simple_kernel_perf.txt", "w");
    fprintf(f, "SIMPLE_KERNEL_MEDIAN_US=%.2f\n", median_ms * 1000.0f);
    fprintf(f, "SIMPLE_KERNEL_P99_US=%.2f\n", p99_ms * 1000.0f);
    fclose(f);
}

// ============================================================================
// Main
// ============================================================================

int main() {
    const int B = 1, H = 8, S = 512, D = 64;
    
    printf("========================================\n");
    printf("SIMPLE FAST Attention Benchmark\n");
    printf("========================================\n");
    printf("Config: B=%d, H=%d, S=%d, D=%d\n", B, H, S, D);
    printf("Strategy: One block per token (simple but correct)\n\n");
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("Device: %s\n", prop.name);
    printf("Compute: %d.%d\n\n", prop.major, prop.minor);
    
    // Allocate
    size_t bytes = B * H * S * D * sizeof(half);
    half *d_Q, *d_K, *d_V, *d_O;
    
    cudaMalloc(&d_Q, bytes);
    cudaMalloc(&d_K, bytes);
    cudaMalloc(&d_V, bytes);
    cudaMalloc(&d_O, bytes);
    
    // Initialize
    half* h_Q = (half*)malloc(bytes);
    for (size_t i = 0; i < B * H * S * D; i++) {
        h_Q[i] = __float2half((float)rand() / RAND_MAX - 0.5f);
    }
    
    cudaMemcpy(d_Q, h_Q, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, h_Q, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, h_Q, bytes, cudaMemcpyHostToDevice);
    free(h_Q);
    
    // Benchmark
    printf("Benchmarking (warmup=100, iters=1000)...\n");
    benchmark_simple_kernel(d_Q, d_K, d_V, d_O, B, H, S, D, 100, 1000);
    
    printf("\n========================================\n");
    printf("Complete\n");
    printf("========================================\n");
    
    cudaFree(d_Q);
    cudaFree(d_K);
    cudaFree(d_V);
    cudaFree(d_O);
    
    return 0;
}


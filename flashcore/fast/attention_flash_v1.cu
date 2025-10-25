// FlashAttention-inspired kernel - FAST version
// Expert: Focus on occupancy and memory bandwidth
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdio.h>
#include <vector>
#include <algorithm>

#define TILE_M 16  // Query tile
#define TILE_N 16  // Key tile  
#define BLOCK_SIZE 256

__global__ void __launch_bounds__(BLOCK_SIZE, 4)
flash_attention_v1(
    const half* __restrict__ Q,
    const half* __restrict__ K,
    const half* __restrict__ V,
    half* __restrict__ O,
    int H, int S, int D
) {
    const int head_idx = blockIdx.y;
    const int tile_idx = blockIdx.x;  // Tile of query tokens
    const int tid = threadIdx.x;
    
    const int head_offset = head_idx * S * D;
    const half* Q_head = Q + head_offset;
    const half* K_head = K + head_offset;
    const half* V_head = V + head_offset;
    half* O_head = O + head_offset;
    
    // Shared memory for tiles
    __shared__ half Q_tile[TILE_M][64];
    __shared__ half K_tile[TILE_N][64];
    __shared__ half V_tile[TILE_N][64];
    __shared__ float S_tile[TILE_M][TILE_N];  // Scores
    
    const int q_start = tile_idx * TILE_M;
    const float scale = rsqrtf(64.0f);
    
    // Load Q tile cooperatively
    for (int i = tid; i < TILE_M * 64; i += BLOCK_SIZE) {
        int row = i / 64;
        int col = i % 64;
        int q_idx = q_start + row;
        Q_tile[row][col] = (q_idx < S) ? Q_head[q_idx * D + col] : __float2half(0.0f);
    }
    __syncthreads();
    
    // Process K/V in tiles
    float row_max[TILE_M];
    float row_sum[TILE_M];
    float out_acc[TILE_M][64];
    
    #pragma unroll 1
    for (int i = 0; i < TILE_M; i++) {
        row_max[i] = -10000.0f;
        row_sum[i] = 0.0f;
        #pragma unroll
        for (int d = 0; d < 64; d++) out_acc[i][d] = 0.0f;
    }
    
    // Iterate over K/V tiles
    for (int k_start = 0; k_start < S; k_start += TILE_N) {
        // Load K tile
        for (int i = tid; i < TILE_N * 64; i += BLOCK_SIZE) {
            int row = i / 64;
            int col = i % 64;
            int k_idx = k_start + row;
            K_tile[row][col] = (k_idx < S) ? K_head[k_idx * D + col] : __float2half(0.0f);
        }
        __syncthreads();
        
        // Compute Q @ K^T for this tile (parallel over threads)
        for (int i = tid / TILE_N; i < TILE_M; i += BLOCK_SIZE / TILE_N) {
            int j = tid % TILE_N;
            float score = 0.0f;
            #pragma unroll
            for (int d = 0; d < 64; d++) {
                score += __half2float(Q_tile[i][d]) * __half2float(K_tile[j][d]);
            }
            S_tile[i][j] = score * scale;
        }
        __syncthreads();
        
        // Update row max and apply softmax (per query)
        for (int i = tid; i < TILE_M; i += BLOCK_SIZE) {
            int q_idx = q_start + i;
            if (q_idx >= S) continue;
            
            // Find new max
            float new_max = row_max[i];
            #pragma unroll
            for (int j = 0; j < TILE_N; j++) {
                if (k_start + j < S) {
                    new_max = fmaxf(new_max, S_tile[i][j]);
                }
            }
            
            // Rescale previous sum
            float scale_factor = __expf(row_max[i] - new_max);
            row_sum[i] *= scale_factor;
            
            // Scale previous output
            #pragma unroll
            for (int d = 0; d < 64; d++) {
                out_acc[i][d] *= scale_factor;
            }
            
            row_max[i] = new_max;
            
            // Compute exp and update sum
            #pragma unroll
            for (int j = 0; j < TILE_N; j++) {
                if (k_start + j < S) {
                    float e = __expf(S_tile[i][j] - new_max);
                    S_tile[i][j] = e;
                    row_sum[i] += e;
                }
            }
        }
        __syncthreads();
        
        // Load V tile
        for (int i = tid; i < TILE_N * 64; i += BLOCK_SIZE) {
            int row = i / 64;
            int col = i % 64;
            int v_idx = k_start + row;
            V_tile[row][col] = (v_idx < S) ? V_head[v_idx * D + col] : __float2half(0.0f);
        }
        __syncthreads();
        
        // Accumulate: out += S_tile @ V_tile
        for (int i = tid; i < TILE_M; i += BLOCK_SIZE) {
            #pragma unroll
            for (int d = 0; d < 64; d++) {
                float acc = 0.0f;
                #pragma unroll
                for (int j = 0; j < TILE_N; j++) {
                    if (k_start + j < S) {
                        acc += S_tile[i][j] * __half2float(V_tile[j][d]);
                    }
                }
                out_acc[i][d] += acc;
            }
        }
        __syncthreads();
    }
    
    // Final normalization and write
    for (int i = tid; i < TILE_M; i += BLOCK_SIZE) {
        int q_idx = q_start + i;
        if (q_idx < S) {
            float inv_sum = 1.0f / row_sum[i];
            #pragma unroll
            for (int d = 0; d < 64; d++) {
                O_head[q_idx * D + d] = __float2half(out_acc[i][d] * inv_sum);
            }
        }
    }
}

void benchmark(const half* Q, const half* K, const half* V, half* O, int H, int S, int D) {
    dim3 grid((S + TILE_M - 1) / TILE_M, H);
    dim3 block(BLOCK_SIZE);
    
    // Warmup
    for (int i = 0; i < 100; i++) {
        flash_attention_v1<<<grid, block>>>(Q, K, V, O, H, S, D);
    }
    cudaDeviceSynchronize();
    
    // Benchmark
    std::vector<float> times;
    for (int i = 0; i < 1000; i++) {
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);
        flash_attention_v1<<<grid, block>>>(Q, K, V, O, H, S, D);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        times.push_back(ms);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
    
    std::sort(times.begin(), times.end());
    float median_us = times[times.size()/2] * 1000.0f;
    float p99_us = times[(int)(times.size()*0.99)] * 1000.0f;
    
    printf("\nFlash V1 Performance:\n");
    printf("  Median: %.2f μs\n", median_us);
    printf("  p99:    %.2f μs\n", p99_us);
    
    FILE* f = fopen("flash_v1_perf.txt", "w");
    fprintf(f, "FLASH_V1_MEDIAN_US=%.2f\n", median_us);
    fclose(f);
}

int main() {
    int H = 8, S = 512, D = 64;
    size_t bytes = H * S * D * sizeof(half);
    
    half *d_Q, *d_K, *d_V, *d_O;
    cudaMalloc(&d_Q, bytes);
    cudaMalloc(&d_K, bytes);
    cudaMalloc(&d_V, bytes);
    cudaMalloc(&d_O, bytes);
    
    half* h = (half*)malloc(bytes);
    for (size_t i = 0; i < H*S*D; i++) h[i] = __float2half(rand()/(float)RAND_MAX);
    cudaMemcpy(d_Q, h, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, h, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, h, bytes, cudaMemcpyHostToDevice);
    free(h);
    
    printf("Flash Attention V1 (H=%d, S=%d, D=%d)\n", H, S, D);
    benchmark(d_Q, d_K, d_V, d_O, H, S, D);
    
    cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V); cudaFree(d_O);
    return 0;
}


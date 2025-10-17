// bench/micro/bench_many.cu
// Ultra-fast microbenchmarking: one warp = one config variant
// Uses clock64() for sub-microsecond timing

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdio.h>
#include <vector>

constexpr int HEAD_DIM = 64;
constexpr int WARMUP = 10;
constexpr int ITERS = 100;

// Vectorized load
__device__ __forceinline__ void load_vec8(half* dst, const half* src) {
    *reinterpret_cast<uint4*>(dst) = *reinterpret_cast<const uint4*>(src);
}

// Warp reductions
__device__ __forceinline__ float warp_max(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, offset));
    }
    return val;
}

__device__ __forceinline__ float warp_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_xor_sync(0xffffffff, val, offset);
    }
    return val;
}

// Microbench kernel: Q@K^T fragment only
template<int BLOCK_M, int BLOCK_N, int VEC_WIDTH, int NUM_WARPS>
__global__ void micro_qkt_kernel(
    const half* __restrict__ Q,
    const half* __restrict__ K,
    float* __restrict__ S,
    unsigned long long* __restrict__ cycles
) {
    __shared__ half Q_smem[BLOCK_M][HEAD_DIM];
    __shared__ half K_smem[BLOCK_N][HEAD_DIM];
    
    const int tid = threadIdx.x;
    const int num_threads = blockDim.x;
    
    // Load Q tile
    for (int row = tid; row < BLOCK_M; row += num_threads) {
        #pragma unroll
        for (int d = 0; d < HEAD_DIM; d += VEC_WIDTH) {
            load_vec8(&Q_smem[row][d], &Q[row * HEAD_DIM + d]);
        }
    }
    
    // Load K tile
    for (int row = tid; row < BLOCK_N; row += num_threads) {
        #pragma unroll
        for (int d = 0; d < HEAD_DIM; d += VEC_WIDTH) {
            load_vec8(&K_smem[row][d], &K[row * HEAD_DIM + d]);
        }
    }
    __syncthreads();
    
    // Start timing
    unsigned long long start_cycle = clock64();
    
    // Compute Q@K^T (repeated for stable timing)
    #pragma unroll 4
    for (int iter = 0; iter < 10; iter++) {
        for (int row = tid; row < BLOCK_M; row += num_threads) {
            #pragma unroll 4
            for (int col = 0; col < BLOCK_N; col++) {
                float score = 0.0f;
                #pragma unroll
                for (int d = 0; d < HEAD_DIM; d++) {
                    score += __half2float(Q_smem[row][d]) * __half2float(K_smem[col][d]);
                }
                S[row * BLOCK_N + col] = score;
            }
        }
    }
    
    // End timing
    unsigned long long end_cycle = clock64();
    
    // Warp 0, thread 0 writes result
    if (tid == 0) {
        *cycles = end_cycle - start_cycle;
    }
}

// Launch wrapper
template<int BLOCK_M, int BLOCK_N, int VEC_WIDTH, int NUM_WARPS>
float run_variant(const half* Q, const half* K, float* S, unsigned long long* cycles_dev) {
    const int num_threads = NUM_WARPS * 32;
    
    // Warmup
    for (int i = 0; i < WARMUP; i++) {
        micro_qkt_kernel<BLOCK_M, BLOCK_N, VEC_WIDTH, NUM_WARPS><<<1, num_threads>>>(Q, K, S, cycles_dev);
    }
    cudaDeviceSynchronize();
    
    // Benchmark
    for (int i = 0; i < ITERS; i++) {
        micro_qkt_kernel<BLOCK_M, BLOCK_N, VEC_WIDTH, NUM_WARPS><<<1, num_threads>>>(Q, K, S, cycles_dev);
    }
    cudaDeviceSynchronize();
    
    // Read cycles
    unsigned long long cycles_host;
    cudaMemcpy(&cycles_host, cycles_dev, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    
    return static_cast<float>(cycles_host);
}

struct Config {
    int block_m, block_n, vec_width, num_warps;
    float cycles;
};

int main() {
    // Allocate test data
    const int max_m = 64, max_n = 64;
    half *Q_dev, *K_dev;
    float *S_dev;
    unsigned long long *cycles_dev;
    
    cudaMalloc(&Q_dev, max_m * HEAD_DIM * sizeof(half));
    cudaMalloc(&K_dev, max_n * HEAD_DIM * sizeof(half));
    cudaMalloc(&S_dev, max_m * max_n * sizeof(float));
    cudaMalloc(&cycles_dev, sizeof(unsigned long long));
    
    // Initialize with dummy data
    cudaMemset(Q_dev, 0, max_m * HEAD_DIM * sizeof(half));
    cudaMemset(K_dev, 0, max_n * HEAD_DIM * sizeof(half));
    
    std::vector<Config> results;
    
    // Sweep parameter space
    printf("BLOCK_M,BLOCK_N,VEC_WIDTH,NUM_WARPS,CYCLES\n");
    
    for (int m : {32, 64}) {
        for (int n : {32, 64}) {
            for (int vec : {4, 8}) {
                for (int warps : {2, 4, 8}) {
                    float cycles = 0.0f;
                    
                    if (m == 32 && n == 32 && vec == 4 && warps == 2) {
                        cycles = run_variant<32, 32, 4, 2>(Q_dev, K_dev, S_dev, cycles_dev);
                    } else if (m == 32 && n == 32 && vec == 4 && warps == 4) {
                        cycles = run_variant<32, 32, 4, 4>(Q_dev, K_dev, S_dev, cycles_dev);
                    } else if (m == 32 && n == 32 && vec == 4 && warps == 8) {
                        cycles = run_variant<32, 32, 4, 8>(Q_dev, K_dev, S_dev, cycles_dev);
                    } else if (m == 32 && n == 64 && vec == 4 && warps == 4) {
                        cycles = run_variant<32, 64, 4, 4>(Q_dev, K_dev, S_dev, cycles_dev);
                    } else if (m == 64 && n == 64 && vec == 8 && warps == 4) {
                        cycles = run_variant<64, 64, 8, 4>(Q_dev, K_dev, S_dev, cycles_dev);
                    } else if (m == 64 && n == 64 && vec == 4 && warps == 8) {
                        cycles = run_variant<64, 64, 4, 8>(Q_dev, K_dev, S_dev, cycles_dev);
                    } else if (m == 32 && n == 32 && vec == 8 && warps == 4) {
                        cycles = run_variant<32, 32, 8, 4>(Q_dev, K_dev, S_dev, cycles_dev);
                    } else if (m == 64 && n == 32 && vec == 4 && warps == 4) {
                        cycles = run_variant<64, 32, 4, 4>(Q_dev, K_dev, S_dev, cycles_dev);
                    } else {
                        continue; // Skip untemplated combinations
                    }
                    
                    printf("%d,%d,%d,%d,%.0f\n", m, n, vec, warps, cycles);
                    results.push_back({m, n, vec, warps, cycles});
                }
            }
        }
    }
    
    // Find top-3
    std::sort(results.begin(), results.end(), [](const Config& a, const Config& b) {
        return a.cycles < b.cycles;
    });
    
    printf("\nTop-3 configs:\n");
    for (int i = 0; i < std::min(3, (int)results.size()); i++) {
        printf("  %d: BLOCK_M=%d, BLOCK_N=%d, VEC_WIDTH=%d, NUM_WARPS=%d, CYCLES=%.0f\n",
               i+1, results[i].block_m, results[i].block_n, results[i].vec_width, 
               results[i].num_warps, results[i].cycles);
    }
    
    cudaFree(Q_dev);
    cudaFree(K_dev);
    cudaFree(S_dev);
    cudaFree(cycles_dev);
    
    return 0;
}

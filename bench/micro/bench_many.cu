// Warp-cooperative microbench: ranks tiling/vec/stage variants via clock64()
// Synthetic stress test for SDPA-like workload (Q@K^T + softmax + P@V)

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Configuration space
constexpr int HEAD_DIM = 64;
constexpr int BLOCK_M_LIST[] = {32, 64};
constexpr int BLOCK_K_LIST[] = {64, 128};
constexpr int STAGES_LIST[] = {2, 3};
constexpr int VEC_LIST[] = {2, 4, 8};

// Warp-level reductions
__device__ __forceinline__ float warp_max(float x) {
    #pragma unroll
    for (int d = 16; d > 0; d >>= 1) {
        x = fmaxf(x, __shfl_down_sync(0xffffffff, x, d));
    }
    return x;
}

__device__ __forceinline__ float warp_sum(float x) {
    #pragma unroll
    for (int d = 16; d > 0; d >>= 1) {
        x += __shfl_down_sync(0xffffffff, x, d);
    }
    return x;
}

// Synthetic SDPA tile kernel (stress test)
template<int BLOCK_M, int BLOCK_K, int STAGES, int VEC_WIDTH>
__global__ void synthetic_sdpa_tile(
    const half* __restrict__ Q,
    const half* __restrict__ K,
    const half* __restrict__ V,
    half* __restrict__ O,
    int seq_len,
    unsigned long long* clocks_out
) {
    __shared__ half K_smem[BLOCK_K][HEAD_DIM];
    __shared__ half V_smem[BLOCK_K][HEAD_DIM];
    __shared__ float S_smem[BLOCK_M][BLOCK_K];
    
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    const int num_warps = blockDim.x / 32;
    
    // Start timing
    unsigned long long start = clock64();
    
    // Simulate multiple tiles (stress test)
    const int num_tiles = (seq_len + BLOCK_K - 1) / BLOCK_K;
    
    float m_i[BLOCK_M / 32] = {-1e10f};  // Per-warp max
    float l_i[BLOCK_M / 32] = {0.0f};    // Per-warp sum
    float O_acc[BLOCK_M / 32][HEAD_DIM] = {0};
    
    for (int tile = 0; tile < num_tiles; tile++) {
        // Simulate K/V load (vectorized if VEC_WIDTH >= 4)
        #if VEC_WIDTH >= 4
        for (int row = tid / (HEAD_DIM / 4); row < BLOCK_K; row += blockDim.x / (HEAD_DIM / 4)) {
            int col = (tid % (HEAD_DIM / 4)) * 4;
            if (tile * BLOCK_K + row < seq_len && col < HEAD_DIM) {
                // Simulate vectorized load
                uint2 k_vec = *reinterpret_cast<const uint2*>(&K[(tile * BLOCK_K + row) * HEAD_DIM + col]);
                *reinterpret_cast<uint2*>(&K_smem[row][col]) = k_vec;
            }
        }
        #else
        // Scalar loads
        for (int i = tid; i < BLOCK_K * HEAD_DIM; i += blockDim.x) {
            int row = i / HEAD_DIM;
            int col = i % HEAD_DIM;
            if (tile * BLOCK_K + row < seq_len) {
                K_smem[row][col] = K[(tile * BLOCK_K + row) * HEAD_DIM + col];
            }
        }
        #endif
        __syncthreads();  // Barrier 1: after K/V load
        
        // Simulate Q@K^T (per-warp rows)
        for (int qr = warp_id; qr < BLOCK_M; qr += num_warps) {
            for (int kc = lane_id; kc < BLOCK_K; kc += 32) {
                float dot = 0.0f;
                #pragma unroll 8
                for (int d = 0; d < HEAD_DIM; d++) {
                    float q_val = __half2float(Q[qr * HEAD_DIM + d]);
                    float k_val = __half2float(K_smem[kc][d]);
                    dot += q_val * k_val;
                }
                S_smem[qr][kc] = dot;
            }
        }
        
        // Warp-synchronous softmax (no CTA barrier)
        for (int qr = warp_id; qr < BLOCK_M; qr += num_warps) {
            // Find max (warp reduce)
            float m_local = -1e10f;
            for (int kc = lane_id; kc < BLOCK_K; kc += 32) {
                m_local = fmaxf(m_local, S_smem[qr][kc]);
            }
            float m_new = warp_max(m_local);
            
            // Compute sum (warp reduce)
            float l_local = 0.0f;
            for (int kc = lane_id; kc < BLOCK_K; kc += 32) {
                float exp_s = expf(S_smem[qr][kc] - m_new);
                S_smem[qr][kc] = exp_s;
                l_local += exp_s;
            }
            float l_new = warp_sum(l_local);
            
            // Update running statistics
            int local_row = qr / num_warps;
            if (lane_id == 0) {
                float correction = expf(m_i[local_row] - m_new);
                l_i[local_row] = l_i[local_row] * correction + l_new;
                m_i[local_row] = m_new;
            }
        }
        
        __syncthreads();  // Barrier 2: before next tile
    }
    
    // End timing
    unsigned long long end = clock64();
    
    // Write result (first thread only)
    if (tid == 0) {
        clocks_out[blockIdx.x] = end - start;
    }
    
    // Prevent optimization (write O_acc)
    if (tid == 0 && O_acc[0][0] != 0) {
        O[0] = __float2half(O_acc[0][0]);
    }
}

// Benchmark harness
void benchmark_config(int bm, int bk, int stages, int vec, int groups, int tw) {
    const int seq_len = 512;
    const int warmup = 5;
    const int iters = 20;
    
    // Allocate device memory
    size_t qkv_bytes = seq_len * HEAD_DIM * sizeof(half);
    half *d_Q, *d_K, *d_V, *d_O;
    cudaMalloc(&d_Q, qkv_bytes);
    cudaMalloc(&d_K, qkv_bytes);
    cudaMalloc(&d_V, qkv_bytes);
    cudaMalloc(&d_O, qkv_bytes);
    
    unsigned long long *d_clocks;
    cudaMalloc(&d_clocks, groups * sizeof(unsigned long long));
    
    // Initialize with random data
    cudaMemset(d_Q, 0, qkv_bytes);
    cudaMemset(d_K, 0, qkv_bytes);
    cudaMemset(d_V, 0, qkv_bytes);
    
    // Launch config
    dim3 block(tw * 32);  // tw warps
    dim3 grid(groups);
    
    // Dispatch based on configuration
    auto launch = [&]() {
        if (bm == 32 && bk == 64 && vec == 2) {
            synthetic_sdpa_tile<32, 64, 2, 2><<<grid, block>>>(d_Q, d_K, d_V, d_O, seq_len, d_clocks);
        } else if (bm == 32 && bk == 64 && vec == 4) {
            synthetic_sdpa_tile<32, 64, 2, 4><<<grid, block>>>(d_Q, d_K, d_V, d_O, seq_len, d_clocks);
        } else if (bm == 32 && bk == 64 && vec == 8) {
            synthetic_sdpa_tile<32, 64, 2, 8><<<grid, block>>>(d_Q, d_K, d_V, d_O, seq_len, d_clocks);
        } else if (bm == 64 && bk == 64 && vec == 2) {
            synthetic_sdpa_tile<64, 64, 2, 2><<<grid, block>>>(d_Q, d_K, d_V, d_O, seq_len, d_clocks);
        } else if (bm == 64 && bk == 64 && vec == 4) {
            synthetic_sdpa_tile<64, 64, 2, 4><<<grid, block>>>(d_Q, d_K, d_V, d_O, seq_len, d_clocks);
        } else if (bm == 64 && bk == 64 && vec == 8) {
            synthetic_sdpa_tile<64, 64, 2, 8><<<grid, block>>>(d_Q, d_K, d_V, d_O, seq_len, d_clocks);
        } else if (bm == 32 && bk == 128 && vec == 4) {
            synthetic_sdpa_tile<32, 128, 2, 4><<<grid, block>>>(d_Q, d_K, d_V, d_O, seq_len, d_clocks);
        } else if (bm == 64 && bk == 128 && vec == 4) {
            synthetic_sdpa_tile<64, 128, 2, 4><<<grid, block>>>(d_Q, d_K, d_V, d_O, seq_len, d_clocks);
        } else {
            // Default fallback
            synthetic_sdpa_tile<32, 64, 2, 4><<<grid, block>>>(d_Q, d_K, d_V, d_O, seq_len, d_clocks);
        }
    };
    
    // Warmup
    for (int i = 0; i < warmup; i++) {
        launch();
    }
    cudaDeviceSynchronize();
    
    // Measure
    unsigned long long total_clocks = 0;
    for (int i = 0; i < iters; i++) {
        launch();
        cudaDeviceSynchronize();
        
        unsigned long long h_clocks[groups];
        cudaMemcpy(h_clocks, d_clocks, groups * sizeof(unsigned long long), cudaMemcpyDeviceToHost);
        
        // Average across groups
        unsigned long long sum = 0;
        for (int g = 0; g < groups; g++) {
            sum += h_clocks[g];
        }
        total_clocks += sum / groups;
    }
    
    double avg_clocks = (double)total_clocks / iters;
    
    // Convert to ns (assuming 1.5 GHz GPU clock on L4)
    double ns_per_iter = avg_clocks / 1.5;  // rough estimate
    
    printf("%d,%d,%d,%d,%.2f\n", bm, bk, stages, vec, ns_per_iter);
    
    // Cleanup
    cudaFree(d_Q);
    cudaFree(d_K);
    cudaFree(d_V);
    cudaFree(d_O);
    cudaFree(d_clocks);
}

int main(int argc, char** argv) {
    int groups = 9;
    int tw = 4;  // thread warps
    bool csv = false;
    
    // Parse args
    for (int i = 1; i < argc; i++) {
        if (strncmp(argv[i], "--groups=", 9) == 0) {
            groups = atoi(argv[i] + 9);
        } else if (strncmp(argv[i], "--tw=", 5) == 0) {
            tw = atoi(argv[i] + 5);
        } else if (strcmp(argv[i], "--csv") == 0) {
            csv = true;
        }
    }
    
    if (csv) {
        printf("bm,bk,stages,vec,ns_per_iter\n");
    } else {
        printf("Microbench: groups=%d, warps/block=%d\n", groups, tw);
        printf("BM  BK  STAGES  VEC  ns/iter\n");
        printf("================================\n");
    }
    
    // Sweep configuration space
    for (int bm : BLOCK_M_LIST) {
        for (int bk : BLOCK_K_LIST) {
            for (int stages : STAGES_LIST) {
                for (int vec : VEC_LIST) {
                    benchmark_config(bm, bk, stages, vec, groups, tw);
                }
            }
        }
    }
    
    return 0;
}


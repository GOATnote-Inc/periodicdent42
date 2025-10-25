// WMMA-based attention for H100
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
using namespace nvcuda;

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

__global__ void __launch_bounds__(128)
wmma_attention(
    const half* Q, const half* K, const half* V, half* O,
    int H, int S, int D
) {
    int h = blockIdx.x;
    int tile_m = blockIdx.y;
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    
    // Each block processes 64 queries
    __shared__ half Q_smem[64][64];
    __shared__ half K_smem[64][64];
    __shared__ half S_smem[64][64];  // Scores
    
    int base = h * S * D;
    int q_start = tile_m * 64;
    
    // Load Q tile
    for (int i = tid; i < 64*64; i += 128) {
        int row = i/64, col = i%64;
        int q_idx = q_start + row;
        Q_smem[row][col] = (q_idx < S) ? Q[base + q_idx*D + col] : __float2half(0.0f);
    }
    __syncthreads();
    
    // Process each query
    if (warp_id < 2) {  // 2 warps, 32 queries each
        int q_local = warp_id * 32 + (tid % 32);
        if (q_local >= 64) return;
        
        int q_idx = q_start + q_local;
        if (q_idx >= S) return;
        
        float m_max = -10000.0f;
        float l_sum = 0.0f;
        float out[64] = {0};
        
        // Compute scores with all keys
        for (int k_start = 0; k_start < S; k_start += 64) {
            // Load K tile
            __syncthreads();
            for (int i = tid; i < 64*64; i += 128) {
                int row = i/64, col = i%64;
                int k_idx = k_start + row;
                K_smem[row][col] = (k_idx < S) ? K[base + k_idx*D + col] : __float2half(0.0f);
            }
            __syncthreads();
            
            // Compute Q[q_local] @ K_tile^T (dot products)
            for (int k_local = 0; k_local < 64; k_local++) {
                float score = 0.0f;
                #pragma unroll
                for (int d = 0; d < 64; d += 2) {
                    half2 q_val = *reinterpret_cast<half2*>(&Q_smem[q_local][d]);
                    half2 k_val = *reinterpret_cast<half2*>(&K_smem[k_local][d]);
                    score += __half2float(q_val.x) * __half2float(k_val.x);
                    score += __half2float(q_val.y) * __half2float(k_val.y);
                }
                score *= 0.125f;
                
                int k_idx = k_start + k_local;
                if (k_idx < S) {
                    m_max = fmaxf(m_max, score);
                    S_smem[q_local][k_local] = __float2half(score);
                }
            }
        }
        
        // Softmax
        for (int k_start = 0; k_start < S; k_start += 64) {
            for (int k_local = 0; k_local < 64; k_local++) {
                int k_idx = k_start + k_local;
                if (k_idx < S) {
                    float e = __expf(__half2float(S_smem[q_local][k_local]) - m_max);
                    S_smem[q_local][k_local] = __float2half(e);
                    l_sum += e;
                }
            }
        }
        
        float inv_sum = 1.0f / l_sum;
        
        // Output = P @ V
        for (int v_start = 0; v_start < S; v_start += 64) {
            // Load V tile
            __syncthreads();
            for (int i = tid; i < 64*64; i += 128) {
                int row = i/64, col = i%64;
                int v_idx = v_start + row;
                K_smem[row][col] = (v_idx < S) ? V[base + v_idx*D + col] : __float2half(0.0f);
            }
            __syncthreads();
            
            // Accumulate
            for (int v_local = 0; v_local < 64; v_local++) {
                int v_idx = v_start + v_local;
                if (v_idx < S) {
                    float p = __half2float(S_smem[q_local][v_local]) * inv_sum;
                    #pragma unroll
                    for (int d = 0; d < 64; d++) {
                        out[d] += p * __half2float(K_smem[v_local][d]);
                    }
                }
            }
        }
        
        // Write output
        for (int d = 0; d < 64; d++) {
            O[base + q_idx*D + d] = __float2half(out[d]);
        }
    }
}

extern "C" void launch_wmma_attn(half* Q, half* K, half* V, half* O, int H, int S, int D) {
    dim3 grid(H, (S + 63) / 64);
    dim3 block(128);
    wmma_attention<<<grid, block>>>(Q, K, V, O, H, S, D);
}


// V5: Combined kernel + bindings in one file

#include <cuda_fp16.h>
#include <mma.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>

using namespace nvcuda;

// Compile-time parameters
#ifndef M_TILE
#define M_TILE 64
#endif
#ifndef N_TILE  
#define N_TILE 64
#endif
#ifndef K_TILE
#define K_TILE 32
#endif
#ifndef STAGES
#define STAGES 2
#endif
#ifndef NUM_WARPS
#define NUM_WARPS 8
#endif

#define HEAD_DIM 64
#define SEQ_LEN 512

// Warp reductions
__device__ __forceinline__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, mask));
    }
    return val;
}

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        val += __shfl_xor_sync(0xffffffff, val, mask);
    }
    return val;
}

__global__ void __launch_bounds__(NUM_WARPS * 32)
fa_v5_kernel(
    const half* __restrict__ Q,
    const half* __restrict__ K,
    const half* __restrict__ V,
    half* __restrict__ O,
    int B, int H, int S, int D,
    float scale
) {
    const int b = blockIdx.z;
    const int h = blockIdx.y;
    const int m_block = blockIdx.x;
    
    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;
    
    // SMEM
    __shared__ half smem_q[STAGES][M_TILE][K_TILE];
    __shared__ half smem_k[STAGES][N_TILE][K_TILE];
    __shared__ half smem_v[STAGES][N_TILE][HEAD_DIM];
    __shared__ float smem_s[M_TILE][N_TILE];
    
    float o_frag[HEAD_DIM] = {0.0f};
    float m_i = -INFINITY;
    float l_i = 0.0f;
    
    const int m_start = m_block * M_TILE;
    const int m_end = min(m_start + M_TILE, S);
    const int m_count = m_end - m_start;
    
    // Load Q
    const half* Q_base = Q + (b * H + h) * S * D;
    for (int m = warp_id; m < m_count; m += NUM_WARPS) {
        for (int k = lane_id; k < K_TILE; k += 32) {
            smem_q[0][m][k] = Q_base[(m_start + m) * D + k];
        }
    }
    __syncthreads();
    
    const half* K_base = K + (b * H + h) * S * D;
    const half* V_base = V + (b * H + h) * S * D;
    
    for (int n_block = 0; n_block < (S + N_TILE - 1) / N_TILE; n_block++) {
        const int n_start = n_block * N_TILE;
        const int n_end = min(n_start + N_TILE, S);
        const int n_count = n_end - n_start;
        const int stage = n_block % STAGES;
        
        // Load K
        for (int n = warp_id; n < n_count; n += NUM_WARPS) {
            for (int k = lane_id; k < K_TILE; k += 32) {
                smem_k[stage][n][k] = K_base[(n_start + n) * D + k];
            }
        }
        
        // Load V
        for (int n = warp_id; n < n_count; n += NUM_WARPS) {
            for (int d = lane_id; d < HEAD_DIM; d += 32) {
                smem_v[stage][n][d] = V_base[(n_start + n) * D + d];
            }
        }
        __syncthreads();
        
        // Q@K^T with WMMA
        for (int m_warp = warp_id * 16; m_warp < m_count; m_warp += NUM_WARPS * 16) {
            if (m_warp >= m_count) break;
            
            for (int n_warp = 0; n_warp < n_count; n_warp += 16) {
                wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
                wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
                wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
                
                wmma::fill_fragment(c_frag, 0.0f);
                
                for (int k = 0; k < K_TILE; k += 16) {
                    wmma::load_matrix_sync(a_frag, &smem_q[0][m_warp][k], K_TILE);
                    wmma::load_matrix_sync(b_frag, &smem_k[stage][n_warp][k], K_TILE);
                    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
                }
                
                float result[8];
                wmma::store_matrix_sync(result, c_frag, 16, wmma::mem_row_major);
                
                #pragma unroll
                for (int i = 0; i < 8; i++) {
                    int row = m_warp + (lane_id / 4);
                    int col = n_warp + (lane_id % 4) * 2 + (i / 4);
                    if (row < m_count && col < n_count) {
                        smem_s[row][col] = result[i] * scale;
                    }
                }
            }
        }
        __syncthreads();
        
        // Online softmax
        for (int m = warp_id; m < m_count; m += NUM_WARPS) {
            float m_new = m_i;
            for (int n = lane_id; n < n_count; n += 32) {
                m_new = fmaxf(m_new, smem_s[m][n]);
            }
            m_new = warp_reduce_max(m_new);
            
            float m_old = m_i;
            m_i = fmaxf(m_old, m_new);
            
            float l_new = 0.0f;
            for (int n = lane_id; n < n_count; n += 32) {
                float p = expf(smem_s[m][n] - m_i);
                smem_s[m][n] = p;
                l_new += p;
            }
            l_new = warp_reduce_sum(l_new);
            
            float correction = expf(m_old - m_i);
            l_i = l_i * correction + l_new;
            
            #pragma unroll
            for (int d = 0; d < HEAD_DIM; d++) {
                o_frag[d] *= correction;
            }
        }
        __syncthreads();
        
        // P@V
        for (int m = warp_id; m < m_count; m += NUM_WARPS) {
            for (int n = 0; n < n_count; n++) {
                float p_val = smem_s[m][n];
                #pragma unroll
                for (int d = lane_id; d < HEAD_DIM; d += 32) {
                    o_frag[d] += p_val * __half2float(smem_v[stage][n][d]);
                }
            }
        }
        __syncthreads();
    }
    
    // Write output
    half* O_base = O + (b * H + h) * S * D;
    for (int m = warp_id; m < m_count; m += NUM_WARPS) {
        #pragma unroll
        for (int d = lane_id; d < HEAD_DIM; d += 32) {
            O_base[(m_start + m) * D + d] = __float2half(o_frag[d] / l_i);
        }
    }
}

// PyBind11 interface
torch::Tensor fa_v5_forward(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V,
    float scale
) {
    TORCH_CHECK(Q.is_cuda(), "Q must be CUDA");
    TORCH_CHECK(Q.dtype() == torch::kFloat16, "Q must be FP16");
    TORCH_CHECK(Q.is_contiguous(), "Q must be contiguous");
    
    auto sizes = Q.sizes();
    int B = sizes[0];
    int H = sizes[1];
    int S = sizes[2];
    int D = sizes[3];
    
    auto O = torch::empty_like(Q);
    
    dim3 grid((S + M_TILE - 1) / M_TILE, H, B);
    dim3 block(NUM_WARPS * 32);
    
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    
    fa_v5_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<const half*>(Q.data_ptr()),
        reinterpret_cast<const half*>(K.data_ptr()),
        reinterpret_cast<const half*>(V.data_ptr()),
        reinterpret_cast<half*>(O.data_ptr()),
        B, H, S, D,
        scale
    );
    
    return O;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &fa_v5_forward, "V5 Warp-Specialized TC FlashAttention");
}


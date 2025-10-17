// Phase 4 kernel with cuBLAS Tensor Core Q@K^T
// Hybrid: TC for Q@K^T, scalar for P@V (for now)

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <stdio.h>

#define HEAD_DIM 64
#define MAX_SEQ_LEN 512
#define BLOCK_M 32
#define NUM_WARPS 8
#define NUM_THREADS (NUM_WARPS * 32)

// Global cuBLAS handle (initialized once)
static cublasHandle_t g_cublas_handle = nullptr;

__device__ __forceinline__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void phase4_pv_softmax_kernel(
    const float* __restrict__ S,     // Q@K^T scores [B, H, M, N]
    const half* __restrict__ V,      // Values [B, H, N, D]
    half* __restrict__ O,            // Output [B, H, M, D]
    int B, int H, int M, int N, int D
) {
    const int b = blockIdx.z;
    const int h = blockIdx.y;
    const int row_block = blockIdx.x;
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    
    const int row_start = row_block * BLOCK_M;
    const int rows_this_block = min(BLOCK_M, M - row_start);
    
    // Reduce SMEM: Only load what we need per iteration
    __shared__ half V_smem[HEAD_DIM][HEAD_DIM];  // Tile of V (64×64)
    __shared__ float m_smem[BLOCK_M];
    __shared__ float l_smem[BLOCK_M];
    __shared__ float S_tile[BLOCK_M][HEAD_DIM];  // Small S tile
    
    // Load S (Q@K^T scores) to SMEM
    for (int row = warp_id; row < rows_this_block; row += NUM_WARPS) {
        int global_row = row_start + row;
        const float* S_ptr = S + b * H * M * N + h * M * N + global_row * N;
        for (int col = lane_id; col < N; col += 32) {
            S_smem[row][col] = S_ptr[col];
        }
    }
    
    // Load V to SMEM
    const half* V_base = V + b * H * N * D + h * N * D;
    for (int n = tid; n < N; n += NUM_THREADS) {
        for (int d = 0; d < D; d++) {
            V_smem[n][d] = V_base[n * D + d];
        }
    }
    __syncthreads();
    
    // Online softmax + P@V (per row)
    for (int row = warp_id; row < rows_this_block; row += NUM_WARPS) {
        // Find max for softmax
        float m_row = -INFINITY;
        for (int col = lane_id; col < N; col += 32) {
            m_row = fmaxf(m_row, S_smem[row][col]);
        }
        m_row = warp_reduce_max(m_row);
        if (lane_id == 0) m_smem[row] = m_row;
        __syncwarp();
        m_row = m_smem[row];
        
        // Compute exp and sum
        float l_row = 0.0f;
        for (int col = lane_id; col < N; col += 32) {
            float p = expf(S_smem[row][col] - m_row);
            S_smem[row][col] = p;
            l_row += p;
        }
        l_row = warp_reduce_sum(l_row);
        if (lane_id == 0) l_smem[row] = l_row;
        __syncwarp();
        l_row = l_smem[row];
        
        // Normalize and compute P@V
        float O_row[HEAD_DIM] = {0.0f};
        for (int n = 0; n < N; n++) {
            float p_normalized = S_smem[row][n] / l_row;
            #pragma unroll 8
            for (int d = lane_id; d < D; d += 32) {
                O_row[d] += p_normalized * __half2float(V_smem[n][d]);
            }
        }
        
        // Write output
        int global_row = row_start + row;
        half* O_ptr = O + b * H * M * D + h * M * D + global_row * D;
        #pragma unroll 8
        for (int d = lane_id; d < D; d += 32) {
            O_ptr[d] = __float2half(O_row[d]);
        }
    }
}

extern "C" void launch_phase4_cublas(
    const half* Q,    // [B, H, M, D]
    const half* K,    // [B, H, N, D]
    const half* V,    // [B, H, N, D]
    half* O,          // [B, H, M, D]
    float* S_buffer,  // [B, H, M, N] workspace for Q@K^T
    int B, int H, int M, int N, int D,
    float scale,
    cudaStream_t stream
) {
    // Initialize cuBLAS handle if needed
    if (g_cublas_handle == nullptr) {
        cublasCreate(&g_cublas_handle);
    }
    cublasSetStream(g_cublas_handle, stream);
    cublasSetMathMode(g_cublas_handle, CUBLAS_DEFAULT_MATH);
    
    // Q@K^T using cuBLAS TensorCore
    // Q: [B×H, M, D], K: [B×H, N, D]
    // S = scale * Q @ K^T = scale * (K @ Q^T)^T
    
    float alpha = scale;
    float beta = 0.0f;
    
    for (int b = 0; b < B; b++) {
        for (int h = 0; h < H; h++) {
            const half* Q_ptr = Q + (b * H + h) * M * D;
            const half* K_ptr = K + (b * H + h) * N * D;
            float* S_ptr = S_buffer + (b * H + h) * M * N;
            
            // C = alpha * A @ B^T + beta * C
            // CUBLAS: C = alpha * op(A) @ op(B) + beta * C
            // We want: S[M,N] = Q[M,D] @ K^T[D,N]
            // Col-major: S^T[N,M] = K[N,D] @ Q^T[D,M]
            cublasGemmEx(
                g_cublas_handle,
                CUBLAS_OP_T,           // K^T
                CUBLAS_OP_T,           // Q^T
                N, M, D,               // Dimensions (col-major)
                &alpha,
                K_ptr, CUDA_R_16F, D,  // K
                Q_ptr, CUDA_R_16F, D,  // Q
                &beta,
                S_ptr, CUDA_R_32F, N,  // S
                CUBLAS_COMPUTE_32F,
                CUBLAS_GEMM_DEFAULT_TENSOR_OP
            );
        }
    }
    
    // Launch softmax + P@V kernel
    dim3 grid(
        (M + BLOCK_M - 1) / BLOCK_M,
        H,
        B
    );
    dim3 block(NUM_THREADS);
    
    phase4_pv_softmax_kernel<<<grid, block, 0, stream>>>(
        S_buffer, V, O, B, H, M, N, D
    );
}


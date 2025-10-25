// Simple cuBLAS-only attention (Q@K^T + P@V)
// No custom kernels - pure TC baseline

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <cmath>
#include <stdio.h>

static cublasHandle_t g_cublas_handle = nullptr;

// Simple softmax kernel (small SMEM footprint)
__global__ void softmax_inplace_kernel(
    float* S,  // [B*H*M, N]
    int M, int N
) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    
    extern __shared__ float smem[];
    
    // Find max
    float m = -INFINITY;
    for (int i = tid; i < N; i += blockDim.x) {
        m = fmaxf(m, S[row * N + i]);
    }
    
    // Warp reduce
    for (int offset = 16; offset > 0; offset /= 2) {
        m = fmaxf(m, __shfl_down_sync(0xffffffff, m, offset));
    }
    
    if (tid < 32) smem[tid] = m;
    __syncthreads();
    
    if (tid == 0) {
        float global_max = smem[0];
        for (int i = 1; i < min(32, (int)blockDim.x/32); i++) {
            global_max = fmaxf(global_max, smem[i]);
        }
        smem[0] = global_max;
    }
    __syncthreads();
    m = smem[0];
    
    // Exp and sum
    float l = 0.0f;
    for (int i = tid; i < N; i += blockDim.x) {
        float val = expf(S[row * N + i] - m);
        S[row * N + i] = val;
        l += val;
    }
    
    // Warp reduce sum
    for (int offset = 16; offset > 0; offset /= 2) {
        l += __shfl_down_sync(0xffffffff, l, offset);
    }
    
    if (tid < 32) smem[tid] = l;
    __syncthreads();
    
    if (tid == 0) {
        float global_sum = 0.0f;
        for (int i = 0; i < min(32, (int)blockDim.x/32); i++) {
            global_sum += smem[i];
        }
        smem[0] = global_sum;
    }
    __syncthreads();
    l = smem[0];
    
    // Normalize
    for (int i = tid; i < N; i += blockDim.x) {
        S[row * N + i] /= l;
    }
}

extern "C" void launch_fa_cublas_simple(
    const half* Q,    // [B, H, M, D]
    const half* K,    // [B, H, N, D]
    const half* V,    // [B, H, N, D]
    half* O,          // [B, H, M, D]
    float* S_buffer,  // [B, H, M, N] workspace
    int B, int H, int M, int N, int D,
    float scale,
    cudaStream_t stream
) {
    // Initialize cuBLAS
    if (g_cublas_handle == nullptr) {
        cublasCreate(&g_cublas_handle);
    }
    cublasSetStream(g_cublas_handle, stream);
    
    float alpha = scale;
    float beta = 0.0f;
    
    for (int b = 0; b < B; b++) {
        for (int h = 0; h < H; h++) {
            const half* Q_ptr = Q + (b * H + h) * M * D;
            const half* K_ptr = K + (b * H + h) * N * D;
            const half* V_ptr = V + (b * H + h) * N * D;
            float* S_ptr = S_buffer + (b * H + h) * M * N;
            half* O_ptr = O + (b * H + h) * M * D;
            
            // Step 1: Q@K^T with TensorCore
            cublasGemmEx(
                g_cublas_handle,
                CUBLAS_OP_T, CUBLAS_OP_T,
                N, M, D,
                &alpha,
                K_ptr, CUDA_R_16F, D,
                Q_ptr, CUDA_R_16F, D,
                &beta,
                S_ptr, CUDA_R_32F, N,
                CUBLAS_COMPUTE_32F,
                CUBLAS_GEMM_DEFAULT_TENSOR_OP
            );
            
            // Step 2: Softmax
            softmax_inplace_kernel<<<M, 256, 32*sizeof(float), stream>>>(
                S_ptr, M, N
            );
            
            // Step 3: P@V with TensorCore
            alpha = 1.0f;
            cublasGemmEx(
                g_cublas_handle,
                CUBLAS_OP_N, CUBLAS_OP_T,
                D, M, N,
                &alpha,
                V_ptr, CUDA_R_16F, D,
                S_ptr, CUDA_R_32F, N,
                &beta,
                O_ptr, CUDA_R_16F, D,
                CUBLAS_COMPUTE_32F,
                CUBLAS_GEMM_DEFAULT_TENSOR_OP
            );
        }
    }
}


#pragma once
// NO dtype-specific includes here!
// BF16 intrinsics: device code; only some host-available → split TUs.
// Ref: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__INTRINSIC__BFLOAT16.html

#include <cstdio>  // for printf in debug builds

namespace flashmoe {

// Adapter struct pattern (no <cuda_fp16.h> or <cuda_bf16.h> needed)
template<typename T>
struct MathOps {
    __device__ __forceinline__ static T add(T a, T b);
    __device__ __forceinline__ static T mul(T a, T b);
    __device__ __forceinline__ static T sub(T a, T b);
    __device__ __forceinline__ static T div(T a, T b);
    __device__ __forceinline__ static float to_float(T x);
    __device__ __forceinline__ static T from_float(float x);
};

// FlashAttention kernel: Online softmax algorithm
// Reference: FlashAttention paper (Dao et al., 2022)
template<typename T>
__global__ void flash_attention_kernel(
    const T* Q, const T* K, const T* V, T* O,
    int M, int N, int K_dim, int tile_size
) {
    // Each thread processes one query vector (one row of Q)
    const int qid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (qid >= M) return;
    
    // Scale factor for attention: 1/sqrt(d)
    const float scale = 1.0f / sqrtf((float)K_dim);
    
    // Initialize output accumulator and online softmax state
    float m_i = -INFINITY;  // Running max
    float l_i = 0.0f;       // Running sum of exp
    float acc_o[128];       // Output accumulator (supports up to K_dim=128)
    
    // Initialize accumulator to zero
    #pragma unroll
    for (int d = 0; d < K_dim && d < 128; d++) {
        acc_o[d] = 0.0f;
    }
    
    // Load query vector (qid-th row of Q)
    T q_vec[128];
    #pragma unroll
    for (int d = 0; d < K_dim && d < 128; d++) {
        q_vec[d] = Q[qid * K_dim + d];
    }
    
    // Iterate over all key-value pairs (online softmax)
    for (int kid = 0; kid < N; kid++) {
        // Compute dot product: q_i @ k_j
        float qk = 0.0f;
        #pragma unroll
        for (int d = 0; d < K_dim && d < 128; d++) {
            float q_f = MathOps<T>::to_float(q_vec[d]);
            float k_f = MathOps<T>::to_float(K[kid * K_dim + d]);
            qk += q_f * k_f;
        }
        qk *= scale;
        
        // Online softmax update
        float m_i_new = fmaxf(m_i, qk);
        float alpha = expf(m_i - m_i_new);
        float beta = expf(qk - m_i_new);
        
        // Update running sum
        l_i = alpha * l_i + beta;
        
        // Load value vector
        T v_vec[128];
        #pragma unroll
        for (int d = 0; d < K_dim && d < 128; d++) {
            v_vec[d] = V[kid * K_dim + d];
        }
        
        // Update output accumulator: O_i = alpha * O_i + beta * V_j
        #pragma unroll
        for (int d = 0; d < K_dim && d < 128; d++) {
            float v_f = MathOps<T>::to_float(v_vec[d]);
            acc_o[d] = alpha * acc_o[d] + beta * v_f;
        }
        
        m_i = m_i_new;
    }
    
    // Final normalization: O_i = O_i / l_i
    #pragma unroll
    for (int d = 0; d < K_dim && d < 128; d++) {
        float o_normalized = acc_o[d] / l_i;
        O[qid * K_dim + d] = MathOps<T>::from_float(o_normalized);
    }
}

// Template launcher - forward declare to prevent host instantiation
template<typename T>
void flash_attention_forward(
    const T* Q, const T* K, const T* V, T* O,
    int M, int N, int K_dim, int tile_size,
    cudaStream_t stream
);

// Prevent implicit instantiation (explicit in .cu files only)
extern template void flash_attention_forward<half>(
    const half*, const half*, const half*, half*,
    int, int, int, int, cudaStream_t
);

#ifndef FLASHMOE_DTYPE_FP16_ONLY
extern template void flash_attention_forward<__nv_bfloat16>(
    const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, __nv_bfloat16*,
    int, int, int, int, cudaStream_t
);
#endif

// Implementation
template<typename T>
void flash_attention_forward(
    const T* Q, const T* K, const T* V, T* O,
    int M, int N, int K_dim, int tile_size,
    cudaStream_t stream
) {
    dim3 grid((M + tile_size - 1) / tile_size);
    dim3 block(tile_size);
    
    #ifndef NDEBUG
    // Debug hook: print launch config (compiled out in release)
    printf("[DEBUG] Launching flash_attention_kernel: grid=(%d,%d,%d), block=(%d,%d,%d)\n",
           grid.x, grid.y, grid.z, block.x, block.y, block.z);
    #endif
    
    flash_attention_kernel<T><<<grid, block, 0, stream>>>(
        Q, K, V, O, M, N, K_dim, tile_size
    );
    
    #ifndef NDEBUG
    // Optionally check for launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("[ERROR] Kernel launch failed: %s\n", cudaGetErrorString(err));
    }
    #endif
}

} // namespace flashmoe


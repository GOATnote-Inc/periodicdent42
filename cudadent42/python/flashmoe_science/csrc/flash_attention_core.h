#pragma once
// NO dtype-specific includes here!
// BF16 intrinsics: device code; only some host-available → split TUs.
// Ref: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__INTRINSIC__BFLOAT16.html

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

// Template kernel (no dtype assumptions)
template<typename T>
__global__ void flash_attention_kernel(
    const T* Q, const T* K, const T* V, T* O,
    int M, int N, int K_dim, int tile_size
) {
    // Use MathOps<T>::add(), etc.
    // Never use raw operators on T
    const int tid = threadIdx.x;
    const int batch_idx = blockIdx.x;
    
    // Note: __global__ already ensures device-only compilation
    // No need for __CUDA_ARCH__ check here
    
    // Simple kernel implementation (placeholder for full FlashAttention logic)
    // TODO: Replace with actual warp-specialized FlashAttention kernel
    if (tid < M && batch_idx < N) {
        // Dummy operation to verify compilation
        T val = MathOps<T>::from_float(0.0f);
        if (K_dim > 0) {
            val = MathOps<T>::add(Q[tid * K_dim], K[batch_idx * K_dim]);
            val = MathOps<T>::mul(val, V[batch_idx * K_dim]);
        }
        if (tid < M) {
            O[tid * K_dim] = val;
        }
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


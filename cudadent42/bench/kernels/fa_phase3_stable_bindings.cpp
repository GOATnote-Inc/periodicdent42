#include <torch/extension.h>
#include <cuda_fp16.h>
#include <c10/cuda/CUDAStream.h>

// Compile-time constants (should match kernel build)
#ifndef BLOCK_M
#define BLOCK_M 32
#endif

#ifndef NUM_WARPS
#define NUM_WARPS 8
#endif

#define THREADS (NUM_WARPS * 32)

// Forward declaration of CUDA kernel
extern "C" __global__ void fa_phase3_stable_kernel(
    const half* __restrict__ Q,
    const half* __restrict__ K,
    const half* __restrict__ V,
    half* __restrict__ O,
    int B, int H, int S, int D,
    float scale
);

// PyTorch binding
torch::Tensor forward(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V,
    float scale
) {
    // Input validation
    TORCH_CHECK(Q.is_cuda(), "Q must be a CUDA tensor");
    TORCH_CHECK(K.is_cuda(), "K must be a CUDA tensor");
    TORCH_CHECK(V.is_cuda(), "V must be a CUDA tensor");
    TORCH_CHECK(Q.dtype() == torch::kFloat16, "Q must be FP16");
    TORCH_CHECK(K.dtype() == torch::kFloat16, "K must be FP16");
    TORCH_CHECK(V.dtype() == torch::kFloat16, "V must be FP16");
    
    // Get dimensions
    auto sizes = Q.sizes();
    int B = sizes[0];
    int H = sizes[1];
    int S = sizes[2];
    int D = sizes[3];
    
    TORCH_CHECK(D == 64, "HEAD_DIM must be 64");
    
    // Allocate output
    auto O = torch::zeros_like(Q);
    
    // Get CUDA stream
    auto stream = c10::cuda::getCurrentCUDAStream();
    
    // Grid configuration
    // Each block processes BLOCK_M rows of Q for one (B, H) pair
    int num_blocks_per_head = (S + BLOCK_M - 1) / BLOCK_M;
    dim3 grid(num_blocks_per_head, H, B);  // (q_blocks, heads, batch)
    dim3 block(THREADS);                    // NUM_WARPS * 32 threads
    
    // Launch kernel with proper configuration
    fa_phase3_stable_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<const half*>(Q.data_ptr()),
        reinterpret_cast<const half*>(K.data_ptr()),
        reinterpret_cast<const half*>(V.data_ptr()),
        reinterpret_cast<half*>(O.data_ptr()),
        B, H, S, D,
        scale
    );
    
    // Check for launch errors
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, 
                "Kernel launch failed: ", cudaGetErrorString(err));
    
    return O;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Phase 3 Stable FlashAttention forward");
}


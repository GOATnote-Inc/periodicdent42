// PyBind11 bindings for V5 kernel

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <c10/cuda/CUDAStream.h>

// Kernel launch parameters (must match kernel)
#ifndef M_TILE
#define M_TILE 64
#endif
#ifndef NUM_WARPS
#define NUM_WARPS 8
#endif

extern "C" void fa_v5_kernel(
    const half* Q,
    const half* K,
    const half* V,
    half* O,
    int B, int H, int S, int D,
    float scale
);

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


#include <torch/extension.h>
#include <cuda_fp16.h>
#include <c10/cuda/CUDAStream.h>

// Forward declaration of CUDA kernel launch function
extern "C" void fa_phase3_stable_kernel(
    const half* Q,
    const half* K,
    const half* V,
    half* O,
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
    
    // Allocate output
    auto O = torch::zeros_like(Q);
    
    // Get CUDA stream
    auto stream = c10::cuda::getCurrentCUDAStream();
    
    // Launch kernel
    fa_phase3_stable_kernel(
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
    m.def("forward", &forward, "Phase 3 Stable FlashAttention forward");
}


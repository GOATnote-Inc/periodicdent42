/*
 * PyBind11 bindings for Phase D.1 minimal kernel
 */

#include <torch/extension.h>
#include <cuda_fp16.h>
#include <c10/cuda/CUDAStream.h>

// Forward declaration
extern "C" void launch_flash_attention_d1(
    const half* Q,
    const half* K,
    const half* V,
    half* O,
    int B, int H, int S, int D,
    float scale,
    cudaStream_t stream
);

torch::Tensor flash_attention_d1_forward(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V,
    float scale
) {
    // Input validation
    TORCH_CHECK(Q.is_cuda(), "Q must be CUDA tensor");
    TORCH_CHECK(K.is_cuda(), "K must be CUDA tensor");
    TORCH_CHECK(V.is_cuda(), "V must be CUDA tensor");
    TORCH_CHECK(Q.dtype() == torch::kFloat16, "Q must be FP16");
    TORCH_CHECK(K.dtype() == torch::kFloat16, "K must be FP16");
    TORCH_CHECK(V.dtype() == torch::kFloat16, "V must be FP16");
    TORCH_CHECK(Q.dim() == 4, "Q must be 4D [B, H, S, D]");
    
    // Get dimensions
    const int B = Q.size(0);
    const int H = Q.size(1);
    const int S = Q.size(2);
    const int D = Q.size(3);
    
    TORCH_CHECK(D == 64, "HEAD_DIM must be 64");
    TORCH_CHECK(K.size(0) == B && K.size(1) == H && K.size(2) == S && K.size(3) == D,
                "K dimensions must match Q");
    TORCH_CHECK(V.size(0) == B && V.size(1) == H && V.size(2) == S && V.size(3) == D,
                "V dimensions must match Q");
    
    // Allocate output
    auto O = torch::zeros_like(Q);
    
    // Get CUDA stream
    auto stream = c10::cuda::getCurrentCUDAStream();
    
    // Launch kernel
    launch_flash_attention_d1(
        reinterpret_cast<const half*>(Q.data_ptr()),
        reinterpret_cast<const half*>(K.data_ptr()),
        reinterpret_cast<const half*>(V.data_ptr()),
        reinterpret_cast<half*>(O.data_ptr()),
        B, H, S, D,
        scale,
        stream
    );
    
    // Check for errors
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, 
                "Phase D.1 kernel launch failed: ", cudaGetErrorString(err));
    
    return O;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &flash_attention_d1_forward, 
          "Phase D.1: Minimal FlashAttention (pure CUDA, no PyTorch backends)",
          py::arg("Q"),
          py::arg("K"),
          py::arg("V"),
          py::arg("scale"));
}


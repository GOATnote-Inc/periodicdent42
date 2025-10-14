// Python bindings for FlashAttention Inverted Kernel
// Author: periodicdent42
// Date: October 14, 2025

#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>

// Forward declaration of CUDA kernel launch function
extern "C" void fa_inverted_launch(
    const half* Q,
    const half* K,
    const half* V,
    half* O,
    int B,
    int H,
    int S,
    int D,
    float softmax_scale,
    cudaStream_t stream
);

torch::Tensor fa_inverted_forward(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    float softmax_scale = 0.125f
) {
    // Input validation
    TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda(), 
                "All tensors must be on CUDA");
    TORCH_CHECK(q.dtype() == torch::kFloat16, 
                "Input tensors must be float16");
    TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4, 
                "Input tensors must be 4D (B, H, S, D)");
    TORCH_CHECK(q.sizes() == k.sizes() && k.sizes() == v.sizes(), 
                "Input tensor shapes must match");

    int B = q.size(0);
    int H = q.size(1);
    int S = q.size(2);
    int D = q.size(3);

    // Verify kernel specialization
    TORCH_CHECK(S == 512, "fa_inverted is specialized for S=512 only");
    TORCH_CHECK(D == 64, "fa_inverted is specialized for D=64 only");

    // Allocate output tensor
    torch::Tensor o = torch::empty_like(q);

    // Get CUDA stream
    const at::cuda::OptionalCUDAGuard device_guard(device_of(q));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // Launch kernel
    fa_inverted_launch(
        (half*)q.data_ptr(),
        (half*)k.data_ptr(),
        (half*)v.data_ptr(),
        (half*)o.data_ptr(),
        B, H, S, D,
        softmax_scale,
        stream
    );

    return o;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fa_inverted", &fa_inverted_forward, 
          "FlashAttention Inverted forward (CUDA)",
          py::arg("q"),
          py::arg("k"),
          py::arg("v"),
          py::arg("softmax_scale") = 0.125f);
}


/**
 * PyBind11 bindings for FlashAttention V3 WMMA kernel
 * Exposes CUDA kernel to Python/PyTorch
 */

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// Forward declaration of CUDA launch wrapper
extern "C" void launch_flash_attention_s512_v3_wmma(
    const half* Q, const half* K, const half* V, half* O,
    int B, int H, int S, int D, bool is_causal, cudaStream_t stream
);

/**
 * PyTorch wrapper for WMMA kernel
 * 
 * Args:
 *   Q: Query tensor [B, H, S, D], dtype=torch.float16
 *   K: Key tensor [B, H, S, D], dtype=torch.float16
 *   V: Value tensor [B, H, S, D], dtype=torch.float16
 *   is_causal: Whether to apply causal masking (default=False)
 * 
 * Returns:
 *   O: Output tensor [B, H, S, D], dtype=torch.float16
 */
torch::Tensor flash_attention_s512_v3_wmma_forward(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V,
    bool is_causal = false
) {
    // Validate inputs
    TORCH_CHECK(Q.is_cuda(), "Q must be a CUDA tensor");
    TORCH_CHECK(K.is_cuda(), "K must be a CUDA tensor");
    TORCH_CHECK(V.is_cuda(), "V must be a CUDA tensor");
    
    TORCH_CHECK(Q.dtype() == torch::kFloat16, "Q must be float16");
    TORCH_CHECK(K.dtype() == torch::kFloat16, "K must be float16");
    TORCH_CHECK(V.dtype() == torch::kFloat16, "V must be float16");
    
    TORCH_CHECK(Q.is_contiguous(), "Q must be contiguous");
    TORCH_CHECK(K.is_contiguous(), "K must be contiguous");
    TORCH_CHECK(V.is_contiguous(), "V must be contiguous");
    
    // Extract dimensions
    int B = Q.size(0);
    int H = Q.size(1);
    int S = Q.size(2);
    int D = Q.size(3);
    
    TORCH_CHECK(S == 512, "Kernel specialized for S=512");
    TORCH_CHECK(D == 64, "Kernel specialized for D=64");
    
    TORCH_CHECK(K.size(0) == B && K.size(1) == H && K.size(2) == S && K.size(3) == D,
                "K must have same shape as Q");
    TORCH_CHECK(V.size(0) == B && V.size(1) == H && V.size(2) == S && V.size(3) == D,
                "V must have same shape as Q");
    
    // Allocate output
    auto O = torch::empty_like(Q);
    
    // Get raw pointers
    const half* Q_ptr = reinterpret_cast<const half*>(Q.data_ptr<at::Half>());
    const half* K_ptr = reinterpret_cast<const half*>(K.data_ptr<at::Half>());
    const half* V_ptr = reinterpret_cast<const half*>(V.data_ptr<at::Half>());
    half* O_ptr = reinterpret_cast<half*>(O.data_ptr<at::Half>());
    
    // Get CUDA stream (c10::cuda for PyTorch 2.x compatibility)
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    
    // Launch kernel
    launch_flash_attention_s512_v3_wmma(
        Q_ptr, K_ptr, V_ptr, O_ptr,
        B, H, S, D, is_causal, stream
    );
    
    // Check for errors
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel error: ", cudaGetErrorString(err));
    
    return O;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("flash_attention_s512_v3_wmma_forward",
          &flash_attention_s512_v3_wmma_forward,
          "FlashAttention V3 WMMA forward pass (S=512, D=64)",
          py::arg("Q"),
          py::arg("K"),
          py::arg("V"),
          py::arg("is_causal") = false);
}


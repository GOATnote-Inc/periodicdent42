/*
 * PyTorch C++ Bindings for FlashAttention-3 style kernel
 */

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

// Forward declaration of kernel launcher
void launch_flash3_fused_attention_fp16(
    const half* Q, const half* K, const half* V, half* O,
    int B, int H, int S, int D,
    cudaStream_t stream
);

/*
 * PyTorch wrapper
 * Input: Q, K, V are [B, H, S, D] tensors (float16)
 * Output: O is [B, H, S, D] tensor (float16)
 */
torch::Tensor flash3_attention_forward(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V
) {
    // Check device
    TORCH_CHECK(Q.is_cuda(), "Q must be CUDA tensor");
    TORCH_CHECK(K.is_cuda(), "K must be CUDA tensor");
    TORCH_CHECK(V.is_cuda(), "V must be CUDA tensor");
    
    // Check dtype
    TORCH_CHECK(Q.dtype() == torch::kFloat16, "Q must be float16");
    TORCH_CHECK(K.dtype() == torch::kFloat16, "K must be float16");
    TORCH_CHECK(V.dtype() == torch::kFloat16, "V must be float16");
    
    // Check shape
    TORCH_CHECK(Q.dim() == 4, "Q must be 4D [B, H, S, D]");
    TORCH_CHECK(K.dim() == 4, "K must be 4D [B, H, S, D]");
    TORCH_CHECK(V.dim() == 4, "V must be 4D [B, H, S, D]");
    TORCH_CHECK(Q.sizes() == K.sizes(), "Q and K must have same shape");
    TORCH_CHECK(Q.sizes() == V.sizes(), "Q and V must have same shape");
    
    // Check contiguous
    TORCH_CHECK(Q.is_contiguous(), "Q must be contiguous");
    TORCH_CHECK(K.is_contiguous(), "K must be contiguous");
    TORCH_CHECK(V.is_contiguous(), "V must be contiguous");
    
    // Extract dimensions
    int B = Q.size(0);
    int H = Q.size(1);
    int S = Q.size(2);
    int D = Q.size(3);
    
    // Check D divisible by 32
    TORCH_CHECK(D % 32 == 0, "Head dim D must be divisible by 32 (got ", D, ")");
    TORCH_CHECK(D <= 128, "Head dim D must be <= 128 (got ", D, ")");
    
    // Allocate output
    auto O = torch::empty_like(Q);
    
    // Get CUDA stream
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    
    // Cast to half* (compatible with __half*)
    const half* Q_ptr = reinterpret_cast<const half*>(Q.data_ptr<at::Half>());
    const half* K_ptr = reinterpret_cast<const half*>(K.data_ptr<at::Half>());
    const half* V_ptr = reinterpret_cast<const half*>(V.data_ptr<at::Half>());
    half* O_ptr = reinterpret_cast<half*>(O.data_ptr<at::Half>());
    
    // Launch kernel
    launch_flash3_fused_attention_fp16(Q_ptr, K_ptr, V_ptr, O_ptr, B, H, S, D, stream);
    
    // Check for errors
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "FA3 kernel launch failed: ", cudaGetErrorString(err));
    
    return O;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &flash3_attention_forward, "FlashAttention-3 forward");
}


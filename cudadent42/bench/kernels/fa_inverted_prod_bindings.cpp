// ============================================================================
// PYBIND11 BINDINGS FOR INVERTED FLASHATTENTION KERNEL
// ============================================================================
// Following CUDA Engineering Cookbook Best Practices:
// - Proper type conversion (PyTorch -> CUDA)
// - Input validation
// - Stream management
// - Error handling
// ============================================================================

#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>

// Forward declaration of the CUDA kernel launch function
extern "C" void launch_flash_attention_inverted(
    const half* Q,
    const half* K,
    const half* V,
    half* O,
    float softmax_scale,
    int batch_size,
    int num_heads,
    int seq_len,
    bool is_causal,
    cudaStream_t stream
);

// Python-facing function
torch::Tensor flash_attention_inverted_forward(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    float softmax_scale,
    bool is_causal
) {
    // Input validation (following cookbook best practices)
    TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda(), 
                "All tensors must be on CUDA");
    TORCH_CHECK(q.dtype() == torch::kFloat16, 
                "Input tensors must be float16");
    TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4, 
                "Input tensors must be 4D (B, H, S, D) - PyTorch SDPA format");
    TORCH_CHECK(q.sizes() == k.sizes() && k.sizes() == v.sizes(), 
                "Input tensor shapes must match");
    TORCH_CHECK(q.is_contiguous() && k.is_contiguous() && v.is_contiguous(),
                "Input tensors must be contiguous");
    
    // Extract dimensions - FIX: PyTorch uses (B, H, S, D) not (B, S, H, D)!
    int batch_size = q.size(0);  // B
    int num_heads = q.size(1);   // H (not S!)
    int seq_len = q.size(2);     // S (not H!)
    int head_dim = q.size(3);    // D
    
    // Validate head_dim (current kernel only supports 64)
    TORCH_CHECK(head_dim == 64, 
                "Only HEAD_DIM=64 supported in current version");
    
    // Allocate output tensor
    torch::Tensor o = torch::empty_like(q);
    
    // Get CUDA stream (proper stream management)
    const at::cuda::OptionalCUDAGuard device_guard(device_of(q));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    
    // Launch kernel
    // Kernel signature: (Q, K, V, O, scale, batch, heads, seq, causal, stream)
    launch_flash_attention_inverted(
        reinterpret_cast<const half*>(q.data_ptr()),
        reinterpret_cast<const half*>(k.data_ptr()),
        reinterpret_cast<const half*>(v.data_ptr()),
        reinterpret_cast<half*>(o.data_ptr()),
        softmax_scale,
        batch_size,   // B
        num_heads,    // H  
        seq_len,      // S
        is_causal,
        stream
    );
    
    return o;
}

// PyBind11 module definition
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &flash_attention_inverted_forward, 
          "FlashAttention Inverted Forward (Production)",
          py::arg("q"),
          py::arg("k"),
          py::arg("v"),
          py::arg("softmax_scale"),
          py::arg("is_causal"));
}


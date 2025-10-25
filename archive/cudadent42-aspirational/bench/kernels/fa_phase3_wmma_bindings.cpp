// ============================================================================
// PYBIND11 BINDINGS FOR PHASE 3 (WMMA/TENSOR CORES)
// ============================================================================

#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>

extern "C" void launch_flash_attention_phase3(
    const half* Q,
    const half* K,
    const half* V,
    half* O,
    float softmax_scale,
    int batch_size,
    int num_heads,
    int seq_len,
    cudaStream_t stream
);

torch::Tensor flash_attention_phase3_forward(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    float softmax_scale
) {
    TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda(), 
                "All tensors must be on CUDA");
    TORCH_CHECK(q.dtype() == torch::kFloat16, 
                "Input tensors must be float16");
    TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4, 
                "Input tensors must be 4D (B, H, S, D)");
    TORCH_CHECK(q.sizes() == k.sizes() && k.sizes() == v.sizes(), 
                "Input tensor shapes must match");
    TORCH_CHECK(q.is_contiguous() && k.is_contiguous() && v.is_contiguous(),
                "Input tensors must be contiguous");
    
    int batch_size = q.size(0);
    int num_heads = q.size(1);
    int seq_len = q.size(2);
    int head_dim = q.size(3);
    
    TORCH_CHECK(head_dim == 64, 
                "Only HEAD_DIM=64 supported");
    
    torch::Tensor o = torch::empty_like(q);
    
    const at::cuda::OptionalCUDAGuard device_guard(device_of(q));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    
    launch_flash_attention_phase3(
        reinterpret_cast<const half*>(q.data_ptr()),
        reinterpret_cast<const half*>(k.data_ptr()),
        reinterpret_cast<const half*>(v.data_ptr()),
        reinterpret_cast<half*>(o.data_ptr()),
        softmax_scale,
        batch_size,
        num_heads,
        seq_len,
        stream
    );
    
    return o;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &flash_attention_phase3_forward, 
          "FlashAttention Phase 3 (Tensor Cores/WMMA)",
          py::arg("q"),
          py::arg("k"),
          py::arg("v"),
          py::arg("softmax_scale"));
}


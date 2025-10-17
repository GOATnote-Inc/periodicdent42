// PyBind11 bindings for Phase 6 (Aggressive Scalar Optimization)

#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>

extern "C" void launch_flash_attention_phase6(
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

torch::Tensor flash_attention_phase6_forward(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    float softmax_scale
) {
    TORCH_CHECK(q.is_cuda(), "Q must be CUDA tensor");
    TORCH_CHECK(k.is_cuda(), "K must be CUDA tensor");
    TORCH_CHECK(v.is_cuda(), "V must be CUDA tensor");
    TORCH_CHECK(q.dtype() == torch::kFloat16, "Q must be FP16");
    TORCH_CHECK(k.dtype() == torch::kFloat16, "K must be FP16");
    TORCH_CHECK(v.dtype() == torch::kFloat16, "V must be FP16");
    
    auto sizes = q.sizes();
    int batch_size = sizes[0];
    int num_heads = sizes[1];
    int seq_len = sizes[2];
    int head_dim = sizes[3];
    
    TORCH_CHECK(head_dim == 64, "Only HEAD_DIM=64 supported");
    
    auto o = torch::empty_like(q);
    
    c10::cuda::CUDAGuard device_guard(q.device());
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    
    launch_flash_attention_phase6(
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
    m.def("forward", &flash_attention_phase6_forward, 
          "FlashAttention Phase 6 (Aggressive Scalar Optimization)",
          py::arg("q"),
          py::arg("k"),
          py::arg("v"),
          py::arg("softmax_scale"));
}


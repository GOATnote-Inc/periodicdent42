// PyBind11 bindings for Phase 4 cuBLAS kernel

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>

extern "C" void launch_phase4_cublas(
    const half* Q,
    const half* K,
    const half* V,
    half* O,
    float* S_buffer,
    int B, int H, int M, int N, int D,
    float scale,
    cudaStream_t stream
);

torch::Tensor flash_attention_phase4_cublas_forward(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V,
    float scale
) {
    TORCH_CHECK(Q.is_cuda(), "Q must be on CUDA");
    TORCH_CHECK(K.is_cuda(), "K must be on CUDA");
    TORCH_CHECK(V.is_cuda(), "V must be on CUDA");
    TORCH_CHECK(Q.dtype() == torch::kFloat16, "Q must be FP16");
    TORCH_CHECK(K.dtype() == torch::kFloat16, "K must be FP16");
    TORCH_CHECK(V.dtype() == torch::kFloat16, "V must be FP16");
    
    auto sizes = Q.sizes();
    int B = sizes[0];
    int H = sizes[1];
    int M = sizes[2];
    int D = sizes[3];
    int N = K.size(2);
    
    auto O = torch::empty_like(Q);
    
    // Allocate workspace for Q@K^T scores (FP32)
    auto S_buffer = torch::empty({B, H, M, N}, 
                                 torch::TensorOptions()
                                     .dtype(torch::kFloat32)
                                     .device(Q.device()));
    
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    
    launch_phase4_cublas(
        reinterpret_cast<const half*>(Q.data_ptr()),
        reinterpret_cast<const half*>(K.data_ptr()),
        reinterpret_cast<const half*>(V.data_ptr()),
        reinterpret_cast<half*>(O.data_ptr()),
        S_buffer.data_ptr<float>(),
        B, H, M, N, D,
        scale,
        stream
    );
    
    return O;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &flash_attention_phase4_cublas_forward, 
          "FlashAttention Phase 4 with cuBLAS TensorCore Q@K^T");
}


#include <torch/extension.h>
#include <cuda_fp16.h>

// Forward declaration
void launch_flashcore_phase1_proven_wmma(
    const half* Q, const half* K, const half* V, half* O,
    float softmax_scale, int B, int H, int S, int D,
    cudaStream_t stream
);

// PyTorch wrapper
torch::Tensor flashcore_phase1_forward(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V,
    float softmax_scale
) {
    TORCH_CHECK(Q.is_cuda(), "Q must be CUDA tensor");
    TORCH_CHECK(K.is_cuda(), "K must be CUDA tensor");
    TORCH_CHECK(V.is_cuda(), "V must be CUDA tensor");
    TORCH_CHECK(Q.dtype() == torch::kFloat16, "Q must be FP16");
    TORCH_CHECK(K.dtype() == torch::kFloat16, "K must be FP16");
    TORCH_CHECK(V.dtype() == torch::kFloat16, "V must be FP16");
    
    const int B = Q.size(0);
    const int H = Q.size(1);
    const int S = Q.size(2);
    const int D = Q.size(3);
    
    auto O = torch::empty_like(Q);
    
    const half* Q_ptr = reinterpret_cast<const half*>(Q.data_ptr<at::Half>());
    const half* K_ptr = reinterpret_cast<const half*>(K.data_ptr<at::Half>());
    const half* V_ptr = reinterpret_cast<const half*>(V.data_ptr<at::Half>());
    half* O_ptr = reinterpret_cast<half*>(O.data_ptr<at::Half>());
    
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    
    launch_flashcore_phase1_proven_wmma(Q_ptr, K_ptr, V_ptr, O_ptr,
                                         softmax_scale, B, H, S, D, stream);
    
    return O;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &flashcore_phase1_forward, "FlashCore Phase 1 forward (CUDA)");
}


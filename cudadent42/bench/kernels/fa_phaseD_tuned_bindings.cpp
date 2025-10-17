// PyBind11 bindings for Phase D tuned kernel

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// Forward declare launcher
extern "C" void launch_phaseD_tuned(
    const half* Q,
    const half* K,
    const half* V,
    half* O,
    int B, int H, int M, int N, int D,
    float scale,
    cudaStream_t stream
);

torch::Tensor phaseD_attention_forward(
    torch::Tensor Q,  // [B, H, M, D]
    torch::Tensor K,  // [B, H, N, D]
    torch::Tensor V   // [B, H, N, D]
) {
    const int B = Q.size(0);
    const int H = Q.size(1);
    const int M = Q.size(2);
    const int D = Q.size(3);
    const int N = K.size(2);
    
    TORCH_CHECK(Q.is_cuda(), "Q must be CUDA tensor");
    TORCH_CHECK(K.is_cuda(), "K must be CUDA tensor");
    TORCH_CHECK(V.is_cuda(), "V must be CUDA tensor");
    TORCH_CHECK(Q.dtype() == torch::kFloat16, "Q must be FP16");
    TORCH_CHECK(K.dtype() == torch::kFloat16, "K must be FP16");
    TORCH_CHECK(V.dtype() == torch::kFloat16, "V must be FP16");
    
    auto O = torch::empty({B, H, M, D}, Q.options());
    
    float scale = 1.0f / sqrtf(static_cast<float>(D));
    
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    
    launch_phaseD_tuned(
        reinterpret_cast<const half*>(Q.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(K.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(V.data_ptr<at::Half>()),
        reinterpret_cast<half*>(O.data_ptr<at::Half>()),
        B, H, M, N, D,
        scale,
        stream
    );
    
    return O;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &phaseD_attention_forward, "Phase D tuned attention forward");
}


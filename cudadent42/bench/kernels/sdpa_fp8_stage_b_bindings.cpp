#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

extern "C" void launch_sdpa_fp8_stage_b(
    const void* Q,
    const void* K,
    const void* V,
    const float* Q_scale,
    const float* K_scale,
    const float* V_scale,
    half* O,
    int B, int H, int S, int D,
    float softmax_scale,
    cudaStream_t stream
);

torch::Tensor sdpa_fp8_stage_b_forward(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V,
    torch::Tensor Q_scale,
    torch::Tensor K_scale,
    torch::Tensor V_scale
) {
    const int B = Q.size(0);
    const int H = Q.size(1);
    const int S = Q.size(2);
    const int D = Q.size(3);
    
    auto O = torch::empty({B, H, S, D}, torch::dtype(torch::kFloat16).device(Q.device()));
    
    const float softmax_scale = 1.0f / sqrtf(static_cast<float>(D));
    
    const at::cuda::OptionalCUDAGuard device_guard(Q.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    
    launch_sdpa_fp8_stage_b(
        Q.data_ptr(),
        K.data_ptr(),
        V.data_ptr(),
        Q_scale.data_ptr<float>(),
        K_scale.data_ptr<float>(),
        V_scale.data_ptr<float>(),
        reinterpret_cast<half*>(O.data_ptr<at::Half>()),
        B, H, S, D,
        softmax_scale,
        stream
    );
    
    return O;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &sdpa_fp8_stage_b_forward, "FP8 SDPA Stage B (FP16 compute)");
}


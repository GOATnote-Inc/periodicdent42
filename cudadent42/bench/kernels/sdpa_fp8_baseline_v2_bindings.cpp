// PyBind11 bindings for FP8 SDPA baseline V2

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

extern "C" void launch_sdpa_fp8_baseline_v2(
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

torch::Tensor sdpa_fp8_v2_forward(
    torch::Tensor Q_fp8,
    torch::Tensor K_fp8,
    torch::Tensor V_fp8,
    torch::Tensor Q_scale,
    torch::Tensor K_scale,
    torch::Tensor V_scale
) {
    TORCH_CHECK(Q_fp8.is_cuda(), "Q must be CUDA tensor");
    TORCH_CHECK(Q_fp8.dtype() == torch::kUInt8, "Q must be uint8");
    TORCH_CHECK(Q_scale.dtype() == torch::kFloat32, "Scales must be float32");
    
    const int B = Q_fp8.size(0);
    const int H = Q_fp8.size(1);
    const int S = Q_fp8.size(2);
    const int D = Q_fp8.size(3);
    
    TORCH_CHECK(D == 64, "Only D=64 supported");
    
    auto O = torch::empty({B, H, S, D}, 
                          torch::TensorOptions()
                              .dtype(torch::kFloat16)
                              .device(Q_fp8.device()));
    
    float softmax_scale = 1.0f / sqrtf(static_cast<float>(D));
    
    launch_sdpa_fp8_baseline_v2(
        Q_fp8.data_ptr(),
        K_fp8.data_ptr(),
        V_fp8.data_ptr(),
        Q_scale.data_ptr<float>(),
        K_scale.data_ptr<float>(),
        V_scale.data_ptr<float>(),
        reinterpret_cast<half*>(O.data_ptr<at::Half>()),
        B, H, S, D,
        softmax_scale,
        0  // default stream
    );
    
    return O;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &sdpa_fp8_v2_forward, "FP8 SDPA forward (V2, fixed quant)");
}


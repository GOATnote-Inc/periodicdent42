#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

extern "C" void launch_sdpa_fp8_fixed(
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

torch::Tensor sdpa_fp8_fixed_forward(
    torch::Tensor Q,        // [B, H, S, D] uint8
    torch::Tensor K,        // [B, H, S, D] uint8
    torch::Tensor V,        // [B, H, S, D] uint8
    torch::Tensor Q_scale,  // [H] float32
    torch::Tensor K_scale,  // [H] float32
    torch::Tensor V_scale   // [H] float32
) {
    TORCH_CHECK(Q.dtype() == torch::kUInt8, "Q must be uint8");
    TORCH_CHECK(K.dtype() == torch::kUInt8, "K must be uint8");
    TORCH_CHECK(V.dtype() == torch::kUInt8, "V must be uint8");
    TORCH_CHECK(Q_scale.dtype() == torch::kFloat32, "Q_scale must be float32");
    TORCH_CHECK(K_scale.dtype() == torch::kFloat32, "K_scale must be float32");
    TORCH_CHECK(V_scale.dtype() == torch::kFloat32, "V_scale must be float32");
    
    TORCH_CHECK(Q.is_cuda(), "Q must be on CUDA");
    TORCH_CHECK(K.is_cuda(), "K must be on CUDA");
    TORCH_CHECK(V.is_cuda(), "V must be on CUDA");
    TORCH_CHECK(Q_scale.is_cuda(), "Q_scale must be on CUDA");
    TORCH_CHECK(K_scale.is_cuda(), "K_scale must be on CUDA");
    TORCH_CHECK(V_scale.is_cuda(), "V_scale must be on CUDA");
    
    const int B = Q.size(0);
    const int H = Q.size(1);
    const int S = Q.size(2);
    const int D = Q.size(3);
    
    TORCH_CHECK(Q_scale.size(0) == H, "Q_scale must have H elements");
    TORCH_CHECK(K_scale.size(0) == H, "K_scale must have H elements");
    TORCH_CHECK(V_scale.size(0) == H, "V_scale must have H elements");
    
    auto O = torch::empty({B, H, S, D}, torch::dtype(torch::kFloat16).device(Q.device()));
    
    const float softmax_scale = 1.0f / sqrtf(static_cast<float>(D));
    
    const at::cuda::OptionalCUDAGuard device_guard(Q.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    
    launch_sdpa_fp8_fixed(
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
    m.def("forward", &sdpa_fp8_fixed_forward, "FP8 SDPA forward (FIXED)");
}


#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <cmath>

extern "C" void launch_sdpa_fp8_stage_c_wmma(
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

namespace {
void validate_inputs(const torch::Tensor& tensor, const char* name) {
    TORCH_CHECK(
        tensor.device().is_cuda(),
        name, " must be a CUDA tensor"
    );
    TORCH_CHECK(
        tensor.is_contiguous(),
        name, " must be contiguous"
    );
}
}  // namespace

torch::Tensor sdpa_fp8_stage_c_wmma_forward(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V,
    torch::Tensor Q_scale,
    torch::Tensor K_scale,
    torch::Tensor V_scale,
    double softmax_scale
) {
    validate_inputs(Q, "Q");
    validate_inputs(K, "K");
    validate_inputs(V, "V");

    TORCH_CHECK(
        Q.dtype() == torch::kUInt8,
        "Q must be uint8 (simulated FP8)"
    );
    TORCH_CHECK(K.dtype() == torch::kUInt8, "K must be uint8 (simulated FP8)");
    TORCH_CHECK(V.dtype() == torch::kUInt8, "V must be uint8 (simulated FP8)");

    TORCH_CHECK(Q.dim() == 4, "Q must have shape [B, H, S, D]");
    TORCH_CHECK(Q.sizes() == K.sizes(), "K must match Q shape");
    TORCH_CHECK(Q.sizes() == V.sizes(), "V must match Q shape");

    const auto sizes = Q.sizes();
    const int64_t B = sizes[0];
    const int64_t H = sizes[1];
    const int64_t S = sizes[2];
    const int64_t D = sizes[3];

    TORCH_CHECK(D == 64, "HEAD_DIM of 64 is required for this kernel");

    TORCH_CHECK(
        Q_scale.dtype() == torch::kFloat32,
        "Q_scale must be float32"
    );
    TORCH_CHECK(K_scale.dtype() == torch::kFloat32, "K_scale must be float32");
    TORCH_CHECK(V_scale.dtype() == torch::kFloat32, "V_scale must be float32");

    TORCH_CHECK(Q_scale.numel() == H, "Q_scale must have one entry per head");
    TORCH_CHECK(K_scale.numel() == H, "K_scale must have one entry per head");
    TORCH_CHECK(V_scale.numel() == H, "V_scale must have one entry per head");

    auto O = torch::empty(
        {B, H, S, D},
        torch::dtype(torch::kFloat16).device(Q.device())
    );

    const at::cuda::OptionalCUDAGuard device_guard(Q.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    launch_sdpa_fp8_stage_c_wmma(
        Q.data_ptr(),
        K.data_ptr(),
        V.data_ptr(),
        Q_scale.data_ptr<float>(),
        K_scale.data_ptr<float>(),
        V_scale.data_ptr<float>(),
        reinterpret_cast<half*>(O.data_ptr<at::Half>()),
        static_cast<int>(B),
        static_cast<int>(H),
        static_cast<int>(S),
        static_cast<int>(D),
        static_cast<float>(softmax_scale),
        stream
    );

    return O;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "forward",
        &sdpa_fp8_stage_c_wmma_forward,
        "FP8 SDPA Stage C (WMMA tensor core pipeline)",
        py::arg("Q"),
        py::arg("K"),
        py::arg("V"),
        py::arg("Q_scale"),
        py::arg("K_scale"),
        py::arg("V_scale"),
        py::arg("softmax_scale")
    );
}


#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

extern "C" void launch_test_qkt_fixed(
    const void* Q, const void* K,
    float Q_scale, float K_scale,
    half* S, int M, int N, int D,
    cudaStream_t stream, bool apply_inv_sqrt_d
);

torch::Tensor test_qkt_fixed_forward(
    torch::Tensor Q_fp8,
    torch::Tensor K_fp8,
    float Q_scale,
    float K_scale,
    bool apply_inv_sqrt_d
) {
    TORCH_CHECK(Q_fp8.is_cuda(), "Q must be CUDA");
    TORCH_CHECK(Q_fp8.dtype() == torch::kUInt8, "Q must be uint8");
    
    const int M = Q_fp8.size(0);
    const int D = Q_fp8.size(1);
    const int N = K_fp8.size(0);
    
    auto S = torch::empty({M, N},
                          torch::TensorOptions()
                              .dtype(torch::kFloat16)
                              .device(Q_fp8.device()));
    
    launch_test_qkt_fixed(
        Q_fp8.data_ptr(),
        K_fp8.data_ptr(),
        Q_scale,
        K_scale,
        reinterpret_cast<half*>(S.data_ptr<at::Half>()),
        M, N, D,
        0,
        apply_inv_sqrt_d
    );
    
    return S;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &test_qkt_fixed_forward, "Test Q@K^T with FIXED indexing");
}


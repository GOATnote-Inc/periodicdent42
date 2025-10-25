#include <torch/extension.h>
#include <cuda_fp16.h>

// Forward declaration (matches flashcore_fused_wmma.cu)
void launch_flashcore_fused_wmma(
    const half* Q,
    const half* K,
    const half* V,
    half* O,
    int B, int H, int S, int D
);

void flashcore_fused_forward(
    torch::Tensor Q,  // [B, H, S, D]
    torch::Tensor K,  // [B, H, S, D]
    torch::Tensor V,  // [B, H, S, D]
    torch::Tensor O   // [B, H, S, D] - output (pre-allocated)
) {
    TORCH_CHECK(Q.is_cuda(), "Q must be on CUDA");
    TORCH_CHECK(K.is_cuda(), "K must be on CUDA");
    TORCH_CHECK(V.is_cuda(), "V must be on CUDA");
    TORCH_CHECK(O.is_cuda(), "O must be on CUDA");
    TORCH_CHECK(Q.dtype() == torch::kFloat16, "Q must be FP16");
    TORCH_CHECK(K.dtype() == torch::kFloat16, "K must be FP16");
    TORCH_CHECK(V.dtype() == torch::kFloat16, "V must be FP16");
    TORCH_CHECK(O.dtype() == torch::kFloat16, "O must be FP16");
    
    const int B = Q.size(0);
    const int H = Q.size(1);
    const int S = Q.size(2);
    const int D = Q.size(3);
    
    TORCH_CHECK(D == 64, "Only D=64 supported for now");
    TORCH_CHECK(O.size(0) == B && O.size(1) == H && O.size(2) == S && O.size(3) == D,
                "Output shape must match input");
    
    // Launch kernel
    launch_flashcore_fused_wmma(
        reinterpret_cast<const half*>(Q.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(K.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(V.data_ptr<at::Half>()),
        reinterpret_cast<half*>(O.data_ptr<at::Half>()),
        B, H, S, D
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &flashcore_fused_forward, "FlashCore Fused WMMA Forward (in-place)");
}


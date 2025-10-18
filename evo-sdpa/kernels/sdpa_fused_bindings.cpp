#include <torch/extension.h>
#include "runtime.hpp"

// Python wrapper for sdpa_fused_forward
torch::Tensor sdpa_fused_forward_py(
    torch::Tensor Q,     // [B, H, L, d]
    torch::Tensor K,     // [B, H, L, d]
    torch::Tensor V,     // [B, H, L, d]
    torch::Tensor O,     // [B, H, L, d] (pre-allocated output)
    bool causal,
    float scale
) {
    TORCH_CHECK(Q.is_cuda(), "Q must be CUDA tensor");
    TORCH_CHECK(K.is_cuda(), "K must be CUDA tensor");
    TORCH_CHECK(V.is_cuda(), "V must be CUDA tensor");
    TORCH_CHECK(O.is_cuda(), "O must be CUDA tensor");
    
    TORCH_CHECK(Q.dim() == 4, "Q must be 4D");
    TORCH_CHECK(K.dim() == 4, "K must be 4D");
    TORCH_CHECK(V.dim() == 4, "V must be 4D");
    
    const int B = Q.size(0);
    const int H = Q.size(1);
    const int L = Q.size(2);
    const int d = Q.size(3);
    
    SdpaParams params;
    params.Q = Q.data_ptr();
    params.K = K.data_ptr();
    params.V = V.data_ptr();
    params.O = O.data_ptr();
    params.B = B;
    params.H = H;
    params.L = L;
    params.d = d;
    params.scale = scale;
    params.causal = causal;
    
    // Use V2c-v6 kernel (Full WMMA pipeline: Q@K^T + P@V)
    cudaError_t err = sdpa_fused_forward_v2c_v6(params);
    TORCH_CHECK(err == cudaSuccess, "CUDA error: ", cudaGetErrorString(err));
    
    return O;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sdpa_fused_forward", &sdpa_fused_forward_py, "Fused SDPA forward pass");
}


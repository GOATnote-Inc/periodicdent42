// CRITICAL: Define BEFORE any includes
#define CUDA_NO_BFLOAT16
#define __CUDA_NO_BFLOAT16_OPERATORS__

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include "flash_attention_dispatch.h"

torch::Tensor flash_attention_fwd(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V
) {
    // === Multi-GPU Safety ===
    TORCH_CHECK(Q.is_cuda(), "Q must be CUDA tensor");
    TORCH_CHECK(K.is_cuda(), "K must be CUDA tensor");
    TORCH_CHECK(V.is_cuda(), "V must be CUDA tensor");
    
    // Set CUDA device guard (multi-GPU safety)
    at::cuda::CUDAGuard device_guard(Q.get_device());
    
    // === Tensor Layout Checks (early, cheap) ===
    TORCH_CHECK(Q.is_contiguous() && K.is_contiguous() && V.is_contiguous(),
                "Q/K/V must be contiguous tensors (call .contiguous() if needed)");
    
    TORCH_CHECK(Q.dim() == 2 && K.dim() == 2 && V.dim() == 2,
                "Expected 2D tensors [M,D], [N,D], [N,D]; got Q=", Q.sizes(),
                " K=", K.sizes(), " V=", V.sizes());
    
    // === Dtype Consistency ===
    TORCH_CHECK(Q.dtype() == K.dtype() && K.dtype() == V.dtype(),
                "Q/K/V dtype mismatch: Q=", Q.dtype(), " K=", K.dtype(), " V=", V.dtype());
    
    // === Shape Consistency ===
    TORCH_CHECK(Q.size(1) == K.size(1) && K.size(0) == V.size(0) && K.size(1) == V.size(1),
                "Shape mismatch: Q=[", Q.size(0), ",", Q.size(1), "], "
                "K=[", K.size(0), ",", K.size(1), "], "
                "V=[", V.size(0), ",", V.size(1), "]");
    
    // === Allocate Output ===
    auto O = torch::empty_like(Q);
    
    // === Dtype Dispatch ===
    int dtype_id;
    if (Q.dtype() == torch::kHalf) {
        dtype_id = 0;
    }
#ifdef FLASHMOE_HAS_BF16
    else if (Q.dtype() == torch::kBFloat16) {
        dtype_id = 1;
    }
#endif
    else {
        TORCH_CHECK(false,
                    "Unsupported dtype: ", Q.dtype(),
                    " (built with FLASHMOE_HAS_BF16=",
#ifdef FLASHMOE_HAS_BF16
                    "1"
#else
                    "0"
#endif
                    ")");
    }
    
    // === Launch Kernel ===
    flashmoe::flash_attention_dispatch(
        Q.data_ptr(),
        K.data_ptr(),
        V.data_ptr(),
        O.data_ptr(),
        Q.size(0),  // M
        K.size(0),  // N
        Q.size(1),  // K_dim
        128,        // tile_size (TODO: make configurable)
        dtype_id,
        at::cuda::getCurrentCUDAStream()
    );
    
    return O;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &flash_attention_fwd, "FlashAttention forward (production-hardened)",
          py::arg("Q"), py::arg("K"), py::arg("V"));
}


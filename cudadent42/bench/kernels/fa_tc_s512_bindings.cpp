// ============================================================================
// PyBind11 Bindings for fa_tc_s512 Tensor Core Kernel
// ============================================================================

#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include "../runtime/tensor_contract.hpp"

// Forward declarations
extern "C" cudaError_t launch_fa_tc_s512_64_64_2(
    const half* Q, const half* K, const half* V, half* O,
    float softmax_scale, int B, int H, int S, bool is_causal, cudaStream_t stream
);

extern "C" cudaError_t launch_fa_tc_s512_128_64_2(
    const half* Q, const half* K, const half* V, half* O,
    float softmax_scale, int B, int H, int S, bool is_causal, cudaStream_t stream
);

torch::Tensor flash_attention_tc_s512_forward(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    float softmax_scale,
    bool is_causal,
    int config_id  // 1=64x64, 2=128x64
) {
    // Contract validation
    cudadent42::runtime::assert_qkv_contract(q, k, v);
    
    int B = q.size(0);
    int H = q.size(1);
    int S = q.size(2);
    int D = q.size(3);
    
    TORCH_CHECK(D == 64, "TC kernel only supports HEAD_DIM=64 (got ", D, ")");
    TORCH_CHECK(S == 512, "TC kernel only supports S=512 (got ", S, ")");
    
    // Allocate output
    torch::Tensor o = torch::empty_like(q);
    
    // Get stream
    const at::cuda::OptionalCUDAGuard device_guard(device_of(q));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    
    // Launch appropriate config
    cudaError_t err;
    switch (config_id) {
        case 1:  // 64x64
            err = launch_fa_tc_s512_64_64_2(
                (half*)q.data_ptr(), (half*)k.data_ptr(), (half*)v.data_ptr(), (half*)o.data_ptr(),
                softmax_scale, B, H, S, is_causal, stream
            );
            break;
        case 2:  // 128x64
            err = launch_fa_tc_s512_128_64_2(
                (half*)q.data_ptr(), (half*)k.data_ptr(), (half*)v.data_ptr(), (half*)o.data_ptr(),
                softmax_scale, B, H, S, is_causal, stream
            );
            break;
        default:
            TORCH_CHECK(false, "Invalid config_id: ", config_id, " (expected 1 or 2)");
    }
    
    TORCH_CHECK(err == cudaSuccess, "TC kernel launch failed: ", cudaGetErrorString(err));
    
    return o;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &flash_attention_tc_s512_forward, "FlashAttention TC S=512 forward (CUDA)");
}


// ============================================================================
// PyBind11 Bindings for fa_s512_v3 Kernel
// ============================================================================

#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include "../runtime/tensor_contract.hpp"

// Forward declarations for template instantiations
// We'll pre-compile a few promising configs to avoid JIT explosion

// Config 1: BLOCK_M=32, BLOCK_N=64, WARPS=4, STAGES=2, SWIZZLE=1, HALF2=1
extern "C" cudaError_t launch_fa_s512_v3_32_64_4_2_1_1(
    const half* Q, const half* K, const half* V, half* O,
    float softmax_scale, int B, int H, int S, bool is_causal, cudaStream_t stream
);

// Config 2: BLOCK_M=32, BLOCK_N=32, WARPS=4, STAGES=2, SWIZZLE=1, HALF2=1
extern "C" cudaError_t launch_fa_s512_v3_32_32_4_2_1_1(
    const half* Q, const half* K, const half* V, half* O,
    float softmax_scale, int B, int H, int S, bool is_causal, cudaStream_t stream
);

// Config 3: BLOCK_M=48, BLOCK_N=64, WARPS=8, STAGES=2, SWIZZLE=1, HALF2=1
extern "C" cudaError_t launch_fa_s512_v3_48_64_8_2_1_1(
    const half* Q, const half* K, const half* V, half* O,
    float softmax_scale, int B, int H, int S, bool is_causal, cudaStream_t stream
);

// Config 4: BLOCK_M=32, BLOCK_N=64, WARPS=4, STAGES=1, SWIZZLE=1, HALF2=1
extern "C" cudaError_t launch_fa_s512_v3_32_64_4_1_1_1(
    const half* Q, const half* K, const half* V, half* O,
    float softmax_scale, int B, int H, int S, bool is_causal, cudaStream_t stream
);

// Config 5: BLOCK_M=16, BLOCK_N=64, WARPS=4, STAGES=2, SWIZZLE=1, HALF2=1
extern "C" cudaError_t launch_fa_s512_v3_16_64_4_2_1_1(
    const half* Q, const half* K, const half* V, half* O,
    float softmax_scale, int B, int H, int S, bool is_causal, cudaStream_t stream
);

torch::Tensor flash_attention_s512_v3_forward(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    float softmax_scale,
    bool is_causal,
    int config_id  // Select which template instantiation
) {
    // ========================================================================
    // CONTRACT VALIDATION: (B, H, S, D) contiguous layout
    // ========================================================================
    cudadent42::runtime::assert_qkv_contract(q, k, v);
    
    int B = q.size(0);
    int H = q.size(1);
    int S = q.size(2);
    int D = q.size(3);
    
    TORCH_CHECK(D == 64, "Only HEAD_DIM=64 supported in S=512 kernel");
    TORCH_CHECK(S == 512, "This kernel is specialized for S=512");
    
    // Allocate output
    torch::Tensor o = torch::empty_like(q);
    
    // Get stream
    const at::cuda::OptionalCUDAGuard device_guard(device_of(q));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    
    // Launch appropriate config
    cudaError_t err;
    switch (config_id) {
        case 1:
            err = launch_fa_s512_v3_32_64_4_2_1_1(
                (half*)q.data_ptr(), (half*)k.data_ptr(), (half*)v.data_ptr(), (half*)o.data_ptr(),
                softmax_scale, B, H, S, is_causal, stream
            );
            break;
        case 2:
            err = launch_fa_s512_v3_32_32_4_2_1_1(
                (half*)q.data_ptr(), (half*)k.data_ptr(), (half*)v.data_ptr(), (half*)o.data_ptr(),
                softmax_scale, B, H, S, is_causal, stream
            );
            break;
        case 3:
            err = launch_fa_s512_v3_48_64_8_2_1_1(
                (half*)q.data_ptr(), (half*)k.data_ptr(), (half*)v.data_ptr(), (half*)o.data_ptr(),
                softmax_scale, B, H, S, is_causal, stream
            );
            break;
        case 4:
            err = launch_fa_s512_v3_32_64_4_1_1_1(
                (half*)q.data_ptr(), (half*)k.data_ptr(), (half*)v.data_ptr(), (half*)o.data_ptr(),
                softmax_scale, B, H, S, is_causal, stream
            );
            break;
        case 5:
            err = launch_fa_s512_v3_16_64_4_2_1_1(
                (half*)q.data_ptr(), (half*)k.data_ptr(), (half*)v.data_ptr(), (half*)o.data_ptr(),
                softmax_scale, B, H, S, is_causal, stream
            );
            break;
        default:
            TORCH_CHECK(false, "Invalid config_id: " + std::to_string(config_id));
    }
    
    TORCH_CHECK(err == cudaSuccess, "Kernel launch failed: " + std::string(cudaGetErrorString(err)));
    
    // Synchronize and check for runtime errors
    err = cudaStreamSynchronize(stream);
    TORCH_CHECK(err == cudaSuccess, "Kernel runtime failed: " + std::string(cudaGetErrorString(err)));
    
    return o;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &flash_attention_s512_v3_forward, "FlashAttention S=512 V3 forward (CUDA)");
}

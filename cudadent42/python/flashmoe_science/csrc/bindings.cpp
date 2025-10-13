/**
 * PyTorch C++ bindings for FlashMoE-Science CUDA kernels.
 * 
 * Provides Python interface to CUDA kernels via torch.utils.cpp_extension.
 * 
 * @author GOATnote Autonomous Research Lab Initiative
 * @date 2025-10-11
 */

#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>

// Forward declarations of CUDA kernels
namespace flashmoe {

template<typename T>
void flash_attention_forward(
    const T* Q, const T* K, const T* V,
    T* O, float* softmax_lse,
    const int batch_size, const int num_heads,
    const int seq_len, const int head_dim,
    const float softmax_scale, const bool causal
);

// DISABLED: Split-K has linking issues
#if 0
template<typename T>
void flash_attention_forward_split_k(
    const T* Q, const T* K, const T* V,
    T* O, float* softmax_lse,
    const int batch_size, const int num_heads,
    const int seq_len, const int head_dim,
    const float softmax_scale, const bool causal
);
#endif

}  // namespace flashmoe

/**
 * FlashAttention forward pass (Python interface).
 */
torch::Tensor flash_attention_forward_cuda(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V,
    bool causal,
    float softmax_scale
) {
    // Set CUDA device
    c10::cuda::CUDAGuard device_guard(Q.device());
    
    // Validate inputs
    TORCH_CHECK(Q.is_cuda(), "Q must be on CUDA device");
    TORCH_CHECK(K.is_cuda(), "K must be on CUDA device");
    TORCH_CHECK(V.is_cuda(), "V must be on CUDA device");
    TORCH_CHECK(Q.dtype() == K.dtype() && K.dtype() == V.dtype(),
                "Q, K, V must have same dtype");
    TORCH_CHECK(Q.is_contiguous() && K.is_contiguous() && V.is_contiguous(),
                "Q, K, V must be contiguous");
    
    // Get dimensions
    const int batch_size = Q.size(0);
    const int num_heads = Q.size(1);
    const int seq_len = Q.size(2);
    const int head_dim = Q.size(3);
    
    TORCH_CHECK(K.size(0) == batch_size && V.size(0) == batch_size,
                "Batch size mismatch");
    TORCH_CHECK(K.size(1) == num_heads && V.size(1) == num_heads,
                "Number of heads mismatch");
    TORCH_CHECK(K.size(2) == seq_len && V.size(2) == seq_len,
                "Sequence length mismatch");
    TORCH_CHECK(K.size(3) == head_dim && V.size(3) == head_dim,
                "Head dimension mismatch");
    
    // Allocate output
    auto O = torch::empty_like(Q);
    
    // Allocate softmax LSE for backward pass
    auto softmax_lse = torch::empty({batch_size, num_heads, seq_len},
                                     torch::dtype(torch::kFloat32).device(Q.device()));
    
    // Dispatch based on dtype
#if !defined(FLASHMOE_DTYPE_FP16_ONLY)
    if (Q.dtype() == torch::kBFloat16) {
        flashmoe::flash_attention_forward<at::BFloat16>(
            reinterpret_cast<const at::BFloat16*>(Q.data_ptr()),
            reinterpret_cast<const at::BFloat16*>(K.data_ptr()),
            reinterpret_cast<const at::BFloat16*>(V.data_ptr()),
            reinterpret_cast<at::BFloat16*>(O.data_ptr()),
            softmax_lse.data_ptr<float>(),
            batch_size, num_heads, seq_len, head_dim,
            softmax_scale, causal
        );
    } else
#endif
    if (Q.dtype() == torch::kFloat16) {
        flashmoe::flash_attention_forward<at::Half>(
            reinterpret_cast<const at::Half*>(Q.data_ptr()),
            reinterpret_cast<const at::Half*>(K.data_ptr()),
            reinterpret_cast<const at::Half*>(V.data_ptr()),
            reinterpret_cast<at::Half*>(O.data_ptr()),
            softmax_lse.data_ptr<float>(),
            batch_size, num_heads, seq_len, head_dim,
            softmax_scale, causal
        );
    } else {
        TORCH_CHECK(false, "Unsupported dtype (only FP16 and BF16 supported)");
    }
    
    return O;
}

// DISABLED: Split-K has linking issues
#if 0
/**
 * FlashAttention-2 Split-K forward pass (Python interface).
 * 
 * This version parallelizes across both query and K/V tiles for better
 * SM utilization and reduced memory traffic.
 */
torch::Tensor flash_attention_forward_split_k_cuda(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V,
    bool causal,
    float softmax_scale
) {
    // Set CUDA device
    c10::cuda::CUDAGuard device_guard(Q.device());
    
    // Validate inputs
    TORCH_CHECK(Q.is_cuda(), "Q must be on CUDA device");
    TORCH_CHECK(K.is_cuda(), "K must be on CUDA device");
    TORCH_CHECK(V.is_cuda(), "V must be on CUDA device");
    TORCH_CHECK(Q.dtype() == K.dtype() && K.dtype() == V.dtype(),
                "Q, K, V must have same dtype");
    TORCH_CHECK(Q.is_contiguous() && K.is_contiguous() && V.is_contiguous(),
                "Q, K, V must be contiguous");
    
    // Get dimensions
    const int batch_size = Q.size(0);
    const int num_heads = Q.size(1);
    const int seq_len = Q.size(2);
    const int head_dim = Q.size(3);
    
    TORCH_CHECK(K.size(0) == batch_size && V.size(0) == batch_size,
                "Batch size mismatch");
    TORCH_CHECK(K.size(1) == num_heads && V.size(1) == num_heads,
                "Number of heads mismatch");
    TORCH_CHECK(K.size(2) == seq_len && V.size(2) == seq_len,
                "Sequence length mismatch");
    TORCH_CHECK(K.size(3) == head_dim && V.size(3) == head_dim,
                "Head dimension mismatch");
    
    // Allocate output
    auto O = torch::empty_like(Q);
    
    // Allocate softmax LSE for backward pass
    auto softmax_lse = torch::empty({batch_size, num_heads, seq_len},
                                     torch::dtype(torch::kFloat32).device(Q.device()));
    
    // Dispatch based on dtype
#if !defined(FLASHMOE_DTYPE_FP16_ONLY)
    if (Q.dtype() == torch::kBFloat16) {
        flashmoe::flash_attention_forward_split_k<at::BFloat16>(
            reinterpret_cast<const at::BFloat16*>(Q.data_ptr()),
            reinterpret_cast<const at::BFloat16*>(K.data_ptr()),
            reinterpret_cast<const at::BFloat16*>(V.data_ptr()),
            reinterpret_cast<at::BFloat16*>(O.data_ptr()),
            softmax_lse.data_ptr<float>(),
            batch_size, num_heads, seq_len, head_dim,
            softmax_scale, causal
        );
    } else
#endif
    if (Q.dtype() == torch::kFloat16) {
        flashmoe::flash_attention_forward_split_k<at::Half>(
            reinterpret_cast<const at::Half*>(Q.data_ptr()),
            reinterpret_cast<const at::Half*>(K.data_ptr()),
            reinterpret_cast<const at::Half*>(V.data_ptr()),
            reinterpret_cast<at::Half*>(O.data_ptr()),
            softmax_lse.data_ptr<float>(),
            batch_size, num_heads, seq_len, head_dim,
            softmax_scale, causal
        );
    } else {
        TORCH_CHECK(false, "Unsupported dtype (only FP16 and BF16 supported)");
    }
    
    return O;
}
#endif  // End of disabled split_k code

// DISABLED: Backward pass not implemented yet
#if 0
/**
 * FlashAttention backward pass (Python interface).
 */
std::vector<torch::Tensor> flash_attention_backward_cuda(
    torch::Tensor grad_output,
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V,
    torch::Tensor O,
    torch::Tensor softmax_lse,
    bool causal,
    float softmax_scale
) {
    // Set CUDA device
    c10::cuda::CUDAGuard device_guard(Q.device());
    
    // Get dimensions
    const int batch_size = Q.size(0);
    const int num_heads = Q.size(1);
    const int seq_len = Q.size(2);
    const int head_dim = Q.size(3);
    
    // Allocate gradients
    auto dQ = torch::empty_like(Q);
    auto dK = torch::empty_like(K);
    auto dV = torch::empty_like(V);
    
    // Dispatch based on dtype
#if !defined(FLASHMOE_DTYPE_FP16_ONLY)
    if (Q.dtype() == torch::kBFloat16) {
        flashmoe::flash_attention_backward<at::BFloat16>(
            reinterpret_cast<const at::BFloat16*>(grad_output.data_ptr()),
            reinterpret_cast<const at::BFloat16*>(Q.data_ptr()),
            reinterpret_cast<const at::BFloat16*>(K.data_ptr()),
            reinterpret_cast<const at::BFloat16*>(V.data_ptr()),
            reinterpret_cast<const at::BFloat16*>(O.data_ptr()),
            softmax_lse.data_ptr<float>(),
            reinterpret_cast<at::BFloat16*>(dQ.data_ptr()),
            reinterpret_cast<at::BFloat16*>(dK.data_ptr()),
            reinterpret_cast<at::BFloat16*>(dV.data_ptr()),
            batch_size, num_heads, seq_len, head_dim,
            softmax_scale, causal
        );
    } else
#endif
    if (Q.dtype() == torch::kFloat16) {
        flashmoe::flash_attention_backward<at::Half>(
            reinterpret_cast<const at::Half*>(grad_output.data_ptr()),
            reinterpret_cast<const at::Half*>(Q.data_ptr()),
            reinterpret_cast<const at::Half*>(K.data_ptr()),
            reinterpret_cast<const at::Half*>(V.data_ptr()),
            reinterpret_cast<const at::Half*>(O.data_ptr()),
            softmax_lse.data_ptr<float>(),
            reinterpret_cast<at::Half*>(dQ.data_ptr()),
            reinterpret_cast<at::Half*>(dK.data_ptr()),
            reinterpret_cast<at::Half*>(dV.data_ptr()),
            batch_size, num_heads, seq_len, head_dim,
            softmax_scale, causal
        );
    }
    
    return {dQ, dK, dV};
}
#endif  // End of disabled backward

// DISABLED: Warp-specialized kernel never properly implemented
// Commenting out to fix compilation error
#if 0
/**
 * FlashAttention warp-specialized forward pass (Python interface).
 * 
 * This version uses FlashAttention-4 style warp specialization for better
 * parallelism. Intended for benchmarking and production use.
 */
torch::Tensor flash_attention_warp_specialized_cuda(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V,
    bool causal,
    float softmax_scale
) {
    // Set CUDA device
    c10::cuda::CUDAGuard device_guard(Q.device());
    
    // Validate inputs
    TORCH_CHECK(Q.is_cuda(), "Q must be on CUDA device");
    TORCH_CHECK(K.is_cuda(), "K must be on CUDA device");
    TORCH_CHECK(V.is_cuda(), "V must be on CUDA device");
    TORCH_CHECK(Q.dtype() == K.dtype() && K.dtype() == V.dtype(),
                "Q, K, V must have same dtype");
    TORCH_CHECK(Q.is_contiguous() && K.is_contiguous() && V.is_contiguous(),
                "Q, K, V must be contiguous");
    
    // Get dimensions
    const int batch_size = Q.size(0);
    const int num_heads = Q.size(1);
    const int seq_len = Q.size(2);
    const int head_dim = Q.size(3);
    
    TORCH_CHECK(K.size(0) == batch_size && V.size(0) == batch_size,
                "Batch size mismatch");
    TORCH_CHECK(K.size(1) == num_heads && V.size(1) == num_heads,
                "Number of heads mismatch");
    TORCH_CHECK(K.size(2) == seq_len && V.size(2) == seq_len,
                "Sequence length mismatch");
    TORCH_CHECK(K.size(3) == head_dim && V.size(3) == head_dim,
                "Head dimension mismatch");
    
    // Allocate output
    auto O = torch::empty_like(Q);
    
    // Allocate softmax LSE for backward pass
    auto softmax_lse = torch::empty({batch_size, num_heads, seq_len},
                                     torch::dtype(torch::kFloat32).device(Q.device()));
    
    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream(Q.device().index());
    
    // Dispatch based on dtype
#if !defined(FLASHMOE_DTYPE_FP16_ONLY)
    if (Q.dtype() == torch::kBFloat16) {
        flashmoe::flash_attention_warp_specialized_launch<at::BFloat16>(
            reinterpret_cast<const at::BFloat16*>(Q.data_ptr()),
            reinterpret_cast<const at::BFloat16*>(K.data_ptr()),
            reinterpret_cast<const at::BFloat16*>(V.data_ptr()),
            reinterpret_cast<at::BFloat16*>(O.data_ptr()),
            softmax_lse.data_ptr<float>(),
            batch_size, num_heads, seq_len, head_dim,
            softmax_scale, causal, stream
        );
    } else
#endif
    if (Q.dtype() == torch::kFloat16) {
        flashmoe::flash_attention_warp_specialized_launch<at::Half>(
            reinterpret_cast<const at::Half*>(Q.data_ptr()),
            reinterpret_cast<const at::Half*>(K.data_ptr()),
            reinterpret_cast<const at::Half*>(V.data_ptr()),
            reinterpret_cast<at::Half*>(O.data_ptr()),
            softmax_lse.data_ptr<float>(),
            batch_size, num_heads, seq_len, head_dim,
            softmax_scale, causal, stream
        );
    } else {
        TORCH_CHECK(false, "Unsupported dtype (only FP16 and BF16 supported)");
    }
    
    return O;
}
#endif  // End of disabled warp-specialized code

/**
 * Fused MoE forward pass (Python interface) - stub.
 */
torch::Tensor fused_moe_forward_cuda(
    torch::Tensor tokens,
    torch::Tensor expert_weights,
    torch::Tensor routing_weights,
    int top_k
) {
    // TODO: Implement full MoE kernel binding
    // For now, return input unchanged
    return tokens;
}

// Python module definition
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("flash_attention_forward", &flash_attention_forward_cuda,
          "FlashAttention-Science forward pass");
    // DISABLED: split_k has linking issues
    // m.def("flash_attention_forward_split_k", &flash_attention_forward_split_k_cuda,
    //       "FlashAttention-2 Split-K forward pass (parallel K/V tiles)");
    // DISABLED: warp_specialized never implemented
    // m.def("flash_attention_warp_specialized", &flash_attention_warp_specialized_cuda,
    //       "FlashAttention-Science warp-specialized forward pass (Phase 1)");
    // DISABLED: backward not implemented yet
    // m.def("flash_attention_backward", &flash_attention_backward_cuda,
    //       "FlashAttention-Science backward pass");
    m.def("fused_moe_forward", &fused_moe_forward_cuda,
          "Fused MoE forward pass");
}


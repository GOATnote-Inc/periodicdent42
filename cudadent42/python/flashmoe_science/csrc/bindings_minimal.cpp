/**
 * Minimal PyTorch C++ bindings for FlashAttention (Split-K version).
 * 
 * Only includes:
 * - flash_attention_forward (FA-1 style, sequential K/V)
 * - flash_attention_forward_split_k (FA-2 style, parallel K/V)
 */

#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>

// Forward declarations
namespace flashmoe {

template<typename T>
void flash_attention_forward(
    const T* Q, const T* K, const T* V,
    T* O, float* softmax_lse,
    const int batch_size, const int num_heads,
    const int seq_len, const int head_dim,
    const float softmax_scale, const bool causal
);

template<typename T>
void flash_attention_forward_split_k(
    const T* Q, const T* K, const T* V,
    T* O, float* softmax_lse,
    const int batch_size, const int num_heads,
    const int seq_len, const int head_dim,
    const float softmax_scale, const bool causal
);

}  // namespace flashmoe

/**
 * FlashAttention forward pass (FA-1 style).
 */
torch::Tensor flash_attention_forward_cuda(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V,
    bool causal,
    float softmax_scale
) {
    c10::cuda::CUDAGuard device_guard(Q.device());
    
    TORCH_CHECK(Q.is_cuda() && K.is_cuda() && V.is_cuda(), "Inputs must be on CUDA");
    TORCH_CHECK(Q.dtype() == K.dtype() && K.dtype() == V.dtype(), "Dtype mismatch");
    TORCH_CHECK(Q.is_contiguous() && K.is_contiguous() && V.is_contiguous(), "Inputs must be contiguous");
    
    const int batch_size = Q.size(0);
    const int num_heads = Q.size(1);
    const int seq_len = Q.size(2);
    const int head_dim = Q.size(3);
    
    auto O = torch::empty_like(Q);
    auto softmax_lse = torch::empty({batch_size, num_heads, seq_len},
                                     torch::dtype(torch::kFloat32).device(Q.device()));
    
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
    } else if (Q.dtype() == torch::kFloat16) {
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
        TORCH_CHECK(false, "Unsupported dtype (only FP16 and BF16)");
    }
    
    return O;
}

/**
 * FlashAttention-2 Split-K forward pass (FA-2 style).
 */
torch::Tensor flash_attention_forward_split_k_cuda(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V,
    bool causal,
    float softmax_scale
) {
    c10::cuda::CUDAGuard device_guard(Q.device());
    
    TORCH_CHECK(Q.is_cuda() && K.is_cuda() && V.is_cuda(), "Inputs must be on CUDA");
    TORCH_CHECK(Q.dtype() == K.dtype() && K.dtype() == V.dtype(), "Dtype mismatch");
    TORCH_CHECK(Q.is_contiguous() && K.is_contiguous() && V.is_contiguous(), "Inputs must be contiguous");
    
    const int batch_size = Q.size(0);
    const int num_heads = Q.size(1);
    const int seq_len = Q.size(2);
    const int head_dim = Q.size(3);
    
    auto O = torch::empty_like(Q);
    auto softmax_lse = torch::empty({batch_size, num_heads, seq_len},
                                     torch::dtype(torch::kFloat32).device(Q.device()));
    
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
    } else if (Q.dtype() == torch::kFloat16) {
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
        TORCH_CHECK(false, "Unsupported dtype (only FP16 and BF16)");
    }
    
    return O;
}

// Python module definition
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("flash_attention_forward", &flash_attention_forward_cuda,
          "FlashAttention forward pass (FA-1 style, sequential K/V)");
    m.def("flash_attention_forward_split_k", &flash_attention_forward_split_k_cuda,
          "FlashAttention-2 Split-K forward pass (FA-2 style, parallel K/V)");
}


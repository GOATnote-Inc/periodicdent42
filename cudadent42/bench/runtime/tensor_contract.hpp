#pragma once

#include <torch/extension.h>
#include <stdexcept>
#include <sstream>

namespace cudadent42 {
namespace runtime {

// ============================================================================
// Tensor Contract: (B, H, S, D) Contiguous Layout
// ============================================================================

/**
 * @brief Validate tensor layout matches PyTorch (B, H, S, D) contiguous format
 * 
 * Expected strides: [H*S*D, S*D, D, 1]
 * Expected dtype: torch.float16 (half)
 * Expected device: CUDA
 * 
 * @param tensor Input tensor to validate
 * @param name Tensor name for error messages
 * @throws std::runtime_error if contract violated
 */
inline void assert_bhsd_contract(const torch::Tensor& tensor, const char* name) {
    // Check device
    if (!tensor.is_cuda()) {
        std::ostringstream oss;
        oss << name << " must be on CUDA device, got " << tensor.device();
        throw std::runtime_error(oss.str());
    }
    
    // Check dtype
    if (tensor.dtype() != torch::kFloat16) {
        std::ostringstream oss;
        oss << name << " must be FP16, got " << tensor.dtype();
        throw std::runtime_error(oss.str());
    }
    
    // Check rank
    if (tensor.dim() != 4) {
        std::ostringstream oss;
        oss << name << " must have 4 dimensions (B,H,S,D), got " << tensor.dim();
        throw std::runtime_error(oss.str());
    }
    
    // Extract dimensions
    int64_t B = tensor.size(0);
    int64_t H = tensor.size(1);
    int64_t S = tensor.size(2);
    int64_t D = tensor.size(3);
    
    // Check D is multiple of 8 (half2 vectorization safety)
    if (D % 8 != 0) {
        std::ostringstream oss;
        oss << name << " head dimension D=" << D << " must be multiple of 8";
        throw std::runtime_error(oss.str());
    }
    
    // Check contiguous with expected strides: [H*S*D, S*D, D, 1]
    auto strides = tensor.strides();
    int64_t expected_stride_b = H * S * D;
    int64_t expected_stride_h = S * D;
    int64_t expected_stride_s = D;
    int64_t expected_stride_d = 1;
    
    if (strides[0] != expected_stride_b ||
        strides[1] != expected_stride_h ||
        strides[2] != expected_stride_s ||
        strides[3] != expected_stride_d) {
        std::ostringstream oss;
        oss << name << " strides must be [H*S*D, S*D, D, 1] = ["
            << expected_stride_b << ", " << expected_stride_h << ", "
            << expected_stride_s << ", " << expected_stride_d << "], got ["
            << strides[0] << ", " << strides[1] << ", "
            << strides[2] << ", " << strides[3] << "]. "
            << "Tensor may not be contiguous in (B,H,S,D) layout. "
            << "Call .contiguous() before passing to kernel.";
        throw std::runtime_error(oss.str());
    }
}

/**
 * @brief Validate Q, K, V tensors have matching shapes and layouts
 * 
 * @param Q Query tensor (B, H, S, D)
 * @param K Key tensor (B, H, S, D)
 * @param V Value tensor (B, H, S, D)
 * @throws std::runtime_error if contract violated
 */
inline void assert_qkv_contract(
    const torch::Tensor& Q,
    const torch::Tensor& K,
    const torch::Tensor& V
) {
    // Validate individual tensors
    assert_bhsd_contract(Q, "Q");
    assert_bhsd_contract(K, "K");
    assert_bhsd_contract(V, "V");
    
    // Check shapes match
    if (Q.sizes() != K.sizes() || Q.sizes() != V.sizes()) {
        std::ostringstream oss;
        oss << "Q, K, V must have identical shapes, got Q=" << Q.sizes()
            << ", K=" << K.sizes() << ", V=" << V.sizes();
        throw std::runtime_error(oss.str());
    }
}

/**
 * @brief Validate output tensor matches input shape and layout
 * 
 * @param O Output tensor (B, H, S, D)
 * @param Q Query tensor (reference for shape)
 * @throws std::runtime_error if contract violated
 */
inline void assert_output_contract(
    const torch::Tensor& O,
    const torch::Tensor& Q
) {
    assert_bhsd_contract(O, "O");
    
    if (O.sizes() != Q.sizes()) {
        std::ostringstream oss;
        oss << "O must have same shape as Q, got O=" << O.sizes()
            << ", Q=" << Q.sizes();
        throw std::runtime_error(oss.str());
    }
}

} // namespace runtime
} // namespace cudadent42


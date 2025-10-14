/**
 * PyTorch C++ bindings for fa_s512 kernel (S=512 specialized FlashAttention)
 * 
 * Minimal bindings for pre-compiled extension (bypasses JIT timeout).
 * 
 * @author GOATnote Autonomous Research Lab Initiative
 * @date 2025-10-14
 */

#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>

// Forward declaration of CUDA kernel
// Note: Actual kernel implementation is in ../cudadent42/bench/kernels/fa_s512.cu
extern "C" void fa_s512_launch(
    const at::Half* Q,
    const at::Half* K,
    const at::Half* V,
    at::Half* O,
    int B,
    int H,
    int S,
    int D,
    float softmax_scale,
    cudaStream_t stream
);

/**
 * Python-facing function: fa_s512(Q, K, V) -> O
 * 
 * Args:
 *   Q: Query tensor [B, H, S, D] (FP16)
 *   K: Key tensor [B, H, S, D] (FP16)
 *   V: Value tensor [B, H, S, D] (FP16)
 * 
 * Returns:
 *   O: Output tensor [B, H, S, D] (FP16)
 * 
 * Note: S must be 512 (compile-time specialized)
 */
torch::Tensor fa_s512_forward(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V
) {
    // Input validation
    TORCH_CHECK(Q.device().is_cuda(), "Q must be on CUDA");
    TORCH_CHECK(K.device().is_cuda(), "K must be on CUDA");
    TORCH_CHECK(V.device().is_cuda(), "V must be on CUDA");
    
    TORCH_CHECK(Q.dtype() == torch::kFloat16, "Q must be FP16");
    TORCH_CHECK(K.dtype() == torch::kFloat16, "K must be FP16");
    TORCH_CHECK(V.dtype() == torch::kFloat16, "V must be FP16");
    
    TORCH_CHECK(Q.dim() == 4, "Q must be 4D [B, H, S, D]");
    TORCH_CHECK(K.dim() == 4, "K must be 4D [B, H, S, D]");
    TORCH_CHECK(V.dim() == 4, "V must be 4D [B, H, S, D]");
    
    auto B = Q.size(0);
    auto H = Q.size(1);
    auto S = Q.size(2);
    auto D = Q.size(3);
    
    TORCH_CHECK(S == 512, "fa_s512 is specialized for S=512 only");
    TORCH_CHECK(D == 64, "fa_s512 is specialized for D=64 only");
    
    TORCH_CHECK(K.size(0) == B && K.size(1) == H && K.size(2) == S && K.size(3) == D, 
                "K shape mismatch");
    TORCH_CHECK(V.size(0) == B && V.size(1) == H && V.size(2) == S && V.size(3) == D, 
                "V shape mismatch");
    
    // Ensure contiguous
    Q = Q.contiguous();
    K = K.contiguous();
    V = V.contiguous();
    
    // Allocate output
    auto O = torch::empty_like(Q);
    
    // CUDA stream
    c10::cuda::CUDAGuard device_guard(Q.device());
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    
    // Softmax scale (1 / sqrt(D))
    float softmax_scale = 1.0f / std::sqrt(static_cast<float>(D));
    
    // Launch kernel
    fa_s512_launch(
        reinterpret_cast<const at::Half*>(Q.data_ptr<at::Half>()),
        reinterpret_cast<const at::Half*>(K.data_ptr<at::Half>()),
        reinterpret_cast<const at::Half*>(V.data_ptr<at::Half>()),
        reinterpret_cast<at::Half*>(O.data_ptr<at::Half>()),
        static_cast<int>(B),
        static_cast<int>(H),
        static_cast<int>(S),
        static_cast<int>(D),
        softmax_scale,
        stream
    );
    
    // Check for CUDA errors
    auto cuda_err = cudaGetLastError();
    TORCH_CHECK(cuda_err == cudaSuccess, 
                "fa_s512 kernel failed: ", cudaGetErrorString(cuda_err));
    
    return O;
}

// PyBind11 module definition
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fa_s512", &fa_s512_forward, 
          "FlashAttention forward (S=512 specialized, FP16)",
          py::arg("Q"),
          py::arg("K"),
          py::arg("V"));
}


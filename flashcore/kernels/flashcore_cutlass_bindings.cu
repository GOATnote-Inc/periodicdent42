/*
 * FlashCore: PyTorch C++ Bindings for CUTLASS FMHA
 */

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include "cutlass/half.h"

// Forward declaration
void launch_cutlass_fmha(
    const cutlass::half_t* Q,
    const cutlass::half_t* K,
    const cutlass::half_t* V,
    cutlass::half_t* O,
    int B, int H, int S, int D,
    cudaStream_t stream
);

/*
 * PyTorch wrapper
 * Input: Q, K, V are [B, H, S, D] tensors (float16)
 * Output: O is [B, H, S, D] tensor (float16)
 */
torch::Tensor fmha_cutlass_forward(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V
) {
    // Check device
    TORCH_CHECK(Q.is_cuda(), "Q must be CUDA tensor");
    TORCH_CHECK(K.is_cuda(), "K must be CUDA tensor");
    TORCH_CHECK(V.is_cuda(), "V must be CUDA tensor");
    
    // Check dtype
    TORCH_CHECK(Q.dtype() == torch::kFloat16, "Q must be float16");
    TORCH_CHECK(K.dtype() == torch::kFloat16, "K must be float16");
    TORCH_CHECK(V.dtype() == torch::kFloat16, "V must be float16");
    
    // Check shape
    TORCH_CHECK(Q.dim() == 4, "Q must be 4D [B, H, S, D]");
    TORCH_CHECK(K.dim() == 4, "K must be 4D [B, H, S, D]");
    TORCH_CHECK(V.dim() == 4, "V must be 4D [B, H, S, D]");
    TORCH_CHECK(Q.sizes() == K.sizes(), "Q and K must have same shape");
    TORCH_CHECK(Q.sizes() == V.sizes(), "Q and V must have same shape");
    
    // Check contiguous
    TORCH_CHECK(Q.is_contiguous(), "Q must be contiguous");
    TORCH_CHECK(K.is_contiguous(), "K must be contiguous");
    TORCH_CHECK(V.is_contiguous(), "V must be contiguous");
    
    // Extract dimensions
    int B = Q.size(0);
    int H = Q.size(1);
    int S = Q.size(2);
    int D = Q.size(3);
    
    // CUTLASS expects [B, S, H, D] layout (BMHK)
    // PyTorch gives us [B, H, S, D] layout (BHSD)
    // Need to permute: [B, H, S, D] -> [B, S, H, D]
    auto Q_bmhk = Q.permute({0, 2, 1, 3}).contiguous();
    auto K_bmhk = K.permute({0, 2, 1, 3}).contiguous();
    auto V_bmhk = V.permute({0, 2, 1, 3}).contiguous();
    
    // Allocate output in BMHK format
    auto O_bmhk = torch::empty({B, S, H, D}, Q.options());
    
    // Get CUDA stream
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    
    // Cast to cutlass::half_t* (compatible with __half*)
    const cutlass::half_t* Q_ptr = reinterpret_cast<const cutlass::half_t*>(Q_bmhk.data_ptr<at::Half>());
    const cutlass::half_t* K_ptr = reinterpret_cast<const cutlass::half_t*>(K_bmhk.data_ptr<at::Half>());
    const cutlass::half_t* V_ptr = reinterpret_cast<const cutlass::half_t*>(V_bmhk.data_ptr<at::Half>());
    cutlass::half_t* O_ptr = reinterpret_cast<cutlass::half_t*>(O_bmhk.data_ptr<at::Half>());
    
    // Launch kernel
    launch_cutlass_fmha(Q_ptr, K_ptr, V_ptr, O_ptr, B, H, S, D, stream);
    
    // Permute output back to PyTorch format: [B, S, H, D] -> [B, H, S, D]
    auto O = O_bmhk.permute({0, 2, 1, 3}).contiguous();
    
    return O;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fmha", &fmha_cutlass_forward, "CUTLASS FMHA forward");
}


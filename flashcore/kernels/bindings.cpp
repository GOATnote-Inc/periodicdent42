// ============================================================================
// FlashCore Baseline PyTorch C++ Bindings
// ============================================================================
// Wraps flashcore_baseline.cu kernel for PyTorch usage
//
// Usage:
//   import flashcore_baseline
//   O = flashcore_baseline.forward(Q, K, V, scale)
// ============================================================================

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <vector>

// Forward declaration of CUDA kernel (defined in flashcore_baseline.cu)
__global__ void flash_attention_minimal_kernel(
    const half* __restrict__ Q,
    const half* __restrict__ K,
    const half* __restrict__ V,
    half* __restrict__ O,
    float softmax_scale,
    int batch_size,
    int num_heads,
    int seq_len
);

// ============================================================================
// Helper Functions
// ============================================================================

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_FP16(x) TORCH_CHECK(x.dtype() == torch::kFloat16, #x " must be FP16")
#define CHECK_DIM(x, d) TORCH_CHECK(x.dim() == d, #x " must be " #d "D tensor")

void check_tensor(const torch::Tensor& t, const char* name, int expected_dim) {
    CHECK_CUDA(t);
    CHECK_CONTIGUOUS(t);
    CHECK_FP16(t);
    CHECK_DIM(t, expected_dim);
}

// ============================================================================
// Forward Pass
// ============================================================================

torch::Tensor flashcore_baseline_forward(
    torch::Tensor Q,  // [B, H, S, D]
    torch::Tensor K,  // [B, H, S, D]
    torch::Tensor V,  // [B, H, S, D]
    float scale
) {
    // Input validation
    check_tensor(Q, "Q", 4);
    check_tensor(K, "K", 4);
    check_tensor(V, "V", 4);
    
    const auto B = Q.size(0);  // Batch size
    const auto H = Q.size(1);  // Num heads
    const auto S = Q.size(2);  // Sequence length
    const auto D = Q.size(3);  // Head dimension
    
    // Validate shapes match
    TORCH_CHECK(K.size(0) == B && K.size(1) == H && K.size(2) == S && K.size(3) == D,
                "K shape mismatch");
    TORCH_CHECK(V.size(0) == B && V.size(1) == H && V.size(2) == S && V.size(3) == D,
                "V shape mismatch");
    
    // Currently only support D=64 (hardcoded in kernel)
    TORCH_CHECK(D == 64, "Only D=64 supported in baseline kernel");
    
    // Allocate output
    auto O = torch::empty_like(Q);
    
    // Launch configuration
    // Grid: (S, H, B) - one block per query row per head per batch
    // Block: 128 threads
    dim3 grid(S, H, B);
    dim3 block(128);
    
    // Launch kernel
    flash_attention_minimal_kernel<<<grid, block>>>(
        reinterpret_cast<const half*>(Q.data_ptr()),
        reinterpret_cast<const half*>(K.data_ptr()),
        reinterpret_cast<const half*>(V.data_ptr()),
        reinterpret_cast<half*>(O.data_ptr()),
        scale,
        static_cast<int>(B),
        static_cast<int>(H),
        static_cast<int>(S)
    );
    
    // Check for launch errors
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess,
                "Kernel launch failed: ", cudaGetErrorString(err));
    
    // Synchronize (optional for debugging)
    // cudaDeviceSynchronize();
    
    return O;
}

// ============================================================================
// Python Module Definition
// ============================================================================

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &flashcore_baseline_forward,
          "FlashCore baseline forward pass (CUDA)",
          py::arg("Q"),
          py::arg("K"),
          py::arg("V"),
          py::arg("scale"));
    
    m.doc() = "FlashCore baseline attention kernel\n\n"
              "Args:\n"
              "    Q (Tensor): Query tensor [B, H, S, D], FP16\n"
              "    K (Tensor): Key tensor [B, H, S, D], FP16\n"
              "    V (Tensor): Value tensor [B, H, S, D], FP16\n"
              "    scale (float): Softmax scale factor (typically 1/sqrt(D))\n"
              "\n"
              "Returns:\n"
              "    O (Tensor): Output tensor [B, H, S, D], FP16\n"
              "\n"
              "Note: Currently only supports D=64 (hardcoded in kernel)";
}


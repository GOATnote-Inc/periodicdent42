// ============================================================================
// BlackwellSparseK: PyTorch C++ Extension Bindings
// ============================================================================
// Provides Python interface to CUDA kernels via pybind11.
// ============================================================================

#include <torch/extension.h>
#include <cuda_fp16.h>
#include <vector>
#include <string>

// Forward declaration from kernel_dispatch.cu
void attention_forward(
    const half* Q,
    const half* K,
    const half* V,
    half* O,
    float softmax_scale,
    int B, int H, int S, int D,
    cudaStream_t stream
);

// ============================================================================
// INPUT VALIDATION
// ============================================================================

void validate_attention_inputs(
    const torch::Tensor& Q,
    const torch::Tensor& K,
    const torch::Tensor& V
) {
    // Check device
    TORCH_CHECK(Q.is_cuda(), "Q must be a CUDA tensor");
    TORCH_CHECK(K.is_cuda(), "K must be a CUDA tensor");
    TORCH_CHECK(V.is_cuda(), "V must be a CUDA tensor");
    
    // Check dtype
    TORCH_CHECK(Q.scalar_type() == torch::kFloat16, "Q must be FP16 (torch.float16)");
    TORCH_CHECK(K.scalar_type() == torch::kFloat16, "K must be FP16 (torch.float16)");
    TORCH_CHECK(V.scalar_type() == torch::kFloat16, "V must be FP16 (torch.float16)");
    
    // Check dimensions
    TORCH_CHECK(Q.dim() == 4, "Q must be 4D [B, H, S, D]");
    TORCH_CHECK(K.dim() == 4, "K must be 4D [B, H, S, D]");
    TORCH_CHECK(V.dim() == 4, "V must be 4D [B, H, S, D]");
    
    // Check shapes match
    TORCH_CHECK(Q.sizes() == K.sizes(), "Q and K must have the same shape");
    TORCH_CHECK(Q.sizes() == V.sizes(), "Q and V must have the same shape");
    
    // Check head dimension
    int64_t D = Q.size(3);
    TORCH_CHECK(D == 64 || D == 128, 
                "Head dimension must be 64 or 128, got ", D);
    
    // Check contiguity
    TORCH_CHECK(Q.is_contiguous(), "Q must be contiguous");
    TORCH_CHECK(K.is_contiguous(), "K must be contiguous");
    TORCH_CHECK(V.is_contiguous(), "V must be contiguous");
}

// ============================================================================
// PYTORCH INTERFACE
// ============================================================================

torch::Tensor attention_forward_cuda(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V,
    double softmax_scale
) {
    // Validate inputs
    validate_attention_inputs(Q, K, V);
    
    // Get dimensions
    const int64_t B = Q.size(0);
    const int64_t H = Q.size(1);
    const int64_t S = Q.size(2);
    const int64_t D = Q.size(3);
    
    // Allocate output tensor
    auto O = torch::empty_like(Q);
    
    // Get CUDA stream
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    
    // Call CUDA kernel
    attention_forward(
        reinterpret_cast<const half*>(Q.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(K.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(V.data_ptr<at::Half>()),
        reinterpret_cast<half*>(O.data_ptr<at::Half>()),
        static_cast<float>(softmax_scale),
        static_cast<int>(B),
        static_cast<int>(H),
        static_cast<int>(S),
        static_cast<int>(D),
        stream
    );
    
    return O;
}

// ============================================================================
// VERSION INFO
// ============================================================================

std::string get_version() {
    return "0.1.0";
}

std::string get_cuda_arch() {
    int device;
    cudaGetDevice(&device);
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    
    return "sm_" + std::to_string(prop.major) + std::to_string(prop.minor);
}

std::vector<std::string> get_supported_archs() {
    return {"sm_90a", "sm_100"};
}

// ============================================================================
// PYBIND11 MODULE DEFINITION
// ============================================================================

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "BlackwellSparseK: Production CUDA kernels for Blackwell sparse attention";
    
    // Main kernel function
    m.def(
        "attention_forward",
        &attention_forward_cuda,
        "Compute scaled dot-product attention using BlackwellSparseK kernels",
        py::arg("Q"),
        py::arg("K"),
        py::arg("V"),
        py::arg("softmax_scale")
    );
    
    // Version and info functions
    m.def("version", &get_version, "Get BlackwellSparseK version");
    m.def("cuda_arch", &get_cuda_arch, "Get current CUDA architecture");
    m.def("supported_archs", &get_supported_archs, "Get supported CUDA architectures");
    
    // Module version attribute
    m.attr("__version__") = "0.1.0";
}


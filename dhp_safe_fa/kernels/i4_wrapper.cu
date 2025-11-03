// PyTorch C++ Extension for I4 Kernel
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// Include I4 kernel
extern __global__ void dhp_i4_fused_softmax_pv(
    const __half* __restrict__ scores,
    const __half* __restrict__ V,
    __half* __restrict__ out,
    const uint32_t S_max,
    const uint32_t S_actual,
    const uint32_t d,
    const uint32_t batch_size
);

torch::Tensor i4_fused_softmax_pv_forward(
    torch::Tensor scores,  // [B*H, S, S]
    torch::Tensor V,       // [B*H, S, d]
    int S_max,
    int S_actual
) {
    // Input validation
    TORCH_CHECK(scores.is_cuda(), "scores must be CUDA tensor");
    TORCH_CHECK(V.is_cuda(), "V must be CUDA tensor");
    TORCH_CHECK(scores.dtype() == torch::kFloat16, "scores must be FP16");
    TORCH_CHECK(V.dtype() == torch::kFloat16, "V must be FP16");
    
    auto batch_size = scores.size(0);
    auto d = V.size(2);
    
    // Allocate output
    auto out = torch::empty({batch_size, S_max, d}, 
                           torch::TensorOptions()
                               .dtype(torch::kFloat16)
                               .device(scores.device()));
    
    // Launch kernel
    const int threads = 256;
    const int blocks = (batch_size * S_max + threads - 1) / threads;
    
    dhp_i4_fused_softmax_pv<<<blocks, threads>>>(
        reinterpret_cast<const __half*>(scores.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(V.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(out.data_ptr<at::Half>()),
        S_max,
        S_actual,
        d,
        batch_size
    );
    
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &i4_fused_softmax_pv_forward, "I4 Fused Softmax+PV");
}


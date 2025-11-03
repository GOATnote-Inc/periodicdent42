// PyTorch C++ Extension for I5 Kernel
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// Include I5 kernel
extern __global__ void dhp_i5_warp_cooperative(
    const __half* __restrict__ scores,
    const __half* __restrict__ V,
    __half* __restrict__ out,
    const uint32_t S_max,
    const uint32_t S_actual,
    const uint32_t d,
    const uint32_t batch_size
);

torch::Tensor i5_warp_cooperative_forward(
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
    
    // Launch kernel - 256 threads per block (8 warps)
    const int threads = 256;
    const int blocks = (batch_size * S_max + threads - 1) / threads;
    
    dhp_i5_warp_cooperative<<<blocks, threads>>>(
        reinterpret_cast<const __half*>(scores.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(V.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(out.data_ptr<at::Half>()),
        S_max,
        S_actual,
        d,
        batch_size
    );
    
    // Check for errors
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "I5 kernel launch failed: ", cudaGetErrorString(err));
    
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &i5_warp_cooperative_forward, "I5 Warp-Cooperative Softmax+PV");
}


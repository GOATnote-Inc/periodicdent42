#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

extern __global__ void dhp_i8_warp_optimized(
    const __half* __restrict__ Q,
    const __half* __restrict__ K,
    const __half* __restrict__ V,
    __half* __restrict__ out,
    const uint32_t S_max,
    const uint32_t S_actual,
    const uint32_t batch_size
);

torch::Tensor i8_forward(torch::Tensor Q, torch::Tensor K, torch::Tensor V, int S_max, int S_actual) {
    TORCH_CHECK(Q.is_cuda() && K.is_cuda() && V.is_cuda());
    TORCH_CHECK(Q.dtype() == torch::kFloat16);
    
    auto batch_size = Q.size(0);
    auto d = Q.size(2);
    TORCH_CHECK(d == 64, "I8 requires d=64");
    
    auto out = torch::empty_like(Q);
    
    dim3 grid(batch_size, (S_max + 255) / 256);
    dim3 block(256);
    
    dhp_i8_warp_optimized<<<grid, block>>>(
        reinterpret_cast<const __half*>(Q.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(K.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(V.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(out.data_ptr<at::Half>()),
        S_max, S_actual, batch_size
    );
    
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "I8 kernel failed: ", cudaGetErrorString(err));
    
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &i8_forward, "I8 Warp-Optimized Deterministic Attention");
}


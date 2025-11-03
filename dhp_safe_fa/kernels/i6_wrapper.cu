// PyTorch C++ Extension for I6 Kernel
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// Include I6 kernel
extern __global__ void dhp_i6_block_parallel(
    const __half* __restrict__ Q,
    const __half* __restrict__ K,
    const __half* __restrict__ V,
    __half* __restrict__ out,
    const uint32_t S_max,
    const uint32_t S_actual,
    const uint32_t d,
    const uint32_t batch_size
);

torch::Tensor i6_block_parallel_forward(
    torch::Tensor Q,       // [B*H, S, d]
    torch::Tensor K,       // [B*H, S, d]
    torch::Tensor V,       // [B*H, S, d]
    int S_max,
    int S_actual
) {
    // Input validation
    TORCH_CHECK(Q.is_cuda(), "Q must be CUDA tensor");
    TORCH_CHECK(K.is_cuda(), "K must be CUDA tensor");
    TORCH_CHECK(V.is_cuda(), "V must be CUDA tensor");
    TORCH_CHECK(Q.dtype() == torch::kFloat16, "Q must be FP16");
    TORCH_CHECK(K.dtype() == torch::kFloat16, "K must be FP16");
    TORCH_CHECK(V.dtype() == torch::kFloat16, "V must be FP16");
    
    auto batch_size = Q.size(0);
    auto d = Q.size(2);
    
    // Allocate output
    auto out = torch::empty({batch_size, S_max, d}, 
                           torch::TensorOptions()
                               .dtype(torch::kFloat16)
                               .device(Q.device()));
    
    // Launch configuration
    // Grid: (B*H, ceil(S_max/BM))
    // Block: 128 threads (4 warps)
    constexpr int BM = 64;
    constexpr int THREADS = 128;
    
    const int num_tiles = (S_max + BM - 1) / BM;
    
    dim3 grid(batch_size, num_tiles);
    dim3 block(THREADS);
    
    dhp_i6_block_parallel<<<grid, block>>>(
        reinterpret_cast<const __half*>(Q.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(K.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(V.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(out.data_ptr<at::Half>()),
        S_max,
        S_actual,
        d,
        batch_size
    );
    
    // Check for errors
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "I6 kernel launch failed: ", cudaGetErrorString(err));
    
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &i6_block_parallel_forward, "I6 Block-Parallel Attention");
}


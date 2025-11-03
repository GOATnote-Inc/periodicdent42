#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

extern "C" void init_cublas();

extern "C" void launch_i9_cublas(
    const __half* Q,
    const __half* K,
    const __half* V,
    __half* scores_tmp,
    __half* out,
    int batch_size,
    int S_max,
    int S_actual,
    cudaStream_t stream
);

torch::Tensor i9_forward(torch::Tensor Q, torch::Tensor K, torch::Tensor V, int S_max, int S_actual) {
    TORCH_CHECK(Q.is_cuda() && K.is_cuda() && V.is_cuda());
    TORCH_CHECK(Q.dtype() == torch::kFloat16);
    
    auto batch_size = Q.size(0);
    auto d = Q.size(2);
    TORCH_CHECK(d == 64, "I9 requires d=64");
    
    // Initialize cuBLAS handle once
    static bool initialized = false;
    if (!initialized) {
        init_cublas();
        initialized = true;
    }
    
    // Allocate temporary scores buffer
    auto scores = torch::empty({batch_size, S_max, S_max}, 
                               torch::TensorOptions()
                                   .dtype(torch::kFloat16)
                                   .device(Q.device()));
    
    auto out = torch::empty_like(Q);
    
    // Get CUDA stream (use default stream for now)
    cudaStream_t stream = 0;
    
    // Launch cuBLAS-based attention
    launch_i9_cublas(
        reinterpret_cast<const __half*>(Q.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(K.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(V.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(scores.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(out.data_ptr<at::Half>()),
        batch_size, S_max, S_actual, stream
    );
    
    // Check for errors
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "I9 cuBLAS attention failed: ", cudaGetErrorString(err));
    
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &i9_forward, "I9 cuBLAS-accelerated attention");
}

#include <torch/extension.h>
#include <cuda_fp16.h>

// Forward declaration
void launch_flashcore_fused_wmma_cpasync(
    const half* Q, const half* K, const half* V, half* O,
    int B, int H, int S, int D, cudaStream_t stream
);

torch::Tensor flashcore_cpasync_forward(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V,
    torch::Tensor O
) {
    const int B = Q.size(0);
    const int H = Q.size(1);
    const int S = Q.size(2);
    const int D = Q.size(3);
    
    launch_flashcore_fused_wmma_cpasync(
        reinterpret_cast<const half*>(Q.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(K.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(V.data_ptr<at::Half>()),
        reinterpret_cast<half*>(O.data_ptr<at::Half>()),
        B, H, S, D,
        0  // default stream
    );
    
    return O;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &flashcore_cpasync_forward, "FlashCore cp.async forward");
}


#include <torch/extension.h>
#include <cuda_fp16.h>

// Forward declaration
void launch_flash_attention_wmma(
    const half* Q, const half* K, const half* V, half* O,
    float softmax_scale, int B, int H, int S,
    cudaStream_t stream
);

torch::Tensor flashcore_wmma_forward(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V,
    float scale
) {
    const int B = Q.size(0);
    const int H = Q.size(1);
    const int S = Q.size(2);
    const int D = Q.size(3);
    
    auto O = torch::empty_like(Q);
    
    launch_flash_attention_wmma(
        reinterpret_cast<const half*>(Q.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(K.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(V.data_ptr<at::Half>()),
        reinterpret_cast<half*>(O.data_ptr<at::Half>()),
        scale, B, H, S,
        0  // default stream
    );
    
    return O;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &flashcore_wmma_forward, "FlashCore WMMA forward");
}


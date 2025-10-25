// PyTorch bindings for simplified FA-3 kernel

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

void launch_flash3_simple(
    const half* Q, const half* K, const half* V, half* O,
    int B, int H, int S, int D,
    cudaStream_t stream
);

torch::Tensor flash3_simple_forward(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V
) {
    TORCH_CHECK(Q.is_cuda(), "Q must be CUDA");
    TORCH_CHECK(K.is_cuda(), "K must be CUDA");
    TORCH_CHECK(V.is_cuda(), "V must be CUDA");
    TORCH_CHECK(Q.dtype() == torch::kFloat16, "Q must be float16");
    TORCH_CHECK(K.dtype() == torch::kFloat16, "K must be float16");
    TORCH_CHECK(V.dtype() == torch::kFloat16, "V must be float16");
    TORCH_CHECK(Q.dim() == 4, "Q must be 4D [B,H,S,D]");
    TORCH_CHECK(Q.is_contiguous(), "Q must be contiguous");
    TORCH_CHECK(K.is_contiguous(), "K must be contiguous");
    TORCH_CHECK(V.is_contiguous(), "V must be contiguous");

    int B = Q.size(0), H = Q.size(1), S = Q.size(2), D = Q.size(3);
    TORCH_CHECK(D % 32 == 0, "D must be divisible by 32");
    TORCH_CHECK(D <= 128, "D must be <= 128");

    auto O = torch::empty_like(Q);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    launch_flash3_simple(
        reinterpret_cast<const half*>(Q.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(K.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(V.data_ptr<at::Half>()),
        reinterpret_cast<half*>(O.data_ptr<at::Half>()),
        B, H, S, D, stream
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "Kernel failed: ", cudaGetErrorString(err));

    return O;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &flash3_simple_forward, "FA-3 Simple forward");
}


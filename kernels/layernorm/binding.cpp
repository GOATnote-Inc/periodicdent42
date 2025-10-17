#include <torch/extension.h>
#include <cuda_fp16.h>
#include <c10/cuda/CUDAStream.h>

extern "C" void layernorm_forward_launcher(
    const half*, half*, const half*, const half*, int, int, cudaStream_t, int, int, int, int);

torch::Tensor layernorm_forward(torch::Tensor x, c10::optional<torch::Tensor> gamma, c10::optional<torch::Tensor> beta,
                                int threads, int rows_per_cta, int vec_width, int use_warp){
    TORCH_CHECK(x.is_cuda(), "x must be CUDA");
    TORCH_CHECK(x.dtype()==torch::kFloat16, "x fp16 required");
    auto sizes = x.sizes(); // [B,H,S,D]
    int B=sizes[0], H=sizes[1], S=sizes[2], D=sizes[3];
    int R=B*H*S;
    auto y = torch::empty_like(x);
    half* g = gamma.has_value()? (half*)gamma->data_ptr() : nullptr;
    half* b = beta.has_value()?  (half*)beta->data_ptr()  : nullptr;
    auto stream = c10::cuda::getCurrentCUDAStream();
    layernorm_forward_launcher(
        (half*)x.data_ptr(), (half*)y.data_ptr(), g, b, R, D, stream.stream(),
        threads, rows_per_cta, vec_width, use_warp);
    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
    m.def("forward",&layernorm_forward,"layernorm forward");
}


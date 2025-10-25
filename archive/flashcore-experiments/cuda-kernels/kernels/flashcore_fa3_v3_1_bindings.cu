// flashcore_fa3_v3_1_bindings.cu
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

void launch_flash3_v3_1(
    const half* Q, const half* K, const half* V, half* O,
    int B, int H, int S, int D, bool is_causal, cudaStream_t stream);

torch::Tensor flash3_forward(
    torch::Tensor Q, 
    torch::Tensor K, 
    torch::Tensor V, 
    bool is_causal=false)
{
  TORCH_CHECK(Q.is_cuda() && K.is_cuda() && V.is_cuda(), "Q/K/V must be CUDA");
  TORCH_CHECK(Q.dtype() == torch::kFloat16, "Q must be float16");
  TORCH_CHECK(K.dtype() == torch::kFloat16, "K must be float16");
  TORCH_CHECK(V.dtype() == torch::kFloat16, "V must be float16");
  TORCH_CHECK(Q.dim() == 4, "Q must be 4D [B,H,S,D]");
  TORCH_CHECK(Q.sizes() == K.sizes() && Q.sizes() == V.sizes(), "Shapes must match");
  TORCH_CHECK(Q.is_contiguous(), "Q must be contiguous");
  TORCH_CHECK(K.is_contiguous(), "K must be contiguous");
  TORCH_CHECK(V.is_contiguous(), "V must be contiguous");

  const int64_t B = Q.size(0), H = Q.size(1), S = Q.size(2), D = Q.size(3);
  TORCH_CHECK(D % 32 == 0, "D must be divisible by 32");
  TORCH_CHECK(D <= 128, "D must be <= 128");

  auto O = torch::empty_like(Q);
  auto stream = at::cuda::getCurrentCUDAStream();

  const half* Qp = reinterpret_cast<const half*>(Q.data_ptr<at::Half>());
  const half* Kp = reinterpret_cast<const half*>(K.data_ptr<at::Half>());
  const half* Vp = reinterpret_cast<const half*>(V.data_ptr<at::Half>());
  half*       Op = reinterpret_cast<half*>(O.data_ptr<at::Half>());

  launch_flash3_v3_1(Qp, Kp, Vp, Op, (int)B, (int)H, (int)S, (int)D, is_causal, stream);

  cudaError_t err = cudaGetLastError();
  TORCH_CHECK(err == cudaSuccess, "Kernel launch failed: ", cudaGetErrorString(err));

  return O;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &flash3_forward, "FA-3 v3.1 (fixed state management)",
        py::arg("Q"), py::arg("K"), py::arg("V"), py::arg("is_causal")=false);
}


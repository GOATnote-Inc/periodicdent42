// flashcore_fa3_bindings_v2.cu
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

void launch_flash3_fused_attention_fp16_v2(
    const half* Q, const half* K, const half* V, half* O,
    int B, int H, int S, int D, bool is_causal, cudaStream_t stream);

torch::Tensor flash3_forward(torch::Tensor Q, torch::Tensor K, torch::Tensor V, bool is_causal=false) {
  TORCH_CHECK(Q.is_cuda() && K.is_cuda() && V.is_cuda(), "Q/K/V must be CUDA");
  TORCH_CHECK(Q.dtype() == torch::kFloat16 && K.dtype() == torch::kFloat16 && V.dtype() == torch::kFloat16, "FP16 only");
  TORCH_CHECK(Q.dim() == 4 && K.dim() == 4 && V.dim() == 4, "Expected [B,H,S,D]");
  TORCH_CHECK(Q.sizes() == K.sizes() && Q.sizes() == V.sizes(), "Shapes must match");

  const int64_t B = Q.size(0), H = Q.size(1), S = Q.size(2), D = Q.size(3);
  TORCH_CHECK(D % 32 == 0, "D must be divisible by 32");
  TORCH_CHECK(D <= 128, "D must be <= 128 for this kernel");

  auto O = torch::empty_like(Q);
  auto stream = at::cuda::getCurrentCUDAStream();

  const half* Qp = reinterpret_cast<const half*>(Q.data_ptr<at::Half>());
  const half* Kp = reinterpret_cast<const half*>(K.data_ptr<at::Half>());
  const half* Vp = reinterpret_cast<const half*>(V.data_ptr<at::Half>());
  half*       Op = reinterpret_cast<half*>(O.data_ptr<at::Half>());

  launch_flash3_fused_attention_fp16_v2(Qp, Kp, Vp, Op, (int)B, (int)H, (int)S, (int)D, is_causal, stream);

  cudaError_t err = cudaGetLastError();
  TORCH_CHECK(err == cudaSuccess, "Kernel launch failed: ", cudaGetErrorString(err));

  return O;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &flash3_forward, "FlashAttention3-style fused attention (fp16)",
        py::arg("Q"), py::arg("K"), py::arg("V"), py::arg("is_causal")=false);
}


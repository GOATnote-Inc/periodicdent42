#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#include <torch/extension.h>

#include <vector>

extern "C" void flashcore_v6_launch_qkt(
    const half* Q,
    const half* K,
    float* Scores,
    int B,
    int H,
    int S,
    int D,
    float scale,
    cudaStream_t stream);

extern "C" void flashcore_v7_1_launch_pv(
    const half* P,
    const half* V,
    half* O,
    int B,
    int H,
    int S,
    int D,
    cudaStream_t stream);

extern "C" void flashcore_fused_launch(
    const half* Q,
    const half* K,
    const half* V,
    half* O,
    int B,
    int H,
    int S,
    int D,
    float scale,
    cudaStream_t stream);

extern "C" void flashcore_fused_phase2_launch(
    const half* Q,
    const half* K,
    const half* V,
    half* O,
    int B,
    int H,
    int S,
    int D,
    float scale,
    cudaStream_t stream);

extern "C" void flashcore_fused_phase2_2_launch(
    const half* Q,
    const half* K,
    const half* V,
    half* O,
    int B,
    int H,
    int S,
    int D,
    float scale,
    cudaStream_t stream);

namespace {

torch::Tensor launch_qkt(torch::Tensor q, torch::Tensor k, double scale) {
    TORCH_CHECK(q.device().is_cuda(), "Q must be on CUDA");
    TORCH_CHECK(k.device().is_cuda(), "K must be on CUDA");
    TORCH_CHECK(q.is_contiguous(), "Q must be contiguous");
    TORCH_CHECK(k.is_contiguous(), "K must be contiguous");
    TORCH_CHECK(q.dtype() == torch::kHalf, "Q must be half");
    TORCH_CHECK(k.dtype() == torch::kHalf, "K must be half");
    TORCH_CHECK(q.sizes() == k.sizes(), "Q and K must have identical shapes");
    TORCH_CHECK(q.dim() == 4, "Expected Q of shape [B, H, S, D]");

    const int64_t B = q.size(0);
    const int64_t H = q.size(1);
    const int64_t S = q.size(2);
    const int64_t D = q.size(3);

    auto options = q.options().dtype(torch::kFloat);
    auto scores = torch::empty({B, H, S, S}, options);

    auto stream = at::cuda::getCurrentCUDAStream();

    flashcore_v6_launch_qkt(
        reinterpret_cast<const half*>(q.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(k.data_ptr<at::Half>()),
        scores.data_ptr<float>(),
        static_cast<int>(B),
        static_cast<int>(H),
        static_cast<int>(S),
        static_cast<int>(D),
        static_cast<float>(scale),
        stream);

    return scores;
}

torch::Tensor launch_pv(torch::Tensor p, torch::Tensor v) {
    TORCH_CHECK(p.device().is_cuda(), "P must be on CUDA");
    TORCH_CHECK(v.device().is_cuda(), "V must be on CUDA");
    TORCH_CHECK(p.is_contiguous(), "P must be contiguous");
    TORCH_CHECK(v.is_contiguous(), "V must be contiguous");
    TORCH_CHECK(p.dtype() == torch::kHalf, "P must be half");
    TORCH_CHECK(v.dtype() == torch::kHalf, "V must be half");
    TORCH_CHECK(p.dim() == 4, "Expected P of shape [B, H, S, S]");
    TORCH_CHECK(v.dim() == 4, "Expected V of shape [B, H, S, D]");
    TORCH_CHECK(p.size(0) == v.size(0), "Batch mismatch");
    TORCH_CHECK(p.size(1) == v.size(1), "Head mismatch");
    TORCH_CHECK(p.size(2) == v.size(2), "Sequence mismatch");

    const int64_t B = p.size(0);
    const int64_t H = p.size(1);
    const int64_t S = p.size(2);
    const int64_t D = v.size(3);

    auto options = v.options();
    auto output = torch::empty({B, H, S, D}, options);

    auto stream = at::cuda::getCurrentCUDAStream();

    flashcore_v7_1_launch_pv(
        reinterpret_cast<const half*>(p.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(v.data_ptr<at::Half>()),
        reinterpret_cast<half*>(output.data_ptr<at::Half>()),
        static_cast<int>(B),
        static_cast<int>(H),
        static_cast<int>(S),
        static_cast<int>(D),
        stream);

    return output;
}

torch::Tensor launch_fused(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    double scale) {
    TORCH_CHECK(q.device().is_cuda(), "Q must be on CUDA");
    TORCH_CHECK(k.device().is_cuda(), "K must be on CUDA");
    TORCH_CHECK(v.device().is_cuda(), "V must be on CUDA");
    TORCH_CHECK(q.is_contiguous(), "Q must be contiguous");
    TORCH_CHECK(k.is_contiguous(), "K must be contiguous");
    TORCH_CHECK(v.is_contiguous(), "V must be contiguous");
    TORCH_CHECK(q.dtype() == torch::kHalf, "Q must be half");
    TORCH_CHECK(k.dtype() == torch::kHalf, "K must be half");
    TORCH_CHECK(v.dtype() == torch::kHalf, "V must be half");
    TORCH_CHECK(q.dim() == 4, "Expected Q of shape [B, H, S, D]");
    TORCH_CHECK(k.dim() == 4, "Expected K of shape [B, H, S, D]");
    TORCH_CHECK(v.dim() == 4, "Expected V of shape [B, H, S, D]");
    TORCH_CHECK(q.sizes() == k.sizes(), "Q and K must have identical shapes");
    TORCH_CHECK(q.sizes() == v.sizes(), "Q and V must have identical shapes");

    const int64_t B = q.size(0);
    const int64_t H = q.size(1);
    const int64_t S = q.size(2);
    const int64_t D = q.size(3);

    auto options = q.options();
    auto output = torch::empty({B, H, S, D}, options);

    auto stream = at::cuda::getCurrentCUDAStream();

    flashcore_fused_launch(
        reinterpret_cast<const half*>(q.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(k.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(v.data_ptr<at::Half>()),
        reinterpret_cast<half*>(output.data_ptr<at::Half>()),
        static_cast<int>(B),
        static_cast<int>(H),
        static_cast<int>(S),
        static_cast<int>(D),
        static_cast<float>(scale),
        stream);

    return output;
}

torch::Tensor launch_fused_phase2(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    double scale) {
    TORCH_CHECK(q.device().is_cuda(), "Q must be on CUDA");
    TORCH_CHECK(k.device().is_cuda(), "K must be on CUDA");
    TORCH_CHECK(v.device().is_cuda(), "V must be on CUDA");
    TORCH_CHECK(q.is_contiguous(), "Q must be contiguous");
    TORCH_CHECK(k.is_contiguous(), "K must be contiguous");
    TORCH_CHECK(v.is_contiguous(), "V must be contiguous");
    TORCH_CHECK(q.dtype() == torch::kHalf, "Q must be half");
    TORCH_CHECK(k.dtype() == torch::kHalf, "K must be half");
    TORCH_CHECK(v.dtype() == torch::kHalf, "V must be half");
    TORCH_CHECK(q.dim() == 4, "Expected Q of shape [B, H, S, D]");
    TORCH_CHECK(k.dim() == 4, "Expected K of shape [B, H, S, D]");
    TORCH_CHECK(v.dim() == 4, "Expected V of shape [B, H, S, D]");
    TORCH_CHECK(q.sizes() == k.sizes(), "Q and K must have identical shapes");
    TORCH_CHECK(q.sizes() == v.sizes(), "Q and V must have identical shapes");

    const int64_t B = q.size(0);
    const int64_t H = q.size(1);
    const int64_t S = q.size(2);
    const int64_t D = q.size(3);

    auto options = q.options();
    auto output = torch::empty({B, H, S, D}, options);

    auto stream = at::cuda::getCurrentCUDAStream();

    flashcore_fused_phase2_launch(
        reinterpret_cast<const half*>(q.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(k.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(v.data_ptr<at::Half>()),
        reinterpret_cast<half*>(output.data_ptr<at::Half>()),
        static_cast<int>(B),
        static_cast<int>(H),
        static_cast<int>(S),
        static_cast<int>(D),
        static_cast<float>(scale),
        stream);

    return output;
}

torch::Tensor launch_fused_phase2_2(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    double scale) {
    TORCH_CHECK(q.device().is_cuda(), "Q must be on CUDA");
    TORCH_CHECK(k.device().is_cuda(), "K must be on CUDA");
    TORCH_CHECK(v.device().is_cuda(), "V must be on CUDA");
    TORCH_CHECK(q.is_contiguous(), "Q must be contiguous");
    TORCH_CHECK(k.is_contiguous(), "K must be contiguous");
    TORCH_CHECK(v.is_contiguous(), "V must be contiguous");
    TORCH_CHECK(q.dtype() == torch::kHalf, "Q must be half");
    TORCH_CHECK(k.dtype() == torch::kHalf, "K must be half");
    TORCH_CHECK(v.dtype() == torch::kHalf, "V must be half");
    TORCH_CHECK(q.dim() == 4, "Expected Q of shape [B, H, S, D]");
    TORCH_CHECK(k.dim() == 4, "Expected K of shape [B, H, S, D]");
    TORCH_CHECK(v.dim() == 4, "Expected V of shape [B, H, S, D]");
    TORCH_CHECK(q.sizes() == k.sizes(), "Q and K must have identical shapes");
    TORCH_CHECK(q.sizes() == v.sizes(), "Q and V must have identical shapes");

    const int64_t B = q.size(0);
    const int64_t H = q.size(1);
    const int64_t S = q.size(2);
    const int64_t D = q.size(3);

    auto options = q.options();
    auto output = torch::empty({B, H, S, D}, options);

    auto stream = at::cuda::getCurrentCUDAStream();

    flashcore_fused_phase2_2_launch(
        reinterpret_cast<const half*>(q.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(k.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(v.data_ptr<at::Half>()),
        reinterpret_cast<half*>(output.data_ptr<at::Half>()),
        static_cast<int>(B),
        static_cast<int>(H),
        static_cast<int>(S),
        static_cast<int>(D),
        static_cast<float>(scale),
        stream);

    return output;
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("qkt", &launch_qkt, "FlashCore WMMA QK^T kernel", py::arg("q"), py::arg("k"), py::arg("scale"));
    m.def("pv", &launch_pv, "FlashCore WMMA P*V kernel", py::arg("p"), py::arg("v"));
    m.def("fused", &launch_fused, "FlashCore Fused Attention kernel",
          py::arg("q"), py::arg("k"), py::arg("v"), py::arg("scale"));
    m.def("fused_phase2", &launch_fused_phase2, "FlashCore Phase 2 Fused Attention (64×64 dynamic SMEM)",
          py::arg("q"), py::arg("k"), py::arg("v"), py::arg("scale"));
    m.def("fused_phase2_2", &launch_fused_phase2_2, "FlashCore Phase 2.2 Optimized (48×48 + cp.async)",
          py::arg("q"), py::arg("k"), py::arg("v"), py::arg("scale"));
}


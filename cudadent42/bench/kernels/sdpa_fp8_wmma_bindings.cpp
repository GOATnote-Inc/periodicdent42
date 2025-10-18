#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

// Forward declaration
extern "C" void launch_sdpa_fp8_wmma(
    const void* Q, const void* K, const void* V,
    const float* Qs, const float* Ks, const float* Vs,
    void* O, int B, int H, int S, int D,
    float softmax_scale, cudaStream_t stream
);

torch::Tensor sdpa_fp8_wmma_forward(
    torch::Tensor Q,  // [B,H,S,D] uint8
    torch::Tensor K,  // [B,H,S,D] uint8
    torch::Tensor V,  // [B,H,S,D] uint8
    torch::Tensor Q_scale,  // [H] float32
    torch::Tensor K_scale,  // [H] float32
    torch::Tensor V_scale,  // [H] float32
    float softmax_scale
){
    const int B = Q.size(0);
    const int H = Q.size(1);
    const int S = Q.size(2);
    const int D = Q.size(3);

    auto O = torch::empty({B, H, S, D}, torch::dtype(torch::kFloat16).device(Q.device()));

    const at::cuda::OptionalCUDAGuard device_guard(Q.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    launch_sdpa_fp8_wmma(
        Q.data_ptr(), K.data_ptr(), V.data_ptr(),
        Q_scale.data_ptr<float>(), K_scale.data_ptr<float>(), V_scale.data_ptr<float>(),
        O.data_ptr(), B, H, S, D, softmax_scale, stream
    );

    return O;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &sdpa_fp8_wmma_forward, "FP8 SDPA with WMMA (Tensor Cores)");
}

// Launcher implementation
#include <cuda_runtime.h>

extern __global__ void sdpa_fp8_wmma_kernel(
    const uint8_t* Q, const uint8_t* K, const uint8_t* V,
    const float* Qs, const float* Ks, const float* Vs,
    half* O, int B, int H, int S, int D, float softmax_scale
);

void launch_sdpa_fp8_wmma(
    const void* Q, const void* K, const void* V,
    const float* Qs, const float* Ks, const float* Vs,
    void* O, int B, int H, int S, int D,
    float softmax_scale, cudaStream_t stream
){
    const int TILE_M = 32;
    dim3 block(256);  // 8 warps × 32 threads
    dim3 grid((S + TILE_M - 1) / TILE_M, H, B);

    sdpa_fp8_wmma_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<const uint8_t*>(Q),
        reinterpret_cast<const uint8_t*>(K),
        reinterpret_cast<const uint8_t*>(V),
        Qs, Ks, Vs,
        reinterpret_cast<half*>(O),
        B, H, S, D, softmax_scale
    );
}


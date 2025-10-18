#include <cuda_runtime.h>
#include <cuda_fp16.h>
<cstdint>

// Forward declaration of kernel
extern __global__ void sdpa_fp8_wmma_kernel(
    const uint8_t* Q, const uint8_t* K, const uint8_t* V,
    const float* Qs, const float* Ks, const float* Vs,
    half* O, int B, int H, int S, int D, float softmax_scale
);

extern "C" void launch_sdpa_fp8_wmma(
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


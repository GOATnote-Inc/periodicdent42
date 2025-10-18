#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// SDPA kernel parameters
struct SdpaParams {
    const void* Q;      // [B, H, L, d]
    const void* K;      // [B, H, L, d]
    const void* V;      // [B, H, L, d]
    void* O;            // [B, H, L, d]
    int B;              // batch size
    int H;              // number of heads
    int L;              // sequence length
    int d;              // head dimension
    float scale;        // 1/sqrt(d)
    bool causal;        // causal masking flag
};

// Forward declarations
cudaError_t sdpa_fused_forward(const SdpaParams& params, cudaStream_t stream = 0);
cudaError_t sdpa_fused_forward_v2(const SdpaParams& params, cudaStream_t stream = 0);
cudaError_t sdpa_fused_forward_v2b(const SdpaParams& params, cudaStream_t stream = 0);


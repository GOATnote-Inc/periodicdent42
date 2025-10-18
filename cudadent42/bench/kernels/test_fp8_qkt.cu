// Minimal test: Q@K^T only (no softmax, no attention)
// Goal: Isolate FP8 dequantization bug

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

__device__ __forceinline__ float dequant_sim_fp8(uint8_t val_uint8, float scale) {
    float val = (float(val_uint8) / 255.0f) * (2.0f * 448.0f) - 448.0f;
    return val * scale;
}

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Minimal Q@K^T kernel (single block, small problem)
__global__ void test_qkt_kernel(
    const uint8_t* __restrict__ Q,    // [M, D]
    const uint8_t* __restrict__ K,    // [N, D]
    const float Q_scale,
    const float K_scale,
    half* __restrict__ S,             // [M, N] output
    const int M,
    const int N,
    const int D
) {
    const int m = blockIdx.x * blockDim.x + threadIdx.x;
    const int n = blockIdx.y;
    
    if (m >= M || n >= N) return;
    
    const int lane_id = threadIdx.x % 32;
    
    // Compute dot product: Q[m] · K[n]
    float score = 0.0f;
    
    for (int d = lane_id; d < D; d += 32) {
        float q_val = dequant_sim_fp8(Q[m * D + d], Q_scale);
        float k_val = dequant_sim_fp8(K[n * D + d], K_scale);
        score += q_val * k_val;
    }
    
    // Warp reduction
    score = warp_reduce_sum(score);
    
    // Write result (only lane 0)
    if (lane_id == 0) {
        S[m * N + n] = __float2half(score);
    }
}

extern "C" void launch_test_qkt(
    const void* Q,
    const void* K,
    float Q_scale,
    float K_scale,
    half* S,
    int M, int N, int D,
    cudaStream_t stream
) {
    dim3 block(32);
    dim3 grid(M, N);
    
    test_qkt_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<const uint8_t*>(Q),
        reinterpret_cast<const uint8_t*>(K),
        Q_scale,
        K_scale,
        S,
        M, N, D
    );
}


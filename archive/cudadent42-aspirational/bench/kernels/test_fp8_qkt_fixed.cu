// FIXED: Q@K^T with correct thread mapping
// One warp per (m,n) pair (not 32 rows per warp!)

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cstdio>

__device__ __forceinline__ float dequant_sim_fp8(uint8_t u, float scale) {
    // Map [0,255] -> [-448, +448] then apply external scale
    const float inv255 = 1.0f / 255.0f;
    float x = (float(u) * inv255) * (2.0f * 448.0f) - 448.0f;
    return x * scale;
}

__device__ __forceinline__ float warp_reduce_sum(float v) {
    #pragma unroll
    for (int ofs = 16; ofs > 0; ofs >>= 1) {
        v += __shfl_down_sync(0xffffffff, v, ofs);
    }
    return v;
}

__global__ void test_qkt_kernel_fixed(
    const uint8_t* __restrict__ Q,   // [M, D]
    const uint8_t* __restrict__ K,   // [N, D]
    const float Q_scale,              // per-tensor scale
    const float K_scale,
    half* __restrict__ S,             // [M, N]
    const int M, const int N, const int D,
    const float inv_sqrt_d            // pass 1.0f if you don't want it
){
    const int m = blockIdx.x;         // ✅ ONE WARP PER (m,n) PAIR
    const int n = blockIdx.y;
    const int lane = threadIdx.x & 31;
    
    if (m >= M || n >= N) return;

    const int q_base = m * D;
    const int k_base = n * D;

    float acc = 0.f;

    // Split D across lanes; works for any D
    for (int d = lane; d < D; d += 32) {
        float q = dequant_sim_fp8(Q[q_base + d], Q_scale);
        float k = dequant_sim_fp8(K[k_base + d], K_scale);
        acc += q * k;
    }

    // Warp reduction to a single scalar for (m,n)
    acc = warp_reduce_sum(acc);

    if (lane == 0) {
        acc *= inv_sqrt_d;
        S[m * N + n] = __float2half(acc);
    }
}

extern "C" void launch_test_qkt_fixed(
    const void* Q, const void* K,
    float Q_scale, float K_scale,
    half* S, int M, int N, int D,
    cudaStream_t stream, bool apply_inv_sqrt_d
){
    const float inv_sqrt_d = apply_inv_sqrt_d ? rsqrtf((float)D) : 1.0f;
    dim3 block(32);
    dim3 grid(M, N);
    test_qkt_kernel_fixed<<<grid, block, 0, stream>>>(
        reinterpret_cast<const uint8_t*>(Q),
        reinterpret_cast<const uint8_t*>(K),
        Q_scale, K_scale, S, M, N, D, inv_sqrt_d
    );
}


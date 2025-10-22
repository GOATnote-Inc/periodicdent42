// flashcore_fa3_kernel_v2.cu
// FA-3 style fused attention (FP16) for L4 (sm_89)
// Fixes: (1) K/V preload per row, (2) fewer barriers, (3) SMEM padding to avoid bank conflicts.
// Pattern: Stream K/V tiles, double buffer, online softmax + fused P·V.

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math.h>

#ifndef WARPS_PER_BLOCK
#define WARPS_PER_BLOCK 4           // 4 warps = 128 threads
#endif
#ifndef M_TILE
#define M_TILE 64                   // queries per block
#endif
#ifndef N_TILE
#define N_TILE 128                  // keys per streamed tile
#endif
#ifndef PAD
#define PAD 8                       // SMEM leading-dim padding (fp16/bank-conflict relief)
#endif

__device__ __forceinline__ float warp_sum(float v) {
  #pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1) {
    v += __shfl_xor_sync(0xffffffff, v, mask);
  }
  return v;
}

__global__ void __launch_bounds__(WARPS_PER_BLOCK * 32)
flash3_fused_attention_fp16_kernel_v2(
    const half* __restrict__ Q,  // [B,H,S,D]
    const half* __restrict__ K,  // [B,H,S,D]
    const half* __restrict__ V,  // [B,H,S,D]
    half* __restrict__ O,        // [B,H,S,D]
    int B, int H, int S, int D,
    int is_causal,               // 0/1
    float scale)                 // 1/sqrt(D)
{
  const int warp_id  = threadIdx.x / 32;
  const int lane_id  = threadIdx.x % 32;
  const int nwarps   = blockDim.x / 32;

  const int q_tile_start = blockIdx.x * M_TILE;
  const int h            = blockIdx.y;
  const int b            = blockIdx.z;

  if (h >= H || b >= B) return;

  const int q_tile_len = min(M_TILE, S - q_tile_start);
  if (q_tile_len <= 0) return;

  const size_t base = ((size_t)b * H + (size_t)h) * (size_t)S * (size_t)D;

  const half* __restrict__ Q_bh = Q + base;
  const half* __restrict__ K_bh = K + base;
  const half* __restrict__ V_bh = V + base;
  half*       __restrict__ O_bh = O + base;

  // Shared memory layout with padded leading dimensions:
  // Q:  M_TILE x (D+PAD)
  // K0: N_TILE x (D+PAD)
  // V0: N_TILE x (D+PAD)
  // K1: N_TILE x (D+PAD)
  // V1: N_TILE x (D+PAD)
  const int LDQ = D + PAD;
  const int LDK = D + PAD;
  const int LDV = D + PAD;

  extern __shared__ half smem[];
  half* smem_Q  = smem;
  half* smem_K0 = smem_Q  + (size_t)M_TILE * LDQ;
  half* smem_V0 = smem_K0 + (size_t)N_TILE * LDK;
  half* smem_K1 = smem_V0 + (size_t)N_TILE * LDV;
  half* smem_V1 = smem_K1 + (size_t)N_TILE * LDK;

  // Load Q tile once per block
  {
    const int q_elems = q_tile_len * D;
    for (int idx = threadIdx.x; idx < q_elems; idx += blockDim.x) {
      const int r = idx / D;      // row within Q tile
      const int d = idx % D;
      smem_Q[r * LDQ + d] = Q_bh[(q_tile_start + r) * D + d];
    }
  }
  __syncthreads();

  // Each warp handles rows in the Q tile (strided by #warps)
  for (int q_local = warp_id; q_local < q_tile_len; q_local += nwarps) {
    const int q_abs = q_tile_start + q_local;

    // --- Preload K/V TILE 0 *per row* (critical correctness fix) ---
    int buf = 0;
    int k_start = 0;
    {
      const int k_len0 = min(N_TILE, S - k_start);
      const int elems  = k_len0 * D;
      half* dstK = buf ? smem_K1 : smem_K0;
      half* dstV = buf ? smem_V1 : smem_V0;
      for (int idx = threadIdx.x; idx < elems; idx += blockDim.x) {
        const int j = idx / D;
        const int d = idx % D;
        dstK[j * LDK + d] = K_bh[(k_start + j) * D + d];
        dstV[j * LDV + d] = V_bh[(k_start + j) * D + d];
      }
    }
    __syncthreads();

    // Register-resident slice of Q row for this lane
    const int elems_per_lane = D / 32;        // D ∈ {32,64,96,128}, assumed divisible by 32
    float q_reg[4];                            // supports up to D=128
    #pragma unroll
    for (int c = 0; c < elems_per_lane; ++c) {
      const int d_idx = c * 32 + lane_id;
      q_reg[c] = __half2float(smem_Q[q_local * LDQ + d_idx]);
    }

    // Online softmax state for row q_abs
    float m_i = -INFINITY;
    float l_i = 0.f;
    float out_acc[4];                          // numerator accumulator (FP32)
    #pragma unroll
    for (int c = 0; c < elems_per_lane; ++c) out_acc[c] = 0.f;

    // --- Stream over K/V tiles ---
    for (k_start = 0; k_start < S; k_start += N_TILE) {
      const int k_len = min(N_TILE, S - k_start);

      // Prefetch next tile into the other buffer (if exists)
      if (k_start + N_TILE < S) {
        const int next_len = min(N_TILE, S - (k_start + N_TILE));
        const int elems    = next_len * D;
        half* nxtK = (buf ^ 1) ? smem_K1 : smem_K0;
        half* nxtV = (buf ^ 1) ? smem_V1 : smem_V0;
        for (int idx = threadIdx.x; idx < elems; idx += blockDim.x) {
          const int j = idx / D;
          const int d = idx % D;
          nxtK[j * LDK + d] = K_bh[(k_start + N_TILE + j) * D + d];
          nxtV[j * LDV + d] = V_bh[(k_start + N_TILE + j) * D + d];
        }
      }
      __syncthreads();  // ensure prefetch complete before we switch buffers

      // Active buffers for this tile
      half* Kbuf = buf ? smem_K1 : smem_K0;
      half* Vbuf = buf ? smem_V1 : smem_V0;

      // Iterate keys in this tile
      #pragma unroll 4
      for (int j = 0; j < k_len; ++j) {
        const int key_abs = k_start + j;

        // Dot(Q_row, K_j) across D via warp
        float partial = 0.f;
        #pragma unroll
        for (int c = 0; c < elems_per_lane; ++c) {
          const int d_idx = c * 32 + lane_id;
          const float kf  = __half2float(Kbuf[j * LDK + d_idx]);
          partial += q_reg[c] * kf;
        }

        float score = warp_sum(partial) * scale;
        if (is_causal && key_abs > q_abs) {
          score = -INFINITY;
        }

        // Online softmax update
        const float m_new = fmaxf(m_i, score);
        const float alpha = __expf(m_i - m_new);
        const float beta  = __expf(score - m_new);
        const float l_new = l_i * alpha + beta;

        // Fused P·V numerator accumulation
        #pragma unroll
        for (int c = 0; c < elems_per_lane; ++c) {
          const int d_idx = c * 32 + lane_id;
          const float vf  = __half2float(Vbuf[j * LDV + d_idx]);
          out_acc[c] = out_acc[c] * alpha + beta * vf;
        }

        m_i = m_new;
        l_i = l_new;
      }

      buf ^= 1;
      __syncthreads(); // ensure all readers finished before next prefetch overwrites
    }

    // Normalize and store result row
    #pragma unroll
    for (int c = 0; c < elems_per_lane; ++c) {
      const int d_idx = c * 32 + lane_id;
      const float outv = (l_i > 0.f) ? (out_acc[c] / l_i) : 0.f;
      O_bh[q_abs * D + d_idx] = __float2half(outv);
    }
  }
}

// -------------------- Host launcher --------------------
void launch_flash3_fused_attention_fp16_v2(
    const half* Q, const half* K, const half* V, half* O,
    int B, int H, int S, int D,
    bool is_causal,
    cudaStream_t stream)
{
  const float scale = 1.0f / sqrtf((float)D);

  dim3 block(WARPS_PER_BLOCK * 32);
  dim3 grid((S + M_TILE - 1) / M_TILE, H, B);

  // dynamic smem size (padded LDs)
  const size_t smem_elems =
      (size_t)M_TILE * (D + PAD)   // Q
    + (size_t)N_TILE * (D + PAD)   // K0
    + (size_t)N_TILE * (D + PAD)   // V0
    + (size_t)N_TILE * (D + PAD)   // K1
    + (size_t)N_TILE * (D + PAD);  // V1
  const size_t smem_bytes = smem_elems * sizeof(half);

  cudaFuncSetAttribute(
      flash3_fused_attention_fp16_kernel_v2,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      (int)smem_bytes);

  flash3_fused_attention_fp16_kernel_v2<<<grid, block, smem_bytes, stream>>>(
      Q, K, V, O, B, H, S, D, (int)is_causal, scale);
}


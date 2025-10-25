// flashcore_fa3_v7_wmma_pv.cu
// Phase 2: WMMA for both Q·K^T and P·V (Full Tensor Cores!)
// Expected: 447 → 100-150 μs (3-4× speedup)
// Target: <40 μs after phases 3-4

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <math.h>

using namespace nvcuda;

#ifndef WARPS_PER_BLOCK
#define WARPS_PER_BLOCK 4
#endif
#ifndef M_TILE
#define M_TILE 64
#endif
#ifndef N_TILE
#define N_TILE 64
#endif
#ifndef PAD
#define PAD 8
#endif

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

__device__ __forceinline__ float warp_sum(float v) {
  #pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1) {
    v += __shfl_xor_sync(0xffffffff, v, mask);
  }
  return v;
}

__device__ __forceinline__ float warp_max(float v) {
  #pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1) {
    v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, mask));
  }
  return v;
}

__global__ void __launch_bounds__(WARPS_PER_BLOCK * 32, 2)
flash3_wmma_full_kernel(
    const half* __restrict__ Q,
    const half* __restrict__ K,
    const half* __restrict__ V,
    half* __restrict__ O,
    int B, int H, int S, int D,
    int is_causal,
    float scale)
{
  const int warp_id = threadIdx.x / 32;
  const int lane_id = threadIdx.x % 32;

  const int q_tile_start = blockIdx.x * M_TILE;
  const int h = blockIdx.y;
  const int b = blockIdx.z;

  if (h >= H || b >= B) return;

  const int q_tile_len = min(M_TILE, S - q_tile_start);
  if (q_tile_len <= 0) return;

  const size_t base = ((size_t)b * H + (size_t)h) * (size_t)S * (size_t)D;

  const half* __restrict__ Q_bh = Q + base;
  const half* __restrict__ K_bh = K + base;
  const half* __restrict__ V_bh = V + base;
  half*       __restrict__ O_bh = O + base;

  const int LDQ = D + PAD;
  const int LDK = D + PAD;
  const int LDV = D + PAD;
  const int LDP = N_TILE + PAD;  // For attention probs (reuse sS memory)

  extern __shared__ char smem_bytes[];
  
  half* sQ = reinterpret_cast<half*>(smem_bytes);
  half* sK = sQ + M_TILE * LDQ;
  half* sV = sK + N_TILE * LDK;
  
  size_t float_offset = (M_TILE * LDQ + N_TILE * LDK + N_TILE * LDV) * sizeof(half);
  float_offset = (float_offset + 3) & ~3;
  
  float* sS = reinterpret_cast<float*>(smem_bytes + float_offset);
  half*  sP = reinterpret_cast<half*>(sS);  // Reuse sS buffer for FP16 probs
  float* sM = sS + M_TILE * LDP;
  float* sL = sM + M_TILE;
  float* sO = sL + M_TILE;

  // Prologue: Load Q
  for (int idx = threadIdx.x; idx < q_tile_len * D; idx += blockDim.x) {
    int m = idx / D;
    int d = idx % D;
    sQ[m * LDQ + d] = Q_bh[(q_tile_start + m) * D + d];
  }

  for (int m = threadIdx.x; m < q_tile_len; m += blockDim.x) {
    sM[m] = -INFINITY;
    sL[m] = 0.0f;
  }
  for (int idx = threadIdx.x; idx < q_tile_len * D; idx += blockDim.x) {
    int m = idx / D;
    int d = idx % D;
    sO[m * (D + PAD) + d] = 0.0f;
  }
  __syncthreads();

  // Main loop
  for (int k_start = 0; k_start < S; k_start += N_TILE) {
    const int k_len = min(N_TILE, S - k_start);

    // Load K/V
    for (int idx = threadIdx.x; idx < k_len * D; idx += blockDim.x) {
      int n = idx / D;
      int d = idx % D;
      sK[n * LDK + d] = K_bh[(k_start + n) * D + d];
      sV[n * LDV + d] = V_bh[(k_start + n) * D + d];
    }
    __syncthreads();

    // === WMMA Q·K^T ===
    const int warp_m_start = warp_id * WMMA_M;
    if (warp_m_start < q_tile_len) {
      const int warp_m_end = min(warp_m_start + WMMA_M, q_tile_len);

      for (int n_wmma = 0; n_wmma < k_len; n_wmma += WMMA_N) {
        const int n_end = min(n_wmma + WMMA_N, k_len);
        
        if (n_end - n_wmma == WMMA_N && warp_m_end - warp_m_start == WMMA_M) {
          wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> q_frag;
          wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> k_frag;
          wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> s_frag;
          
          wmma::fill_fragment(s_frag, 0.0f);

          #pragma unroll
          for (int k_wmma = 0; k_wmma < D; k_wmma += WMMA_K) {
            wmma::load_matrix_sync(q_frag, &sQ[warp_m_start * LDQ + k_wmma], LDQ);
            wmma::load_matrix_sync(k_frag, &sK[n_wmma * LDK + k_wmma], LDK);
            wmma::mma_sync(s_frag, q_frag, k_frag, s_frag);
          }

          wmma::store_matrix_sync(&sS[warp_m_start * LDP + n_wmma], s_frag, LDP, wmma::mem_row_major);
        }
      }
    }
    __syncthreads();

    // === Online Softmax + Convert to FP16 ===
    for (int m_local = warp_id; m_local < q_tile_len; m_local += WARPS_PER_BLOCK) {
      const int m_abs = q_tile_start + m_local;
      
      float m_prev = sM[m_local];
      float l_prev = sL[m_local];

      // Find max
      float m_tile = -INFINITY;
      for (int n = lane_id; n < k_len; n += 32) {
        const int k_abs = k_start + n;
        float score = sS[m_local * LDP + n] * scale;
        if (is_causal && k_abs > m_abs) score = -INFINITY;
        m_tile = fmaxf(m_tile, score);
      }
      m_tile = warp_max(m_tile);

      float m_new = fmaxf(m_prev, m_tile);
      float alpha = expf(m_prev - m_new);
      
      // Compute exp and sum
      float l_tile = 0.0f;
      for (int n = lane_id; n < k_len; n += 32) {
        const int k_abs = k_start + n;
        float score = sS[m_local * LDP + n] * scale;
        if (is_causal && k_abs > m_abs) score = -INFINITY;
        
        float prob = expf(score - m_new);
        l_tile += prob;
        
        // Store as FP16 for WMMA PV
        sP[m_local * LDP + n] = __float2half(prob);
      }
      l_tile = warp_sum(l_tile);

      float l_new = l_prev * alpha + l_tile;

      // Scale previous O accumulator
      for (int d = lane_id; d < D; d += 32) {
        sO[m_local * (D + PAD) + d] *= alpha;
      }

      if (lane_id == 0) {
        sM[m_local] = m_new;
        sL[m_local] = l_new;
      }
    }
    __syncthreads();

    // === WMMA P·V ===
    if (warp_m_start < q_tile_len) {
      const int warp_m_end = min(warp_m_start + WMMA_M, q_tile_len);

      // Process D dimension in WMMA_K chunks
      for (int d_wmma = 0; d_wmma < D; d_wmma += WMMA_K) {
        
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> o_frag;
        wmma::fill_fragment(o_frag, 0.0f);

        // N dimension loop (across attention weights)
        #pragma unroll
        for (int n_wmma = 0; n_wmma < k_len; n_wmma += WMMA_K) {
          if (n_wmma + WMMA_K <= k_len && warp_m_end - warp_m_start == WMMA_M) {
            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> p_frag;
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> v_frag;
            
            // Load P (attention probs)
            wmma::load_matrix_sync(p_frag, &sP[warp_m_start * LDP + n_wmma], LDP);
            
            // Load V (transpose view for WMMA: need [N,K] to multiply [M,N] @ [N,K])
            wmma::load_matrix_sync(v_frag, &sV[n_wmma * LDV + d_wmma], LDV);
            
            // O += P @ V
            wmma::mma_sync(o_frag, p_frag, v_frag, o_frag);
          }
        }

        // Accumulate into sO (add to scaled previous values)
        float o_tile[8];  // WMMA accumulator fragment has 8 elements per thread
        wmma::store_matrix_sync(o_tile, o_frag, WMMA_N, wmma::mem_row_major);
        
        for (int i = 0; i < 8; ++i) {
          int row_offset = i / 4;  // Which row within 16-row tile
          int col_offset = (i % 4);  // Which col within 16-col tile
          
          int m_idx = warp_m_start + row_offset + (lane_id / 4) * 2;
          int d_idx = d_wmma + col_offset * 4 + (lane_id % 4);
          
          if (m_idx < q_tile_len && d_idx < D) {
            atomicAdd(&sO[m_idx * (D + PAD) + d_idx], o_tile[i]);
          }
        }
      }
    }
    __syncthreads();
  }

  // Epilogue: Normalize
  for (int m_local = warp_id; m_local < q_tile_len; m_local += WARPS_PER_BLOCK) {
    const int m_abs = q_tile_start + m_local;
    float l = sL[m_local];

    for (int d = lane_id; d < D; d += 32) {
      float o_val = sO[m_local * (D + PAD) + d];
      o_val = (l > 0.0f) ? (o_val / l) : 0.0f;
      O_bh[m_abs * D + d] = __float2half(o_val);
    }
  }
}

void launch_flash3_v7_wmma_pv(
    const half* Q, const half* K, const half* V, half* O,
    int B, int H, int S, int D,
    bool is_causal,
    cudaStream_t stream)
{
  const float scale = 1.0f / sqrtf((float)D);

  dim3 block(WARPS_PER_BLOCK * 32);
  dim3 grid((S + M_TILE - 1) / M_TILE, H, B);

  const size_t half_elems = M_TILE * (D + PAD) + N_TILE * (D + PAD) + N_TILE * (D + PAD);
  const size_t float_elems = M_TILE * (N_TILE + PAD) + M_TILE + M_TILE + M_TILE * (D + PAD);
  const size_t smem_bytes = half_elems * sizeof(half) + float_elems * sizeof(float) + 4;

  cudaFuncSetAttribute(
      flash3_wmma_full_kernel,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      (int)smem_bytes);

  flash3_wmma_full_kernel<<<grid, block, smem_bytes, stream>>>(
      Q, K, V, O, B, H, S, D, (int)is_causal, scale);
}


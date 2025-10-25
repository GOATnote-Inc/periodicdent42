#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <cstdint>
#include <cmath>

using namespace nvcuda;

// --- Tunables (L4 sm_89, Full WMMA) ---
#define HEAD_DIM 64
#define TILE_M   32      // Q rows per block
#define TILE_N   32      // KV rows per tile
#define NUM_WARPS 8
#define THREADS_PER_BLOCK (NUM_WARPS * 32)
#define D_PAD    80      // 16B aligned (64 + 16)

// WMMA tile size (16×16×16 for FP16 on sm_89)
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

// Warp reductions (FP32)
__device__ __forceinline__ float warp_reduce_sum(float v){
    #pragma unroll
    for (int o=16; o>0; o>>=1) v += __shfl_down_sync(0xffffffff, v, o);
    return v;
}

__device__ __forceinline__ float warp_reduce_max(float v){
    #pragma unroll
    for (int o=16; o>0; o>>=1) v = fmaxf(v, __shfl_down_sync(0xffffffff, v, o));
    return v;
}

// Sim-FP8 dequant
__device__ __forceinline__ float dequant_sim_fp8(uint8_t u, float s){
    float x = (float(u) / 255.0f) * (2.0f * 448.0f) - 448.0f;
    return x * s;
}

__launch_bounds__(THREADS_PER_BLOCK, 2)
__global__ void sdpa_fp8_wmma_kernel(
    const uint8_t* __restrict__ Q,   // [B,H,S,D]
    const uint8_t* __restrict__ K,   // [B,H,S,D]
    const uint8_t* __restrict__ V,   // [B,H,S,D]
    const float*   __restrict__ Qs,  // [H]
    const float*   __restrict__ Ks,  // [H]
    const float*   __restrict__ Vs,  // [H]
    half*          __restrict__ O,   // [B,H,S,D]
    int B, int H, int S, int D, float softmax_scale
){
    const int b = blockIdx.z;
    const int h = blockIdx.y;
    const int q_block = blockIdx.x;
    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane    = tid & 31;

    const int q_start = q_block * TILE_M;
    const int q_end   = min(q_start + TILE_M, S);
    const int rows_in_tile = q_end - q_start;
    if (rows_in_tile <= 0) return;

    const size_t BHSD = (size_t)S * D;
    const uint8_t* Qbh = Q + ((size_t)b * H + h) * BHSD;
    const uint8_t* Kbh = K + ((size_t)b * H + h) * BHSD;
    const uint8_t* Vbh = V + ((size_t)b * H + h) * BHSD;
    half*         Obh  = O + ((size_t)b * H + h) * BHSD;

    const float q_s = Qs[h], k_s = Ks[h], v_s = Vs[h];

    // --- Shared memory ---
    // For WMMA: need proper layout (row-major for A, col-major for B)
    __shared__ alignas(16) half sQ[TILE_M][D_PAD];      // 5.1 KB (row-major)
    __shared__ alignas(16) half sK_T[D_PAD][TILE_N];    // 5.1 KB (col-major, transposed for WMMA)
    __shared__ alignas(16) half sV[TILE_N][D_PAD];      // 5.1 KB (row-major)
    
    __shared__ float kLUT[256];  // K dequant lookup
    __shared__ float vLUT[256];  // V dequant lookup
    __shared__ float m_smem[TILE_M];
    __shared__ float l_smem[TILE_M];
    __shared__ alignas(16) float U_smem[TILE_M][D_PAD]; // 10.2 KB
    
    // Attention scores in SMEM (for softmax)
    __shared__ float S_smem[TILE_M][TILE_N];  // 4 KB

    // --- Build LUTs ---
    if (tid < 256) {
        const int u = tid;
        float x = (float(u) / 255.0f) * (2.0f * 448.0f) - 448.0f;
        kLUT[u] = x * k_s;
        vLUT[u] = x * v_s;
    }

    // --- Load Q tile (uint8) and convert to FP16 (row-major) ---
    for (int idx = tid; idx < rows_in_tile * D; idx += blockDim.x) {
        int r = idx / D;
        int d = idx % D;
        uint8_t q_u8 = Qbh[(size_t)(q_start + r) * D + d];
        float f = dequant_sim_fp8(q_u8, q_s);
        sQ[r][d] = __float2half(f);
    }

    // Init stats and U
    for (int idx = tid; idx < rows_in_tile * D; idx += blockDim.x) {
        int r = idx / D;
        int d = idx % D;
        U_smem[r][d] = 0.f;
    }
    for (int r = tid; r < rows_in_tile; r += blockDim.x) {
        m_smem[r] = -INFINITY;
        l_smem[r] = 0.f;
    }
    __syncthreads();

    const int nTiles = (S + TILE_N - 1) / TILE_N;

    for (int t = 0; t < nTiles; ++t) {
        const int kv_start = t * TILE_N;
        const int kv_end   = min(kv_start + TILE_N, S);
        const int kv_len   = kv_end - kv_start;

        // --- Load K tile and convert to FP16 (transposed for WMMA) ---
        // We need K^T for Q@K^T, so load as col-major: sK_T[d][n]
        for (int idx = tid; idx < kv_len * D; idx += blockDim.x) {
            int n = idx / D, d = idx % D;
            uint8_t k_u8 = Kbh[(size_t)(kv_start + n) * D + d];
            sK_T[d][n] = __float2half(kLUT[k_u8]);
        }

        // --- Load V tile and convert to FP16 (row-major) ---
        for (int idx = tid; idx < kv_len * D; idx += blockDim.x) {
            int n = idx / D, d = idx % D;
            uint8_t v_u8 = Vbh[(size_t)(kv_start + n) * D + d];
            sV[n][d] = __float2half(vLUT[v_u8]);
        }
        __syncthreads();

        // --- WMMA: Q @ K^T → S ---
        // Q: [TILE_M, D] row-major
        // K^T: [D, TILE_N] col-major (same as K: [TILE_N, D] row-major transposed)
        // S: [TILE_M, TILE_N]
        //
        // WMMA computes C = A @ B where:
        // - A is MxK (row-major or col-major)
        // - B is KxN (row-major or col-major)
        // - C is MxN
        //
        // For Q@K^T:
        // - A = Q[TILE_M, D] row-major
        // - B = K^T[D, TILE_N] col-major
        // - C = S[TILE_M, TILE_N]
        //
        // TILE_M=32, D=64, TILE_N=32
        // Need to decompose into 16×16×16 WMMA tiles
        // Each warp handles 2×2 output tiles (32÷16=2 in each dimension)

        // Warp assignment: 8 warps, 2×2 tiles per warp
        // Warp 0: rows [0,15], cols [0,15]
        // Warp 1: rows [0,15], cols [16,31]
        // Warp 2: rows [16,31], cols [0,15]
        // Warp 3: rows [16,31], cols [16,31]
        // Warps 4-7: duplicate work or idle (we have 8 warps but only need 4)

        const int warp_m = (warp_id / 2) * 16;  // 0 or 16
        const int warp_n = (warp_id % 2) * 16;  // 0 or 16
        
        if (warp_id < 4) {  // Only first 4 warps do WMMA compute
            // Declare WMMA fragments
            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag;
            wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

            // Initialize accumulator
            wmma::fill_fragment(c_frag, 0.0f);

            // Compute Q[warp_m:warp_m+16, :] @ K^T[:, warp_n:warp_n+16]
            // D=64 → 4 WMMA_K tiles (64÷16=4)
            #pragma unroll
            for (int k_tile = 0; k_tile < 4; ++k_tile) {
                int k_offset = k_tile * WMMA_K;
                
                // Load A: Q[warp_m:warp_m+16, k_offset:k_offset+16]
                wmma::load_matrix_sync(a_frag, &sQ[warp_m][k_offset], D_PAD);
                
                // Load B: K^T[k_offset:k_offset+16, warp_n:warp_n+16]
                wmma::load_matrix_sync(b_frag, &sK_T[k_offset][warp_n], D_PAD);
                
                // Compute C += A @ B
                wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
            }

            // Store result to S_smem[warp_m:warp_m+16, warp_n:warp_n+16]
            wmma::store_matrix_sync(&S_smem[warp_m][warp_n], c_frag, TILE_N, wmma::mem_row_major);
        }
        __syncthreads();

        // --- Apply softmax_scale and compute online softmax ---
        for (int r = warp_id; r < rows_in_tile; r += NUM_WARPS) {
            float S_row[TILE_N];
            
            // Load scores and apply scale
            for (int n = lane; n < kv_len; n += 32) {
                S_row[n] = S_smem[r][n] * softmax_scale;
            }
            
            // Broadcast to all lanes (WMMA output is per-warp, need all lanes to have same values)
            #pragma unroll
            for (int n = 0; n < kv_len; ++n) {
                float val = 0.f;
                if (lane < kv_len) val = S_row[n];
                // Each lane gets value from lane n (if n < 32)
                if (n < 32) {
                    S_row[n] = __shfl_sync(0xffffffff, val, n);
                }
            }

            // Online softmax update
            float m_old = m_smem[r];
            float m_new = m_old;
            #pragma unroll
            for (int n = 0; n < kv_len; ++n) {
                m_new = fmaxf(m_new, S_row[n]);
            }

            float l_old = l_smem[r];
            float l_add = 0.f;
            #pragma unroll
            for (int n = 0; n < kv_len; ++n) {
                S_row[n] = __expf(S_row[n] - m_new);
                l_add += S_row[n];
            }
            
            float rescale = __expf(m_old - m_new);
            float l_new = l_old * rescale + l_add;

            // Scale U
            for (int d = lane; d < D; d += 32) {
                U_smem[r][d] *= rescale;
            }

            // Accumulate P·V (still using scalar for now, can WMMA-ify later)
            for (int n = 0; n < kv_len; ++n) {
                float p = S_row[n];
                for (int d = lane; d < D; d += 32) {
                    float v = __half2float(sV[n][d]);
                    U_smem[r][d] += p * v;
                }
            }

            if (lane == 0) {
                m_smem[r] = m_new;
                l_smem[r] = l_new;
            }
        }
        __syncthreads();
    }

    // --- Write O = U / l ---
    for (int r = warp_id; r < rows_in_tile; r += NUM_WARPS) {
        float l_final = l_smem[r];
        half* out = Obh + (size_t)(q_start + r) * D;
        for (int d = lane; d < D; d += 32) {
            float o = U_smem[r][d] / l_final;
            out[d] = __float2half(o);
        }
    }
}


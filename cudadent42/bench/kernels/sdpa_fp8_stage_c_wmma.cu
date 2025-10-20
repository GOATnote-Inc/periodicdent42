#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <cstdint>
#include <cmath>
#include <cstdio>

using namespace nvcuda;

// Debug flag: enable with -DDEBUG_PRINT during compilation
// #define DEBUG_PRINT 1

// --- Tunables (L4 sm_89, full WMMA) ---
#define HEAD_DIM 64
#define TILE_M   32      // Q rows per block (2 WMMA tiles)
#define TILE_N   32      // KV rows per tile (2 WMMA tiles)
#define NUM_WARPS 4      // REDUCED for WMMA (each warp handles one 16×16 tile)
#define THREADS_PER_BLOCK (NUM_WARPS * 32)
#define D_PAD    64      // No padding needed for WMMA (64 is 16-aligned)

// WMMA tile size
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

// Sim-FP8 dequant (symmetric, zero maps exactly to 0)
__device__ __forceinline__ float dequant_sim_fp8(uint8_t u, float s){
    constexpr float INV_MAX = 1.0f / 127.0f;
    float centered = (static_cast<float>(static_cast<int>(u) - 128)) * INV_MAX;
    return centered * 448.0f * s;
}

__launch_bounds__(THREADS_PER_BLOCK, 4)
__global__ void sdpa_fp8_stage_c_wmma_kernel(
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
    // Q: row-major [TILE_M][D]
    // K: col-major [D][TILE_N] (transposed for WMMA)
    // V: row-major [TILE_N][D]
    __shared__ alignas(16) half sQ[TILE_M][D_PAD];     // 4 KB
    __shared__ alignas(16) half sKT[D_PAD][TILE_N];    // 4 KB (transposed!)
    __shared__ alignas(16) half sV[TILE_N][D_PAD];     // 4 KB
    
    __shared__ float kLUT[256];  // K dequant lookup
    __shared__ float vLUT[256];  // V dequant lookup
    __shared__ float m_smem[TILE_M];
    __shared__ float l_smem[TILE_M];
    __shared__ alignas(16) float U_smem[TILE_M][D_PAD];  // 8 KB

    // Total SMEM: 4+4+4+8+3 = 23 KB < 48 KB ✅

    // --- Build LUTs ---
    if (tid < 256) {
        const int u = tid;
        constexpr float INV_MAX = 1.0f / 127.0f;
        float centered = (static_cast<float>(u) - 128.0f) * INV_MAX;
        float decoded = centered * 448.0f;
        kLUT[u] = decoded * k_s;
        vLUT[u] = decoded * v_s;
    }

    // --- Load Q tile (uint8→FP16, row-major) ---
    for (int idx = tid; idx < rows_in_tile * D; idx += blockDim.x) {
        int r = idx / D;
        int d = idx % D;
        uint8_t q_u8 = Qbh[(size_t)(q_start + r) * D + d];
        float f = dequant_sim_fp8(q_u8, q_s);
        sQ[r][d] = __float2half(f);
    }

#ifdef DEBUG_PRINT
    __syncthreads();
    if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && tid == 0) {
        printf("[DEBUG] Q tile loaded (row 0, d=0:5): ");
        for (int d = 0; d < 5; d++) {
            printf("%.4f ", __half2float(sQ[0][d]));
        }
        printf("\n");
    }
#endif

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

        // --- Load K tile (uint8→FP16, TRANSPOSED to col-major) ---
        // sKT[d][n] = K[n][d] (transposed for WMMA B matrix)
        for (int idx = tid; idx < kv_len * D; idx += blockDim.x) {
            int n = idx / D;
            int d = idx % D;
            uint8_t k_u8 = Kbh[(size_t)(kv_start + n) * D + d];
            sKT[d][n] = __float2half(kLUT[k_u8]);  // Transposed!
        }

        // --- Load V tile (uint8→FP16, row-major) ---
        for (int idx = tid; idx < kv_len * D; idx += blockDim.x) {
            int n = idx / D;
            int d = idx % D;
            uint8_t v_u8 = Vbh[(size_t)(kv_start + n) * D + d];
            sV[n][d] = __float2half(vLUT[v_u8]);
        }
        __syncthreads();

#ifdef DEBUG_PRINT
        if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && tid == 0 && t == 0) {
            printf("[DEBUG] V tile loaded (row 0, d=0:5): ");
            for (int d = 0; d < 5; d++) {
                printf("%.4f ", __half2float(sV[0][d]));
            }
            printf("\n");
        }
#endif

        // =========================================
        // WMMA COMPUTE: Q @ K^T → S (32×32)
        // =========================================
        // Each warp handles one 16×16 output tile
        // 4 warps cover 2×2 = 32×32 output
        
        // Warp mapping:
        // warp_id=0 → S[0:16,  0:16]
        // warp_id=1 → S[0:16, 16:32]
        // warp_id=2 → S[16:32, 0:16]
        // warp_id=3 → S[16:32,16:32]
        
        const int warp_m = (warp_id / 2) * WMMA_M;  // 0 or 16
        const int warp_n = (warp_id % 2) * WMMA_N;  // 0 or 16

        // WMMA fragments
        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag;
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> c_frag;

        // Zero accumulator
        wmma::fill_fragment(c_frag, __float2half(0.0f));

        // Compute Q@K^T in 16×16×16 chunks (4 chunks for D=64)
        #pragma unroll
        for (int k = 0; k < D; k += WMMA_K) {
            // Load A: Q[warp_m:warp_m+16, k:k+16] (row-major)
            wmma::load_matrix_sync(a_frag, &sQ[warp_m][k], D_PAD);
            
            // Load B: K^T[k:k+16, warp_n:warp_n+16] (col-major)
            wmma::load_matrix_sync(b_frag, &sKT[k][warp_n], TILE_N);
            
            // MMA: C += A * B
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }

        // Store S to shared memory temporarily for softmax
        __shared__ alignas(16) half sS[TILE_M][TILE_N];
        wmma::store_matrix_sync(&sS[warp_m][warp_n], c_frag, TILE_N, wmma::mem_row_major);
        __syncthreads();

#ifdef DEBUG_PRINT
        if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && tid == 0 && t == 0) {
            printf("[DEBUG] Q@K^T scores (row 0, n=0:5): ");
            for (int n = 0; n < 5; n++) {
                printf("%.4f ", __half2float(sS[0][n]));
            }
            printf("\n");
        }
#endif

        // =========================================
        // ONLINE SOFTMAX (per row, scalar path)
        // =========================================
        // Each warp handles 32/4 = 8 rows
        for (int r = warp_id; r < rows_in_tile; r += NUM_WARPS) {
            // PRIORITY 1 FIX: Each lane loads ALL scores (no stride, no broadcast)
            // Previous bug: Only lane N loaded S_row[N], leaving most elements uninitialized
            // Correct: Each lane loads full S_row[] sequentially → all lanes see same data
            float S_row[TILE_N];
            #pragma unroll
            for (int n = 0; n < kv_len; ++n) {
                S_row[n] = __half2float(sS[r][n]) * softmax_scale;
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

#ifdef DEBUG_PRINT
            if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && r == 0 && warp_id == 0 && lane == 0 && t == 0) {
                printf("[DEBUG] Softmax (row 0): m_old=%.4f m_new=%.4f l_old=%.4f l_add=%.4f rescale=%.4f\n",
                       m_old, m_new, l_old, l_add, rescale);
                printf("[DEBUG] Attention weights P[0:5]: ");
                for (int n = 0; n < 5; n++) {
                    printf("%.4f ", S_row[n]);
                }
                printf("\n");
            }
#endif

            // Scale U
            for (int d = lane; d < D; d += 32) {
                U_smem[r][d] *= rescale;
            }

            // Accumulate P·V (unnormalized)
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
        
#ifdef DEBUG_PRINT
        if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && r == 0 && warp_id == 0 && lane == 0) {
            printf("[DEBUG] Final output (row 0): l_final=%.4f\n", l_final);
            printf("[DEBUG] U_smem[0][0:5] / l_final: ");
            for (int d = 0; d < 5; d++) {
                printf("%.4f ", U_smem[0][d] / l_final);
            }
            printf("\n");
        }
#endif
        
        for (int d = lane; d < D; d += 32) {
            float o = U_smem[r][d] / l_final;
            out[d] = __float2half(o);
        }
    }
}

extern "C" void launch_sdpa_fp8_stage_c_wmma(
    const void* Q,
    const void* K,
    const void* V,
    const float* Q_scale,
    const float* K_scale,
    const float* V_scale,
    half* O,
    int B, int H, int S, int D,
    float softmax_scale,
    cudaStream_t stream
) {
    dim3 grid((S + TILE_M - 1) / TILE_M, H, B);
    dim3 block(THREADS_PER_BLOCK);

    sdpa_fp8_stage_c_wmma_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<const uint8_t*>(Q),
        reinterpret_cast<const uint8_t*>(K),
        reinterpret_cast<const uint8_t*>(V),
        Q_scale,
        K_scale,
        V_scale,
        O,
        B, H, S, D,
        softmax_scale
    );
}


// flashcore_fused_wmma_cpasync.cu
// -------------------------------------------------------------
// Fused QK^T + online softmax + PV using WMMA and cp.async.
// Tile: 64x32 (M x N), D=64, no atomics, no K-transpose in SMEM.
// -------------------------------------------------------------
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>
#include <cstdint>
#include <cmath>
#include <stdexcept>
#include "cuda_check.h"

using namespace nvcuda;
namespace cg = cooperative_groups;

// ------------------------- Tunables -------------------------
#ifndef HEAD_DIM
#define HEAD_DIM 64
#endif

#ifndef TILE_M
#define TILE_M 64      // rows of Q per CTA
#endif

#ifndef TILE_N
#define TILE_N 32      // rows of K/V per CTA (columns of scores)
#endif

#ifndef NUM_WARPS
#define NUM_WARPS 4    // 4 warps -> 128 threads (warp grid 4x1 over M)
#endif

#define THREADS_PER_BLOCK (NUM_WARPS * 32)
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16
#define STAGES 2        // cp.async double-buffering

// --------------------- Helpers & Layout ---------------------
constexpr int smem_stride(int d) {
    // pad to 16 and add +16 when multiple of 32 to prevent bank conflicts
    return (d % 32 == 0) ? d + 16 : ((d + 15) / 16) * 16;
}
#define HEAD_DIM_SMEM smem_stride(HEAD_DIM)

static_assert(HEAD_DIM % WMMA_K == 0, "HEAD_DIM must be multiple of 16");
static_assert(TILE_M % WMMA_M == 0,    "TILE_M must be multiple of 16");
static_assert(TILE_N % WMMA_N == 0,    "TILE_N must be multiple of 16");
static_assert(NUM_WARPS == (TILE_M / WMMA_M), "Use 4 warps for 64x32 (4x1 warp grid)");

#if (HEAD_DIM % 8) != 0
#error "Vectorized int4 loads require HEAD_DIM % 8 == 0"
#endif

template <typename T>
__device__ __forceinline__ void vload_int4(T* __restrict__ dst, const T* __restrict__ src) {
    *reinterpret_cast<int4*>(dst) = *reinterpret_cast<const int4*>(src);
}

__device__ __forceinline__ float clamp_exp(float x) {
    // Clamp exponent argument for stability (+/-20 is safe and fast)
    x = fminf(20.0f, fmaxf(-20.0f, x));
    return __expf(x);
}

// --------------------------- Kernel -------------------------
__global__ void __launch_bounds__(THREADS_PER_BLOCK, 2)
flashcore_fused_wmma_cpasync_kernel(
    const half* __restrict__ Q,   // [B,H,S,D]
    const half* __restrict__ K,   // [B,H,S,D]  (row-major)
    const half* __restrict__ V,   // [B,H,S,D]
    half* __restrict__ O,         // [B,H,S,D]
    float softmax_scale,          // 1/sqrt(D)
    int B, int H, int S, int D    // D==HEAD_DIM (64)
) {
    // ---- CTA coordinates ----
    const int batch_idx = blockIdx.z;
    const int head_idx  = blockIdx.y;
    const int qtile_idx = blockIdx.x;

    const int tid   = threadIdx.x;
    const int warp  = tid / 32;
    const int lane  = tid % 32;

    // Warp grid: 4x1 along M (each warp handles 16 rows × full 32 cols)
    const int warp_m      = warp;           // 0..3
    const int warp_m_row0 = warp_m * WMMA_M;

    // Query range for this CTA
    const int q_row0      = qtile_idx * TILE_M;
    if (q_row0 >= S) return;
    const int q_rows      = min(TILE_M, S - q_row0);

    // ---- Base pointers ----
    const size_t stride_bh = (size_t)S * D;
    const half* Q_bh = Q + ((size_t)batch_idx * H + head_idx) * stride_bh;
    const half* K_bh = K + ((size_t)batch_idx * H + head_idx) * stride_bh;
    const half* V_bh = V + ((size_t)batch_idx * H + head_idx) * stride_bh;
    half*       O_bh = O + ((size_t)batch_idx * H + head_idx) * stride_bh;

    // ---- Dynamic SMEM layout ----
    extern __shared__ unsigned char smem_raw[];
    unsigned char* ptr = smem_raw;

    auto align16 = [&](size_t bytes) {
        size_t rounded = (bytes + 15) & ~size_t(15);
        unsigned char* p = ptr;
        ptr += rounded;
        return p;
    };

    // sQ: [TILE_M][HEAD_DIM_SMEM] half
    half* sQ = reinterpret_cast<half*>(align16(TILE_M * HEAD_DIM_SMEM * sizeof(half)));

    // sK stages: [STAGES][TILE_N][HEAD_DIM_SMEM] half  (no transpose, row-major N×D)
    half* sK[STAGES];
    for (int s = 0; s < STAGES; ++s) {
        sK[s] = reinterpret_cast<half*>(align16(TILE_N * HEAD_DIM_SMEM * sizeof(half)));
    }

    // sV stages: [STAGES][TILE_N][HEAD_DIM_SMEM] half
    half* sV[STAGES];
    for (int s = 0; s < STAGES; ++s) {
        sV[s] = reinterpret_cast<half*>(align16(TILE_N * HEAD_DIM_SMEM * sizeof(half)));
    }

    // sS_f32: [TILE_M][TILE_N] float  (scores scratch per tile)
    float* sS = reinterpret_cast<float*>(align16(TILE_M * TILE_N * sizeof(float)));

    // sP_fp16: [TILE_M][TILE_N] half  (P tile for WMMA PV)
    half* sP = reinterpret_cast<half*>(align16(TILE_M * TILE_N * sizeof(half)));

    // m/l stats: [TILE_M] float
    float* m_smem = reinterpret_cast<float*>(align16(TILE_M * sizeof(float)));
    float* l_smem = reinterpret_cast<float*>(align16(TILE_M * sizeof(float)));

    // U_smem: [TILE_M][HEAD_DIM_SMEM] float (unnormalized PV accumulator)
    float* U_smem = reinterpret_cast<float*>(align16(TILE_M * HEAD_DIM_SMEM * sizeof(float)));

    // ---- Prefetch Q once (pre‑scaled) ----
    const half scale_h = __float2half(softmax_scale);

    // Vectorized loads per-CTA: int4 (8 halfs = 16B)
    const int Q_vec = D / 8;  // number of int4 per row
    for (int i = tid; i < q_rows * Q_vec; i += THREADS_PER_BLOCK) {
        int m  = i / Q_vec;
        int dv = i % Q_vec;
        const half* src = Q_bh + (size_t)(q_row0 + m) * D + dv * 8;
        half*       dst = &sQ[m * HEAD_DIM_SMEM + dv * 8];
        vload_int4(dst, src);
        #pragma unroll
        for (int t = 0; t < 8; ++t) dst[t] = __hmul(dst[t], scale_h);
    }
    // Zero-pad Q (rows beyond q_rows or D pad to HEAD_DIM_SMEM)
    for (int i = tid + q_rows * HEAD_DIM_SMEM; i < TILE_M * HEAD_DIM_SMEM; i += THREADS_PER_BLOCK)
        reinterpret_cast<half*>(sQ)[i] = __float2half(0.f);

    // Init stats + U
    for (int m = tid; m < TILE_M; m += THREADS_PER_BLOCK) { m_smem[m] = -INFINITY; l_smem[m] = 0.f; }
    for (int i = tid; i < TILE_M * HEAD_DIM_SMEM; i += THREADS_PER_BLOCK) U_smem[i] = 0.f;

    __syncthreads();

    // ---- Tile loop over K/V ----
    const int kv_tiles = (S + TILE_N - 1) / TILE_N;
    auto block = cg::this_thread_block();

    // Stage 0 prefetch
    int kv0 = 0;
    int kv_start = kv0 * TILE_N;
    int kv_len0  = min(TILE_N, S - kv_start);

    if (kv_len0 > 0) {
        size_t bytes = (size_t)kv_len0 * D * sizeof(half);
        cg::memcpy_async(block, sK[0], K_bh + (size_t)kv_start * D, bytes);
        cg::memcpy_async(block, sV[0], V_bh + (size_t)kv_start * D, bytes);
        cg::wait(block);  // Wait for all pending memcpy_async
    }
    __syncthreads();

    for (int kv = 0; kv < kv_tiles; ++kv) {
        const int stage     = kv & 1;
        const int next_kv   = kv + 1;
        const int kv_base   = kv * TILE_N;
        const int kv_len    = min(TILE_N, S - kv_base);

        // Launch next-stage prefetch
        if (next_kv < kv_tiles) {
            const int next_stage = next_kv & 1;
            const int next_base  = next_kv * TILE_N;
            const int next_len   = min(TILE_N, S - next_base);
            size_t bytes = (size_t)next_len * D * sizeof(half);
            cg::memcpy_async(block, sK[next_stage], K_bh + (size_t)next_base * D, bytes);
            cg::memcpy_async(block, sV[next_stage], V_bh + (size_t)next_base * D, bytes);
        }

        // Zero-pad K/V stage tails (D pad and N tail)
        // K: [kv_len x D] filled, pad to [TILE_N x HEAD_DIM_SMEM]
        for (int i = tid + kv_len * D; i < TILE_N * D; i += THREADS_PER_BLOCK)
            sK[stage][i] = __float2half(0.f);
        for (int i = tid + TILE_N * D; i < TILE_N * HEAD_DIM_SMEM; i += THREADS_PER_BLOCK)
            sK[stage][i] = __float2half(0.f);

        for (int i = tid + kv_len * D; i < TILE_N * D; i += THREADS_PER_BLOCK)
            sV[stage][i] = __float2half(0.f);
        for (int i = tid + TILE_N * D; i < TILE_N * HEAD_DIM_SMEM; i += THREADS_PER_BLOCK)
            sV[stage][i] = __float2half(0.f);

        // Zero sS and sP for rows beyond q_rows or N tail
        for (int i = tid; i < TILE_M * TILE_N; i += THREADS_PER_BLOCK) {
            int m = i / TILE_N; int n = i % TILE_N;
            if (m < q_rows && n < kv_len) { sS[i] = 0.f; /* sP filled later */ }
            else                           { sS[i] = 0.f; sP[i] = __float2half(0.f); }
        }
        __syncthreads();

        // ---- QK^T on Tensor Cores (A=row_major, B=col_major) ----
        // Each warp computes 16x32 by issuing two 16x16 MMAs over N=32.
        for (int part = 0; part < TILE_N / WMMA_N; ++part) { // part=0..1 for 32
            const int n_off = part * WMMA_N;

            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_qk;
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_qk; // NOTE: col_major -> no K^T in smem
            wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_qk;
            wmma::fill_fragment(c_qk, 0.0f);

            #pragma unroll
            for (int k = 0; k < HEAD_DIM; k += WMMA_K) {
                // Q: [TILE_M x D] ld = HEAD_DIM_SMEM
                wmma::load_matrix_sync(a_qk, &sQ[(warp_m_row0) * HEAD_DIM_SMEM + k], HEAD_DIM_SMEM);
                // K: [TILE_N x D] row-major in smem -> use col_major with ld = HEAD_DIM_SMEM
                wmma::load_matrix_sync(b_qk, &sK[stage][n_off * HEAD_DIM_SMEM + k], HEAD_DIM_SMEM);
                wmma::mma_sync(c_qk, a_qk, b_qk, c_qk);
            }

            // Scatter to shared sS (scores) for softmax
            wmma::store_matrix_sync(&sS[(warp_m_row0) * TILE_N + n_off], c_qk, TILE_N, wmma::mem_row_major);
        }

        __syncthreads();

        // ---- Online softmax update (FP32) per row ----
        for (int m_local = warp_m_row0 + lane; m_local < warp_m_row0 + WMMA_M; m_local += 32) {
            if (m_local >= q_rows) continue;

            // 1) Max on this kv slice
            float m_tile = -INFINITY;
            #pragma unroll
            for (int n = 0; n < kv_len; ++n) {
                m_tile = fmaxf(m_tile, sS[m_local * TILE_N + n]);
            }

            // 2) Merge with running max, compute scale_old
            const float m_old = m_smem[m_local];
            const float m_new = fmaxf(m_old, m_tile);
            const float scale_old = __expf(m_old - m_new);

            // 3) Scale running U by scale_old
            float* U_row = &U_smem[m_local * HEAD_DIM_SMEM];
            #pragma unroll
            for (int d = 0; d < HEAD_DIM; ++d) U_row[d] *= scale_old;

            // 4) Compute l_add on this kv slice; also materialize unnormalized P (FP16) into sP
            float l_add = 0.f;
            #pragma unroll
            for (int n = 0; n < kv_len; ++n) {
                const float s = sS[m_local * TILE_N + n];
                const float p = clamp_exp(s - m_new);   // unnormalized
                l_add += p;
                sP[m_local * TILE_N + n] = __float2half(p);
            }
            // zero pad remainder of N
            for (int n = kv_len; n < TILE_N; ++n) sP[m_local * TILE_N + n] = __float2half(0.f);

            // 5) Update running l and m
            const float l_old = l_smem[m_local];
            l_smem[m_local] = l_old * scale_old + l_add;
            m_smem[m_local] = m_new;
        }

        __syncthreads();

        // ---- PV on Tensor Cores (no atomics, warp-local rows) ----
        // Each warp computes (16 x 64) by accumulating 4 d-tiles, each with 2 k-partitions (N=32 -> 2x16).
        for (int d_tile = 0; d_tile < HEAD_DIM / WMMA_N; ++d_tile) {
            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half,  wmma::row_major> a_pv;
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half,  wmma::row_major> b_pv;
            wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_pv;
            wmma::fill_fragment(c_pv, 0.0f);

            #pragma unroll
            for (int n_part = 0; n_part < TILE_N / WMMA_K; ++n_part) { // 2 parts (16+16)
                const int n_off = n_part * WMMA_K;

                // A=P[16x32] -> 2 loads of 16
                wmma::load_matrix_sync(a_pv, &sP[(warp_m_row0) * TILE_N + n_off], TILE_N);
                // B=V[32x64] row-major: load 16x16 tile starting at [n_off, d_tile*16], ld = HEAD_DIM_SMEM
                wmma::load_matrix_sync(b_pv, &sV[stage][n_off * HEAD_DIM_SMEM + d_tile * WMMA_N], HEAD_DIM_SMEM);
                wmma::mma_sync(c_pv, a_pv, b_pv, c_pv);
            }

            // Accumulate into U_smem
            // We store to a 16x16 temporary on stack then add to U row. This is fast and avoids atomics.
            float tmp[WMMA_M * WMMA_N];
            wmma::store_matrix_sync(tmp, c_pv, WMMA_N, wmma::mem_row_major);

            // Scatter add (row bounds guarded)
            for (int e = lane; e < WMMA_M * WMMA_N; e += 32) {
                const int r = e / WMMA_N; const int c = e % WMMA_N;
                const int m_row = warp_m_row0 + r;
                if (m_row < q_rows) {
                    U_smem[m_row * HEAD_DIM_SMEM + d_tile * WMMA_N + c] += tmp[e];
                }
            }
        }

        __syncthreads();

        // Wait for next-stage prefetch to complete before swapping
        if (next_kv < kv_tiles) {
            cg::wait(block);  // Wait for pending memcpy_async
        }
        __syncthreads();
    }

    // ---- Final normalization: O = U / l ----
    for (int i = tid; i < q_rows * D; i += THREADS_PER_BLOCK) {
        int m = i / D;
        int d = i % D;
        const float u = U_smem[m * HEAD_DIM_SMEM + d];
        const float l = fmaxf(l_smem[m], 1e-6f);
        O_bh[(size_t)(q_row0 + m) * D + d] = __float2half(u / l);
    }
}

// --------------------------- Host ---------------------------
void launch_flashcore_fused_wmma_cpasync(
    const half* Q, const half* K, const half* V, half* O,
    int B, int H, int S, int D, cudaStream_t stream = 0)
{
    // Compile-time constraints
    if (D != HEAD_DIM) { throw std::runtime_error("D must be 64 for this build"); }

    const float softmax_scale = 1.0f / sqrtf((float)D);

    const int num_qtiles = (S + TILE_M - 1) / TILE_M;
    dim3 grid(num_qtiles, H, B);
    dim3 block(THREADS_PER_BLOCK);

    // Dynamic SMEM size (must match kernel layout; keep multiples of 16)
    size_t smem_bytes = 0;
    auto roundup16 = [](size_t x) { return (x + 15) & ~size_t(15); };
    smem_bytes += roundup16(TILE_M * HEAD_DIM_SMEM * sizeof(half));                     // sQ
    smem_bytes += STAGES * roundup16(TILE_N * HEAD_DIM_SMEM * sizeof(half));            // sK
    smem_bytes += STAGES * roundup16(TILE_N * HEAD_DIM_SMEM * sizeof(half));            // sV
    smem_bytes += roundup16(TILE_M * TILE_N * sizeof(float));                           // sS
    smem_bytes += roundup16(TILE_M * TILE_N * sizeof(half));                            // sP
    smem_bytes += roundup16(TILE_M * sizeof(float));                                    // m
    smem_bytes += roundup16(TILE_M * sizeof(float));                                    // l
    smem_bytes += roundup16(TILE_M * HEAD_DIM_SMEM * sizeof(float));                    // U

    // Query device SMEM limits
    int dev = 0;
    CUDA_CHECK(cudaGetDevice(&dev));
    
    int max_optin = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&max_optin,
        cudaDevAttrMaxSharedMemoryPerBlockOptin, dev));
    
    fprintf(stderr, "[FlashCore cp.async] SMEM: requested=%zu KB, max_optin=%d KB\n",
            smem_bytes / 1024, max_optin / 1024);
    
    if (smem_bytes > (size_t)max_optin) {
        fprintf(stderr, "WARNING: Requested SMEM %zu > opt-in max %d; clamping.\n",
                smem_bytes, max_optin);
        smem_bytes = max_optin;
    }
    
    // Opt-in for larger SMEM
    CUDA_CHECK(cudaFuncSetAttribute(
        flashcore_fused_wmma_cpasync_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        (int)smem_bytes));
    
    // Prefer shared memory carveout
    cudaFuncSetAttribute(
        flashcore_fused_wmma_cpasync_kernel,
        cudaFuncAttributePreferredSharedMemoryCarveout, 100);

    // Launch with explicit error checking
    flashcore_fused_wmma_cpasync_kernel<<<grid, block, smem_bytes, stream>>>(
        Q, K, V, O, softmax_scale, B, H, S, D
    );
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}


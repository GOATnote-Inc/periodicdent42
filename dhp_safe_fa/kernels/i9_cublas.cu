// I9: cuBLAS GEMM + online softmax+PV
// Proves architecture before CUTLASS migration
// Target: <10 μs/head using optimal GEMM

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include "../include/dhp_ct_enhanced.cuh"

// Global cuBLAS handle (initialized once)
static cublasHandle_t cublas_handle = nullptr;

// Initialize cuBLAS handle (call once)
extern "C" void init_cublas() {
    if (cublas_handle == nullptr) {
        cublasCreate(&cublas_handle);
    }
}

// Softmax + P@V kernel (unchanged from I5)
__global__ void __launch_bounds__(256)
dhp_softmax_pv_from_scores(
    const __half* __restrict__ scores,  // [B*H, S, S]
    const __half* __restrict__ V,       // [B*H, S, 64]
    __half* __restrict__ out,           // [B*H, S, 64]
    const uint32_t S_max,
    const uint32_t S_actual,
    const uint32_t batch_size
) {
    const int batch_head = blockIdx.x;
    const int row = blockIdx.y * blockDim.x + threadIdx.x;
    
    if (batch_head >= batch_size || row >= S_max) return;
    
    uint32_t row_valid = ct_lt_u32(row, S_actual);
    
    float m = -INFINITY;
    float l = 0.0f;
    float acc[64];
    
    #pragma unroll
    for (int i = 0; i < 64; ++i) {
        acc[i] = 0.0f;
    }
    
    // Online softmax over precomputed scores
    for (int col = 0; col < S_max; ++col) {
        int score_idx = batch_head * S_max * S_max + row * S_max + col;
        float score = __half2float(scores[score_idx]);
        
        // Update running max
        uint32_t gt = ct_gt_f32(score, m);
        float m_new = ct_select_f32(m, score, gt);
        
        // Rescale accumulator
        float alpha = expf(m - m_new);
        l *= alpha;
        #pragma unroll
        for (int d = 0; d < 64; ++d) {
            acc[d] *= alpha;
        }
        
        // Add contribution
        float p = safe_exp(score - m_new);
        l += p;
        
        #pragma unroll
        for (int d = 0; d < 64; ++d) {
            int v_idx = batch_head * S_max * 64 + col * 64 + d;
            acc[d] += p * __half2float(V[v_idx]);
        }
        
        m = m_new;
    }
    
    // Final normalization
    float l_safe = ct_select_f32(1.0f, l, row_valid);
    #pragma unroll
    for (int d = 0; d < 64; ++d) {
        float val = acc[d] / l_safe;
        val = ct_select_f32(0.0f, val, row_valid);
        int out_idx = batch_head * S_max * 64 + row * 64 + d;
        out[out_idx] = __float2half(val);
    }
}

// Mask kernel: Apply causal mask and scale to scores
__global__ void __launch_bounds__(256)
apply_causal_mask_scale(
    __half* __restrict__ scores,        // [B*H, S, S] (inplace)
    const uint32_t S_max,
    const uint32_t S_actual,
    const uint32_t batch_size,
    const float scale
) {
    const int batch_head = blockIdx.x;
    const int row = blockIdx.y * blockDim.x + threadIdx.x;
    
    if (batch_head >= batch_size || row >= S_max) return;
    
    uint32_t row_valid = ct_lt_u32(row, S_actual);
    
    for (int col = 0; col < S_max; ++col) {
        uint32_t col_valid = ct_lt_u32(col, S_actual);
        uint32_t causal = ct_le_u32(col, row);
        uint32_t valid = ct_and_u32(ct_and_u32(row_valid, col_valid), causal);
        
        int idx = batch_head * S_max * S_max + row * S_max + col;
        float score = __half2float(scores[idx]) * scale;
        score = ct_select_f32(-INFINITY, score, valid);
        scores[idx] = __float2half(score);
    }
}

// Host-side launch function
extern "C" void launch_i9_cublas(
    const __half* Q,      // [B*H, S, 64]
    const __half* K,      // [B*H, S, 64]
    const __half* V,      // [B*H, S, 64]
    __half* scores_tmp,   // [B*H, S, S] temporary
    __half* out,          // [B*H, S, 64]
    int batch_size,
    int S_max,
    int S_actual,
    cudaStream_t stream
) {
    if (cublas_handle == nullptr) {
        init_cublas();
    }
    
    cublasSetStream(cublas_handle, stream);
    
    const float alpha = 1.0f;
    const float beta = 0.0f;
    const float scale = 0.125f; // 1/sqrt(64)
    
    // For each batch/head: scores = Q @ K^T
    // cuBLAS: C = alpha * A @ B + beta * C
    // We want: scores[b] = Q[b] @ K[b]^T
    // In column-major: C = B^T @ A^T
    // So: scores^T = K @ Q^T (column-major)
    
    for (int b = 0; b < batch_size; ++b) {
        const __half* Q_batch = Q + b * S_max * 64;
        const __half* K_batch = K + b * S_max * 64;
        __half* scores_batch = scores_tmp + b * S_max * S_max;
        
        // cublasHgemm for FP16
        cublasHgemm(
            cublas_handle,
            CUBLAS_OP_T,     // K^T
            CUBLAS_OP_N,     // Q
            S_max,           // m (rows of scores)
            S_max,           // n (cols of scores)
            64,              // k (reduction dim)
            (__half*)&alpha,
            K_batch,         // A (K)
            64,              // lda
            Q_batch,         // B (Q)
            64,              // ldb
            (__half*)&beta,
            scores_batch,    // C (scores)
            S_max            // ldc
        );
    }
    
    // Apply causal mask and scale
    dim3 grid_mask(batch_size, (S_max + 255) / 256);
    dim3 block_mask(256);
    apply_causal_mask_scale<<<grid_mask, block_mask, 0, stream>>>(
        scores_tmp, S_max, S_actual, batch_size, scale
    );
    
    // Softmax + P@V
    dim3 grid_softmax(batch_size, (S_max + 255) / 256);
    dim3 block_softmax(256);
    dhp_softmax_pv_from_scores<<<grid_softmax, block_softmax, 0, stream>>>(
        scores_tmp, V, out, S_max, S_actual, batch_size
    );
}


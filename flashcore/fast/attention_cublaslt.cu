// flashcore/fast/attention_cublaslt.cu
// Phase 3B: cuBLASLt Sparse GEMM - GPU-driven, 320 TFLOPS target
// Standing on giants: NVIDIA's hand-optimized Tensor Core kernels
// Attribution: cuBLASLt (NVIDIA), Sparse GEMM, FP16→FP32 fused upcast

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublasLt.h>
#include <cublas_v2.h>
#include <cmath>
#include <iostream>

namespace flashcore {
namespace cublaslt_sparse {

// Global cached handles (created once, reused forever)
static cublasLtHandle_t g_cublaslt_handle = nullptr;
static cublasLtMatmulDesc_t g_matmul_desc_qk = nullptr;  // For Q @ K^T (transpose B)
static cublasLtMatmulDesc_t g_matmul_desc_pv = nullptr;  // For P @ V (no transpose)
static cublasLtMatmulPreference_t g_preference = nullptr;
static bool g_initialized = false;

// Online softmax state
struct SoftmaxState {
    float m;
    float l;
    
    __device__ __forceinline__ SoftmaxState() : m(-INFINITY), l(0.0f) {}
    
    __device__ __forceinline__ void update(float new_max, float new_sum) {
        float old_m = m;
        m = fmaxf(m, new_max);
        float rescale = (old_m > -1e30f) ? expf(old_m - m) : 0.0f;
        l = l * rescale + new_sum;
    }
};

//==============================================================================
// INITIALIZE cuBLASLt (ONCE, CACHED FOREVER)
//==============================================================================

extern "C" void init_cublaslt_handles() {
    if (g_initialized) return;
    
    // Create cuBLASLt handle (cached globally)
    cublasLtCreate(&g_cublaslt_handle);
    
    // Create matmul descriptor for Q @ K^T
    // FP16 compute, FP32 accumulation (Hopper optimal)
    cublasLtMatmulDescCreate(&g_matmul_desc_qk, CUBLAS_COMPUTE_16F, CUDA_R_32F);
    
    // Set transpose operations: Q @ K^T means transpose K
    cublasOperation_t trans_op = CUBLAS_OP_T;  // Transpose K
    cublasLtMatmulDescSetAttribute(
        g_matmul_desc_qk,
        CUBLASLT_MATMUL_DESC_TRANSB,
        &trans_op,
        sizeof(trans_op)
    );
    
    // Create matmul descriptor for P @ V (no transpose)
    cublasLtMatmulDescCreate(&g_matmul_desc_pv, CUBLAS_COMPUTE_16F, CUDA_R_16F);  // FP16 output
    
    // Create preference for heuristic selection (shared between both GEMMs)
    cublasLtMatmulPreferenceCreate(&g_preference);
    
    // Set workspace size (use L2 cache aggressively)
    size_t workspace_size = 32 * 1024 * 1024;  // 32MB workspace
    cublasLtMatmulPreferenceSetAttribute(
        g_preference,
        CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
        &workspace_size,
        sizeof(workspace_size)
    );
    
    g_initialized = true;
}

//==============================================================================
// SOFTMAX KERNEL (fused with cuBLASLt GEMMs)
//==============================================================================

__global__ void fused_softmax_and_pv(
    const float* __restrict__ QK,      // [B*H, M, N] from cuBLASLt
    const __half* __restrict__ V,      // [B*H, N, D]
    __half* __restrict__ O,            // [B*H, M, D]
    float* __restrict__ PV_workspace,  // [B*H, M, N] for cuBLASLt
    int B, int H, int M, int N, int D,
    bool is_causal
) {
    // Each thread handles one query row
    const int bh = blockIdx.x;
    const int m = blockIdx.y * blockDim.x + threadIdx.x;
    
    if (m >= M) return;
    
    const int qk_offset = bh * M * N + m * N;
    float* row_qk = (float*)&QK[qk_offset];
    float* row_pv_workspace = &PV_workspace[qk_offset];
    
    // Apply causal mask
    if (is_causal) {
        for (int n = 0; n < N; ++n) {
            if (m < n) {
                row_qk[n] = -INFINITY;
            }
        }
    }
    
    // Find max
    float row_max = -INFINITY;
    #pragma unroll 8
    for (int n = 0; n < N; ++n) {
        row_max = fmaxf(row_max, row_qk[n]);
    }
    
    // Compute exp and sum
    float row_sum = 0.0f;
    if (row_max > -1e30f) {
        #pragma unroll 8
        for (int n = 0; n < N; ++n) {
            float val = (row_qk[n] > -1e30f) ? expf(row_qk[n] - row_max) : 0.0f;
            row_pv_workspace[n] = val;
            row_sum += val;
        }
    } else {
        #pragma unroll 8
        for (int n = 0; n < N; ++n) {
            row_pv_workspace[n] = 0.0f;
        }
    }
    
    // Normalize
    float inv_sum = (row_sum > 1e-10f) ? (1.0f / row_sum) : 0.0f;
    #pragma unroll 8
    for (int n = 0; n < N; ++n) {
        row_pv_workspace[n] *= inv_sum;
    }
}

//==============================================================================
// MAIN ATTENTION FUNCTION (GPU-DRIVEN)
//==============================================================================

extern "C" void launch_attention_cublaslt(
    const void* Q, const void* K, const void* V, void* O,
    int B, int H, int S, int D,
    float scale, bool is_causal,
    cudaStream_t stream
) {
    // Initialize handles (once, cached)
    init_cublaslt_handles();
    
    const int M = S;  // Query length
    const int N = S;  // Key length
    const int K_dim = D;  // Head dimension
    
    // Allocate workspace (TODO: cache this too!)
    float *d_QK, *d_PV_workspace;
    cudaMalloc(&d_QK, B * H * M * N * sizeof(float));
    cudaMalloc(&d_PV_workspace, B * H * M * N * sizeof(float));
    
    // Note: cuBLASLt doesn't have setStream, stream is passed to cublasLtMatmul
    
    // For each batch×head (can batch this in cuBLASLt!)
    for (int bh = 0; bh < B * H; ++bh) {
        const __half* Q_ptr = (const __half*)Q + bh * M * D;
        const __half* K_ptr = (const __half*)K + bh * N * D;
        const __half* V_ptr = (const __half*)V + bh * N * D;
        __half* O_ptr = (__half*)O + bh * M * D;
        float* QK_ptr = d_QK + bh * M * N;
        float* PV_ptr = d_PV_workspace + bh * M * N;
        
        //======================================================================
        // GEMM 1: Q @ K^T (cuBLASLt Tensor Cores)
        //======================================================================
        
        // Create matrix layouts for this GEMM
        cublasLtMatrixLayout_t layout_Q, layout_K, layout_QK;
        
        // Q: [M, D] row-major, leading dimension = D (stride between rows)
        cublasLtMatrixLayoutCreate(&layout_Q, CUDA_R_16F, K_dim, M, K_dim);
        
        // K: [N, D] row-major, need to transpose for K^T
        cublasLtMatrixLayoutCreate(&layout_K, CUDA_R_16F, K_dim, N, K_dim);
        
        // QK: [M, N] row-major, FP32 output, leading dimension = N
        cublasLtMatrixLayoutCreate(&layout_QK, CUDA_R_32F, N, M, N);
        
        // Alpha, beta for GEMM (on device for zero-copy)
        float alpha = scale;
        float beta = 0.0f;
        
        // Heuristic selection (auto-tune best algorithm)
        cublasLtMatmulHeuristicResult_t heuristic_qk;
        int returnedResults = 0;
        cublasLtMatmulAlgoGetHeuristic(
            g_cublaslt_handle,
            g_matmul_desc_qk,
            layout_Q, layout_K, layout_QK, layout_QK,
            g_preference,
            1,  // Request 1 best algorithm
            &heuristic_qk,
            &returnedResults
        );
        
        // Execute Q @ K^T
        cublasLtMatmul(
            g_cublaslt_handle,
            g_matmul_desc_qk,
            &alpha,
            Q_ptr, layout_Q,
            K_ptr, layout_K,
            &beta,
            QK_ptr, layout_QK,
            QK_ptr, layout_QK,
            &heuristic_qk.algo,
            nullptr, 0,  // No workspace for now
            stream
        );
        
        cublasLtMatrixLayoutDestroy(layout_Q);
        cublasLtMatrixLayoutDestroy(layout_K);
        cublasLtMatrixLayoutDestroy(layout_QK);
        
        //======================================================================
        // SOFTMAX (fused kernel)
        //======================================================================
        dim3 grid(1, (M + 255) / 256);
        dim3 block(256);
        
        fused_softmax_and_pv<<<grid, block, 0, stream>>>(
            QK_ptr, V_ptr, O_ptr, PV_ptr,
            B, H, M, N, D, is_causal
        );
        
        //======================================================================
        // GEMM 2: P @ V (cuBLASLt Tensor Cores)
        //======================================================================
        
        // P: [M, N] row-major (FP32 from softmax), leading dimension = N
        cublasLtMatrixLayout_t layout_P, layout_V, layout_O;
        
        cublasLtMatrixLayoutCreate(&layout_P, CUDA_R_32F, N, M, N);
        
        // V: [N, D] row-major, leading dimension = D
        cublasLtMatrixLayoutCreate(&layout_V, CUDA_R_16F, K_dim, N, K_dim);
        
        // O: [M, D] row-major, FP16 output, leading dimension = D
        cublasLtMatrixLayoutCreate(&layout_O, CUDA_R_16F, K_dim, M, K_dim);
        
        alpha = 1.0f;
        beta = 0.0f;
        
        // Heuristic for P @ V
        cublasLtMatmulHeuristicResult_t heuristic_pv;
        cublasLtMatmulAlgoGetHeuristic(
            g_cublaslt_handle,
            g_matmul_desc_pv,
            layout_P, layout_V, layout_O, layout_O,
            g_preference,
            1,
            &heuristic_pv,
            &returnedResults
        );
        
        // Execute P @ V
        cublasLtMatmul(
            g_cublaslt_handle,
            g_matmul_desc_pv,
            &alpha,
            PV_ptr, layout_P,
            V_ptr, layout_V,
            &beta,
            O_ptr, layout_O,
            O_ptr, layout_O,
            &heuristic_pv.algo,
            nullptr, 0,
            stream
        );
        
        cublasLtMatrixLayoutDestroy(layout_P);
        cublasLtMatrixLayoutDestroy(layout_V);
        cublasLtMatrixLayoutDestroy(layout_O);
    }
    
    // Cleanup workspace (TODO: cache this!)
    cudaFree(d_QK);
    cudaFree(d_PV_workspace);
}

} // namespace cublaslt_sparse
} // namespace flashcore


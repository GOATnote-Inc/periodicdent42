// flashcore/fast/attention_cublaslt_sparse.cu
// Phase 3B/3C: cuBLASLt Sparse-Paged Attention — GPU-driven
// Targets: H100 (SM90a), FP16 inputs, FP32 accum; streaming softmax
// Attribution: uses cuBLASLt for GEMMs; block-sparse pager & online softmax by FlashCore

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublasLt.h>
#include <cublas_v2.h>
#include <cmath>
#include <cstdlib>  // for std::getenv
#include <string>   // for std::stoul
#include <iostream>
#include <vector>

namespace flashcore {
namespace cublaslt_sparse {

//------------------------------------------------------------------------------
// Sparse Pager (simple block list; extend to CSR/CSC if desired)
//------------------------------------------------------------------------------
struct SparsePager {
    // Columns are partitioned into fixed-size pages of width B (page_cols).
    // We only visit pages listed in active_pages[0..num_active_pages-1].
    int page_cols;                 // B (e.g., 128 or 256)
    int num_pages_total;           // ceil(N / B)
    int num_active_pages;          // K (<= num_pages_total)
    const int* __restrict__ active_pages; // length K, each in [0, num_pages_total)
    // (Optional) future: per-page pointers for K/V if already packed per-page.
};

//------------------------------------------------------------------------------
// Global cached handles
//------------------------------------------------------------------------------
static cublasLtHandle_t g_cublaslt_handle = nullptr;
static cublasLtMatmulDesc_t g_desc_qk = nullptr;   // Q @ K_block^T (16F->32F)
static cublasLtMatmulDesc_t g_desc_pv = nullptr;   // P_block @ V_block (32F->16F)
static cublasLtMatmulPreference_t g_preference = nullptr;
static bool g_initialized = false;

//------------------------------------------------------------------------------
// SOFTMAX STATE ARRAYS (rowwise m,l) helpers
//------------------------------------------------------------------------------
__global__ void kernel_fill_float(float* __restrict__ x, float val, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = val;
}

// Convert FP16 to FP32 (for scores after cuBLASLt GEMM)
__global__ void kernel_convert_fp16_to_fp32(
    const __half* __restrict__ src,
    float* __restrict__ dst,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        dst[i] = __half2float(src[i]);
    }
}

// Pad matrix: copy M×D to M×D_padded (zero-pad remaining columns)
__global__ void kernel_pad_matrix(
    const __half* __restrict__ src,  // M × D
    __half* __restrict__ dst,        // M × D_padded (already zero-initialized)
    int M, int D, int D_padded
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M) return;
    
    // Copy D elements from src to dst
    for (int d = 0; d < D; ++d) {
        dst[row * D_padded + d] = src[row * D + d];
    }
    // Remaining (D_padded - D) elements already zero from cudaMemset
}

// Unpad matrix: copy M×D_padded to M×D (discard padding columns)
__global__ void kernel_unpad_matrix(
    const __half* __restrict__ src,  // M × D_padded
    __half* __restrict__ dst,        // M × D
    int M, int D, int D_padded
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M) return;
    
    // Copy D elements from src to dst
    for (int d = 0; d < D; ++d) {
        dst[row * D + d] = src[row * D_padded + d];
    }
}

// Convert FP16 to FP32 with scaling (for S_block after Q@K^T)
__global__ void kernel_convert_fp16_to_fp32_scaled(
    const __half* __restrict__ src,
    float* __restrict__ dst,
    float scale,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        dst[i] = __half2float(src[i]) * scale;
    }
}

// Convert FP32 to FP16 (for P before cuBLASLt GEMM)
__global__ void kernel_convert_fp32_to_fp16(
    const float* __restrict__ src,
    __half* __restrict__ dst,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        dst[i] = __float2half(src[i]);
    }
}

// Rowwise scale O by factors r[m] (each row m gets r[m])
__global__ void kernel_scale_rows_half(__half* __restrict__ O, const float* __restrict__ r,
                                       int M, int D)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M) return;
    float rf = r[row];
    __half* row_ptr = O + row * D;
    for (int d = 0; d < D; ++d) {
        float v = __half2float(row_ptr[d]);
        row_ptr[d] = __float2half(v * rf);
    }
}

// Given scores S_block[M×B] (FP32), previous m,l (length M),
// compute new m', l', write normalized P_block = exp(S - m') / l' into P_block[M×B] (FP32),
// and produce rowwise rescale factor r = (l * exp(m - m')) / l' (or 0 if l==0) for O rows.
__global__ void kernel_online_softmax_block(
    const float* __restrict__ S_block, // [M,B]
    float* __restrict__ P_block,       // [M,B] (output)
    float* __restrict__ m,             // [M] (in/out)
    float* __restrict__ l,             // [M] (in/out)
    float* __restrict__ r_out,         // [M] rowwise rescale for O
    int M, int B, bool causal, int col_start // col_start = page_id * B
){
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M) return;

    const float* s = S_block + row * B;
    float row_max = -INFINITY;

    // Apply causal mask inline: columns beyond row index are masked
    // in absolute column coordinates [col_start, col_start+B)
    // Effective when processing along sequence time.
    if (causal) {
        for (int j = 0; j < B; ++j) {
            int col = col_start + j;
            float v = (row < col) ? -INFINITY : s[j];
            row_max = fmaxf(row_max, v);
        }
    } else {
        for (int j = 0; j < B; ++j) row_max = fmaxf(row_max, s[j]);
    }

    float m_old = m[row];
    float l_old = l[row];

    float m_new = fmaxf(m_old, row_max);
    // compute sumexp with shift by m_new
    float sumexp_block = 0.f;
    float* p = P_block + row * B;

    if (causal) {
        for (int j = 0; j < B; ++j) {
            int col = col_start + j;
            float v = (row < col) ? -INFINITY : s[j];
            // Clamp exp input to avoid denorm/Inf drift
            float exp_input = fmaxf(v - m_new, -80.f);
            float e = (v <= -1e30f) ? 0.f : expf(exp_input);
            p[j] = e; // temp store unnormalized; we'll divide by l_new below
            sumexp_block += e;
        }
    } else {
        for (int j = 0; j < B; ++j) {
            // Clamp exp input to avoid denorm/Inf drift
            float exp_input = fmaxf(s[j] - m_new, -80.f);
            float e = (s[j] <= -1e30f) ? 0.f : expf(exp_input);
            p[j] = e;
            sumexp_block += e;
        }
    }

    float l_new = (l_old > 0.f) ? (l_old * expf(m_old - m_new) + sumexp_block) : sumexp_block;

    // Rowwise rescale factor for O (to keep previous contributions consistent)
    float r = (l_old > 0.f && l_new > 0.f) ? (l_old * expf(m_old - m_new) / l_new) : 0.f;
    // Guard against NaN/Inf in rescale factor
    r_out[row] = isfinite(r) ? r : 0.f;

    // Normalize P_block row by l_new
    float inv_l_new = (l_new > 0.f) ? (1.0f / l_new) : 0.f;
    for (int j = 0; j < B; ++j) p[j] *= inv_l_new;

    // Commit new state
    m[row] = m_new;
    l[row] = l_new;
}

//------------------------------------------------------------------------------
// INITIALIZE cuBLASLt (once)
//------------------------------------------------------------------------------
inline void checkCublas(cublasStatus_t s, const char* where){
    if (s != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "[cuBLASLt] error at " << where << " status=" << (int)s << std::endl;
        std::abort();
    }
}

extern "C" void init_cublaslt_handles() {
    if (g_initialized) return;

    checkCublas(cublasLtCreate(&g_cublaslt_handle), "cublasLtCreate");

    // QK: FP16×FP16 -> FP16 output (for Tensor Core optimization!)
    // Einstein Inversion: Output FP16 to get optimized algorithms, then convert to FP32
    checkCublas(cublasLtMatmulDescCreate(
        &g_desc_qk, CUBLAS_COMPUTE_32F, /*scaleType*/ CUDA_R_32F), "desc_qk");
    {
        // Q not transposed, K transposed
        cublasOperation_t transa = CUBLAS_OP_N;
        cublasOperation_t transb = CUBLAS_OP_T;
        checkCublas(cublasLtMatmulDescSetAttribute(
            g_desc_qk, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa)), "desc_qk.transa");
        checkCublas(cublasLtMatmulDescSetAttribute(
            g_desc_qk, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transb)), "desc_qk.transb");
    }

    // PV: FP16×FP16 -> FP16 output (P converted from FP32 before GEMM)
    // Use CUBLAS_COMPUTE_32F for FP32 accumulation, CUDA_R_32F for scale type (matches float alpha/beta)
    checkCublas(cublasLtMatmulDescCreate(
        &g_desc_pv, CUBLAS_COMPUTE_32F, /*scaleType*/ CUDA_R_32F), "desc_pv");
    {
        // Both non-transposed
        cublasOperation_t transa = CUBLAS_OP_N;
        cublasOperation_t transb = CUBLAS_OP_N;
        checkCublas(cublasLtMatmulDescSetAttribute(
            g_desc_pv, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa)), "desc_pv.transa");
        checkCublas(cublasLtMatmulDescSetAttribute(
            g_desc_pv, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transb)), "desc_pv.transb");
    }

    // Make pointer mode explicit (we pass alpha/beta from host)
    {
        cublasLtPointerMode_t pm = CUBLASLT_POINTER_MODE_HOST;
        checkCublas(cublasLtMatmulDescSetAttribute(
            g_desc_qk, CUBLASLT_MATMUL_DESC_POINTER_MODE, &pm, sizeof(pm)), "desc_qk.pmode");
        checkCublas(cublasLtMatmulDescSetAttribute(
            g_desc_pv, CUBLASLT_MATMUL_DESC_POINTER_MODE, &pm, sizeof(pm)), "desc_pv.pmode");
    }

    checkCublas(cublasLtMatmulPreferenceCreate(&g_preference), "preference");
    
    // Configurable workspace: FLASHCORE_CUBLASLT_WS_MB env var (default 256 MB)
    const char* env_ws = std::getenv("FLASHCORE_CUBLASLT_WS_MB");
    size_t workspace_size = env_ws 
        ? static_cast<size_t>(std::stoul(env_ws)) * 1024ull * 1024ull
        : 256ull * 1024ull * 1024ull;  // Default 256 MB for H100
    
    std::cout << "[cuBLASLt] Workspace preference: " << (workspace_size / 1024.0 / 1024.0) << " MB "
              << (env_ws ? "(from FLASHCORE_CUBLASLT_WS_MB)" : "(default)") << "\n" << std::flush;
    
    checkCublas(cublasLtMatmulPreferenceSetAttribute(
        g_preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
        &workspace_size, sizeof(workspace_size)), "pref.workspace");

    g_initialized = true;
}

//------------------------------------------------------------------------------
// MAIN: sparse-paged attention
//------------------------------------------------------------------------------
extern "C" void launch_attention_cublaslt_sparse(
    const void* Q, const void* K, const void* V, void* O,
    int Batches, int Heads, int S, int D,       // S=M=N, head dim=D
    float scale, bool is_causal,
    const SparsePager* pager,                    // <-- NEW
    cudaStream_t stream
) {
    init_cublaslt_handles();

    const int M = S;              // queries per head
    const int N = S;              // keys/values per head
    const int BH = Batches * Heads;
    const int page_cols = pager->page_cols;
    
    // CRITICAL FIX: H100 Tensor Cores require K ≥ 128!
    // Pad D to next multiple of 128 for Tensor Core algorithms
    const int D_padded = ((D + 127) / 128) * 128;  // Round up to 128
    const bool needs_padding = (D_padded != D);
    
    if (needs_padding && M == S) {  // Only log once
        std::cout << "[cuBLASLt] Padding head dimension: " << D << " → " << D_padded 
                  << " (H100 Tensor Core requirement)\n" << std::flush;
    }

    // Einstein Inversion: Use FP16 output from Q@K^T for Tensor Core algorithms!
    // Allocate small per-BH persistent state: m[M], l[M], r[M], plus S_block/P_block (M×B).
    float *d_m = nullptr, *d_l = nullptr, *d_r = nullptr;
    __half *d_S_block_fp16 = nullptr;  // FP16 output from cuBLASLt Q@K^T (Tensor Cores!)
    float *d_S_block = nullptr;        // FP32 converted for softmax
    float *d_P_block = nullptr;        // FP32 from softmax
    __half *d_P_block_fp16 = nullptr;  // FP16 conversion for cuBLASLt P@V
    
    // Padded buffers for Tensor Core alignment
    __half *d_Q_padded = nullptr;      // M × D_padded (padded Q)
    __half *d_K_padded = nullptr;      // N × D_padded (padded K)
    __half *d_V_padded = nullptr;      // N × D_padded (padded V)
    __half *d_O_padded = nullptr;      // M × D_padded (padded O)

    cudaMalloc(&d_m, M * sizeof(float));
    cudaMalloc(&d_l, M * sizeof(float));
    cudaMalloc(&d_r, M * sizeof(float));

    // Workspace for current page (scores and normalized probs)
    cudaMalloc(&d_S_block_fp16, (size_t)M * page_cols * sizeof(__half));  // NEW: FP16 from Q@K^T
    cudaMalloc(&d_S_block, (size_t)M * page_cols * sizeof(float));       // FP32 for softmax
    cudaMalloc(&d_P_block, (size_t)M * page_cols * sizeof(float));
    cudaMalloc(&d_P_block_fp16, (size_t)M * page_cols * sizeof(__half));
    
    // Allocate padded buffers if needed
    if (needs_padding) {
        cudaMalloc(&d_Q_padded, (size_t)M * D_padded * sizeof(__half));
        cudaMalloc(&d_K_padded, (size_t)N * D_padded * sizeof(__half));
        cudaMalloc(&d_V_padded, (size_t)N * D_padded * sizeof(__half));
        cudaMalloc(&d_O_padded, (size_t)M * D_padded * sizeof(__half));
        cudaMemset(d_Q_padded, 0, (size_t)M * D_padded * sizeof(__half));  // Zero padding
        cudaMemset(d_K_padded, 0, (size_t)N * D_padded * sizeof(__half));
        cudaMemset(d_V_padded, 0, (size_t)N * D_padded * sizeof(__half));
        cudaMemset(d_O_padded, 0, (size_t)M * D_padded * sizeof(__half));
    }

    // Zero/initialize per-BH state for each head independently
    {
        int tpb = 256;
        int b = (M + tpb - 1) / tpb;
        kernel_fill_float<<<b, tpb, 0, stream>>>(d_m, -INFINITY, M);
        kernel_fill_float<<<b, tpb, 0, stream>>>(d_l, 0.f, M);
        kernel_fill_float<<<b, tpb, 0, stream>>>(d_r, 0.f, M);
    }

    // Create matrix layouts (reuse across blocks where possible)
    cublasLtMatrixLayout_t layout_Q = nullptr;
    cublasLtMatrixLayout_t layout_Kb = nullptr;
    cublasLtMatrixLayout_t layout_Sb = nullptr;
    cublasLtMatrixLayout_t layout_Pb = nullptr;
    cublasLtMatrixLayout_t layout_Vb = nullptr;
    cublasLtMatrixLayout_t layout_O  = nullptr;

    // Matrix layouts: cuBLASLt convention is (rows, cols, ld)
    // For GEMM C = A × B: A is M×K, B is K×N, C is M×N
    // CRITICAL: Use D_padded for Tensor Core alignment!
    
    // Q: M×D_padded row-major → rows=M, cols=D_padded, ld=D_padded
    checkCublas(cublasLtMatrixLayoutCreate(&layout_Q, 
        CUDA_R_16F, /*rows*/ M, /*cols*/ D_padded, /*ld*/ D_padded), "layout_Q");
    
    // K_block: page_cols×D_padded row-major → rows=page_cols, cols=D_padded, ld=D_padded
    checkCublas(cublasLtMatrixLayoutCreate(&layout_Kb,
        CUDA_R_16F, /*rows*/ page_cols, /*cols*/ D_padded, /*ld*/ D_padded), "layout_Kb");
    
    // S_block: M×page_cols row-major (FP16 output for Tensor Cores!)
    // Einstein Inversion: Use FP16 output to get optimized algorithms
    checkCublas(cublasLtMatrixLayoutCreate(&layout_Sb,
        CUDA_R_16F, /*rows*/ M, /*cols*/ page_cols, /*ld*/ page_cols), "layout_Sb");
    
    // P_block: M×page_cols row-major (FP16 for cuBLASLt, converted from FP32 softmax)
    checkCublas(cublasLtMatrixLayoutCreate(&layout_Pb,
        CUDA_R_16F, /*rows*/ M, /*cols*/ page_cols, /*ld*/ page_cols), "layout_Pb");
    
    // V_block: page_cols×D_padded row-major (FP16)
    checkCublas(cublasLtMatrixLayoutCreate(&layout_Vb,
        CUDA_R_16F, /*rows*/ page_cols, /*cols*/ D_padded, /*ld*/ D_padded), "layout_Vb");
    
    // O: M×D_padded row-major (FP16 output from P@V)
    checkCublas(cublasLtMatrixLayoutCreate(&layout_O,
        CUDA_R_16F, /*rows*/ M, /*cols*/ D_padded, /*ld*/ D_padded), "layout_O");

    // Make row-major explicit (avoid implementation-defined defaults)
    {
        cublasLtOrder_t row = CUBLASLT_ORDER_ROW;
        checkCublas(cublasLtMatrixLayoutSetAttribute(
            layout_Q,  CUBLASLT_MATRIX_LAYOUT_ORDER, &row, sizeof(row)), "layout_Q.order");
        checkCublas(cublasLtMatrixLayoutSetAttribute(
            layout_Kb, CUBLASLT_MATRIX_LAYOUT_ORDER, &row, sizeof(row)), "layout_Kb.order");
        checkCublas(cublasLtMatrixLayoutSetAttribute(
            layout_Sb, CUBLASLT_MATRIX_LAYOUT_ORDER, &row, sizeof(row)), "layout_Sb.order");
        checkCublas(cublasLtMatrixLayoutSetAttribute(
            layout_Pb, CUBLASLT_MATRIX_LAYOUT_ORDER, &row, sizeof(row)), "layout_Pb.order");
        checkCublas(cublasLtMatrixLayoutSetAttribute(
            layout_Vb, CUBLASLT_MATRIX_LAYOUT_ORDER, &row, sizeof(row)), "layout_Vb.order");
        checkCublas(cublasLtMatrixLayoutSetAttribute(
            layout_O,  CUBLASLT_MATRIX_LAYOUT_ORDER, &row, sizeof(row)), "layout_O.order");
    }

    // Einstein Inversion: Request MANY algorithms, pick the best one!
    // Constraint removed: "only request 1 algo" → Request 64 and filter for workspace
    constexpr int kMaxHeuristics = 64;
    cublasLtMatmulHeuristicResult_t h_qk_list[kMaxHeuristics], h_pv_list[kMaxHeuristics];
    int ret_qk = 0, ret_pv = 0;
    
    // Verification: print layout dimensions as cuBLASLt sees them (ONCE)
    static bool printed_dims = false;
    if (!printed_dims) {
        uint64_t mQ, nQ, ldQ, mK, nK, ldK, mS, nS, ldS;
        size_t size = sizeof(uint64_t);
        
        cublasLtMatrixLayoutGetAttribute(layout_Q, CUBLASLT_MATRIX_LAYOUT_ROWS, &mQ, size, nullptr);
        cublasLtMatrixLayoutGetAttribute(layout_Q, CUBLASLT_MATRIX_LAYOUT_COLS, &nQ, size, nullptr);
        cublasLtMatrixLayoutGetAttribute(layout_Q, CUBLASLT_MATRIX_LAYOUT_LD, &ldQ, size, nullptr);
        
        cublasLtMatrixLayoutGetAttribute(layout_Kb, CUBLASLT_MATRIX_LAYOUT_ROWS, &mK, size, nullptr);
        cublasLtMatrixLayoutGetAttribute(layout_Kb, CUBLASLT_MATRIX_LAYOUT_COLS, &nK, size, nullptr);
        cublasLtMatrixLayoutGetAttribute(layout_Kb, CUBLASLT_MATRIX_LAYOUT_LD, &ldK, size, nullptr);
        
        cublasLtMatrixLayoutGetAttribute(layout_Sb, CUBLASLT_MATRIX_LAYOUT_ROWS, &mS, size, nullptr);
        cublasLtMatrixLayoutGetAttribute(layout_Sb, CUBLASLT_MATRIX_LAYOUT_COLS, &nS, size, nullptr);
        cublasLtMatrixLayoutGetAttribute(layout_Sb, CUBLASLT_MATRIX_LAYOUT_LD, &ldS, size, nullptr);
        
        std::cout << "[cuBLASLt Verification] Layout dimensions:\n"
                  << "  Q: rows=" << mQ << ", cols=" << nQ << ", ld=" << ldQ << "\n"
                  << "  K_block: rows=" << mK << ", cols=" << nK << ", ld=" << ldK << " (transposed)\n"
                  << "  S_block: rows=" << mS << ", cols=" << nS << ", ld=" << ldS << "\n"
                  << "  Expected: (" << M << "×" << D_padded << ") @ (" << D_padded << "×" << page_cols << ") = (" << M << "×" << page_cols << ")\n"
                  << "  Note: D=" << D << " padded to D_padded=" << D_padded << " for Tensor Cores\n"
                  << std::flush;
        printed_dims = true;
    }
    
    // Request MANY algorithms for Q@K^T
    cublasStatus_t status_qk = cublasLtMatmulAlgoGetHeuristic(
        g_cublaslt_handle, g_desc_qk,
        layout_Q, layout_Kb, layout_Sb, layout_Sb,
        g_preference, kMaxHeuristics, h_qk_list, &ret_qk);
    
    // Request MANY algorithms for P@V
    cublasStatus_t status_pv = cublasLtMatmulAlgoGetHeuristic(
        g_cublaslt_handle, g_desc_pv,
        layout_Pb, layout_Vb, layout_O, layout_O,
        g_preference, kMaxHeuristics, h_pv_list, &ret_pv);
    
    // Smart picker: prefer algorithms with workspace > 0 (Tensor Cores!)
    auto pick_best_algo = [](cublasLtMatmulHeuristicResult_t* L, int n, const char* name)->int {
        if (n == 0) {
            std::cerr << "[cuBLASLt " << name << "] WARNING: No algos returned!\n" << std::flush;
            return -1;
        }
        
        int best = -1;
        for (int i = 0; i < n; ++i) {
            if (L[i].state != CUBLAS_STATUS_SUCCESS) continue;
            if (best == -1) best = i;
            // Prefer algos with workspace > 0 (usually faster Tensor Core paths)
            if (L[i].workspaceSize > 0 && L[best].workspaceSize == 0) {
                best = i;
            }
            // Among workspace-using algos, prefer larger workspace (more optimized)
            else if (L[i].workspaceSize > 0 && L[best].workspaceSize > 0 &&
                     L[i].workspaceSize > L[best].workspaceSize) {
                best = i;
            }
        }
        
        if (best >= 0) {
            std::cout << "[cuBLASLt " << name << "] Selected algo " << best << " from " << n << " candidates:\n"
                      << "  Workspace: " << (L[best].workspaceSize / 1024.0) << " KB\n"
                      << "  Waves: " << L[best].wavesCount << "\n" << std::flush;
        }
        return best;
    };
    
    int qk_idx = pick_best_algo(h_qk_list, ret_qk, "QK");
    int pv_idx = pick_best_algo(h_pv_list, ret_pv, "PV");

    // Allocate workspace for cuBLASLt (big perf win!)
    size_t ws_qk = (qk_idx >= 0) ? h_qk_list[qk_idx].workspaceSize : 0;
    size_t ws_pv = (pv_idx >= 0) ? h_pv_list[pv_idx].workspaceSize : 0;
    size_t ws_bytes = (ws_qk > ws_pv) ? ws_qk : ws_pv;
    void* d_ws = nullptr;
    if (ws_bytes > 0) {
        cudaMalloc(&d_ws, ws_bytes);
        std::cout << "[cuBLASLt] Allocated workspace: " << (ws_bytes / 1024.0 / 1024.0) << " MB\n"
                  << "  QK needs: " << (ws_qk / 1024.0 / 1024.0) << " MB\n"
                  << "  PV needs: " << (ws_pv / 1024.0 / 1024.0) << " MB\n" << std::flush;
    } else {
        std::cerr << "[cuBLASLt] WARNING: No workspace allocated - performance will be poor!\n" << std::flush;
    }

    // Process each (batch, head) independently
    for (int bh = 0; bh < BH; ++bh) {
        const __half* Q_ptr = static_cast<const __half*>(Q) + bh * M * D;
        const __half* K_ptr = static_cast<const __half*>(K) + bh * N * D;
        const __half* V_ptr = static_cast<const __half*>(V) + bh * N * D;
        __half*       O_ptr = static_cast<__half*>(O)       + bh * M * D;
        
        // Pad Q, K, V if needed for Tensor Core alignment
        const __half* Q_compute = Q_ptr;
        const __half* K_compute = K_ptr;
        const __half* V_compute = V_ptr;
        __half* O_compute = O_ptr;
        
        if (needs_padding) {
            // Pad Q: M×D → M×D_padded
            {
                int tpb = 256, b = (M + tpb - 1) / tpb;
                kernel_pad_matrix<<<b, tpb, 0, stream>>>(Q_ptr, d_Q_padded, M, D, D_padded);
            }
            // Pad K: N×D → N×D_padded
            {
                int tpb = 256, b = (N + tpb - 1) / tpb;
                kernel_pad_matrix<<<b, tpb, 0, stream>>>(K_ptr, d_K_padded, N, D, D_padded);
            }
            // Pad V: N×D → N×D_padded
            {
                int tpb = 256, b = (N + tpb - 1) / tpb;
                kernel_pad_matrix<<<b, tpb, 0, stream>>>(V_ptr, d_V_padded, N, D, D_padded);
            }
            
            Q_compute = d_Q_padded;
            K_compute = d_K_padded;
            V_compute = d_V_padded;
            O_compute = d_O_padded;
        }

        // Reset (m,l,r) for this head (O will be initialized via beta=0 on first page)
        {
            int tpb = 256, b = (M + tpb - 1) / tpb;
            kernel_fill_float<<<b, tpb, 0, stream>>>(d_m, -INFINITY, M);
            kernel_fill_float<<<b, tpb, 0, stream>>>(d_l, 0.f, M);
            kernel_fill_float<<<b, tpb, 0, stream>>>(d_r, 0.f, M);
        }

        // Iterate over active pages only
        bool first_page = true;
        for (int p = 0; p < pager->num_active_pages; ++p) {
            const int page_id = pager->active_pages[p];   // which page
            const int col_start = page_id * page_cols;
            const int Bcols = (page_cols < (N - col_start)) ? page_cols : (N - col_start);
            if (Bcols <= 0) continue;

            // Slice K,V to current page (use padded dimensions!)
            const __half* K_block = K_compute + col_start * D_padded;  // [Bcols, D_padded]
            const __half* V_block = V_compute + col_start * D_padded;  // [Bcols, D_padded]

            // (1) Scores for this block: S_block = Q @ K_block^T -> [M, Bcols] FP16
            // Einstein Inversion: Output FP16 for Tensor Cores, apply scale during conversion
            float alpha_qk = 1.0f, beta_qk = 0.f;  // Scale applied in conversion kernel!
            
            // CRITICAL FIX: Update layouts for current Bcols (TAIL handling)
            // K_block (Bcols×D) and V_block (Bcols×D) change ROWS, not COLS!
            // Must use uint64_t for layout attributes (cuBLASLt requirement)
            uint64_t bcols64 = static_cast<uint64_t>(Bcols);
            checkCublas(cublasLtMatrixLayoutSetAttribute(
                layout_Kb, CUBLASLT_MATRIX_LAYOUT_ROWS, &bcols64, sizeof(bcols64)), "layout_Kb.rows");
            checkCublas(cublasLtMatrixLayoutSetAttribute(
                layout_Vb, CUBLASLT_MATRIX_LAYOUT_ROWS, &bcols64, sizeof(bcols64)), "layout_Vb.rows");
            checkCublas(cublasLtMatrixLayoutSetAttribute(
                layout_Sb, CUBLASLT_MATRIX_LAYOUT_COLS, &bcols64, sizeof(bcols64)), "layout_Sb.cols");
            checkCublas(cublasLtMatrixLayoutSetAttribute(
                layout_Pb, CUBLASLT_MATRIX_LAYOUT_COLS, &bcols64, sizeof(bcols64)), "layout_Pb.cols");

            // Use chosen heuristic algorithm (may be workspace-using Tensor Core path!)
            cublasLtMatmulAlgo_t* algo_qk = (qk_idx >= 0) ? &h_qk_list[qk_idx].algo : nullptr;
            
            checkCublas(
                cublasLtMatmul(
                    g_cublaslt_handle,
                    g_desc_qk,
                    &alpha_qk,
                    Q_compute, layout_Q,        // Use padded Q!
                    K_block, layout_Kb,         // Use padded K_block!
                    &beta_qk,
                    d_S_block_fp16, layout_Sb,  // FP16 output!
                    d_S_block_fp16, layout_Sb,
                    algo_qk,
                    d_ws, ws_qk,
                    stream
                ),
                "matmul_qk_block"
            );
            
            // Convert FP16 → FP32 and apply attention scale
            {
                int n = M * Bcols;
                int tpb = 256;
                int b = (n + tpb - 1) / tpb;
                kernel_convert_fp16_to_fp32_scaled<<<b, tpb, 0, stream>>>(
                    d_S_block_fp16, d_S_block, scale, n);
            }

            // (2) Online softmax update on this block -> produce P_block (normalized) & r (rowwise)
            {
                int tpb = 256, b = (M + tpb - 1) / tpb;
                kernel_online_softmax_block<<<b, tpb, 0, stream>>>(
                    d_S_block, d_P_block, d_m, d_l, d_r, M, Bcols, is_causal, col_start
                );
            }

            // (3) Rescale previous O rows by r (rowwise) to keep consistency under new m,l
            // CRITICAL FIX: Skip on first page to avoid touching uninitialized O (0×NaN = NaN!)
            if (!first_page) {
                int tpb = 256, b = (M + tpb - 1) / tpb;
                // Use padded D dimension for rescaling!
                kernel_scale_rows_half<<<b, tpb, 0, stream>>>(O_compute, d_r, M, D_padded);
            }

            // (4) Accumulate current block contribution: O += P_block @ V_block
            // Convert P_block from FP32 to FP16 for cuBLASLt (FP16×FP16→FP16 path)
            {
                int n = M * Bcols;
                int tpb = 256;
                int b = (n + tpb - 1) / tpb;
                kernel_convert_fp32_to_fp16<<<b, tpb, 0, stream>>>(d_P_block, d_P_block_fp16, n);
            }
            
            // Use beta=0 for first page (initialize O), beta=1 for subsequent (accumulate)
            float alpha_pv = 1.0f;
            float beta_pv = first_page ? 0.0f : 1.0f;
            
            // Use chosen heuristic algorithm
            cublasLtMatmulAlgo_t* algo_pv = (pv_idx >= 0) ? &h_pv_list[pv_idx].algo : nullptr;
            
            checkCublas(
                cublasLtMatmul(
                    g_cublaslt_handle,
                    g_desc_pv,
                    &alpha_pv,
                    d_P_block_fp16, layout_Pb,   // [M,Bcols] FP16
                    V_block,        layout_Vb,   // [Bcols,D_padded] FP16 (padded!)
                    &beta_pv,
                    O_compute,      layout_O,    // [M,D_padded] FP16 (padded!)
                    O_compute,      layout_O,
                    algo_pv,
                    d_ws, ws_pv,
                    stream
                ),
                "matmul_pv_block"
            );
            
            first_page = false; // Subsequent pages use beta=1
        } // pages
        
        // Unpad O: M×D_padded → M×D
        if (needs_padding) {
            int tpb = 256, b = (M + tpb - 1) / tpb;
            kernel_unpad_matrix<<<b, tpb, 0, stream>>>(O_compute, O_ptr, M, D, D_padded);
        }
    } // bh

    // Cleanup
    cublasLtMatrixLayoutDestroy(layout_Q);
    cublasLtMatrixLayoutDestroy(layout_Kb);
    cublasLtMatrixLayoutDestroy(layout_Sb);
    cublasLtMatrixLayoutDestroy(layout_Pb);
    cublasLtMatrixLayoutDestroy(layout_Vb);
    cublasLtMatrixLayoutDestroy(layout_O);

    cudaFree(d_m);
    cudaFree(d_l);
    cudaFree(d_r);
    cudaFree(d_S_block_fp16);  // NEW: FP16 buffer from Q@K^T
    cudaFree(d_S_block);
    cudaFree(d_P_block);
    cudaFree(d_P_block_fp16);
    if (d_ws) cudaFree(d_ws);
    
    // Cleanup padded buffers
    if (needs_padding) {
        cudaFree(d_Q_padded);
        cudaFree(d_K_padded);
        cudaFree(d_V_padded);
        cudaFree(d_O_padded);
    }
}

//------------------------------------------------------------------------------
// Backward compatibility wrapper (dense path): keep your original signature
//------------------------------------------------------------------------------
extern "C" void launch_attention_cublaslt(
    const void* Q, const void* K, const void* V, void* O,
    int B, int H, int S, int D,
    float scale, bool is_causal,
    cudaStream_t stream
) {
    // Dense = all pages active
    const int Bcols = 128; // default pager block width; tune to 128/256 for H100
    int num_pages = (S + Bcols - 1) / Bcols;

    // Build a dense "all pages" list on the fly (small host buffer)
    std::vector<int> pages(num_pages);
    for (int i = 0; i < num_pages; ++i) pages[i] = i;

    SparsePager pager {
        .page_cols = Bcols,
        .num_pages_total = num_pages,
        .num_active_pages = num_pages,
        .active_pages = pages.data()
    };

    launch_attention_cublaslt_sparse(Q, K, V, O, B, H, S, D, scale, is_causal, &pager, stream);
}

} // namespace cublaslt_sparse
} // namespace flashcore


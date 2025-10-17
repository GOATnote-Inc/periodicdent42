// Minimal CUTLASS FP16 GEMM for L4/Ada (sm_89)
// Based on CUTLASS examples/00_basic_gemm

#include <iostream>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// CUTLASS includes
#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/reference/host/gemm.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/reference/host/tensor_compare.h"

// GEMM operator using FP16 inputs and FP32 accumulation
// This matches our FlashAttention use case: Q@K^T and P@V
using ElementA = cutlass::half_t;
using ElementB = cutlass::half_t;
using ElementC = cutlass::half_t;
using ElementAccumulator = float;

using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;  // For B^T
using LayoutC = cutlass::layout::RowMajor;

// Define the CUTLASS GEMM type with default configuration
// This will use Ampere/Ada Tensor Cores automatically
using CutlassGemm = cutlass::gemm::device::Gemm<
    ElementA,
    LayoutA,
    ElementB,
    LayoutB,
    ElementC,
    LayoutC,
    ElementAccumulator
>;

///////////////////////////////////////////////////////////////////////////////////////////////////

// Test function for Q@K^T style GEMM: (M×K) @ (K×N)^T → (M×N)
cudaError_t cutlass_gemm_qk_transpose(
    int M,
    int N,
    int K,
    cutlass::half_t const *A,  // Q: M×K
    int lda,
    cutlass::half_t const *B,  // K: N×K (stored transposed for K^T)
    int ldb,
    cutlass::half_t *C,        // S: M×N
    int ldc,
    float alpha = 1.0f,
    float beta = 0.0f
) {
    // Create CUTLASS GEMM operator
    CutlassGemm gemm_operator;
    
    // Construct arguments
    // Note: For B in ColumnMajor layout (transposed), the leading dimension is K not N
    CutlassGemm::Arguments args(
        {M, N, K},          // Problem dimensions
        {A, lda},           // A tensor (Q): M×K row-major, lda=K
        {B, K},             // B tensor (K^T): K×N column-major, ldb=K
        {C, ldc},           // C tensor (source)
        {C, ldc},           // D tensor (destination, same as C)
        {alpha, beta}       // Scalars
    );
    
    // Launch the kernel
    cutlass::Status status = gemm_operator(args);
    
    if (status != cutlass::Status::kSuccess) {
        std::cerr << "CUTLASS GEMM failed: " << cutlassGetStatusString(status) << std::endl;
        return cudaErrorUnknown;
    }
    
    return cudaSuccess;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, char **argv) {
    std::cout << "CUTLASS FP16 GEMM for Ada/L4 (sm_89)" << std::endl;
    std::cout << "Testing Q@K^T style multiplication: (M×K) @ (K×N)^T → (M×N)" << std::endl;
    std::cout << std::endl;
    
    // Problem size matching our FlashAttention tiles
    int M = 32;   // BLOCK_M
    int N = 64;   // BLOCK_N (or seq_len tile)
    int K = 64;   // HEAD_DIM
    
    float alpha = 1.0f / sqrtf(static_cast<float>(K));  // Softmax scale
    float beta = 0.0f;
    
    std::cout << "Problem: M=" << M << ", N=" << N << ", K=" << K << std::endl;
    std::cout << "alpha=" << alpha << " (1/sqrt(K) for attention)" << std::endl;
    std::endl(std::cout);
    
    // Allocate host tensors
    cutlass::HostTensor<cutlass::half_t, LayoutA> A({M, K});
    cutlass::HostTensor<cutlass::half_t, LayoutB> B({K, N});
    cutlass::HostTensor<cutlass::half_t, LayoutC> C({M, N});
    cutlass::HostTensor<cutlass::half_t, LayoutC> C_ref({M, N});
    
    // Fill with simple test data
    cutlass::reference::host::TensorFillRandomUniform(
        A.host_view(),
        1,
        cutlass::half_t(1.0f),
        cutlass::half_t(-1.0f),
        0
    );
    
    cutlass::reference::host::TensorFillRandomUniform(
        B.host_view(),
        1,
        cutlass::half_t(1.0f),
        cutlass::half_t(-1.0f),
        1
    );
    
    cutlass::reference::host::TensorFill(C.host_view());
    cutlass::reference::host::TensorFill(C_ref.host_view());
    
    // Copy to device
    A.sync_device();
    B.sync_device();
    C.sync_device();
    
    // Compute reference on host
    std::cout << "Computing reference GEMM on CPU..." << std::endl;
    cutlass::reference::host::Gemm<
        ElementA,
        LayoutA,
        ElementB,
        LayoutB,
        ElementC,
        LayoutC,
        ElementAccumulator,
        ElementAccumulator
    > gemm_ref;
    
    gemm_ref(
        {M, N, K},
        alpha,
        A.host_ref(),
        B.host_ref(),
        beta,
        C_ref.host_ref()
    );
    
    // Run CUTLASS GEMM on GPU
    std::cout << "Running CUTLASS GEMM on GPU..." << std::endl;
    
    // Warmup
    for (int i = 0; i < 10; i++) {
        cutlass_gemm_qk_transpose(
            M, N, K,
            A.device_data(), A.capacity(),
            B.device_data(), B.capacity(),
            C.device_data(), C.capacity(),
            alpha, beta
        );
    }
    
    // Benchmark
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    int num_iters = 1000;
    cudaEventRecord(start);
    for (int i = 0; i < num_iters; i++) {
        cutlass_gemm_qk_transpose(
            M, N, K,
            A.device_data(), A.capacity(),
            B.device_data(), B.capacity(),
            C.device_data(), C.capacity(),
            alpha, beta
        );
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float elapsed_ms;
    cudaEventElapsedTime(&elapsed_ms, start, stop);
    float avg_us = (elapsed_ms * 1000.0f) / num_iters;
    
    std::cout << "Average time: " << avg_us << " μs" << std::endl;
    std::cout << std::endl;
    
    // Copy result back
    C.sync_host();
    
    // Verify correctness
    std::cout << "Verifying correctness..." << std::endl;
    bool passed = cutlass::reference::host::TensorEquals(
        C.host_view(),
        C_ref.host_view()
    );
    
    if (passed) {
        std::cout << "✅ CUTLASS FP16 GEMM PASSED!" << std::endl;
        std::cout << "   Performance: " << avg_us << " μs for " << M << "×" << N << " output" << std::endl;
        std::cout << "   FLOPs: " << (2.0 * M * N * K) << std::endl;
        std::cout << "   GFLOP/s: " << (2.0 * M * N * K) / (avg_us * 1e3) << std::endl;
        return 0;
    } else {
        std::cout << "❌ CUTLASS FP16 GEMM FAILED!" << std::endl;
        return 1;
    }
}


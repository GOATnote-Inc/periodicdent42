// Test basic CUTLASS GEMM compilation for sm_89 (Ada/L4)
#include <cuda_fp16.h>
#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/util/host_tensor.h>
#include <iostream>

// CUTLASS GEMM configuration for FP16 -> FP32
// C = alpha * A @ B + beta * C
// For Q@K^T: (32x64) = (32x64) @ (64x64)^T
using Gemm = cutlass::gemm::device::Gemm<
    __half,                                    // Element A
    cutlass::layout::RowMajor,                 // Layout A
    __half,                                    // Element B  
    cutlass::layout::ColumnMajor,              // Layout B (for transpose)
    float,                                     // Element C (output)
    cutlass::layout::RowMajor,                 // Layout C
    float,                                     // Element accumulator
    cutlass::arch::OpClassTensorOp,            // Operator class (Tensor Cores)
    cutlass::arch::Sm89                        // Architecture (Ada)
>;

int main() {
    std::cout << "CUTLASS Basic Test for sm_89 (L4/Ada)" << std::endl;
    
    // Small test matrices
    int M = 32;
    int N = 64;
    int K = 64;
    
    // Allocate host memory
    cutlass::HostTensor<__half, cutlass::layout::RowMajor> A({M, K});
    cutlass::HostTensor<__half, cutlass::layout::ColumnMajor> B({K, N});
    cutlass::HostTensor<float, cutlass::layout::RowMajor> C({M, N});
    
    // Initialize with simple values
    for (int i = 0; i < M * K; i++) A.host_data()[i] = __float2half(1.0f);
    for (int i = 0; i < K * N; i++) B.host_data()[i] = __float2half(1.0f);
    for (int i = 0; i < M * N; i++) C.host_data()[i] = 0.0f;
    
    A.sync_device();
    B.sync_device();
    C.sync_device();
    
    // GEMM arguments
    typename Gemm::Arguments args{
        {M, N, K},                             // Problem size
        {A.device_data(), K},                  // A tensor
        {B.device_data(), K},                  // B tensor  
        {C.device_data(), N},                  // C tensor (source)
        {C.device_data(), N},                  // D tensor (destination)
        {1.0f, 0.0f}                           // alpha, beta
    };
    
    // Create GEMM operator
    Gemm gemm_op;
    
    // Check if problem size is supported
    cutlass::Status status = gemm_op.can_implement(args);
    if (status != cutlass::Status::kSuccess) {
        std::cerr << "❌ GEMM cannot be implemented: " << cutlass::cutlassGetStatusString(status) << std::endl;
        return 1;
    }
    
    // Initialize
    status = gemm_op.initialize(args);
    if (status != cutlass::Status::kSuccess) {
        std::cerr << "❌ Failed to initialize: " << cutlass::cutlassGetStatusString(status) << std::endl;
        return 1;
    }
    
    // Run GEMM
    status = gemm_op();
    if (status != cutlass::Status::kSuccess) {
        std::cerr << "❌ Failed to run GEMM: " << cutlass::cutlassGetStatusString(status) << std::endl;
        return 1;
    }
    
    // Sync and verify
    cudaDeviceSynchronize();
    C.sync_host();
    
    // Expected: C[i][j] = K (since A and B are all 1s)
    bool correct = true;
    float expected = (float)K;
    for (int i = 0; i < M * N; i++) {
        if (fabs(C.host_data()[i] - expected) > 0.1f) {
            std::cerr << "❌ Incorrect result at " << i << ": " << C.host_data()[i] 
                      << " (expected " << expected << ")" << std::endl;
            correct = false;
            break;
        }
    }
    
    if (correct) {
        std::cout << "✅ CUTLASS GEMM test PASSED!" << std::endl;
        std::cout << "   Matrix sizes: M=" << M << ", N=" << N << ", K=" << K << std::endl;
        std::cout << "   Output: " << C.host_data()[0] << " (expected " << expected << ")" << std::endl;
        return 0;
    }
    
    return 1;
}


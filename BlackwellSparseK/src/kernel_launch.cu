/**
 * Kernel launch wrapper for PyTorch C++ extension.
 * 
 * This file provides the C interface between PyTorch bindings and the CUDA kernel.
 */

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

// BSR structure (must match sparse_h100_async.cu)
struct BSR {
    int M_blocks, N_blocks, K_blocks, nnzb;
    int *row_ptr, *col_idx;
    half *vals;
};

// Template kernel declaration (from sparse_h100_async.cu)
template<int BM_, int BN_, int BK_>
__global__ void bsr_spmm_async(
    const BSR A, const BSR B,
    float* __restrict__ C,
    int M, int N, int K, int ldc);

// Helper to divide and round up
inline int div_up(int a, int b) { return (a + b - 1) / b; }

/**
 * Launch the BSR sparse matrix multiplication kernel.
 * 
 * This function is called from Python via the C++ extension bindings.
 */
extern "C" void launch_bsr_spmm_async(
    // Matrix A (sparse BSR)
    const int* A_row_ptr,
    const int* A_col_idx,
    const void* A_vals,
    int A_M_blocks,
    int A_K_blocks,
    int A_nnzb,
    // Matrix B (sparse BSR)
    const int* B_row_ptr,
    const int* B_col_idx,
    const void* B_vals,
    int B_K_blocks,
    int B_N_blocks,
    int B_nnzb,
    // Output C (dense)
    float* C,
    // Problem dimensions
    int M,
    int N,
    int K,
    int ldc,
    // Tile config
    int BM,
    int BN,
    int BK,
    // CUDA stream
    cudaStream_t stream
) {
    // Construct BSR structures
    BSR hA, hB;
    
    // Matrix A
    hA.M_blocks = A_M_blocks;
    hA.K_blocks = A_K_blocks;
    hA.N_blocks = 0;  // Not used
    hA.nnzb = A_nnzb;
    hA.row_ptr = const_cast<int*>(A_row_ptr);
    hA.col_idx = const_cast<int*>(A_col_idx);
    hA.vals = const_cast<half*>(reinterpret_cast<const half*>(A_vals));
    
    // Matrix B
    hB.M_blocks = 0;  // Not used (B is transposed)
    hB.K_blocks = B_K_blocks;
    hB.N_blocks = B_N_blocks;
    hB.nnzb = B_nnzb;
    hB.row_ptr = const_cast<int*>(B_row_ptr);
    hB.col_idx = const_cast<int*>(B_col_idx);
    hB.vals = const_cast<half*>(reinterpret_cast<const half*>(B_vals));
    
    // Calculate grid dimensions
    // grid.x = number of BN-sized tiles in N
    // grid.y = number of BM-sized tiles in M
    dim3 grid(div_up(N, BN), div_up(M, BM));
    
    // Block size (threads per block)
    // For BM=256, BN=128: 4 warps (128 threads)
    int warps_m = BM / 64;  // Assuming WM=64
    int warps_n = BN / 64;  // Assuming WN=64
    int threads = warps_m * warps_n * 32;
    
    dim3 block(threads);
    
    // Launch kernel with matching template parameters
    if (BM == 256 && BN == 128 && BK == 32) {
        bsr_spmm_async<256, 128, 32><<<grid, block, 0, stream>>>(
            hA, hB, C, M, N, K, ldc
        );
    }
    else if (BM == 512 && BN == 256 && BK == 64) {
        bsr_spmm_async<512, 256, 64><<<grid, block, 0, stream>>>(
            hA, hB, C, M, N, K, ldc
        );
    }
    else if (BM == 128 && BN == 64 && BK == 32) {
        bsr_spmm_async<128, 64, 32><<<grid, block, 0, stream>>>(
            hA, hB, C, M, N, K, ldc
        );
    }
    else {
        // Fallback to default config
        bsr_spmm_async<256, 128, 32><<<grid, block, 0, stream>>>(
            hA, hB, C, M, N, K, ldc
        );
    }
}


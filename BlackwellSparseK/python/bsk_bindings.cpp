/**
 * BlackwellSparseK PyTorch C++ Extension Bindings
 * 
 * Connects CUDA kernel to Python/PyTorch API.
 */

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>

// Forward declare CUDA kernel
// Defined in sparse_h100_async.cu
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
);


/**
 * PyTorch interface for sparse_mm operation.
 * 
 * Args:
 *   A_row_ptr: BSR row pointers for matrix A
 *   A_col_idx: BSR column indices for matrix A
 *   A_vals: BSR values for matrix A (FP16)
 *   B_row_ptr: BSR row pointers for matrix B
 *   B_col_idx: BSR column indices for matrix B
 *   B_vals: BSR values for matrix B (FP16)
 *   M, N, K: Problem dimensions
 *   BM, BN, BK: Tile sizes
 * 
 * Returns:
 *   Dense output tensor C (FP32)
 */
torch::Tensor sparse_mm_bsr_cuda(
    torch::Tensor A_row_ptr,
    torch::Tensor A_col_idx,
    torch::Tensor A_vals,
    torch::Tensor B_row_ptr,
    torch::Tensor B_col_idx,
    torch::Tensor B_vals,
    int M,
    int N,
    int K,
    int BM,
    int BN,
    int BK
) {
    // Validate inputs
    TORCH_CHECK(A_row_ptr.is_cuda(), "A_row_ptr must be CUDA tensor");
    TORCH_CHECK(A_col_idx.is_cuda(), "A_col_idx must be CUDA tensor");
    TORCH_CHECK(A_vals.is_cuda(), "A_vals must be CUDA tensor");
    TORCH_CHECK(B_row_ptr.is_cuda(), "B_row_ptr must be CUDA tensor");
    TORCH_CHECK(B_col_idx.is_cuda(), "B_col_idx must be CUDA tensor");
    TORCH_CHECK(B_vals.is_cuda(), "B_vals must be CUDA tensor");
    
    TORCH_CHECK(A_vals.dtype() == torch::kFloat16, "A_vals must be float16");
    TORCH_CHECK(B_vals.dtype() == torch::kFloat16, "B_vals must be float16");
    
    // Get device
    const auto device = A_vals.device();
    const at::cuda::CUDAGuard device_guard(device);
    
    // Calculate dimensions
    int block_size = 16;  // BSR block size (hardcoded for now)
    int A_M_blocks = M / block_size;
    int A_K_blocks = K / block_size;
    int B_K_blocks = K / block_size;
    int B_N_blocks = N / block_size;
    int A_nnzb = A_col_idx.size(0);
    int B_nnzb = B_col_idx.size(0);
    
    // Allocate output tensor (dense, FP32)
    auto options = torch::TensorOptions()
        .dtype(torch::kFloat32)
        .device(device)
        .layout(torch::kStrided);
    
    auto C = torch::zeros({M, N}, options);
    
    // Get raw pointers
    const int* A_row_ptr_data = A_row_ptr.data_ptr<int>();
    const int* A_col_idx_data = A_col_idx.data_ptr<int>();
    const void* A_vals_data = A_vals.data_ptr<at::Half>();
    
    const int* B_row_ptr_data = B_row_ptr.data_ptr<int>();
    const int* B_col_idx_data = B_col_idx.data_ptr<int>();
    const void* B_vals_data = B_vals.data_ptr<at::Half>();
    
    float* C_data = C.data_ptr<float>();
    
    // Get CUDA stream
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(device.index());
    
    // Launch kernel
    launch_bsr_spmm_async(
        A_row_ptr_data, A_col_idx_data, A_vals_data, A_M_blocks, A_K_blocks, A_nnzb,
        B_row_ptr_data, B_col_idx_data, B_vals_data, B_K_blocks, B_N_blocks, B_nnzb,
        C_data,
        M, N, K, N,  // ldc = N (row-major)
        BM, BN, BK,
        stream
    );
    
    // Check for errors
    AT_CUDA_CHECK(cudaGetLastError());
    
    return C;
}


/**
 * Python module definition.
 */
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "sparse_mm_bsr",
        &sparse_mm_bsr_cuda,
        "Sparse BSR matrix multiplication (CUDA)",
        py::arg("A_row_ptr"),
        py::arg("A_col_idx"),
        py::arg("A_vals"),
        py::arg("B_row_ptr"),
        py::arg("B_col_idx"),
        py::arg("B_vals"),
        py::arg("M"),
        py::arg("N"),
        py::arg("K"),
        py::arg("BM") = 256,
        py::arg("BN") = 128,
        py::arg("BK") = 32
    );
}


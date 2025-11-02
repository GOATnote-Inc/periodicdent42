// PyTorch C++ Extension for Sparse Auto-Tuning
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>
#include <string>

// Forward declarations of CUDA kernels
void bsr_gemm_64_cuda(
    const float* block_vals, const int* row_ptr, const int* col_indices,
    const float* B, float* C,
    int M, int N, int K, int M_blocks, int K_blocks, int nnzb, int block_size
);

void cusparse_bsr_gemm_cuda(
    const float* block_vals, const int* row_ptr, const int* col_indices,
    const float* B, float* C,
    int M, int N, int K, int M_blocks, int K_blocks, int nnzb, int block_size
);

// Python-facing function: Auto-tuned sparse matmul
torch::Tensor sparse_matmul_auto(
    torch::Tensor row_ptr,        // [M_blocks + 1]
    torch::Tensor col_indices,    // [nnzb]
    torch::Tensor values,         // [nnzb, block_size, block_size]
    torch::Tensor dense_matrix,   // [K, N]
    int M, int K, int block_size
) {
    // Input validation
    TORCH_CHECK(row_ptr.is_cuda(), "row_ptr must be a CUDA tensor");
    TORCH_CHECK(col_indices.is_cuda(), "col_indices must be a CUDA tensor");
    TORCH_CHECK(values.is_cuda(), "values must be a CUDA tensor");
    TORCH_CHECK(dense_matrix.is_cuda(), "dense_matrix must be a CUDA tensor");
    
    TORCH_CHECK(row_ptr.dtype() == torch::kInt32, "row_ptr must be int32");
    TORCH_CHECK(col_indices.dtype() == torch::kInt32, "col_indices must be int32");
    TORCH_CHECK(values.dtype() == torch::kFloat32, "values must be float32");
    TORCH_CHECK(dense_matrix.dtype() == torch::kFloat32, "dense_matrix must be float32");
    
    int N = dense_matrix.size(1);
    int M_blocks = row_ptr.size(0) - 1;
    int nnzb = col_indices.size(0);
    int K_blocks = K / block_size;
    
    // Allocate output
    auto output = torch::zeros({M, N}, torch::device(torch::kCUDA).dtype(torch::kFloat32));
    
    // Get data pointers
    const float* vals_ptr = values.data_ptr<float>();
    const int* row_ptr_ptr = row_ptr.data_ptr<int>();
    const int* col_idx_ptr = col_indices.data_ptr<int>();
    const float* B_ptr = dense_matrix.data_ptr<float>();
    float* C_ptr = output.data_ptr<float>();
    
    // TODO: Add auto-tuning logic here
    // For now, use the optimized kernel directly
    bsr_gemm_64_cuda(
        vals_ptr, row_ptr_ptr, col_idx_ptr,
        B_ptr, C_ptr,
        M, N, K, M_blocks, K_blocks, nnzb, block_size
    );
    
    return output;
}

// Python-facing function: cuSPARSE baseline
torch::Tensor sparse_matmul_cusparse(
    torch::Tensor row_ptr,
    torch::Tensor col_indices,
    torch::Tensor values,
    torch::Tensor dense_matrix,
    int M, int K, int block_size
) {
    int N = dense_matrix.size(1);
    int M_blocks = row_ptr.size(0) - 1;
    int nnzb = col_indices.size(0);
    int K_blocks = K / block_size;
    
    auto output = torch::zeros({M, N}, torch::device(torch::kCUDA).dtype(torch::kFloat32));
    
    cusparse_bsr_gemm_cuda(
        values.data_ptr<float>(),
        row_ptr.data_ptr<int>(),
        col_indices.data_ptr<int>(),
        dense_matrix.data_ptr<float>(),
        output.data_ptr<float>(),
        M, N, K, M_blocks, K_blocks, nnzb, block_size
    );
    
    return output;
}

// Python-facing function: Benchmark a variant
float benchmark_variant(
    const std::string& variant_name,
    torch::Tensor row_ptr,
    torch::Tensor col_indices,
    torch::Tensor values,
    torch::Tensor dense_matrix,
    int M, int K, int block_size,
    int num_runs = 20
) {
    int N = dense_matrix.size(1);
    auto output = torch::zeros({M, N}, torch::device(torch::kCUDA).dtype(torch::kFloat32));
    
    // Warmup
    for (int i = 0; i < 5; i++) {
        if (variant_name == "custom") {
            output = sparse_matmul_auto(row_ptr, col_indices, values, dense_matrix, M, K, block_size);
        } else if (variant_name == "cusparse") {
            output = sparse_matmul_cusparse(row_ptr, col_indices, values, dense_matrix, M, K, block_size);
        }
    }
    c10::cuda::getCurrentCUDAStream().synchronize();
    
    // Timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, c10::cuda::getCurrentCUDAStream());
    
    for (int i = 0; i < num_runs; i++) {
        if (variant_name == "custom") {
            output = sparse_matmul_auto(row_ptr, col_indices, values, dense_matrix, M, K, block_size);
        } else if (variant_name == "cusparse") {
            output = sparse_matmul_cusparse(row_ptr, col_indices, values, dense_matrix, M, K, block_size);
        }
    }
    
    cudaEventRecord(stop, c10::cuda::getCurrentCUDAStream());
    cudaEventSynchronize(stop);
    
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return ms / num_runs;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sparse_matmul_auto", &sparse_matmul_auto, "Sparse matmul with auto-tuning");
    m.def("sparse_matmul_cusparse", &sparse_matmul_cusparse, "Sparse matmul using cuSPARSE");
    m.def("benchmark_variant", &benchmark_variant, "Benchmark a sparse matmul variant");
}

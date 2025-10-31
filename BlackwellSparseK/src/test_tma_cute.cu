// TMA test using CuTe copy primitive
#include <cuda.h>
#include <cstdio>
#include <vector>
#include <cute/tensor.hpp>
#include <cute/atom/copy_atom.hpp>
#include <cute/atom/copy_traits_sm90_tma.hpp>
#include <cutlass/numeric_types.h>

using namespace cute;
using Element = cutlass::half_t;

__global__ void test_tma_kernel(
    cute::TmaDescriptor const tma_desc,
    Element const* gmem_ptr,
    Element* output,
    int M, int N
) {
    __shared__ Element smem[128][128];
    
    if (threadIdx.x == 0) {
        // Use CuTe copy API
        auto tma_load = make_tma_atom(SM90_TMA_LOAD{});
        auto gmem_layout = make_layout(make_shape(M, N), make_stride(Int<1>{}, M));
        auto smem_layout = make_layout(make_shape(Int<128>{}, Int<128>{}));
        
        Tensor gA = make_tensor(make_gmem_ptr(gmem_ptr), gmem_layout);
        Tensor sA = make_tensor(make_smem_ptr(&smem[0][0]), smem_layout);
        
        // Simple copy (no pipeline)
        copy(tma_desc, gA(make_coord(0,0)), sA);
    }
    __syncthreads();
    
    // Copy smem -> gmem for verification
    for (int i = threadIdx.x; i < 128*128; i += blockDim.x) {
        int row = i / 128;
        int col = i % 128;
        output[i] = smem[row][col];
    }
}

int main() {
    const int M = 128, N = 128;
    
    std::vector<Element> h_in(M * N);
    for (int i = 0; i < M * N; i++) h_in[i] = Element((float)i / 100.0f);
    
    Element *d_in, *d_out;
    cudaMalloc(&d_in, M * N * sizeof(Element));
    cudaMalloc(&d_out, M * N * sizeof(Element));
    cudaMemcpy(d_in, h_in.data(), M * N * sizeof(Element), cudaMemcpyHostToDevice);
    
    // Create TMA on host
    auto gmem_layout = make_layout(make_shape(M, N), make_stride(Int<1>{}, M));
    auto smem_layout = make_layout(make_shape(Int<128>{}, Int<128>{}));
    Tensor g = make_tensor(make_gmem_ptr(d_in), gmem_layout);
    auto tma = make_tma_copy(SM90_TMA_LOAD{}, g, smem_layout);
    auto desc = *tma.get_tma_descriptor();
    
    test_tma_kernel<<<1, 256>>>(desc, d_in, d_out, M, N);
    cudaDeviceSynchronize();
    
    std::vector<Element> h_out(M * N);
    cudaMemcpy(h_out.data(), d_out, M * N * sizeof(Element), cudaMemcpyDeviceToHost);
    
    float max_diff = 0.0f;
    for (int i = 0; i < M * N; i++) {
        float diff = fabsf(float(h_in[i]) - float(h_out[i]));
        max_diff = fmaxf(max_diff, diff);
    }
    
    printf("%s diff=%.6f\n", (max_diff < 1e-3f) ? "✅" : "❌", max_diff);
    
    cudaFree(d_in);
    cudaFree(d_out);
    return (max_diff < 1e-3f) ? 0 : 1;
}


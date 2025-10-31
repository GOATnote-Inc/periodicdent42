// Minimal TMA test - load matrix tile via TMA, verify correctness
// Build: nvcc -O3 -std=c++17 -arch=sm_90a -I$CUTLASS_PATH/include -o test_tma test_tma_simple.cu

#include <cuda.h>
#include <cstdio>
#include <vector>

#include <cute/tensor.hpp>
#include <cute/atom/copy_atom.hpp>
#include <cute/atom/copy_traits_sm90_tma.hpp>
#include <cutlass/numeric_types.h>

using namespace cute;
using Element = cutlass::half_t;

// Kernel: Load 128x128 tile via TMA, write to global
__global__ void test_tma_load(
    cute::TmaDescriptor const* tma_desc,
    Element const* gmem_input,
    Element* gmem_output,
    int M, int N
) {
    extern __shared__ Element smem[];
    
    int tid = threadIdx.x;
    int lane = tid % 32;
    
    // Only warp 0, lane 0 does TMA
    if (tid == 0) {
        // TMA load 128x128 tile
        uint64_t desc_bits = *reinterpret_cast<uint64_t const*>(tma_desc);
        uint32_t mbar_handle = 0; // Simplified: no pipeline
        
        // TMA copy instruction (simplified)
        asm volatile(
            "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes"
            " [%0], [%1, {%2, %3}], [%4];"
            :
            : "r"((uint32_t)__cvta_generic_to_shared(smem)),
              "l"(desc_bits),
              "r"(0),  // tile_x
              "r"(0),  // tile_y  
              "r"(mbar_handle)
        );
    }
    
    __syncthreads();
    
    // All threads copy smem → gmem
    for (int i = tid; i < 128 * 128; i += blockDim.x) {
        gmem_output[i] = smem[i];
    }
}

int main() {
    const int M = 128, N = 128;
    
    // Allocate
    std::vector<Element> h_input(M * N);
    for (int i = 0; i < M * N; i++) {
        h_input[i] = Element((float)i / 100.0f);
    }
    
    Element *d_input, *d_output;
    cudaMalloc(&d_input, M * N * sizeof(Element));
    cudaMalloc(&d_output, M * N * sizeof(Element));
    cudaMemcpy(d_input, h_input.data(), M * N * sizeof(Element), cudaMemcpyHostToDevice);
    cudaMemset(d_output, 0, M * N * sizeof(Element));
    
    // Create TMA descriptor (CuTe helper)
    // Column-major layout (stride-1 in first dimension)
    auto gmem_layout = make_layout(make_shape(M, N), make_stride(Int<1>{}, M));
    Tensor gmem_tensor = make_tensor(make_gmem_ptr(d_input), gmem_layout);
    
    // Shared memory layout (also column-major)
    auto smem_layout = make_layout(make_shape(Int<128>{}, Int<128>{}), make_stride(Int<1>{}, Int<128>{}));
    
    auto tma_load = make_tma_copy(
        SM90_TMA_LOAD{},
        gmem_tensor,
        smem_layout
    );
    
    // Get TMA descriptor
    auto const* tma_desc_ptr = tma_load.get_tma_descriptor();
    
    // Copy descriptor to device
    cute::TmaDescriptor* d_tma_desc;
    cudaMalloc(&d_tma_desc, sizeof(cute::TmaDescriptor));
    cudaMemcpy(d_tma_desc, tma_desc_ptr, sizeof(cute::TmaDescriptor), cudaMemcpyHostToDevice);
    
    // Launch
    int smem_size = 128 * 128 * sizeof(Element);
    test_tma_load<<<1, 128, smem_size>>>(d_tma_desc, d_input, d_output, M, N);
    cudaDeviceSynchronize();
    
    // Verify
    std::vector<Element> h_output(M * N);
    cudaMemcpy(h_output.data(), d_output, M * N * sizeof(Element), cudaMemcpyDeviceToHost);
    
    float max_diff = 0.0f;
    for (int i = 0; i < M * N; i++) {
        float diff = fabsf(float(h_input[i]) - float(h_output[i]));
        max_diff = fmaxf(max_diff, diff);
    }
    
    printf("[TMA Test] Max diff: %.6f %s\n", max_diff, 
           (max_diff < 1e-5f) ? "✅ PASS" : "❌ FAIL");
    
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_tma_desc);
    
    return (max_diff < 1e-5f) ? 0 : 1;
}


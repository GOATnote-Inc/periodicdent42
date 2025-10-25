// Test if WMMA is available on this CUDA setup
#include <cuda_fp16.h>
#include <mma.h>
#include <stdio.h>

using namespace nvcuda::wmma;

__global__ void test_wmma_kernel() {
    fragment<matrix_a, 16, 16, 16, __half, row_major> a_frag;
    fragment<matrix_b, 16, 16, 16, __half, row_major> b_frag;
    fragment<accumulator, 16, 16, 16, float> c_frag;
    
    fill_fragment(c_frag, 0.0f);
    
    // This won't execute, just testing compilation
    printf("WMMA available!\\n");
}

int main() {
    printf("WMMA test compiled successfully!\\n");
    return 0;
}


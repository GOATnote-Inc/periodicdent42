// Minimal kernel to map accumulator lane slots -> (row,col)
#include <torch/extension.h>
#include <mma.h>
using namespace nvcuda;

__global__
void wmma_accum_introspect_kernel_cuda(float* out /*16x16*/) {
    wmma::fragment<wmma::accumulator,16,16,16,float> c;
    #pragma unroll
    for (int i=0; i<c.num_elements; i++) {
        c.x[i] = float(threadIdx.x * 8 + i);
    }
    wmma::store_matrix_sync(out, c, 16, wmma::mem_row_major);
}

void wmma_accum_introspect_kernel_wrapper(torch::Tensor out) {
    TORCH_CHECK(out.is_cuda(), "out must be a CUDA tensor");
    TORCH_CHECK(out.dtype() == torch::kFloat32, "out must be float32");
    TORCH_CHECK(out.size(0) == 16 && out.size(1) == 16, "out must be 16x16");
    
    wmma_accum_introspect_kernel_cuda<<<1, 32>>>(out.data_ptr<float>());
}


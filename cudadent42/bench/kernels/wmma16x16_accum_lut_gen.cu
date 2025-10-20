// Minimal kernel to map accumulator lane slots -> (row,col)
#include <mma.h>
using namespace nvcuda;

extern "C" __global__
void wmma_accum_introspect_kernel(float* out /*16x16*/) {
    wmma::fragment<wmma::accumulator,16,16,16,float> c;
    #pragma unroll
    for (int i=0; i<c.num_elements; i++) {
        c.x[i] = float(threadIdx.x * 8 + i);
    }
    wmma::store_matrix_sync(out, c, 16, wmma::mem_row_major);
}


#include <torch/extension.h>

// Forward declaration
void wmma_accum_introspect_kernel_wrapper(torch::Tensor out);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("wmma_accum_introspect_kernel", &wmma_accum_introspect_kernel_wrapper, "WMMA 16x16 accumulator introspection");
}


#pragma once
#include <cstdio>
#include <stdexcept>
#include <cuda_runtime.h>

#define CUDA_CHECK(expr) do {                                  \
  cudaError_t _err = (expr);                                   \
  if (_err != cudaSuccess) {                                   \
    fprintf(stderr, "CUDA error %s:%d: %s\n",                  \
            __FILE__, __LINE__, cudaGetErrorString(_err));     \
    throw std::runtime_error(cudaGetErrorString(_err));        \
  }                                                            \
} while(0)


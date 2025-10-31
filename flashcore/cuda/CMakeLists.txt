# FlashCore CUDA Kernels
# Build system for Hopper-optimized attention

cmake_minimum_required(VERSION 3.24)
project(flashcore_cuda LANGUAGES CXX CUDA)

# CUDA configuration
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 90)  # Hopper (H100)

# Find CUDA
find_package(CUDAToolkit 12.0 REQUIRED)

# Compiler flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --use_fast_math")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xptxas -v,-warn-lmem-usage")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")

# Include directories
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../..)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../third_party/cutlass/include)

# Build attention kernel
add_library(flashcore_hopper SHARED
    ../fast/attention_hopper_cuda.cu
)

target_link_libraries(flashcore_hopper
    CUDA::cudart
    CUDA::cuda_driver
)

# Set output directory
set_target_properties(flashcore_hopper PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
    CUDA_SEPARABLE_COMPILATION ON
)

# Build test executable
add_executable(test_hopper
    test_hopper_kernel.cu
)

target_link_libraries(test_hopper
    flashcore_hopper
    CUDA::cudart
)

set_target_properties(test_hopper PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# Print configuration
message(STATUS "CUDA Version: ${CUDAToolkit_VERSION}")
message(STATUS "CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "CUDA Flags: ${CMAKE_CUDA_FLAGS}")


# TriageAttention - High-Performance Sparse Attention Kernels
# Copyright © 2025 GOATnote Inc.
# Author: Brandon Dent, MD

cmake_minimum_required(VERSION 3.22 FATAL_ERROR)

project(
  TriageAttention
  VERSION 0.1.0
  DESCRIPTION "High-performance sparse attention kernels for H100/B200"
  LANGUAGES CXX CUDA
)

# ==============================================================================
# Build Options
# ==============================================================================
option(TRIAGEATTENTION_BUILD_TESTS "Build test suite" ON)
option(TRIAGEATTENTION_BUILD_BENCHMARKS "Build benchmarks" ON)
option(TRIAGEATTENTION_BUILD_EXAMPLES "Build examples" ON)
option(TRIAGEATTENTION_BUILD_PYTHON "Build Python bindings" ON)

# ==============================================================================
# CUDA Configuration
# ==============================================================================
find_package(CUDAToolkit 13.0 REQUIRED)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Target architectures: H100 (sm_90a), B200 (sm_100)
set(CMAKE_CUDA_ARCHITECTURES "90" CACHE STRING "CUDA architectures")

# Compiler flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --use_fast_math -lineinfo")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --maxrregcount=255 --threads 0")

# ==============================================================================
# Dependencies
# ==============================================================================

# CUTLASS 4.3.0
find_path(CUTLASS_INCLUDE_DIR 
  NAMES cutlass/cutlass.h
  PATHS /opt/cutlass/include
        ${CMAKE_SOURCE_DIR}/third_party/cutlass/include
)

if(NOT CUTLASS_INCLUDE_DIR)
  message(FATAL_ERROR "CUTLASS 4.3.0 not found. Set CUTLASS_INCLUDE_DIR.")
endif()

message(STATUS "CUTLASS found: ${CUTLASS_INCLUDE_DIR}")

# ==============================================================================
# Include Directories
# ==============================================================================
include_directories(
  ${CMAKE_SOURCE_DIR}/include
  ${CMAKE_SOURCE_DIR}/csrc
  ${CUTLASS_INCLUDE_DIR}
  ${CUDAToolkit_INCLUDE_DIRS}
)

# ==============================================================================
# Core Kernel Library
# ==============================================================================
add_library(triageattention_kernels STATIC
  csrc/kernels/attention_bleeding_edge_tma.cu
)

target_include_directories(triageattention_kernels PUBLIC
  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

target_link_libraries(triageattention_kernels PUBLIC
  CUDA::cudart
  CUDA::cuda_driver
)

set_target_properties(triageattention_kernels PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  CUDA_RESOLVE_DEVICE_SYMBOLS ON
  POSITION_INDEPENDENT_CODE ON
)

# ==============================================================================
# BlackwellSparseK Integration
# ==============================================================================
add_subdirectory(BlackwellSparseK)

# ==============================================================================
# Tests
# ==============================================================================
if(TRIAGEATTENTION_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()

# ==============================================================================
# Benchmarks
# ==============================================================================
if(TRIAGEATTENTION_BUILD_BENCHMARKS)
  add_subdirectory(benchmarks)
endif()

# ==============================================================================
# Examples
# ==============================================================================
if(TRIAGEATTENTION_BUILD_EXAMPLES)
  add_subdirectory(examples)
endif()

# ==============================================================================
# Python Bindings
# ==============================================================================
if(TRIAGEATTENTION_BUILD_PYTHON)
  find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
  add_subdirectory(python)
endif()

# ==============================================================================
# Installation
# ==============================================================================
install(TARGETS triageattention_kernels
  EXPORT TriageAttentionTargets
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION bin
  INCLUDES DESTINATION include
)

install(DIRECTORY include/triageattention
  DESTINATION include
)

# ==============================================================================
# Summary
# ==============================================================================
message(STATUS "")
message(STATUS "TriageAttention Configuration Summary")
message(STATUS "=====================================")
message(STATUS "Version:             ${PROJECT_VERSION}")
message(STATUS "CUDA Version:        ${CUDAToolkit_VERSION}")
message(STATUS "CUDA Architectures:  ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "CUTLASS:             ${CUTLASS_INCLUDE_DIR}")
message(STATUS "Build Tests:         ${TRIAGEATTENTION_BUILD_TESTS}")
message(STATUS "Build Benchmarks:    ${TRIAGEATTENTION_BUILD_BENCHMARKS}")
message(STATUS "Build Examples:      ${TRIAGEATTENTION_BUILD_EXAMPLES}")
message(STATUS "Build Python:        ${TRIAGEATTENTION_BUILD_PYTHON}")
message(STATUS "=====================================")
message(STATUS "")


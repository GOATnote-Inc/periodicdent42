cmake_minimum_required(VERSION 3.18)
project(BlackwellSparseK LANGUAGES CUDA CXX)

# ============================================================================
# PROJECT CONFIGURATION
# ============================================================================

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Export compile commands for IDE integration
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ============================================================================
# CUTLASS INTEGRATION
# ============================================================================

# CUTLASS path
set(CUTLASS_PATH "/opt/cutlass" CACHE PATH "Path to CUTLASS library")
if(NOT EXISTS ${CUTLASS_PATH})
    message(FATAL_ERROR "CUTLASS not found at ${CUTLASS_PATH}. Set CUTLASS_PATH.")
endif()

message(STATUS "CUTLASS Path: ${CUTLASS_PATH}")
include_directories(${CUTLASS_PATH}/include)

# ============================================================================
# CUDA ARCHITECTURE SUPPORT
# ============================================================================

# Dual-architecture support: sm_90a (Hopper H100) and sm_100 (Blackwell B200)
set(CMAKE_CUDA_ARCHITECTURES "90a;100" CACHE STRING "CUDA architectures to build for")

message(STATUS "CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")

# ============================================================================
# COMPILATION FLAGS
# ============================================================================

# CUDA compilation flags
set(CUDA_NVCC_FLAGS
    -O3
    --use_fast_math
    -lineinfo
    --expt-relaxed-constexpr
    --expt-extended-lambda
    -Xcompiler=-fPIC
    -Xcompiler=-fopenmp
)

# Debug flags (optional)
option(BSK_DEBUG "Enable debug mode" OFF)
if(BSK_DEBUG)
    list(APPEND CUDA_NVCC_FLAGS -G -DDEBUG=1 -O0)
    message(STATUS "Debug Mode: ENABLED")
endif()

# Profiling flags (optional)
option(BSK_PROFILE "Enable profiling" OFF)
if(BSK_PROFILE)
    list(APPEND CUDA_NVCC_FLAGS -DPROFILE=1)
    message(STATUS "Profiling: ENABLED")
endif()

# Apply CUDA flags
foreach(flag ${CUDA_NVCC_FLAGS})
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${flag}")
endforeach()

# C++ compilation flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -fPIC -fopenmp")

# ============================================================================
# SOURCE FILES
# ============================================================================

set(KERNEL_SOURCES
    src/blackwell_sparsek/kernels/attention_fmha.cu
    src/blackwell_sparsek/kernels/kernel_dispatch.cu
    src/blackwell_sparsek/kernels/kernel_bindings.cpp
)

# ============================================================================
# PYTORCH INTEGRATION
# ============================================================================

# Find PyTorch
find_package(Torch REQUIRED)

# Include directories
include_directories(${TORCH_INCLUDE_DIRS})
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

# ============================================================================
# BUILD TARGETS
# ============================================================================

# Main library
add_library(blackwell_sparsek_kernels MODULE ${KERNEL_SOURCES})

# Link libraries
target_link_libraries(blackwell_sparsek_kernels ${TORCH_LIBRARIES})
target_link_libraries(blackwell_sparsek_kernels cudart cuda)

# Set output properties
set_target_properties(blackwell_sparsek_kernels PROPERTIES
    PREFIX ""
    OUTPUT_NAME "_C"
    SUFFIX ".so"
)

# Install target
install(TARGETS blackwell_sparsek_kernels
    LIBRARY DESTINATION src/blackwell_sparsek
)

# ============================================================================
# BUILD SUMMARY
# ============================================================================

message(STATUS "========================================")
message(STATUS "BlackwellSparseK Build Configuration")
message(STATUS "========================================")
message(STATUS "  CUDA Version:        ${CMAKE_CUDA_COMPILER_VERSION}")
message(STATUS "  CUDA Architectures:  ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "  CUTLASS Path:        ${CUTLASS_PATH}")
message(STATUS "  PyTorch Include:     ${TORCH_INCLUDE_DIRS}")
message(STATUS "  Debug Mode:          ${BSK_DEBUG}")
message(STATUS "  Profiling:           ${BSK_PROFILE}")
message(STATUS "========================================")


# CUDAdent42: FlashAttention CUDA Kernels with BF16 Support
# Production-grade CMake build system

cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(flashmoe_science LANGUAGES CXX CUDA)

# Find required packages
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
find_package(CUDAToolkit REQUIRED)

# Find PyTorch
execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import torch; print(torch.__version__)"
    OUTPUT_VARIABLE TORCH_VERSION
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import torch; import os; print(os.path.dirname(torch.__file__))"
    OUTPUT_VARIABLE TORCH_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

message(STATUS "Found PyTorch ${TORCH_VERSION}")
message(STATUS "PyTorch location: ${TORCH_DIR}")

# PyTorch include directories
set(TORCH_INCLUDE_DIRS 
    "${TORCH_DIR}/include"
    "${TORCH_DIR}/include/torch/csrc/api/include"
)

# Get PyTorch ABI flag
execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import torch; print('1' if torch._C._GLIBCXX_USE_CXX11_ABI else '0')"
    OUTPUT_VARIABLE TORCH_CXX11_ABI
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

message(STATUS "PyTorch ABI: _GLIBCXX_USE_CXX11_ABI=${TORCH_CXX11_ABI}")

# Detect GPU architecture
if(NOT DEFINED CUDA_ARCHITECTURES)
    execute_process(
        COMMAND ${Python3_EXECUTABLE} -c "import torch; major, minor = torch.cuda.get_device_capability() if torch.cuda.is_available() else (8, 9); print(f'{major}{minor}')"
        OUTPUT_VARIABLE DETECTED_ARCH
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    set(CUDA_ARCHITECTURES ${DETECTED_ARCH})
    message(STATUS "Auto-detected CUDA architecture: SM_${CUDA_ARCHITECTURES}")
else()
    message(STATUS "Using specified CUDA architecture: SM_${CUDA_ARCHITECTURES}")
endif()

# Determine if BF16 is supported (SM80+)
if(${CUDA_ARCHITECTURES} GREATER_EQUAL 80)
    set(HAS_BF16 ON)
    message(STATUS "BF16 support: ENABLED (SM${CUDA_ARCHITECTURES} >= SM80)")
else()
    set(HAS_BF16 OFF)
    message(STATUS "BF16 support: DISABLED (SM${CUDA_ARCHITECTURES} < SM80)")
endif()

# C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Base compilation flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -fPIC")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=${TORCH_CXX11_ABI}")

# Base CUDA flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --use_fast_math -lineinfo")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr --expt-extended-lambda")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=-fPIC,-fno-strict-aliasing,-fno-omit-frame-pointer")

# Architecture-specific CUDA flags
set(CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCHITECTURES})

# Include directories
include_directories(
    ${CMAKE_SOURCE_DIR}/python/flashmoe_science/csrc
    ${TORCH_INCLUDE_DIRS}
    ${Python3_INCLUDE_DIRS}
    ${CUDAToolkit_INCLUDE_DIRS}
)

# FP16 kernel (always compiled)
add_library(flashmoe_fp16 OBJECT
    python/flashmoe_science/csrc/flash_attention_fp16_sm75.cu
)

target_compile_definitions(flashmoe_fp16 PRIVATE
    FLASHMOE_HAS_BF16=$<BOOL:${HAS_BF16}>
    CUDA_NO_BFLOAT16
    __CUDA_NO_BFLOAT16_OPERATORS__
)

set_target_properties(flashmoe_fp16 PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    POSITION_INDEPENDENT_CODE ON
)

# BF16 kernel (only if supported)
if(HAS_BF16)
    add_library(flashmoe_bf16 OBJECT
        python/flashmoe_science/csrc/flash_attention_bf16_sm80.cu
    )
    
    target_compile_definitions(flashmoe_bf16 PRIVATE
        FLASHMOE_HAS_BF16=1
    )
    
    set_target_properties(flashmoe_bf16 PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
        POSITION_INDEPENDENT_CODE ON
    )
    
    message(STATUS "Building BF16 kernel (SM80+)")
endif()

# Python bindings (host code)
add_library(flashmoe_bindings OBJECT
    python/flashmoe_science/csrc/bindings_new.cpp
)

target_compile_definitions(flashmoe_bindings PRIVATE
    FLASHMOE_HAS_BF16=$<BOOL:${HAS_BF16}>
    CUDA_NO_BFLOAT16
    __CUDA_NO_BFLOAT16_OPERATORS__
    TORCH_API_INCLUDE_EXTENSION_H
    TORCH_EXTENSION_NAME=_C
)

set_target_properties(flashmoe_bindings PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)

# Link all together into Python module
if(HAS_BF16)
    set(ALL_OBJECTS $<TARGET_OBJECTS:flashmoe_fp16> $<TARGET_OBJECTS:flashmoe_bf16> $<TARGET_OBJECTS:flashmoe_bindings>)
else()
    set(ALL_OBJECTS $<TARGET_OBJECTS:flashmoe_fp16> $<TARGET_OBJECTS:flashmoe_bindings>)
endif()

# Get Python extension suffix
execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))"
    OUTPUT_VARIABLE PYTHON_EXT_SUFFIX
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

# Create shared library
add_library(_C SHARED ${ALL_OBJECTS})

set_target_properties(_C PROPERTIES
    PREFIX ""
    SUFFIX "${PYTHON_EXT_SUFFIX}"
    OUTPUT_NAME "_C"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/flashmoe_science"
)

# Link PyTorch libraries
target_link_directories(_C PRIVATE "${TORCH_DIR}/lib")
target_link_libraries(_C PRIVATE
    c10
    c10_cuda
    torch
    torch_cpu
    torch_python
    torch_cuda
    ${CUDA_LIBRARIES}
    cudart
)

# Set RPATH for runtime library discovery
set_target_properties(_C PROPERTIES
    BUILD_RPATH "${TORCH_DIR}/lib:${CUDAToolkit_LIBRARY_DIR}"
    INSTALL_RPATH "${TORCH_DIR}/lib:${CUDAToolkit_LIBRARY_DIR}"
)

# Summary
message(STATUS "═══════════════════════════════════════════════════════════")
message(STATUS "  CUDAdent42 Build Configuration")
message(STATUS "═══════════════════════════════════════════════════════════")
message(STATUS "  Python: ${Python3_VERSION}")
message(STATUS "  PyTorch: ${TORCH_VERSION}")
message(STATUS "  CUDA: ${CUDAToolkit_VERSION}")
message(STATUS "  Target GPU: SM_${CUDA_ARCHITECTURES}")
message(STATUS "  BF16 Support: ${HAS_BF16}")
message(STATUS "  ABI: _GLIBCXX_USE_CXX11_ABI=${TORCH_CXX11_ABI}")
message(STATUS "  Output: flashmoe_science/_C${PYTHON_EXT_SUFFIX}")
message(STATUS "═══════════════════════════════════════════════════════════")

# Optional: Install target
install(TARGETS _C
    LIBRARY DESTINATION ${CMAKE_SOURCE_DIR}/flashmoe_science
)


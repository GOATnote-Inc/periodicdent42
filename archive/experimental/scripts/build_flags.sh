#!/bin/bash
# Build Flag Presets for L4 (sm_89) CUDA Kernel Development
# Phase 0: Pre-flight setup

# Export functions for use in other scripts
export CUDA_ARCH="sm_89"
export CUDA_BIN="/usr/local/cuda/bin"

# Sanitizer/Debug flags
export FLAGS_DEBUG="-G -lineinfo -U__CUDA_NO_HALF_OPERATORS__ -U__CUDA_NO_HALF_CONVERSIONS__ -DDEBUG_V3"

# Release/Bench flags
export FLAGS_RELEASE="-O3 -use_fast_math -Xptxas -v --expt-relaxed-constexpr -DNDEBUG"

# Common flags for both modes
export FLAGS_COMMON="-arch=${CUDA_ARCH} --extended-lambda --expt-relaxed-constexpr"

# Helper functions
build_debug() {
    local src=$1
    local out=$2
    ${CUDA_BIN}/nvcc ${FLAGS_COMMON} ${FLAGS_DEBUG} -o "${out}" "${src}" "$@"
}

build_release() {
    local src=$1
    local out=$2
    ${CUDA_BIN}/nvcc ${FLAGS_COMMON} ${FLAGS_RELEASE} -o "${out}" "${src}" "$@"
}

# Print current config
print_config() {
    echo "============================================================"
    echo "Build Configuration (L4 / sm_89)"
    echo "============================================================"
    echo "CUDA Architecture: ${CUDA_ARCH}"
    echo "CUDA Binaries:     ${CUDA_BIN}"
    echo ""
    echo "Debug Flags:"
    echo "  ${FLAGS_COMMON} ${FLAGS_DEBUG}"
    echo ""
    echo "Release Flags:"
    echo "  ${FLAGS_COMMON} ${FLAGS_RELEASE}"
    echo "============================================================"
}

# If sourced, export functions; if executed, print config
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    print_config
fi


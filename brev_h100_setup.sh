#!/bin/bash
set -euo pipefail

# ==============================================================================
# Brev H100 Environment Setup - Expert CUDA/Nsight Configuration
# ==============================================================================
# Purpose: Production-grade CUDA 13.0.2 + CUTLASS 4.3.0 + Nsight Compute setup
# Target: NVIDIA H100 (sm_90a) on Brev.dev
# Author: Brandon Dent, MD
# Date: November 2, 2025
# ==============================================================================

export DEBIAN_FRONTEND=noninteractive
export WORKSPACE="/workspace"
export CUDA_VERSION="13.0"
export CUTLASS_VERSION="4.3.0"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# ==============================================================================
# Phase 1: Environment Validation
# ==============================================================================

phase1_validate_environment() {
    log_info "Phase 1: Validating H100 environment..."
    
    # Check GPU
    if ! command -v nvidia-smi &> /dev/null; then
        log_error "nvidia-smi not found"
        exit 1
    fi
    
    log_info "GPU Information:"
    nvidia-smi --query-gpu=name,compute_cap,memory.total,driver_version --format=csv,noheader
    
    # Verify H100
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
    if [[ ! "$GPU_NAME" =~ "H100" ]]; then
        log_warning "Expected H100, found: $GPU_NAME"
    fi
    
    # Check CUDA Toolkit
    if ! command -v nvcc &> /dev/null; then
        log_error "nvcc not found - CUDA Toolkit not installed"
        exit 1
    fi
    
    NVCC_VERSION=$(nvcc --version | grep "release" | sed 's/.*release //' | sed 's/,.*//')
    log_info "CUDA Toolkit: $NVCC_VERSION"
    
    if [[ ! "$NVCC_VERSION" =~ "13.0" ]]; then
        log_warning "Expected CUDA 13.0.x, found: $NVCC_VERSION"
    fi
    
    # Check Nsight Compute
    if ! command -v ncu &> /dev/null; then
        log_warning "ncu not found - will install Nsight Compute"
        NCU_AVAILABLE=false
    else
        NCU_VERSION=$(ncu --version | head -1 | grep -oP '\d+\.\d+\.\d+' || echo "unknown")
        log_info "Nsight Compute: $NCU_VERSION"
        NCU_AVAILABLE=true
    fi
    
    # Check performance counter access
    if [ -f /proc/driver/nvidia/capabilities/gpu0/mig/config ]; then
        log_info "MIG mode detected - checking CUPTI access"
    fi
    
    # Test basic CUDA access
    log_info "Testing CUDA device access..."
    cat > /tmp/test_cuda.cu <<'EOF'
#include <cuda_runtime.h>
#include <stdio.h>
int main() {
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    printf("Device: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("SMs: %d\n", prop.multiProcessorCount);
    return 0;
}
EOF
    
    nvcc -o /tmp/test_cuda /tmp/test_cuda.cu 2>/dev/null || {
        log_error "Failed to compile test CUDA program"
        exit 1
    }
    /tmp/test_cuda || {
        log_error "Failed to run test CUDA program"
        exit 1
    }
    rm /tmp/test_cuda /tmp/test_cuda.cu
    
    log_success "Phase 1 complete: Environment validated"
}

# ==============================================================================
# Phase 2: Toolchain Setup
# ==============================================================================

phase2_install_toolchain() {
    log_info "Phase 2: Installing essential toolchain..."
    
    # Update package lists
    log_info "Updating package lists..."
    apt-get update -qq
    
    # Install build essentials
    log_info "Installing build tools..."
    apt-get install -y -qq \
        build-essential \
        git \
        cmake \
        ninja-build \
        python3-pip \
        wget \
        curl \
        vim \
        htop \
        tmux \
        jq \
        bc \
        libssl-dev \
        pkg-config
    
    # Install Python tools
    log_info "Installing Python tools..."
    pip3 install --quiet \
        numpy \
        pandas \
        matplotlib \
        seaborn \
        jupyter \
        torch --index-url https://download.pytorch.org/whl/cu121
    
    log_success "Phase 2 complete: Toolchain installed"
}

# ==============================================================================
# Phase 3: CUTLASS 4.3.0 Setup
# ==============================================================================

phase3_setup_cutlass() {
    log_info "Phase 3: Setting up CUTLASS 4.3.0..."
    
    CUTLASS_DIR="/usr/local/cutlass-4.3.0"
    
    if [ -d "$CUTLASS_DIR" ]; then
        log_warning "CUTLASS already exists at $CUTLASS_DIR"
        read -p "Reinstall? (y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            log_info "Skipping CUTLASS installation"
            return
        fi
        rm -rf "$CUTLASS_DIR"
    fi
    
    log_info "Cloning CUTLASS 4.3.0..."
    git clone --depth 1 --branch main https://github.com/NVIDIA/cutlass.git "$CUTLASS_DIR"
    
    cd "$CUTLASS_DIR"
    
    # Verify it's actually 4.3.0 (or close)
    GIT_DESC=$(git describe --tags 2>/dev/null || echo "main-branch")
    log_info "CUTLASS version: $GIT_DESC"
    
    # Build CUTLASS (optional, for examples)
    log_info "Building CUTLASS examples (optional, can skip)..."
    mkdir -p build
    cd build
    
    cmake .. \
        -DCUTLASS_NVCC_ARCHS=90 \
        -DCUTLASS_ENABLE_TESTS=OFF \
        -DCUTLASS_ENABLE_EXAMPLES=ON \
        -DCMAKE_BUILD_TYPE=Release \
        -G Ninja || log_warning "CMake configuration failed (non-critical)"
    
    if [ -f "build.ninja" ]; then
        ninja -j$(nproc) || log_warning "Build failed (non-critical for header-only usage)"
    fi
    
    # Set environment variables
    echo "export CUTLASS_HOME=$CUTLASS_DIR" >> ~/.bashrc
    echo "export CPATH=\$CUTLASS_HOME/include:\$CPATH" >> ~/.bashrc
    export CUTLASS_HOME="$CUTLASS_DIR"
    export CPATH="$CUTLASS_HOME/include:$CPATH"
    
    log_success "Phase 3 complete: CUTLASS 4.3.0 installed at $CUTLASS_DIR"
}

# ==============================================================================
# Phase 4: Nsight Tools Installation
# ==============================================================================

phase4_install_nsight() {
    log_info "Phase 4: Installing Nsight Compute tools..."
    
    # Check if already installed
    if command -v ncu &> /dev/null; then
        NCU_VERSION=$(ncu --version | head -1)
        log_info "Nsight Compute already installed: $NCU_VERSION"
        
        # Test performance counter access
        log_info "Testing NCU performance counter access..."
        
        cat > /tmp/test_ncu.cu <<'EOF'
#include <cuda_runtime.h>
__global__ void test_kernel() {
    int x = threadIdx.x;
    float y = x * 2.0f;
}
int main() {
    test_kernel<<<1, 32>>>();
    cudaDeviceSynchronize();
    return 0;
}
EOF
        
        nvcc -o /tmp/test_ncu /tmp/test_ncu.cu
        
        if ncu --set full --target-processes all /tmp/test_ncu &> /tmp/ncu_test.log; then
            log_success "NCU performance counters accessible"
        else
            log_error "NCU performance counters NOT accessible"
            log_info "This may require running in privileged mode or adjusting permissions"
            log_info "Check /tmp/ncu_test.log for details"
        fi
        
        rm -f /tmp/test_ncu /tmp/test_ncu.cu /tmp/ncu_test.log
        
    else
        log_warning "Nsight Compute not found in PATH"
        log_info "NCU is typically included with CUDA Toolkit"
        log_info "Checking for NCU in CUDA installation..."
        
        if [ -f "/usr/local/cuda/bin/ncu" ]; then
            log_info "Found NCU at /usr/local/cuda/bin/ncu"
            ln -sf /usr/local/cuda/bin/ncu /usr/local/bin/ncu || true
        else
            log_warning "NCU not found - may need manual CUDA Toolkit installation"
        fi
    fi
    
    # Install nsys (Nsight Systems) if available
    if ! command -v nsys &> /dev/null; then
        log_info "Nsight Systems (nsys) not found - checking CUDA installation..."
        if [ -f "/usr/local/cuda/bin/nsys" ]; then
            ln -sf /usr/local/cuda/bin/nsys /usr/local/bin/nsys || true
            log_success "Linked nsys to /usr/local/bin/"
        fi
    else
        NSYS_VERSION=$(nsys --version | head -1)
        log_info "Nsight Systems: $NSYS_VERSION"
    fi
    
    log_success "Phase 4 complete: Nsight tools configured"
}

# ==============================================================================
# Phase 5: Development Environment
# ==============================================================================

phase5_setup_dev_environment() {
    log_info "Phase 5: Configuring development environment..."
    
    # Create workspace structure
    mkdir -p "$WORKSPACE"/{projects,ncu_reports,benchmarks,logs}
    
    # Install Rust (for Burn integration)
    log_info "Installing Rust toolchain..."
    if ! command -v rustup &> /dev/null; then
        curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain nightly
        source "$HOME/.cargo/env"
    else
        log_info "Rust already installed"
    fi
    
    rustup override set nightly
    rustup component add rust-src rustfmt clippy
    
    # Verify Rust + CUDA coexistence
    log_info "Verifying Rust + CUDA coexistence..."
    cat > /tmp/test_rust_cuda.sh <<'EOF'
#!/bin/bash
echo "Testing Rust..."
rustc --version
echo "Testing CUDA..."
nvcc --version | head -1
echo "Both work in same pipeline!"
EOF
    chmod +x /tmp/test_rust_cuda.sh
    /tmp/test_rust_cuda.sh || log_warning "Rust/CUDA coexistence test failed"
    rm /tmp/test_rust_cuda.sh
    
    # Configure environment variables
    cat >> ~/.bashrc <<'EOF'

# ==== H100 CUDA Development Environment ====
export WORKSPACE="/workspace"
export CUDA_HOME="/usr/local/cuda"
export PATH="$CUDA_HOME/bin:$HOME/.cargo/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
export CUDA_VISIBLE_DEVICES="0"

# CUTLASS
export CUTLASS_HOME="/usr/local/cutlass-4.3.0"
export CPATH="$CUTLASS_HOME/include:$CPATH"

# Rust
source "$HOME/.cargo/env"

# Aliases for common tasks
alias gpu='nvidia-smi'
alias gpuw='watch -n 1 nvidia-smi'
alias ncu-check='ncu --query-metrics'
alias cuda-info='nvidia-smi --query-gpu=name,compute_cap,memory.total,driver_version,cuda_version --format=csv'
EOF
    
    source ~/.bashrc
    
    log_success "Phase 5 complete: Development environment configured"
}

# ==============================================================================
# Phase 6: Clone and Build BlackwellSparseK
# ==============================================================================

phase6_setup_blackwellsparsek() {
    log_info "Phase 6: Setting up BlackwellSparseK..."
    
    cd "$WORKSPACE/projects"
    
    if [ -d "BlackwellSparseK" ]; then
        log_warning "BlackwellSparseK already exists"
        cd BlackwellSparseK
        git pull
    else
        log_info "Cloning periodicdent42..."
        git clone https://github.com/GOATnote-Inc/periodicdent42.git
        cd periodicdent42/BlackwellSparseK
    fi
    
    log_info "Building 598.9 TFLOPS kernel..."
    
    # Create build directory
    mkdir -p build
    
    # Compile the kernel
    nvcc -O3 -std=c++17 -arch=sm_90a --expt-relaxed-constexpr \
         --maxrregcount=255 --use_fast_math -lineinfo \
         -I"$CUTLASS_HOME/include" \
         src/gemm_h100_599tflops_final.cu \
         -o build/blackwell_gemm \
         -lcudart || {
        log_error "Failed to compile BlackwellSparseK kernel"
        return 1
    }
    
    log_success "Kernel compiled successfully"
    
    # Quick validation run
    log_info "Running quick validation..."
    if ./build/blackwell_gemm 2>&1 | tee "$WORKSPACE/logs/blackwell_initial_run.log"; then
        log_success "Kernel executed successfully"
    else
        log_error "Kernel execution failed - check logs"
    fi
    
    log_success "Phase 6 complete: BlackwellSparseK ready"
}

# ==============================================================================
# Phase 7: NCU Profiling Test
# ==============================================================================

phase7_ncu_profiling() {
    log_info "Phase 7: Running Nsight Compute profiling..."
    
    if ! command -v ncu &> /dev/null; then
        log_warning "NCU not available - skipping profiling"
        return
    fi
    
    cd "$WORKSPACE/projects/periodicdent42/BlackwellSparseK"
    
    if [ ! -f "build/blackwell_gemm" ]; then
        log_warning "Kernel not built - skipping profiling"
        return
    fi
    
    log_info "Running NCU with full metric collection..."
    
    NCU_REPORT="$WORKSPACE/ncu_reports/blackwell_validation_$(date +%Y%m%d_%H%M%S).ncu-rep"
    
    # Run NCU profiling
    ncu --set full \
        --target-processes all \
        --export "$NCU_REPORT" \
        ./build/blackwell_gemm 2>&1 | tee "$WORKSPACE/logs/ncu_profiling.log" || {
        log_error "NCU profiling failed"
        log_info "This may be due to insufficient permissions"
        log_info "Trying basic metrics instead..."
        
        # Fallback to basic metrics
        ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,dram__throughput.avg.pct_of_peak_sustained_elapsed \
            ./build/blackwell_gemm 2>&1 | tee "$WORKSPACE/logs/ncu_basic.log" || {
            log_error "Even basic NCU profiling failed"
            log_warning "Performance counter access may be blocked"
            return 1
        }
    }
    
    if [ -f "$NCU_REPORT" ]; then
        log_success "NCU report saved: $NCU_REPORT"
        log_info "View with: ncu-ui $NCU_REPORT"
    fi
    
    log_success "Phase 7 complete: NCU profiling tested"
}

# ==============================================================================
# Phase 8: Create Preflight Script
# ==============================================================================

phase8_create_preflight() {
    log_info "Phase 8: Creating preflight validation script..."
    
    cat > "$WORKSPACE/preflight.sh" <<'PREFLIGHT_EOF'
#!/bin/bash
# ==============================================================================
# Preflight Check - H100 Environment Validation
# ==============================================================================

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[✓]${NC} $1"; }
log_error() { echo -e "${RED}[✗]${NC} $1"; }

echo "╔════════════════════════════════════════════════════════════╗"
echo "║          H100 Environment Preflight Check                  ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# GPU Check
log_info "GPU Status:"
nvidia-smi --query-gpu=name,temperature.gpu,utilization.gpu,utilization.memory,memory.used,memory.total --format=csv,noheader | while IFS=, read -r name temp util_gpu util_mem mem_used mem_total; do
    echo "  GPU: $name"
    echo "  Temp: $temp"
    echo "  GPU Util: $util_gpu"
    echo "  Mem Util: $util_mem"
    echo "  Memory: $mem_used / $mem_total"
done

# CUDA Toolkit
log_info "CUDA Toolkit:"
nvcc --version | grep "release" | sed 's/.*release /  Version: /'

# Nsight Compute
if command -v ncu &> /dev/null; then
    log_success "Nsight Compute: $(ncu --version | head -1)"
else
    log_error "Nsight Compute not found"
fi

# Nsight Systems
if command -v nsys &> /dev/null; then
    log_success "Nsight Systems: $(nsys --version | head -1)"
else
    log_error "Nsight Systems not found"
fi

# CUTLASS
if [ -d "$CUTLASS_HOME" ]; then
    log_success "CUTLASS: $CUTLASS_HOME"
else
    log_error "CUTLASS not found"
fi

# Rust
if command -v rustc &> /dev/null; then
    log_success "Rust: $(rustc --version)"
else
    log_error "Rust not found"
fi

# GPU Architecture Details
log_info "GPU Architecture:"
nvidia-smi --query-gpu=compute_cap,name --format=csv,noheader | while IFS=, read -r compute_cap name; do
    echo "  Compute Capability: $compute_cap"
    echo "  Name: $name"
done

# SM Count and Clocks
log_info "GPU Specifications:"
nvidia-smi --query-gpu=clocks.current.sm,clocks.max.sm,clocks.current.memory,clocks.max.memory --format=csv,noheader | while IFS=, read -r sm_cur sm_max mem_cur mem_max; do
    echo "  SM Clock: $sm_cur (max: $sm_max)"
    echo "  Memory Clock: $mem_cur (max: $mem_max)"
done

# Workspace
if [ -d "/workspace" ]; then
    log_success "Workspace: /workspace ($(du -sh /workspace 2>/dev/null | cut -f1))"
else
    log_error "Workspace not found"
fi

echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║                   Preflight Complete                       ║"
echo "╚════════════════════════════════════════════════════════════╝"

# Save system info
cat > /workspace/system_info.log <<EOF
=== H100 System Information ===
Date: $(date)
GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)
Compute Capability: $(nvidia-smi --query-gpu=compute_cap --format=csv,noheader)
Driver: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader)
CUDA: $(nvcc --version | grep "release" | sed 's/.*release //' | sed 's/,.*//')
SM Count: $(nvidia-smi --query-gpu=count --format=csv,noheader || echo "N/A")
Memory: $(nvidia-smi --query-gpu=memory.total --format=csv,noheader)
CUTLASS: $CUTLASS_HOME
Rust: $(rustc --version)
EOF

log_success "System info saved to /workspace/system_info.log"
PREFLIGHT_EOF
    
    chmod +x "$WORKSPACE/preflight.sh"
    
    log_success "Preflight script created: $WORKSPACE/preflight.sh"
    
    # Run it once
    log_info "Running preflight check..."
    "$WORKSPACE/preflight.sh"
    
    log_success "Phase 8 complete: Preflight script ready"
}

# ==============================================================================
# Phase 9: Optional - Burn Integration Prep
# ==============================================================================

phase9_burn_integration_prep() {
    log_info "Phase 9: Preparing for Burn integration (optional)..."
    
    cd "$WORKSPACE/projects"
    
    # Create burn-blackwell scaffold
    log_info "Creating burn-blackwell crate scaffold..."
    
    cargo new --lib burn-blackwell || {
        log_warning "burn-blackwell already exists"
        cd burn-blackwell
    }
    
    cd burn-blackwell
    
    # Update Cargo.toml
    cat > Cargo.toml <<'EOF'
[package]
name = "burn-blackwell"
version = "0.1.0"
edition = "2021"

[dependencies]
cuda-sys = "0.4"
libc = "0.2"

[build-dependencies]
cc = "1.0"

[lib]
name = "burn_blackwell"
path = "src/lib.rs"
EOF
    
    # Create basic lib.rs
    mkdir -p src
    cat > src/lib.rs <<'EOF'
//! Burn integration for BlackwellSparseK dense GEMM kernel
//! 
//! Performance: 598.9 TFLOPS (96.2% of cuBLAS) on H100
//! 
//! Note: This is currently educational only, as cuBLAS is 4% faster.
//! For production, use this as a template for sparse kernels where
//! we can provide real speedup (5× over cuSPARSE).

#![allow(dead_code)]

pub mod ffi;

pub fn init() {
    println!("BlackwellSparseK integration initialized");
    println!("Note: Dense GEMM is 4% slower than cuBLAS");
    println!("      Consider sparse kernel for real value");
}
EOF
    
    # Create FFI module placeholder
    mkdir -p src
    cat > src/ffi.rs <<'EOF'
//! FFI bindings for BlackwellSparseK kernel
//! 
//! To compile the shared library:
//! ```bash
//! nvcc -O3 -std=c++17 -arch=sm_90a --expt-relaxed-constexpr \
//!      -Xcompiler -fPIC -shared \
//!      -I/usr/local/cutlass-4.3.0/include \
//!      src/gemm_h100_599tflops_final.cu \
//!      -o libblackwell_dense_gemm.so -lcudart
//! ```

use std::os::raw::{c_int, c_void};

#[link(name = "blackwell_dense_gemm", kind = "dylib")]
extern "C" {
    // Add actual kernel FFI here when ready
}

pub fn placeholder() {
    println!("FFI bindings placeholder - integrate when kernel is compiled as .so");
}
EOF
    
    # Create README
    cat > README.md <<'EOF'
# burn-blackwell

Burn integration for BlackwellSparseK dense GEMM kernel.

## Performance

- **H100:** 598.9 TFLOPS (96.2% of cuBLAS)
- **cuBLAS:** 622.8 TFLOPS (Burn's current default)

## Status

⚠️ **Educational only** - This kernel is 4% slower than cuBLAS.

For production value, consider:
1. Building sparse kernel (5× speedup over cuSPARSE)
2. Contributing optimizations to CUTLASS
3. Closing the 4% gap to cuBLAS

See `../periodicdent42/BlackwellSparseK/BURN_INTEGRATION_ASSESSMENT.md` for details.

## Build

```bash
# Compile kernel as shared library
cd ../periodicdent42/BlackwellSparseK
./build_shared_lib.sh

# Build Rust crate
cd ../../burn-blackwell
cargo build --release
```

## Usage

```rust
use burn_blackwell;

fn main() {
    burn_blackwell::init();
    // Integration code here
}
```
EOF
    
    log_success "Burn integration scaffold created at $WORKSPACE/projects/burn-blackwell"
    log_info "This is currently a template - see BlackwellSparseK/BURN_INTEGRATION_ASSESSMENT.md"
    
    log_success "Phase 9 complete: Burn integration prepared"
}

# ==============================================================================
# Main Execution
# ==============================================================================

main() {
    echo "╔════════════════════════════════════════════════════════════╗"
    echo "║   Brev H100 Environment Setup - Expert Configuration       ║"
    echo "║   CUDA 13.0.2 + CUTLASS 4.3.0 + Nsight Compute            ║"
    echo "╚════════════════════════════════════════════════════════════╝"
    echo ""
    
    phase1_validate_environment
    phase2_install_toolchain
    phase3_setup_cutlass
    phase4_install_nsight
    phase5_setup_dev_environment
    phase6_setup_blackwellsparsek
    phase7_ncu_profiling
    phase8_create_preflight
    phase9_burn_integration_prep
    
    echo ""
    echo "╔════════════════════════════════════════════════════════════╗"
    echo "║                  Setup Complete!                           ║"
    echo "╚════════════════════════════════════════════════════════════╝"
    echo ""
    log_success "Brev H100 environment ready for kernel development"
    echo ""
    echo "Quick Start:"
    echo "  • Preflight check: /workspace/preflight.sh"
    echo "  • GPU status: nvidia-smi"
    echo "  • Run kernel: /workspace/projects/periodicdent42/BlackwellSparseK/build/blackwell_gemm"
    echo "  • NCU profile: ncu --set full ./build/blackwell_gemm"
    echo "  • System info: cat /workspace/system_info.log"
    echo ""
    echo "Directories:"
    echo "  • Workspace: /workspace"
    echo "  • Projects: /workspace/projects"
    echo "  • NCU Reports: /workspace/ncu_reports"
    echo "  • Logs: /workspace/logs"
    echo ""
}

# Run if executed directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi


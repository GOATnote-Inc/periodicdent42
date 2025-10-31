#!/bin/bash
# ============================================================================
# BlackwellSparseK Environment Validation
# ============================================================================
# Validates H100 environment is ready for kernel development
# Does not require Docker (for RunPod compatibility)
# ============================================================================

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

log_success() { echo -e "${GREEN}✅ $1${NC}"; }
log_error() { echo -e "${RED}❌ $1${NC}"; }
log_info() { echo -e "${CYAN}ℹ️  $1${NC}"; }
log_step() { echo -e "${BLUE}▶ $1${NC}"; }

echo ""
echo "=========================================="
echo "  BlackwellSparseK Environment Validation"
echo "=========================================="
echo ""

# Check 1: NVIDIA GPU
log_step "Checking NVIDIA GPU..."
if ! command -v nvidia-smi &> /dev/null; then
    log_error "nvidia-smi not found"
    exit 1
fi

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader)
GPU_ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader)
GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader)

log_success "GPU: $GPU_NAME"
log_info "Compute Capability: $GPU_ARCH"
log_info "Memory: $GPU_MEM"

# Check 2: CUDA Toolkit
log_step "Checking CUDA Toolkit..."
if command -v nvcc &> /dev/null; then
    CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $5}' | sed 's/,//')
    log_success "CUDA: $CUDA_VERSION"
else
    log_error "nvcc not found"
    exit 1
fi

# Check 3: Python
log_step "Checking Python..."
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version | awk '{print $2}')
    log_success "Python: $PYTHON_VERSION"
else
    log_error "python3 not found"
    exit 1
fi

# Check 4: PyTorch
log_step "Checking PyTorch..."
if python3 -c "import torch; print(torch.__version__)" &> /dev/null; then
    TORCH_VERSION=$(python3 -c "import torch; print(torch.__version__)")
    TORCH_CUDA=$(python3 -c "import torch; print('Available' if torch.cuda.is_available() else 'Not Available')")
    log_success "PyTorch: $TORCH_VERSION"
    log_info "CUDA Support: $TORCH_CUDA"
else
    log_error "PyTorch not installed"
fi

# Check 5: Project structure
log_step "Checking BlackwellSparseK structure..."
if [ -f "pyproject.toml" ]; then
    log_success "pyproject.toml found"
else
    log_error "pyproject.toml not found"
    exit 1
fi

if [ -d "src/blackwell_sparsek" ]; then
    log_success "src/blackwell_sparsek/ exists"
else
    log_error "src/blackwell_sparsek/ not found"
    exit 1
fi

# Check 6: Scripts
log_step "Checking scripts..."
SCRIPTS=("h100_orchestrator.sh" "collect_logs.sh" "remote_h100_deploy.sh" "validate_environment.sh")
for script in "${SCRIPTS[@]}"; do
    if [ -f "scripts/$script" ]; then
        if bash -n "scripts/$script" 2>/dev/null; then
            log_success "$script: syntax OK"
        else
            log_error "$script: syntax error"
            exit 1
        fi
    else
        log_error "$script: not found"
        exit 1
    fi
done

# Check 7: CUTLASS (optional)
log_step "Checking CUTLASS..."
if [ -d "/opt/cutlass" ]; then
    log_success "CUTLASS found at /opt/cutlass"
elif [ -d "$HOME/cutlass" ]; then
    log_success "CUTLASS found at $HOME/cutlass"
else
    log_info "CUTLASS not found (will be needed for kernel compilation)"
fi

# Summary
echo ""
echo "=========================================="
echo "  Environment Validation Complete"
echo "=========================================="
echo ""
log_success "Environment is ready for BlackwellSparseK development"
echo ""
log_info "Next steps:"
echo "  1. Implement CUDA kernels (src/blackwell_sparsek/kernels/)"
echo "  2. Build PyTorch extension (pip install -e .)"
echo "  3. Run tests (pytest tests/)"
echo "  4. Benchmark performance (python benchmarks/perf.py)"
echo ""


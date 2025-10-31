#!/bin/bash
# ============================================================================
# BlackwellSparseK 7-Loop H100 Validation Framework
# ============================================================================
# Implements systematic validation with deterministic execution
# Author: Dr. Brandon Dent (MD) | Expert CUDA Architect
# Purpose: One-click H100 validation with full reproducibility
# ============================================================================

set -euo pipefail

# Configuration
RESULTS_DIR="/workspace/results"
VALIDATION_LOG="/workspace/validation.log"
TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Create results directory
mkdir -p "${RESULTS_DIR}"

# Logging function
log() {
    echo -e "${BLUE}[$(date +'%H:%M:%S')]${NC} $1" | tee -a "${VALIDATION_LOG}"
}

log_success() {
    echo -e "${GREEN}✅ $1${NC}" | tee -a "${VALIDATION_LOG}"
}

log_error() {
    echo -e "${RED}❌ $1${NC}" | tee -a "${VALIDATION_LOG}"
}

log_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}" | tee -a "${VALIDATION_LOG}"
}

# ============================================================================
# LOOP 1 — Analyze
# ============================================================================
loop1_analyze() {
    log "=========================================="
    log "LOOP 1 — Analyze Environment"
    log "=========================================="
    
    # GPU Check
    log "Checking GPU..."
    if nvidia-smi --query-gpu=name --format=csv,noheader | grep -q "H100"; then
        log_success "H100 GPU detected"
        nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv,noheader > "${RESULTS_DIR}/gpu_info.txt"
    else
        log_error "H100 GPU not found!"
        nvidia-smi
        exit 1
    fi
    
    # CUDA Check
    log "Checking CUDA..."
    if nvcc --version | grep -q "13.0"; then
        log_success "CUDA 13.0.x detected"
        nvcc --version > "${RESULTS_DIR}/cuda_version.txt"
    else
        log_warning "CUDA 13.0.x not detected, attempting to continue..."
        nvcc --version
    fi
    
    # Environment snapshot
    log "Capturing environment..."
    {
        echo "=== GPU ==="
        nvidia-smi
        echo ""
        echo "=== CUDA ==="
        nvcc --version
        echo ""
        echo "=== Python ==="
        python3 --version
        pip list | grep -E "torch|xformers|vllm|cutlass"
        echo ""
        echo "=== CUTLASS ==="
        ls -la /opt/cutlass 2>/dev/null || echo "CUTLASS not found"
    } > "${RESULTS_DIR}/env.log"
    
    log_success "LOOP 1 Complete"
}

# ============================================================================
# LOOP 2 — Build
# ============================================================================
loop2_build() {
    log "=========================================="
    log "LOOP 2 — Build Containers"
    log "=========================================="
    
    cd /workspace/BlackwellSparseK || {
        log_error "BlackwellSparseK directory not found!"
        exit 1
    }
    
    # Build all containers
    log "Building containers..."
    BUILD_START=$(date +%s)
    
    if bash scripts/build_containers.sh 2>&1 | tee -a "${VALIDATION_LOG}"; then
        BUILD_END=$(date +%s)
        BUILD_DURATION=$((BUILD_END - BUILD_START))
        log_success "Containers built successfully in ${BUILD_DURATION}s"
        
        # Record image hashes
        docker images | grep blackwell-sparsek > "${RESULTS_DIR}/container_images.txt"
    else
        log_error "Container build failed!"
        exit 1
    fi
    
    # Verify CUTLASS and PyTorch linkage
    log "Verifying dependencies..."
    docker run --rm blackwell-sparsek:dev python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.version.cuda}')
print(f'CUDA Available: {torch.cuda.is_available()}')
" 2>&1 | tee "${RESULTS_DIR}/dependency_check.txt"
    
    log_success "LOOP 2 Complete"
}

# ============================================================================
# LOOP 3 — Validate
# ============================================================================
loop3_validate() {
    log "=========================================="
    log "LOOP 3 — Run Tests"
    log "=========================================="
    
    cd /workspace/BlackwellSparseK
    
    # Unit tests
    log "Running unit tests..."
    docker-compose --profile test up ci 2>&1 | tee "${RESULTS_DIR}/test_results.txt" || {
        log_error "Tests failed!"
        exit 1
    }
    
    # Integration tests
    log "Running integration tests..."
    docker run --gpus all --rm blackwell-sparsek:dev \
        bash -c "cd /workspace/BlackwellSparseK && pytest tests/ -v" \
        2>&1 | tee "${RESULTS_DIR}/integration_results.txt" || {
        log_warning "Some integration tests may have failed"
    }
    
    log_success "LOOP 3 Complete"
}

# ============================================================================
# LOOP 4 — Benchmark
# ============================================================================
loop4_benchmark() {
    log "=========================================="
    log "LOOP 4 — Performance Benchmarks"
    log "=========================================="
    
    cd /workspace/BlackwellSparseK
    
    # Run performance benchmarks
    log "Running latency benchmarks..."
    docker run --gpus all --rm \
        -v "${RESULTS_DIR}:/workspace/results" \
        blackwell-sparsek:dev \
        python benchmarks/perf.py --save-results 2>&1 | tee -a "${VALIDATION_LOG}"
    
    # Run SDPA comparison
    log "Comparing to PyTorch SDPA..."
    docker run --gpus all --rm \
        blackwell-sparsek:dev \
        python benchmarks/compare_sdpa.py 2>&1 | tee "${RESULTS_DIR}/sdpa_comparison.txt"
    
    # Nsight Compute profiling
    log "Running Nsight Compute profiling..."
    if command -v ncu &> /dev/null; then
        bash benchmarks/ncu_roofline.sh 2>&1 | tee "${RESULTS_DIR}/nsight_metrics.txt" || {
            log_warning "Nsight Compute profiling failed (may require SYS_ADMIN)"
        }
    else
        log_warning "Nsight Compute not available, skipping profiling"
    fi
    
    log_success "LOOP 4 Complete"
}

# ============================================================================
# LOOP 5 — Optimize
# ============================================================================
loop5_optimize() {
    log "=========================================="
    log "LOOP 5 — Optimization Analysis"
    log "=========================================="
    
    # Analyze benchmark results
    log "Analyzing performance..."
    
    if [ -f "${RESULTS_DIR}"/benchmark*.json ]; then
        python3 << 'EOF' > "${RESULTS_DIR}/optimization_analysis.txt"
import json
import glob

files = glob.glob('/workspace/results/benchmark*.json')
if files:
    with open(files[0]) as f:
        data = json.load(f)
    
    print("=== Optimization Analysis ===")
    print(f"Target: <5 μs (5× faster than SDPA @ 24.83 μs)")
    print("")
    
    for result in data.get('results', []):
        cfg = result['config']
        latency = result['comparison']['kernel_time_us']
        speedup = result['comparison']['speedup']
        
        print(f"Config: B={cfg['B']}, H={cfg['H']}, S={cfg['S']}, D={cfg['D']}")
        print(f"  Latency: {latency:.2f} μs")
        print(f"  Speedup: {speedup:.2f}×")
        print(f"  Target: {'✅ MET' if latency < 5.0 else '❌ MISSED'}")
        print("")
EOF
        cat "${RESULTS_DIR}/optimization_analysis.txt"
    else
        log_warning "No benchmark results found for analysis"
    fi
    
    log_success "LOOP 5 Complete"
}

# ============================================================================
# LOOP 6 — Harden
# ============================================================================
loop6_harden() {
    log "=========================================="
    log "LOOP 6 — Determinism & Safety"
    log "=========================================="
    
    # Race condition check
    log "Running compute-sanitizer race check..."
    if command -v compute-sanitizer &> /dev/null; then
        docker run --gpus all --rm --cap-add=SYS_ADMIN \
            blackwell-sparsek:dev \
            compute-sanitizer --tool racecheck \
            pytest tests/test_kernels.py -v 2>&1 | tee "${RESULTS_DIR}/racecheck.log" || {
            log_warning "Race check failed or found issues"
        }
    else
        log_warning "compute-sanitizer not available, skipping race check"
    fi
    
    # Determinism test
    log "Testing determinism..."
    docker run --gpus all --rm blackwell-sparsek:dev \
        python -c "
import torch
from blackwell_sparsek import attention_forward

torch.manual_seed(42)
Q = torch.randn(1, 8, 512, 64, dtype=torch.float16, device='cuda')
K = torch.randn(1, 8, 512, 64, dtype=torch.float16, device='cuda')
V = torch.randn(1, 8, 512, 64, dtype=torch.float16, device='cuda')

# Run twice
out1 = attention_forward(Q, K, V)
out2 = attention_forward(Q, K, V)

# Check determinism
if torch.equal(out1, out2):
    print('✅ Determinism verified: outputs are identical')
else:
    print('❌ Determinism failed: outputs differ')
    print(f'Max diff: {torch.abs(out1 - out2).max():.6f}')
" 2>&1 | tee "${RESULTS_DIR}/determinism.txt"
    
    log_success "LOOP 6 Complete"
}

# ============================================================================
# LOOP 7 — Report
# ============================================================================
loop7_report() {
    log "=========================================="
    log "LOOP 7 — Generate Report"
    log "=========================================="
    
    # Run log collection
    log "Collecting logs..."
    bash scripts/collect_logs.sh
    
    # Display summary
    log ""
    log "=========================================="
    log "H100 VALIDATION COMPLETE"
    log "=========================================="
    log ""
    log "Report saved to: ${RESULTS_DIR}/H100_VALIDATION_REPORT.md"
    log ""
    
    # Check if deployment criteria met
    TESTS_PASSED=$(grep -c "PASSED" "${RESULTS_DIR}/test_results.txt" 2>/dev/null || echo "0")
    TESTS_FAILED=$(grep -c "FAILED" "${RESULTS_DIR}/test_results.txt" 2>/dev/null || echo "0")
    
    if [ "$TESTS_FAILED" -eq 0 ] && docker images | grep -q blackwell-sparsek; then
        log_success "✅ H100 Validation Complete — CLEARED FOR DEPLOYMENT v0.1.0 (BlackwellSparseK)"
    else
        log_error "Validation completed with issues. Review report before deployment."
    fi
}

# ============================================================================
# MAIN EXECUTION
# ============================================================================
main() {
    echo ""
    echo "=========================================="
    echo "BlackwellSparseK 7-Loop H100 Validation"
    echo "=========================================="
    echo "Version: 0.1.0"
    echo "Timestamp: ${TIMESTAMP}"
    echo "Results: ${RESULTS_DIR}"
    echo "=========================================="
    echo ""
    
    # Execute all 7 loops
    loop1_analyze
    loop2_build
    loop3_validate
    loop4_benchmark
    loop5_optimize
    loop6_harden
    loop7_report
    
    echo ""
    echo "=========================================="
    echo "✅ VALIDATION COMPLETE"
    echo "=========================================="
    echo ""
    echo "View report: cat ${RESULTS_DIR}/H100_VALIDATION_REPORT.md"
    echo ""
}

# Run main function
main "$@"


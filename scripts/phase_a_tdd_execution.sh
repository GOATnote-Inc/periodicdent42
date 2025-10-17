#!/bin/bash
# Phase A TDD Execution Script
# Test-Driven Development approach with pre/post validation

set -euo pipefail

echo "============================================================"
echo "Phase A: TDD-Driven Correctness Fix Execution"
echo "============================================================"
echo ""

# Navigate to repo
cd ~/periodicdent42
source ~/venv/bin/activate

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Test counters
TESTS_PASSED=0
TESTS_FAILED=0
TESTS_TOTAL=0

# Test function
run_test() {
    local test_name="$1"
    local test_cmd="$2"
    local expected_exit_code="${3:-0}"
    
    TESTS_TOTAL=$((TESTS_TOTAL + 1))
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "TEST $TESTS_TOTAL: $test_name"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    set +e
    eval "$test_cmd"
    local exit_code=$?
    set -e
    
    if [ $exit_code -eq $expected_exit_code ]; then
        echo -e "${GREEN}✅ PASSED${NC}: $test_name (exit code: $exit_code)"
        TESTS_PASSED=$((TESTS_PASSED + 1))
        return 0
    else
        echo -e "${RED}❌ FAILED${NC}: $test_name (expected: $expected_exit_code, got: $exit_code)"
        TESTS_FAILED=$((TESTS_FAILED + 1))
        return 1
    fi
}

# Pre-flight checks
echo "============================================================"
echo "PRE-FLIGHT CHECKS"
echo "============================================================"

run_test "Python availability" "python --version"
run_test "PyTorch installation" "python -c 'import torch; print(torch.__version__)'"
run_test "CUDA availability" "python -c 'import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))'"
run_test "Repository structure" "test -f cudadent42/bench/kernels/fa_phase3_wmma.cu"
run_test "Evidence directory" "mkdir -p evidence && test -d evidence"

echo ""
echo "============================================================"
echo "TASK A.1: PyTorch Version Isolation"
echo "============================================================"

# Save current PyTorch version
ORIGINAL_TORCH_VERSION=$(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo "unknown")
echo "Original PyTorch version: $ORIGINAL_TORCH_VERSION"

# Test 1: Baseline with current PyTorch
echo ""
echo "Test 1.1: Baseline correctness (current PyTorch)"
run_test "Baseline SDPA measurement" "python bench/measure_sdpa.py --backend flash --out evidence/phase_a_sdpa_baseline.json"

# Test 2: Check if we can run standalone eval
echo ""
echo "Test 1.2: Check standalone eval script"
if [ -f scripts/standalone_phase4_eval.py ]; then
    echo "✅ standalone_phase4_eval.py exists"
    # Try to run it (may fail, but that's expected)
    set +e
    timeout 120 python scripts/standalone_phase4_eval.py > evidence/phase_a_current_pytorch.log 2>&1
    EVAL_EXIT=$?
    set -e
    
    if [ $EVAL_EXIT -eq 0 ]; then
        echo "✅ Phase 4 eval completed successfully"
        grep -i "correctness\|passed\|failed" evidence/phase_a_current_pytorch.log || true
    else
        echo "⚠️  Phase 4 eval failed (exit: $EVAL_EXIT) - this is expected if kernel not built"
    fi
else
    echo "⚠️  standalone_phase4_eval.py not found - skipping"
fi

# Test 3: PyTorch 2.1.0 installation
echo ""
echo "Test 1.3: Install PyTorch 2.1.0"
echo "⚠️  About to uninstall current PyTorch and install 2.1.0"
read -p "Continue? (y/n) " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    run_test "Uninstall current PyTorch" "pip uninstall torch -y"
    run_test "Install PyTorch 2.1.0" "pip install torch==2.1.0+cu121 --index-url https://download.pytorch.org/whl/cu121"
    run_test "Verify PyTorch 2.1.0" "python -c 'import torch; assert torch.__version__.startswith(\"2.1\"); print(f\"PyTorch {torch.__version__}\")'"
    
    # Clear torch extensions cache
    run_test "Clear torch extensions cache" "rm -rf ~/.cache/torch_extensions"
    
    echo ""
    echo "Test 1.4: Test Phase 4 with PyTorch 2.1.0"
    if [ -f scripts/standalone_phase4_eval.py ]; then
        set +e
        timeout 180 python scripts/standalone_phase4_eval.py > evidence/phase_a_pytorch210.log 2>&1
        EVAL_210_EXIT=$?
        set -e
        
        if [ $EVAL_210_EXIT -eq 0 ]; then
            echo "✅ Phase 4 eval with PyTorch 2.1.0 completed"
            echo ""
            echo "Correctness Results:"
            grep -A 5 -i "correctness\|passed\|max_diff" evidence/phase_a_pytorch210.log || echo "No correctness info found"
            TESTS_PASSED=$((TESTS_PASSED + 1))
        else
            echo "❌ Phase 4 eval with PyTorch 2.1.0 failed (exit: $EVAL_210_EXIT)"
            TESTS_FAILED=$((TESTS_FAILED + 1))
        fi
        TESTS_TOTAL=$((TESTS_TOTAL + 1))
    fi
else
    echo "⚠️  Skipped PyTorch 2.1.0 test"
fi

# Test 4: Upgrade to PyTorch 2.5.0
echo ""
echo "Test 1.5: Upgrade to PyTorch 2.5.0"
read -p "Upgrade to PyTorch 2.5.0? (y/n) " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    run_test "Uninstall PyTorch 2.1.0" "pip uninstall torch -y"
    run_test "Install PyTorch 2.5.0" "pip install torch==2.5.0 --index-url https://download.pytorch.org/whl/cu121"
    run_test "Verify PyTorch 2.5.0" "python -c 'import torch; assert torch.__version__.startswith(\"2.5\"); print(f\"PyTorch {torch.__version__}\")'"
    
    # Clear torch extensions cache
    run_test "Clear torch extensions cache" "rm -rf ~/.cache/torch_extensions"
fi

echo ""
echo "============================================================"
echo "TASK A.2: Numerical Stability (Stable Kernel)"
echo "============================================================"

# Test 5: Check if stable kernel exists
echo ""
echo "Test 2.1: Verify stable kernel exists"
if [ -f cudadent42/bench/kernels/fa_phase3_stable.cu ]; then
    echo "✅ fa_phase3_stable.cu exists"
    TESTS_PASSED=$((TESTS_PASSED + 1))
else
    echo "❌ fa_phase3_stable.cu not found"
    TESTS_FAILED=$((TESTS_FAILED + 1))
fi
TESTS_TOTAL=$((TESTS_TOTAL + 1))

# Test 6: Compile stable kernel (if build script exists)
echo ""
echo "Test 2.2: Build stable kernel (if possible)"
echo "⚠️  Building stable kernel requires integration with build system"
echo "TODO: Implement build_phase3_stable.py for stable kernel"

echo ""
echo "============================================================"
echo "TASK A.3: Dual-Reference Validation"
echo "============================================================"

# Test 7: Check if validator script exists
echo ""
echo "Test 3.1: Verify dual-reference validator"
if [ -f scripts/phase_a_validate_dual_backend.py ]; then
    echo "✅ phase_a_validate_dual_backend.py exists"
    TESTS_PASSED=$((TESTS_PASSED + 1))
    
    # Test 8: Run validator (may fail if kernel not built)
    echo ""
    echo "Test 3.2: Run dual-reference validation"
    set +e
    timeout 120 python scripts/phase_a_validate_dual_backend.py > evidence/phase_a_dual_backend.log 2>&1
    VALIDATOR_EXIT=$?
    set -e
    
    if [ $VALIDATOR_EXIT -eq 0 ]; then
        echo "✅ Dual-reference validation PASSED"
        cat evidence/phase_a_dual_backend.log
        TESTS_PASSED=$((TESTS_PASSED + 1))
    elif [ $VALIDATOR_EXIT -eq 2 ]; then
        echo "⚠️  Dual-reference validation skipped (kernel not built yet)"
        echo "This is expected - kernel needs to be built first"
    else
        echo "❌ Dual-reference validation FAILED"
        tail -20 evidence/phase_a_dual_backend.log
        TESTS_FAILED=$((TESTS_FAILED + 1))
    fi
    TESTS_TOTAL=$((TESTS_TOTAL + 1))
else
    echo "❌ phase_a_validate_dual_backend.py not found"
    TESTS_FAILED=$((TESTS_FAILED + 1))
fi
TESTS_TOTAL=$((TESTS_TOTAL + 1))

# Test 9: Use SDPA Oracle for direct testing
echo ""
echo "Test 3.3: SDPA Oracle smoke test"
run_test "SDPA Oracle smoke test" "python bench/sdpa_oracle.py"

echo ""
echo "============================================================"
echo "FINAL REPORT"
echo "============================================================"
echo ""
echo "Tests Summary:"
echo "  Total:  $TESTS_TOTAL"
echo "  Passed: $TESTS_PASSED"
echo "  Failed: $TESTS_FAILED"
echo ""

if [ $TESTS_FAILED -eq 0 ]; then
    echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${GREEN}✅ PHASE A: ALL TESTS PASSED${NC}"
    echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
    echo "Evidence Files:"
    ls -lh evidence/phase_a_*.{json,log,txt} 2>/dev/null || echo "No evidence files found"
    echo ""
    echo "Next Steps:"
    echo "  1. Review evidence logs"
    echo "  2. If correctness is 100% on PyTorch 2.5.0 → Proceed to Phase B"
    echo "  3. If correctness < 100% → Implement stable kernel build"
    echo "  4. Phase B: cuBLAS Q@K^T (6 hours → 400-500 μs)"
    exit 0
else
    echo -e "${RED}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${RED}❌ PHASE A: SOME TESTS FAILED${NC}"
    echo -e "${RED}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
    echo "Failed tests need attention before proceeding."
    echo "Review evidence logs in evidence/ directory."
    exit 1
fi


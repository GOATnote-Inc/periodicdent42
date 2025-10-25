#!/bin/bash
# Phase A.2 TDD Test Script
# Progressive testing: Build → Launch → Correctness → Performance

set -euo pipefail

cd ~/periodicdent42
source ~/venv/bin/activate
export PYTHONPATH=$(pwd):${PYTHONPATH:-}

echo "============================================================"
echo "Phase A.2: TDD Testing - Stable Kernel"
echo "============================================================"
echo ""

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
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "TEST $TESTS_TOTAL: $test_name"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    set +e
    eval "$test_cmd"
    local exit_code=$?
    set -e
    
    if [ $exit_code -eq $expected_exit_code ]; then
        echo "✅ PASSED: $test_name"
        TESTS_PASSED=$((TESTS_PASSED + 1))
        return 0
    else
        echo "❌ FAILED: $test_name (expected: $expected_exit_code, got: $exit_code)"
        TESTS_FAILED=$((TESTS_FAILED + 1))
        return 1
    fi
}

# Test 1: Build (should pass - already worked)
echo "Test 1: Build Stable Kernel"
run_test "Build stable kernel" "python -c 'from bench.build_phase3_stable import build_phase3_stable; build_phase3_stable()' > /tmp/build_test.log 2>&1"

# Test 2: Simple forward pass (launch test)
echo ""
echo "Test 2: Kernel Launch (smoke test)"
run_test "Kernel launch smoke test" "python -c \"
import torch, sys
sys.path.insert(0, '.')
from bench.build_phase3_stable import build_phase3_stable

# Build
module = build_phase3_stable()

# Simple forward pass
q = torch.randn(1, 8, 512, 64, device='cuda', dtype=torch.float16)
k, v = q.clone(), q.clone()
scale = 1.0 / 64**0.5

try:
    o = module.forward(q, k, v, scale)
    print(f'✅ Kernel launched successfully, output shape: {o.shape}')
except Exception as e:
    print(f'❌ Kernel launch failed: {e}')
    sys.exit(1)
\" > /tmp/launch_test.log 2>&1"

# Test 3: Correctness (basic check)
echo ""
echo "Test 3: Basic Correctness Check"
run_test "Basic correctness" "python -c \"
import torch, sys
sys.path.insert(0, '.')
from bench.build_phase3_stable import build_phase3_stable

# Build
module = build_phase3_stable()

# Generate inputs
torch.manual_seed(42)
q = torch.randn(1, 8, 512, 64, device='cuda', dtype=torch.float16)
k = torch.randn(1, 8, 512, 64, device='cuda', dtype=torch.float16)
v = torch.randn(1, 8, 512, 64, device='cuda', dtype=torch.float16)
scale = 1.0 / 64**0.5

# Forward pass
o = module.forward(q, k, v, scale)

# Basic sanity checks
assert o.shape == q.shape, f'Shape mismatch: {o.shape} vs {q.shape}'
assert not torch.isnan(o).any(), 'Output contains NaN'
assert not torch.isinf(o).any(), 'Output contains Inf'
assert o.abs().max() < 100.0, f'Output values too large: {o.abs().max()}'

print(f'✅ Basic correctness: shape={o.shape}, max_abs={o.abs().max():.3f}, no NaN/Inf')
\" > /tmp/correctness_test.log 2>&1"

# Test 4: SDPA Oracle (full validation)
echo ""
echo "Test 4: SDPA Oracle Validation (full correctness)"
echo "⏱️  This may take 30-60 seconds..."
set +e
timeout 120 python scripts/test_phase3_stable.py > /tmp/oracle_test.log 2>&1
ORACLE_EXIT=$?
set -e

if [ $ORACLE_EXIT -eq 0 ]; then
    echo "✅ SDPA Oracle validation PASSED"
    TESTS_PASSED=$((TESTS_PASSED + 1))
    
    # Extract results
    echo ""
    echo "Results:"
    grep -A 3 "Correctness:\|Performance:\|Speedup:" /tmp/oracle_test.log | head -10 || true
else
    echo "❌ SDPA Oracle validation FAILED (exit: $ORACLE_EXIT)"
    TESTS_FAILED=$((TESTS_FAILED + 1))
    
    # Show errors
    echo ""
    echo "Last 20 lines of output:"
    tail -20 /tmp/oracle_test.log
fi
TESTS_TOTAL=$((TESTS_TOTAL + 1))

# Final report
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
    echo "✅ PHASE A.2: ALL TESTS PASSED"
    echo ""
    echo "Stable Kernel Status:"
    echo "  ✅ Build successful"
    echo "  ✅ Kernel launches"
    echo "  ✅ Basic correctness (no NaN/Inf)"
    echo "  ✅ SDPA Oracle validation"
    echo ""
    echo "Evidence Files:"
    ls -lh evidence/phase3_stable_*.json 2>/dev/null || echo "  (see evidence/ directory)"
    echo ""
    echo "Next: Phase B - cuBLAS Q@K^T (6 hours → 400-500 μs)"
    exit 0
else
    echo "❌ PHASE A.2: SOME TESTS FAILED"
    echo ""
    echo "Failed tests: $TESTS_FAILED"
    echo "Review logs in /tmp/*_test.log"
    exit 1
fi


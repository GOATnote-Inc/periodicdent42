#!/bin/bash
# TDD Test Runner for DHP I4
set -e

echo "════════════════════════════════════════════════════════════════════════════════"
echo "  DHP I4 TEST SUITE (TDD Methodology)"
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""

# Check if on H100
if ! nvidia-smi | grep -q "H100"; then
    echo "⚠️  WARNING: Not running on H100"
    echo "   Tests may not reflect target performance"
    echo ""
fi

TESTS_PASSED=0
TESTS_FAILED=0
TESTS_SKIPPED=0

# Function to run test
run_test() {
    local test_name=$1
    local test_cmd=$2
    
    echo "──────────────────────────────────────────────────────────────────────────────"
    echo "  $test_name"
    echo "──────────────────────────────────────────────────────────────────────────────"
    echo ""
    
    if eval $test_cmd; then
        TESTS_PASSED=$((TESTS_PASSED + 1))
        echo ""
    else
        exit_code=$?
        if [ $exit_code -eq 77 ]; then
            TESTS_SKIPPED=$((TESTS_SKIPPED + 1))
        else
            TESTS_FAILED=$((TESTS_FAILED + 1))
        fi
        echo ""
    fi
}

# Compile kernel first
echo "──────────────────────────────────────────────────────────────────────────────"
echo "  COMPILATION"
echo "──────────────────────────────────────────────────────────────────────────────"
echo ""

if python setup.py build_ext --inplace 2>&1 | tee build.log; then
    echo "✅ Compilation successful"
    echo ""
    
    # Check register usage
    if grep -q "registers" build.log; then
        echo "Register usage:"
        grep "registers" build.log | head -5
        echo ""
    fi
else
    echo "❌ Compilation failed"
    exit 1
fi

# Run tests
run_test "Test 1: Correctness" "python3 tests/test_i4_correctness.py"
run_test "Test 2: Security" "python3 tests/test_i4_security.py"
run_test "Test 3: Performance" "python3 tests/test_i4_performance.py"

# Summary
echo "════════════════════════════════════════════════════════════════════════════════"
echo "  TEST SUMMARY"
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""
echo "Passed:  $TESTS_PASSED"
echo "Failed:  $TESTS_FAILED"
echo "Skipped: $TESTS_SKIPPED"
echo ""

if [ $TESTS_FAILED -gt 0 ]; then
    echo "❌ TESTS FAILED"
    exit 1
else
    echo "✅ ALL TESTS PASSED"
    exit 0
fi


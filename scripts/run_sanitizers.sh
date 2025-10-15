#!/bin/bash
# Phase 9: CUDA Sanitizer Suite
# Runs comprehensive memory safety and race condition checks
#
# Usage:
#   ./scripts/run_sanitizers.sh                    # Run all sanitizers
#   ./scripts/run_sanitizers.sh --tool memcheck    # Run specific tool
#   ./scripts/run_sanitizers.sh --shape canonical  # Test specific shape

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# ============================================================================
# Configuration
# ============================================================================

TOOL="${TOOL:-all}"
SHAPE="${SHAPE:-small}"
OUTPUT_DIR="${OUTPUT_DIR:-artifacts/sanitizers}"
KERNEL="${KERNEL:-v3}"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --tool)
            TOOL="$2"
            shift 2
            ;;
        --shape)
            SHAPE="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --kernel)
            KERNEL="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Create output directory
mkdir -p "$OUTPUT_DIR"

# ============================================================================
# Helper Functions
# ============================================================================

print_header() {
    echo ""
    echo "════════════════════════════════════════════════════════════════════════════════"
    echo "  $1"
    echo "════════════════════════════════════════════════════════════════════════════════"
    echo ""
}

print_section() {
    echo ""
    echo "────────────────────────────────────────────────────────────────────────────────"
    echo "  $1"
    echo "────────────────────────────────────────────────────────────────────────────────"
}

print_pass() {
    echo "✅ PASS: $1"
}

print_fail() {
    echo "❌ FAIL: $1"
}

run_sanitizer_tool() {
    local tool_name=$1
    local test_script=$2
    local output_file="$OUTPUT_DIR/${KERNEL}_${tool_name}_${SHAPE}.log"
    
    print_section "Running: $tool_name"
    
    echo "Command: compute-sanitizer --tool $tool_name $test_script"
    echo "Output:  $output_file"
    echo ""
    
    if compute-sanitizer \
        --tool "$tool_name" \
        --log-file "$output_file" \
        --print-limit 100 \
        python3 "$test_script" --shape "$SHAPE" --kernel "$KERNEL" 2>&1 | tee "${output_file}.stdout"; then
        
        # Check for errors in output
        if grep -q "ERROR SUMMARY: 0 errors" "$output_file" 2>/dev/null; then
            print_pass "$tool_name - No errors detected"
            return 0
        elif grep -q "No kernels were profiled" "$output_file" 2>/dev/null; then
            print_fail "$tool_name - No kernels profiled (test may have failed)"
            return 1
        else
            # Count errors
            error_count=$(grep "ERROR SUMMARY:" "$output_file" | head -1 | awk '{print $3}' || echo "unknown")
            print_fail "$tool_name - $error_count errors detected (see $output_file)"
            return 1
        fi
    else
        print_fail "$tool_name - Command failed (exit code: $?)"
        return 1
    fi
}

# ============================================================================
# Main Sanitizer Suite
# ============================================================================

print_header "CUDA Sanitizer Suite - Phase 9"

# Verify compute-sanitizer is available
if ! command -v compute-sanitizer &> /dev/null; then
    print_fail "compute-sanitizer not found (install CUDA toolkit)"
    exit 1
fi

print_pass "compute-sanitizer available: $(compute-sanitizer --version | head -1)"

# Create test script
TEST_SCRIPT="$OUTPUT_DIR/sanitizer_test.py"
cat > "$TEST_SCRIPT" << 'PYEOF'
#!/usr/bin/env python3
"""
Sanitizer test harness - runs single kernel call for sanitizer validation
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "cudadent42"))

import argparse
import torch

# Import kernels
try:
    from bench.fa_s512_v3 import flash_attention_s512_v3_forward as fa_v3
except ImportError:
    fa_v3 = None

try:
    from bench.fa_inverted_prod import flash_attention_inverted_forward as fa_prod
except ImportError:
    fa_prod = None

def get_shape_config(shape_name: str):
    """Get shape configuration"""
    shapes = {
        "small": (1, 4, 512, 64),
        "medium": (4, 8, 1024, 64),
        "canonical": (4, 16, 2048, 128),
    }
    return shapes.get(shape_name, shapes["small"])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shape", default="small", choices=["small", "medium", "canonical"])
    parser.add_argument("--kernel", default="v3", choices=["v3", "prod"])
    parser.add_argument("--causal", action="store_true", default=True)
    args = parser.parse_args()
    
    # Get shape
    B, H, S, D = get_shape_config(args.shape)
    
    # Select kernel
    if args.kernel == "v3":
        if fa_v3 is None:
            print("ERROR: V3 kernel not available")
            return 1
        kernel_fn = lambda Q, K, V: fa_v3(Q, K, V, is_causal=args.causal, config_id=1)
    else:
        if fa_prod is None:
            print("ERROR: Prod kernel not available")
            return 1
        kernel_fn = lambda Q, K, V: fa_prod(Q, K, V, is_causal=args.causal)
    
    # Generate inputs
    torch.manual_seed(42)
    Q = torch.randn(B, H, S, D, device='cuda', dtype=torch.float16)
    K = torch.randn(B, H, S, D, device='cuda', dtype=torch.float16)
    V = torch.randn(B, H, S, D, device='cuda', dtype=torch.float16)
    
    print(f"Running {args.kernel} kernel on shape B={B}, H={H}, S={S}, D={D}, causal={args.causal}")
    
    # Run kernel (single call for sanitizer)
    try:
        output = kernel_fn(Q, K, V)
        torch.cuda.synchronize()
        
        # Verify output
        assert output.shape == (B, H, S, D), f"Wrong shape: {output.shape}"
        assert torch.isfinite(output).all(), "Output contains NaN/Inf"
        
        print(f"✅ Kernel executed successfully")
        print(f"   Output range: [{output.min().item():.4f}, {output.max().item():.4f}]")
        return 0
        
    except Exception as e:
        print(f"❌ Kernel failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
PYEOF

chmod +x "$TEST_SCRIPT"

# ============================================================================
# Run Sanitizer Tools
# ============================================================================

TOOLS_TO_RUN=()

if [ "$TOOL" = "all" ]; then
    TOOLS_TO_RUN=(memcheck racecheck initcheck synccheck)
else
    TOOLS_TO_RUN=("$TOOL")
fi

echo "Tools to run: ${TOOLS_TO_RUN[*]}"
echo "Shape: $SHAPE"
echo "Kernel: $KERNEL"
echo ""

PASSED=0
FAILED=0

for tool in "${TOOLS_TO_RUN[@]}"; do
    if run_sanitizer_tool "$tool" "$TEST_SCRIPT"; then
        ((PASSED++))
    else
        ((FAILED++))
    fi
done

# ============================================================================
# Summary Report
# ============================================================================

print_header "Sanitizer Suite Summary"

echo "Results:"
echo "  ✅ Passed: $PASSED"
echo "  ❌ Failed: $FAILED"
echo ""
echo "Logs: $OUTPUT_DIR/"
echo ""

if [ $FAILED -eq 0 ]; then
    print_pass "All sanitizer checks passed!"
    
    # Generate summary report
    cat > "$OUTPUT_DIR/sanitizer_summary.md" << EOF
# Sanitizer Suite Summary

**Date**: $(date '+%Y-%m-%d %H:%M:%S')  
**Kernel**: $KERNEL  
**Shape**: $SHAPE

## Results

| Tool | Status | Errors |
|------|--------|--------|
EOF
    
    for tool in "${TOOLS_TO_RUN[@]}"; do
        log_file="$OUTPUT_DIR/${KERNEL}_${tool}_${SHAPE}.log"
        if [ -f "$log_file" ]; then
            errors=$(grep "ERROR SUMMARY:" "$log_file" | head -1 | awk '{print $3}' || echo "?")
            status=$([ "$errors" = "0" ] && echo "✅ PASS" || echo "❌ FAIL")
            echo "| $tool | $status | $errors |" >> "$OUTPUT_DIR/sanitizer_summary.md"
        fi
    done
    
    echo "" >> "$OUTPUT_DIR/sanitizer_summary.md"
    echo "## Conclusion" >> "$OUTPUT_DIR/sanitizer_summary.md"
    echo "" >> "$OUTPUT_DIR/sanitizer_summary.md"
    echo "✅ **ALL CHECKS PASSED** - Kernel is memory-safe" >> "$OUTPUT_DIR/sanitizer_summary.md"
    
    print_pass "Summary report: $OUTPUT_DIR/sanitizer_summary.md"
    exit 0
else
    print_fail "Sanitizer checks failed!"
    echo ""
    echo "Review logs in: $OUTPUT_DIR/"
    exit 1
fi


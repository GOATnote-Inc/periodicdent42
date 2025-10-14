#!/bin/bash
# Run compute-sanitizer on V3 debug build
# Usage: ./run_sanitizer.sh [memcheck|racecheck|initcheck|all]

set -e

TOOL=${1:-memcheck}
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
ARTIFACTS_DIR="${SCRIPT_DIR}/../../artifacts/sanitizers"

mkdir -p "$ARTIFACTS_DIR"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🔍 Compute Sanitizer: $TOOL"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

run_tool() {
    local tool=$1
    local log_file="${ARTIFACTS_DIR}/${tool}_$(date +%Y%m%d_%H%M%S).log"
    
    echo ""
    echo "Running $tool..."
    echo "Log: $log_file"
    echo ""
    
    case $tool in
        memcheck)
            compute-sanitizer \
                --tool memcheck \
                --leak-check full \
                --report-api-errors yes \
                python3 "${SCRIPT_DIR}/test_v3_smoke_debug.py" \
                2>&1 | tee "$log_file"
            ;;
        racecheck)
            compute-sanitizer \
                --tool racecheck \
                python3 "${SCRIPT_DIR}/test_v3_smoke_debug.py" \
                2>&1 | tee "$log_file"
            ;;
        initcheck)
            compute-sanitizer \
                --tool initcheck \
                python3 "${SCRIPT_DIR}/test_v3_smoke_debug.py" \
                2>&1 | tee "$log_file"
            ;;
        *)
            echo "Unknown tool: $tool"
            exit 1
            ;;
    esac
    
    # Check result
    if grep -q "ERROR SUMMARY: 0 errors" "$log_file"; then
        echo "✅ $tool: CLEAN"
        return 0
    else
        echo "❌ $tool: ERRORS DETECTED"
        echo ""
        echo "First 50 lines of errors:"
        grep -A 50 "ERROR SUMMARY" "$log_file" || true
        return 1
    fi
}

if [ "$TOOL" = "all" ]; then
    echo "Running all sanitizer tools..."
    
    SUCCESS=0
    run_tool memcheck || SUCCESS=1
    run_tool racecheck || SUCCESS=1
    run_tool initcheck || SUCCESS=1
    
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    if [ $SUCCESS -eq 0 ]; then
        echo "✅ ALL SANITIZERS CLEAN"
    else
        echo "❌ SOME SANITIZERS FAILED"
    fi
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    exit $SUCCESS
else
    run_tool "$TOOL"
fi


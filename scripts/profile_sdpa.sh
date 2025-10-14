#!/usr/bin/env bash
#
# Nsight Compute Profile Wrapper for PyTorch SDPA
#
# Profiles PyTorch SDPA with comprehensive metrics and saves to artifacts/ncu/
#
# Usage:
#   S=512 B=32 H=8 D=64 bash scripts/profile_sdpa.sh
#   bash scripts/profile_sdpa.sh  # Uses defaults
#
# Output:
#   artifacts/ncu/sdpa_s512_b32_h8_d64.ncu-rep (binary report)
#   artifacts/ncu/sdpa_s512_b32_h8_d64.txt (text summary)
#
# Author: GOATnote Autonomous Research Lab Initiative
# Date: 2025-10-14

set -euo pipefail

# Configuration (defaults)
S=${S:-512}
B=${B:-32}
H=${H:-8}
D=${D:-64}

# Paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PROFILE_SCRIPT="$REPO_ROOT/cudadent42/bench/profile_sdpa_once.py"
OUTPUT_DIR="$REPO_ROOT/artifacts/ncu"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Output files
OUTPUT_BASE="sdpa_s${S}_b${B}_h${H}_d${D}"
OUTPUT_REP="$OUTPUT_DIR/${OUTPUT_BASE}.ncu-rep"
OUTPUT_TXT="$OUTPUT_DIR/${OUTPUT_BASE}.txt"

# Check if ncu is available
if ! command -v ncu &> /dev/null; then
    echo "❌ Error: Nsight Compute (ncu) not found in PATH"
    echo "   Install CUDA toolkit or add ncu to PATH"
    exit 1
fi

# Check if profile script exists
if [ ! -f "$PROFILE_SCRIPT" ]; then
    echo "❌ Error: Profile script not found: $PROFILE_SCRIPT"
    exit 1
fi

echo "=============================================================="
echo "Nsight Compute Profiling: PyTorch SDPA"
echo "=============================================================="
echo ""
echo "Configuration:"
echo "  Batch size (B):     $B"
echo "  Attention heads (H): $H"
echo "  Sequence length (S): $S"
echo "  Head dimension (D):  $D"
echo ""
echo "Output:"
echo "  Binary report: $OUTPUT_REP"
echo "  Text summary:  $OUTPUT_TXT"
echo ""
echo "Running Nsight Compute..."
echo ""

# Run Nsight Compute with full metric set
ncu \
    --set full \
    --target-processes all \
    --kernel-name-base mangled \
    --launch-skip-before-match 0 \
    --launch-count 1 \
    -o "$OUTPUT_REP" \
    python3 "$PROFILE_SCRIPT" --b "$B" --h "$H" --s "$S" --d "$D"

echo ""
echo "✅ Profile complete: $OUTPUT_REP"
echo ""

# Generate text summary
echo "Generating text summary..."
ncu --import "$OUTPUT_REP" --page raw --csv > "$OUTPUT_TXT" 2>&1 || true

if [ -f "$OUTPUT_TXT" ]; then
    echo "✅ Text summary: $OUTPUT_TXT"
    echo ""
    
    # Print key metrics
    echo "Key Metrics:"
    echo "============"
    grep -E "(Duration|Throughput|sm__throughput|dram__throughput|l2__hit|smsp__pipe)" "$OUTPUT_TXT" | head -20 || true
else
    echo "⚠️  Failed to generate text summary"
fi

echo ""
echo "=============================================================="
echo "Next Steps:"
echo "=============================================================="
echo ""
echo "1. Open in Nsight Compute UI:"
echo "   ncu-ui $OUTPUT_REP"
echo ""
echo "2. View text summary:"
echo "   cat $OUTPUT_TXT"
echo ""
echo "3. Compare to baseline:"
echo "   python3 bench/ci_compare.py \\"
echo "     --baseline .ci/baseline_s${S}.json \\"
echo "     --candidate artifacts/summary.json"
echo ""


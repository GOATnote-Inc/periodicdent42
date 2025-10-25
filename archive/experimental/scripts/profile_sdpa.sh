#!/usr/bin/env bash
set -euo pipefail

#
# Nsight Compute Profile Wrapper for PyTorch SDPA
#
# Usage:
#   S=512 B=32 H=8 D=64 bash scripts/profile_sdpa.sh
#
# Environment Variables:
#   S - Sequence length (default: 512)
#   B - Batch size (default: 32)
#   H - Number of heads (default: 8)
#   D - Head dimension (default: 64)
#
# Output:
#   - .ncu-rep file (Nsight Compute binary)
#   - summary.csv (Human-readable metrics)
#
# Author: GOATnote Autonomous Research Lab Initiative
# Date: 2025-10-14
#

# Configuration (override via environment)
S=${S:-512}
B=${B:-32}
H=${H:-8}
D=${D:-64}

# Output directory
OUTPUT_DIR="bench/artifacts/ncu"
mkdir -p "$OUTPUT_DIR"

# Output base name
OUTPUT_BASE="sdpa_s${S}_b${B}_h${H}_d${D}"
OUTPUT_REP="${OUTPUT_DIR}/${OUTPUT_BASE}.ncu-rep"
OUTPUT_CSV="${OUTPUT_DIR}/${OUTPUT_BASE}_summary.csv"

# Find ncu binary
NCU_BIN="ncu"
if ! command -v ncu &> /dev/null; then
    # Try common locations
    if [ -f "/opt/nvidia/nsight-compute/2024.1.1/ncu" ]; then
        NCU_BIN="/opt/nvidia/nsight-compute/2024.1.1/ncu"
    elif [ -f "/usr/local/cuda/nsight-compute/ncu" ]; then
        NCU_BIN="/usr/local/cuda/nsight-compute/ncu"
    else
        echo "❌ Error: ncu (Nsight Compute) not found"
        echo "   Install from: https://developer.nvidia.com/nsight-compute"
        echo "   Or on GPU: sudo apt-get install nsight-compute-2024.1.1"
        exit 1
    fi
fi

echo "🔬 Profiling PyTorch SDPA with Nsight Compute"
echo "=============================================="
echo "Configuration:"
echo "  Batch (B):    $B"
echo "  Heads (H):    $H"
echo "  Sequence (S): $S"
echo "  Dimension (D): $D"
echo ""
echo "Output:"
echo "  Report: $OUTPUT_REP"
echo "  CSV:    $OUTPUT_CSV"
echo ""
echo "⏱️  This will take 2-3 minutes (38 passes per kernel)..."
echo ""

# Run Nsight Compute profiling
"$NCU_BIN" \
    --set full \
    --target-processes all \
    --force-overwrite \
    -o "$OUTPUT_REP" \
    python3 bench/profile_sdpa_once.py --b "$B" --h "$H" --s "$S" --d "$D"

# Check if profile was created
if [ ! -f "${OUTPUT_REP}.ncu-rep" ] && [ ! -f "$OUTPUT_REP" ]; then
    echo ""
    echo "❌ Error: Profile not created"
    exit 1
fi

# Determine actual profile path (ncu adds .ncu-rep extension)
if [ -f "${OUTPUT_REP}.ncu-rep" ]; then
    ACTUAL_REP="${OUTPUT_REP}.ncu-rep"
else
    ACTUAL_REP="$OUTPUT_REP"
fi

echo ""
echo "✅ Profile captured!"
echo ""

# Generate CSV summary
echo "📊 Generating CSV summary..."
"$NCU_BIN" \
    --import "$ACTUAL_REP" \
    --page summary \
    --csv \
    > "$OUTPUT_CSV" 2>/dev/null || true

if [ -f "$OUTPUT_CSV" ]; then
    echo "✅ CSV summary generated: $OUTPUT_CSV"
    echo ""
    echo "Preview (first 20 lines):"
    head -20 "$OUTPUT_CSV" || cat "$OUTPUT_CSV"
else
    echo "⚠️  CSV summary generation failed (non-critical)"
fi

echo ""
echo "=============================================="
echo "✅ Profiling complete!"
echo ""
echo "View in GUI:"
echo "  ncu-ui $ACTUAL_REP"
echo ""
echo "Extract specific metrics:"
echo "  ncu --import $ACTUAL_REP --page raw --csv"
echo ""

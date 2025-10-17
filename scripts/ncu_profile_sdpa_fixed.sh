#!/bin/bash
#
# NCU Profile: SDPA Kernel Only (Fixed)
#
# Fix: Pre-generate tensors, profile ONLY SDPA call
#

set -e

echo "════════════════════════════════════════════════════════════════════════════════"
echo "NCU PROFILING: SDPA Kernel Only (FIXED)"
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""

cd ~/periodicdent42
source ~/venv/bin/activate
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

# Use full path to NCU
NCU_BIN="/usr/local/cuda/bin/ncu"
if [ ! -f "$NCU_BIN" ]; then
    echo "❌ NCU not found at $NCU_BIN"
    exit 1
fi

echo "Step 1: Verify NCU..."
$NCU_BIN --version
echo ""

# Step 2: Full profile with kernel name filtering
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "STEP 2: Profile SDPA Kernels (Full Analysis)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "This will take 5-10 minutes..."
echo ""
echo "Filtering for attention-related kernels:"
echo "  - fmha (Fused Multi-Head Attention)"
echo "  - attention"
echo "  - gemm (matrix multiply)"
echo "  - softmax"
echo ""

$NCU_BIN \
  --set full \
  --target-processes all \
  --kernel-name regex:"(fmha|attention|gemm|softmax)" \
  --launch-skip 5 \
  --launch-count 5 \
  --force-overwrite \
  -o evidence/ncu_sdpa_only \
  python scripts/ncu_profile_sdpa_only.py 2>&1 | tee evidence/ncu_sdpa_only.log

echo ""

# Check if report was generated
if [ -f evidence/ncu_sdpa_only.ncu-rep ]; then
    echo "✅ NCU report generated: evidence/ncu_sdpa_only.ncu-rep"
    echo ""
    
    # Quick summary
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "QUICK SUMMARY"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    
    # Extract kernel names
    echo "Kernels profiled:"
    $NCU_BIN --import evidence/ncu_sdpa_only.ncu-rep --page raw --csv 2>/dev/null | \
      grep "Kernel Name" | head -10 || echo "  (see report for details)"
    
    echo ""
else
    echo "⚠️  NCU report not generated"
    echo "   Check evidence/ncu_sdpa_only.log for errors"
fi

echo ""
echo "════════════════════════════════════════════════════════════════════════════════"
echo "NCU PROFILING COMPLETE"
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""
echo "Evidence:"
echo "  - evidence/ncu_sdpa_only.ncu-rep (full profile, open in NCU UI)"
echo "  - evidence/ncu_sdpa_only.log (text log)"
echo ""
echo "Next: Analyze bottlenecks with:"
echo "  $NCU_BIN --import evidence/ncu_sdpa_only.ncu-rep --page details"
echo ""


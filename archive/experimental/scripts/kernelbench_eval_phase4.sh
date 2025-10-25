#!/bin/bash
# KernelBench evaluation of Phase 4 kernel on L4 GPU

set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
KB_ROOT="$REPO_ROOT/ext/KernelBench"

echo "🔬 KernelBench Evaluation: Phase 4 vs PyTorch SDPA"
echo "=" 
echo "Problem: FlashAttention (B=1, H=8, S=512, D=64)"
echo "Hardware: NVIDIA L4 (sm_89)"
echo "Baseline: PyTorch SDPA (47 μs measured)"
echo ""

cd "$KB_ROOT"

# Ensure dependencies
if ! python -c "import torch" 2>/dev/null; then
    echo "❌ PyTorch not found. Install: pip install torch==2.5.0"
    exit 1
fi

# Run evaluation
echo "📊 Running evaluation (100 correctness tests, 100 timing trials)..."
python scripts/run_and_check.py \
    --problem "KernelBench/level3/X_FlashAttentionL4_periodicdent42.py" \
    --solution "solutions/phase4_periodicdent42.py" \
    --n_correctness 100 \
    --n_trial 100 \
    --device cuda \
    2>&1 | tee "$REPO_ROOT/evidence/kernelbench_phase4.log"

echo ""
echo "✅ Evaluation complete"
echo "📁 Results saved to: evidence/kernelbench_phase4.log"
echo ""
echo "Expected metrics:"
echo "  fast_0 (correctness): 100%"
echo "  fast_1 (faster than PyTorch): 0% (839 vs 47 μs)"
echo "  Speedup: ~0.056× (17.8× slower)"


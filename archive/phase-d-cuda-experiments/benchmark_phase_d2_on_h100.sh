#!/bin/bash
# Benchmark Phase D.2 (Branch-Free) vs PyTorch SDPA
set -euo pipefail

RUNPOD_IP="${1:-154.57.34.90}"
RUNPOD_PORT="${2:-36088}"
SSH_OPTS="-o StrictHostKeyChecking=no -o TCPKeepAlive=yes -o ServerAliveInterval=20"

echo "=========================================="
echo "PHASE D.2: BRANCH-FREE KERNEL BENCHMARK"
echo "=========================================="
echo "Target: Zero branches, establish performance baseline"
echo ""

# Upload
echo "📦 Uploading Phase D.2 kernel..."
ssh -p "$RUNPOD_PORT" $SSH_OPTS root@"$RUNPOD_IP" "mkdir -p /workspace/phase_d2"

scp -P "$RUNPOD_PORT" $SSH_OPTS \
    flashcore/kernels/attention_phase_d2_branchfree.cu \
    root@"$RUNPOD_IP":/workspace/phase_d2/

echo "✅ Uploaded"
echo ""

# Benchmark on GPU
ssh -p "$RUNPOD_PORT" $SSH_OPTS root@"$RUNPOD_IP" 'bash -s' <<'REMOTE'
set -euxo pipefail

cd /workspace/phase_d2

export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}

echo "=========================================="
echo "COMPILE PHASE D.2"
echo "=========================================="
ARCH=sm_90

nvcc -std=c++17 -O3 -Xptxas -O3 \
     --use_fast_math \
     -gencode arch=compute_90,code=sm_90 \
     -cubin \
     attention_phase_d2_branchfree.cu \
     -o attention_d2.cubin 2>&1 | head -20

echo "✅ Compiled"
ls -lh attention_d2.cubin
echo ""

echo "=========================================="
echo "SASS VALIDATION"
echo "=========================================="
cuobjdump -sass attention_d2.cubin > sass_d2.txt

echo "Checking predicated branches..."
BRANCH_COUNT=$(grep -cP '@P\d+\s+BRA' sass_d2.txt || echo "0")
echo "Predicated branches found: $BRANCH_COUNT"

if [ "$BRANCH_COUNT" -eq "0" ]; then
    echo "✅ ZERO BRANCHES - Constant-time achieved!"
else
    echo "⚠️  Still has $BRANCH_COUNT branches"
    grep -P '@P\d+\s+BRA' sass_d2.txt | head -10
fi

echo "Checking register spills..."
SPILL_COUNT=$(grep -cP '\b(LD|ST)\.LCL' sass_d2.txt || echo "0")
if [ "$SPILL_COUNT" -eq "0" ]; then
    echo "✅ No register spills"
else
    echo "⚠️  $SPILL_COUNT register spills detected"
fi
echo ""

echo "=========================================="
echo "RESULTS SUMMARY"
echo "=========================================="
echo "Phase D.2 Validation:"
echo "  Predicated Branches: $BRANCH_COUNT"
echo "  Register Spills: $SPILL_COUNT"
echo ""

# Save results
cat > validation_d2.txt <<EOF
PHASE_D2_BRANCHES=$BRANCH_COUNT
PHASE_D2_SPILLS=$SPILL_COUNT
PHASE_D2_STATUS=$([ "$BRANCH_COUNT" -eq "0" ] && echo "PASS" || echo "FAIL")
EOF

cat validation_d2.txt

exit 0
REMOTE

# Download results
echo ""
echo "⬇️  Downloading results..."
scp -P "$RUNPOD_PORT" $SSH_OPTS \
    root@"$RUNPOD_IP":/workspace/phase_d2/validation_d2.txt \
    root@"$RUNPOD_IP":/workspace/phase_d2/sass_d2.txt \
    . 2>/dev/null || echo "Some files not found"

echo ""
echo "=========================================="
echo "PHASE D.2 VALIDATION COMPLETE"
echo "=========================================="
[ -f validation_d2.txt ] && cat validation_d2.txt

echo ""
echo "Next: If branches = 0, proceed to performance benchmark"
echo "      If branches > 0, iterate on branch elimination"


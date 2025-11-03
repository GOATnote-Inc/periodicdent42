#!/bin/bash
# Deploy to H100 and run TDD workflow

# Config
BREV_INSTANCE=${1:-"awesome-gpu-name"}
REMOTE_DIR="/workspace/dhp_safe_fa"

echo "════════════════════════════════════════════════════════════════════════════════"
echo "  DEPLOYING DHP I4 TO H100"
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""
echo "Instance: $BREV_INSTANCE"
echo "Remote:   $REMOTE_DIR"
echo ""

# Step 1: Login to Brev
echo "──────────────────────────────────────────────────────────────────────────────"
echo "Step 1: Brev Login"
echo "──────────────────────────────────────────────────────────────────────────────"
echo ""

if [ ! -z "$BREV_TOKEN" ]; then
    brev login --token $BREV_TOKEN
else
    echo "⚠️  BREV_TOKEN not set. Login manually if needed."
fi

# Step 2: Create remote directory
echo ""
echo "──────────────────────────────────────────────────────────────────────────────"
echo "Step 2: Create Remote Directory"
echo "──────────────────────────────────────────────────────────────────────────────"
echo ""

brev shell $BREV_INSTANCE -- "mkdir -p $REMOTE_DIR"

# Step 3: Upload files
echo ""
echo "──────────────────────────────────────────────────────────────────────────────"
echo "Step 3: Upload Files"
echo "──────────────────────────────────────────────────────────────────────────────"
echo ""

brev scp -r include/ $BREV_INSTANCE:$REMOTE_DIR/
brev scp -r kernels/ $BREV_INSTANCE:$REMOTE_DIR/
brev scp -r tests/ $BREV_INSTANCE:$REMOTE_DIR/
brev scp -r benchmarks/ $BREV_INSTANCE:$REMOTE_DIR/
brev scp setup.py $BREV_INSTANCE:$REMOTE_DIR/
brev scp run_tests.sh $BREV_INSTANCE:$REMOTE_DIR/
brev scp ncu_validate.sh $BREV_INSTANCE:$REMOTE_DIR/

echo "✅ Files uploaded"

# Step 4: Setup environment
echo ""
echo "──────────────────────────────────────────────────────────────────────────────"
echo "Step 4: Setup Environment"
echo "──────────────────────────────────────────────────────────────────────────────"
echo ""

brev shell $BREV_INSTANCE -- "cd $REMOTE_DIR && bash -c '\
    echo \"Verifying CUDA 13.0...\"; \
    /usr/local/cuda-13.0/bin/nvcc --version || echo \"⚠️  CUDA 13.0 not found\"; \
    echo \"\"; \
    echo \"Verifying PyTorch...\"; \
    python3 -c \"import torch; print(f\\\"PyTorch {torch.__version__}\\\")\"; \
    echo \"\"; \
    echo \"Verifying H100...\"; \
    nvidia-smi --query-gpu=name --format=csv,noheader; \
    echo \"\"; \
    echo \"✅ Environment ready\"; \
'"

# Step 5: Run tests
echo ""
echo "──────────────────────────────────────────────────────────────────────────────"
echo "Step 5: Run TDD Test Suite"
echo "──────────────────────────────────────────────────────────────────────────────"
echo ""

brev shell $BREV_INSTANCE -- "cd $REMOTE_DIR && bash run_tests.sh"

# Step 6: NCU profiling (optional)
echo ""
echo "──────────────────────────────────────────────────────────────────────────────"
echo "Step 6: NCU Profiling (Optional)"
echo "──────────────────────────────────────────────────────────────────────────────"
echo ""

read -p "Run NCU profiling? (y/N) " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    brev shell $BREV_INSTANCE -- "cd $REMOTE_DIR && sudo bash ncu_validate.sh i4 quick"
fi

echo ""
echo "════════════════════════════════════════════════════════════════════════════════"
echo "  DEPLOYMENT COMPLETE"
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""
echo "To access results:"
echo "  brev shell $BREV_INSTANCE"
echo "  cd $REMOTE_DIR"
echo "  cat audits/*.csv"
echo ""


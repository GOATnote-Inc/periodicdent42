#!/bin/bash
# Copyright 2025 GOATnote Inc.
# Licensed under Apache License 2.0

# Deploy and run LLaMA 3.1 validation on RunPod H100
# Usage: ./deploy_llama_validation_h100.sh [IP] [PORT]

set -e

# Configuration
DEFAULT_IP="154.57.34.90"
DEFAULT_PORT="23673"
IP="${1:-$DEFAULT_IP}"
PORT="${2:-$DEFAULT_PORT}"
SSH_OPTS="-o StrictHostKeyChecking=no -o TCPKeepAlive=yes -o ServerAliveInterval=20"

echo "=============================================================================="
echo "FlashCore LLaMA 3.1 Validation Deployment (H100)"
echo "=============================================================================="
echo "Target: root@${IP}:${PORT}"
echo ""

# Verify SSH connection
echo "Step 1: Verifying SSH connection..."
if ssh -p $PORT $SSH_OPTS root@$IP "echo 'SSH connection successful'"; then
    echo "✅ SSH connection established"
else
    echo "❌ SSH connection failed"
    echo "Please verify:"
    echo "  1. Pod is in 'Ready' state"
    echo "  2. IP and Port are correct (check RunPod 'Connect' tab)"
    echo "  3. SSH service has had 60-90s to initialize"
    exit 1
fi

# Verify GPU
echo ""
echo "Step 2: Verifying GPU..."
GPU_INFO=$(ssh -p $PORT $SSH_OPTS root@$IP "nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader" || echo "FAILED")
if [[ "$GPU_INFO" == *"H100"* ]]; then
    echo "✅ GPU detected: $GPU_INFO"
else
    echo "⚠️ Warning: GPU not detected or not H100: $GPU_INFO"
fi

# Create workspace
echo ""
echo "Step 3: Setting up workspace..."
ssh -p $PORT $SSH_OPTS root@$IP "mkdir -p /workspace/flashcore_llama"
echo "✅ Workspace created"

# Copy FlashCore code
echo ""
echo "Step 4: Deploying FlashCore code..."
echo "  - Copying flashcore/ directory..."
scp -P $PORT $SSH_OPTS -r flashcore root@$IP:/workspace/flashcore_llama/
echo "  - Copying tests/ directory..."
scp -P $PORT $SSH_OPTS -r tests root@$IP:/workspace/flashcore_llama/
echo "  - Copying setup files..."
scp -P $PORT $SSH_OPTS setup.py pyproject.toml README.md root@$IP:/workspace/flashcore_llama/
echo "✅ Code deployed"

# Install dependencies
echo ""
echo "Step 5: Installing dependencies..."
ssh -p $PORT $SSH_OPTS root@$IP << 'ENDSSH'
cd /workspace/flashcore_llama

# Check Python and CUDA
echo "  Python: $(python3 --version)"
echo "  PyTorch: $(python3 -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'Not installed')"
echo "  CUDA: $(python3 -c 'import torch; print(torch.version.cuda)' 2>/dev/null || echo 'N/A')"
echo "  Triton: $(python3 -c 'import triton; print(triton.__version__)' 2>/dev/null || echo 'Not installed')"

# Install FlashCore
echo ""
echo "  Installing FlashCore..."
pip install -e . --quiet

# Install transformers if needed
echo "  Installing transformers..."
pip install transformers>=4.36.0 --quiet

# Install test dependencies
echo "  Installing test dependencies..."
pip install pytest --quiet

echo ""
echo "✅ Dependencies installed"
ENDSSH

# Verify installation
echo ""
echo "Step 6: Verifying installation..."
ssh -p $PORT $SSH_OPTS root@$IP << 'ENDSSH'
cd /workspace/flashcore_llama

python3 << 'ENDPYTHON'
import sys
try:
    # Test imports
    import torch
    import triton
    import transformers
    from flashcore.fast.attention_production import attention_with_kv_cache
    from flashcore.llama_integration import replace_llama_attention_with_flashcore
    
    print("✅ All imports successful")
    print(f"  - PyTorch: {torch.__version__}")
    print(f"  - Triton: {triton.__version__}")
    print(f"  - Transformers: {transformers.__version__}")
    print(f"  - CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  - GPU: {torch.cuda.get_device_name(0)}")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)
ENDPYTHON
ENDSSH

if [ $? -eq 0 ]; then
    echo "✅ Installation verified"
else
    echo "❌ Installation verification failed"
    exit 1
fi

# Check HuggingFace token
echo ""
echo "Step 7: Checking HuggingFace authentication..."
ssh -p $PORT $SSH_OPTS root@$IP << 'ENDSSH'
python3 << 'ENDPYTHON'
import os
from huggingface_hub import HfFolder

token = HfFolder.get_token()
if token:
    print("✅ HuggingFace token found")
else:
    print("⚠️ No HuggingFace token found")
    print("To access LLaMA 3.1, you need to:")
    print("  1. Request access at: https://huggingface.co/meta-llama/Llama-3.1-8B")
    print("  2. Login: huggingface-cli login")
    print("")
    print("Alternatively, set HF_TOKEN environment variable")
ENDPYTHON
ENDSSH

# Create test runner script
echo ""
echo "Step 8: Creating test runner script..."
ssh -p $PORT $SSH_OPTS root@$IP << 'ENDSSH'
cat > /workspace/flashcore_llama/run_validation.sh << 'ENDRUNNER'
#!/bin/bash
# Run LLaMA 3.1 validation tests

cd /workspace/flashcore_llama

echo "=============================================================================="
echo "FlashCore LLaMA 3.1 Validation Tests"
echo "=============================================================================="
echo ""

# Check GPU
echo "GPU Status:"
nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free --format=csv
echo ""

# Run tests
echo "Running validation tests..."
echo ""

# Option 1: Run with pytest
if command -v pytest &> /dev/null; then
    echo "Running with pytest..."
    pytest tests/test_llama31_validation.py -v -s
else
    # Option 2: Run directly
    echo "Running directly..."
    python3 tests/test_llama31_validation.py
fi

# Check exit code
if [ $? -eq 0 ]; then
    echo ""
    echo "=============================================================================="
    echo "✅ ALL TESTS PASSED"
    echo "=============================================================================="
else
    echo ""
    echo "=============================================================================="
    echo "❌ SOME TESTS FAILED"
    echo "=============================================================================="
fi
ENDRUNNER

chmod +x /workspace/flashcore_llama/run_validation.sh
echo "✅ Test runner created: /workspace/flashcore_llama/run_validation.sh"
ENDSSH

# Deployment complete
echo ""
echo "=============================================================================="
echo "✅ DEPLOYMENT COMPLETE"
echo "=============================================================================="
echo ""
echo "Next steps:"
echo ""
echo "1. SSH into the machine:"
echo "   ssh -p $PORT $SSH_OPTS root@$IP"
echo ""
echo "2. Run the validation tests:"
echo "   cd /workspace/flashcore_llama"
echo "   ./run_validation.sh"
echo ""
echo "3. Or run individual tests:"
echo "   cd /workspace/flashcore_llama"
echo "   python3 tests/test_llama31_validation.py"
echo ""
echo "4. Monitor GPU usage during tests:"
echo "   watch -n 1 nvidia-smi"
echo ""
echo "Notes:"
echo "  - First run will download LLaMA 3.1 8B (~16GB)"
echo "  - Tests take ~10-15 minutes to complete"
echo "  - HuggingFace token required for model access"
echo ""
echo "=============================================================================="


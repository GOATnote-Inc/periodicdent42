#!/bin/bash
# Phase A GPU Setup Script
# Run this on the L4 GPU instance to start Phase A

set -euo pipefail

echo "============================================================"
echo "Phase A: Correctness Fix (PyTorch 2.5.0 Compatibility)"
echo "============================================================"
echo ""

# Navigate to repo
cd ~/periodicdent42
source ~/venv/bin/activate

echo "📍 Current directory: $(pwd)"
echo "🐍 Python: $(python --version)"
echo "🔥 PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "💻 GPU: $(python -c 'import torch; print(torch.cuda.get_device_name(0))')"
echo ""

# Task 1: Test with PyTorch 2.1.0 (isolate version issue)
echo "============================================================"
echo "Task 1: Test with PyTorch 2.1.0 (isolate version)"
echo "============================================================"
echo ""

read -p "Install PyTorch 2.1.0 and test? (y/n) " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "📦 Installing PyTorch 2.1.0..."
    pip uninstall torch -y
    pip install torch==2.1.0+cu121 --index-url https://download.pytorch.org/whl/cu121
    
    echo ""
    echo "🧪 Testing Phase 4 with PyTorch 2.1.0..."
    rm -rf ~/.cache/torch_extensions
    PYTHONPATH=. python scripts/standalone_phase4_eval.py | tee evidence/phase_a_pytorch210_test.log
    
    echo ""
    echo "✅ Task 1 complete. Check evidence/phase_a_pytorch210_test.log"
    echo ""
fi

# Task 2: Upgrade back to PyTorch 2.5.0
echo "============================================================"
echo "Task 2: Upgrade to PyTorch 2.5.0"
echo "============================================================"
echo ""

read -p "Upgrade to PyTorch 2.5.0? (y/n) " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "📦 Installing PyTorch 2.5.0..."
    pip uninstall torch -y
    pip install torch==2.5.0 --index-url https://download.pytorch.org/whl/cu121
    
    echo ""
    echo "✅ PyTorch 2.5.0 installed"
    echo ""
fi

# Task 3: Dual-reference validation
echo "============================================================"
echo "Task 3: Dual-Reference Validation"
echo "============================================================"
echo ""

read -p "Run dual-reference validation? (y/n) " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "🧪 Testing Flash vs Math SDPA backends..."
    rm -rf ~/.cache/torch_extensions
    python scripts/phase_a_validate_dual_backend.py | tee evidence/phase_a_dual_backend.log
    
    echo ""
    echo "✅ Task 3 complete. Check evidence/phase_a_dual_backend.log"
    echo ""
fi

echo "============================================================"
echo "Phase A Setup Complete"
echo "============================================================"
echo ""
echo "Next Steps:"
echo "  1. Review evidence/phase_a_pytorch210_test.log"
echo "  2. Review evidence/phase_a_dual_backend.log"
echo "  3. If correctness issues persist, implement numerical stability fixes"
echo "  4. Proceed to Phase B (cuBLAS Q@K^T)"
echo ""


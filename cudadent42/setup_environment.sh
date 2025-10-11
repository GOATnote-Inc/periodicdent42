#!/bin/bash
# FlashMoE-Science Environment Setup
# Run this script on a machine with CUDA GPU (H100/A100)

set -e

echo "🚀 Setting up FlashMoE-Science development environment..."

# 1. Create conda environment
echo "📦 Creating conda environment..."
conda create -n flashmoe python=3.10 cuda-toolkit=12.3 -c nvidia -y

# 2. Activate environment
echo "🔧 Activating environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate flashmoe

# 3. Install PyTorch with CUDA 12.3
echo "⚡ Installing PyTorch with CUDA..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu123

# 4. Install development dependencies
echo "📚 Installing dependencies..."
pip install -r requirements.txt

# 5. Verify CUDA availability
echo "✅ Verifying CUDA setup..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'GPU count: {torch.cuda.device_count()}'); print(f'GPU name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

echo ""
echo "✅ Environment setup complete!"
echo "Next steps:"
echo "  conda activate flashmoe"
echo "  python setup.py build_ext --inplace"
echo "  pytest tests/ -v"


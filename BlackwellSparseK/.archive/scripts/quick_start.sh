#!/bin/bash
# ============================================================================
# BlackwellSparseK Quick Start
# ============================================================================
# One-command setup for development environment
# ============================================================================

set -e

GPU_ID=${1:-0}
IMAGE_NAME="blackwell-sparsek:dev"

echo "🚀 BlackwellSparseK Quick Start"
echo "================================"
echo "GPU: ${GPU_ID}"
echo ""

# Check if image exists
if ! docker image inspect ${IMAGE_NAME} &> /dev/null; then
    echo "📦 Image not found. Building ${IMAGE_NAME}..."
    echo ""
    docker build -f docker/blackwell-sparsek-dev.dockerfile -t ${IMAGE_NAME} .
    echo ""
    echo "✅ Build complete!"
    echo ""
fi

# Run interactive session
echo "🏃 Starting interactive development session..."
echo ""

docker run --gpus "device=${GPU_ID}" \
    --rm -it \
    --shm-size=8g \
    -v $(pwd):/workspace/BlackwellSparseK \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -e CUDA_VISIBLE_DEVICES=${GPU_ID} \
    ${IMAGE_NAME} \
    /bin/bash -c "
        echo '✅ Environment ready!'
        echo ''
        echo '=========================================='
        echo 'BlackwellSparseK Development Environment'
        echo '=========================================='
        echo ''
        python -c '
import torch, sys
try:
    import blackwell_sparsek
    print(f\"BlackwellSparseK: {blackwell_sparsek.__version__}\")
except: pass
print(f\"PyTorch: {torch.__version__}\")
print(f\"CUDA: {torch.version.cuda}\")
if torch.cuda.is_available():
    print(f\"GPU: {torch.cuda.get_device_name(0)}\")
print(f\"Python: {sys.version.split()[0]}\")
        '
        echo ''
        echo 'Quick commands:'
        echo '  python examples/basic_attention.py'
        echo '  python benchmarks/perf.py'
        echo '  pytest tests/'
        echo ''
        exec /bin/bash
    "


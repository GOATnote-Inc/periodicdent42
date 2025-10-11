#!/bin/bash
# GCE Startup Script for Automated CUDAdent42 Benchmark Execution
# This script runs automatically when a GCE GPU instance starts
# Results are uploaded to Google Cloud Storage

set -e
set -x  # Verbose mode for debugging

# Log all output
exec > >(tee -a /var/log/cuda-benchmark.log)
exec 2>&1

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🚀 CUDAdent42 Automated Benchmark Execution"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Start Time: $(date)"
echo "Host: $(hostname)"
echo "User: $(whoami)"
echo ""

# Wait for CUDA to be ready
echo "Waiting for CUDA to be ready..."
for i in {1..30}; do
    if nvidia-smi &>/dev/null; then
        echo "✅ CUDA ready"
        break
    fi
    echo "Waiting... ($i/30)"
    sleep 2
done

# Display GPU info
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🎮 GPU Information"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
nvidia-smi

# Setup environment
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📦 Setting Up Environment"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Install Python and tools if needed
apt-get update -qq
apt-get install -y -qq python3-pip python3-venv git

# Clone repository
WORK_DIR="/tmp/cudadent42_benchmark"
mkdir -p "$WORK_DIR"
cd "$WORK_DIR"

echo "Cloning repository..."
git clone https://github.com/GOATnote-Inc/periodicdent42.git
cd periodicdent42/cudadent42

# Setup Python environment
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip --quiet
pip install torch torchvision numpy pytest --quiet

echo "✅ Environment ready"

# Build library
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🔧 Building CUDAdent42 Library"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

bash build_manual.sh || { echo "❌ Build failed"; exit 1; }

echo "✅ Build complete"

# Run correctness tests
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ Running Correctness Tests"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

python3 tests/test_correctness.py || { echo "❌ Correctness tests failed"; exit 1; }

echo "✅ All correctness tests passed"

# Run benchmarks
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📊 Running SOTA Benchmarks (50 repeats)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="benchmark_results/sota_${TIMESTAMP}"
mkdir -p "$OUTPUT_DIR"

python3 benches/bench_correctness_and_speed.py \
    --output-dir "$OUTPUT_DIR" \
    --repeats 50 \
    --warmup 10 \
    --save-csv \
    --verbose \
    | tee "$OUTPUT_DIR/benchmark_log.txt"

# Collect system info
nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv,noheader > "$OUTPUT_DIR/gpu_info.txt"
pip freeze > "$OUTPUT_DIR/python_env.txt"
nvcc --version > "$OUTPUT_DIR/cuda_version.txt" 2>&1 || echo "nvcc not in PATH" > "$OUTPUT_DIR/cuda_version.txt"

# Create summary
cat > "$OUTPUT_DIR/SYSTEM_INFO.md" << EOF
# System Information

**Date**: $(date)
**Host**: $(hostname)
**GPU**: $(nvidia-smi --query-gpu=name --format=csv,noheader)
**CUDA Driver**: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader)
**Compute Capability**: SM$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | sed 's/\.//')
**PyTorch Version**: $(python3 -c "import torch; print(torch.__version__)")

## Build Configuration
- Manual build (proven working on L4)
- FP16: Always enabled
- BF16: $(python3 -c "import torch; print('Enabled' if torch.cuda.get_device_capability()[0] >= 8 else 'Disabled')")

## Test Configuration
- Warmup iterations: 10
- Timing repeats: 50
- Timing method: CUDA events
- Memory tracking: PyTorch CUDA allocator

## Baseline
- PyTorch \`F.scaled_dot_product_attention\`
- Backend: flash_sdp (FlashAttention 2.x)
EOF

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "☁️ Uploading Results to Cloud Storage"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Upload to GCS
BUCKET="gs://periodicdent42-benchmarks"
gsutil -m cp -r "$OUTPUT_DIR" "$BUCKET/cudadent42/" || {
    echo "⚠️ GCS upload failed, results saved locally at: $OUTPUT_DIR"
}

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ Benchmark Complete!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Results location: $OUTPUT_DIR"
echo "Cloud Storage: $BUCKET/cudadent42/$(basename $OUTPUT_DIR)"
echo "End Time: $(date)"
echo ""
echo "Instance will shut down in 60 seconds..."
sleep 60

# Auto-shutdown to save costs
sudo shutdown -h now


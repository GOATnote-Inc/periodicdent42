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

# Setup environment
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📦 Setting Up Environment"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Install essential tools
apt-get update -qq 2>&1 > /dev/null || true
apt-get install -y -qq git python3-pip 2>&1 > /dev/null || true

# Clone repository
WORK_DIR="/tmp/cudadent42_benchmark"
mkdir -p "$WORK_DIR"
cd "$WORK_DIR"

echo "Cloning repository..."
git clone https://github.com/GOATnote-Inc/periodicdent42.git
cd periodicdent42/cudadent42

# Install Python dependencies
python3 -m pip install --user pybind11 torch --quiet || {
    echo "❌ Failed to install Python dependencies"
    exit 1
}

# Add CUDA to PATH (preflight will check this)
export PATH="/usr/local/cuda/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"

# Run preflight checks (self-healing, validates GPU + CUDA + PyTorch)
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ Running Preflight Checks"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
bash scripts/gen_preflight.sh
bash tools/preflight.sh || {
    echo "❌ Preflight failed - environment not ready"
    exit 1
}
echo "✅ Environment validated"

# Build library (inline build - Phase 2 manual build commands)
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🔧 Building CUDAdent42 Library"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Get PyTorch paths
TORCH_INCLUDE=$(python3 -c 'import torch; print(torch.utils.cmake_prefix_path + "/include")')
TORCH_INCLUDE_API=$(python3 -c 'import torch; print(torch.utils.cmake_prefix_path + "/include/torch/csrc/api/include")')
TORCH_LIB=$(python3 -c 'import torch; import os; print(os.path.join(os.path.dirname(torch.__file__), "lib"))')
PYBIND_INCLUDE=$(python3 -c 'import pybind11; print(pybind11.get_include())')

# Detect SM architecture
SM_ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | tr -d '.' | head -1)
echo "Detected SM architecture: SM_$SM_ARCH"

# Compile FP16 kernel
echo "Compiling FP16 kernel..."
nvcc -c python/flashmoe_science/csrc/flash_attention_science.cu \
    -o python/flashmoe_science/flash_attention_science_fp16.o \
    --compiler-options '-fPIC' \
    -arch=sm_$SM_ARCH \
    -O3 \
    -std=c++17 \
    -I/usr/local/cuda/include \
    -I$TORCH_INCLUDE \
    -I$TORCH_INCLUDE_API

# Compile BF16 kernel (if SM80+)
if [ "$SM_ARCH" -ge "80" ]; then
    echo "Compiling BF16 kernel..."
    nvcc -c python/flashmoe_science/csrc/flash_attention_science_bf16.cu \
        -o python/flashmoe_science/flash_attention_science_bf16.o \
        --compiler-options '-fPIC' \
        -arch=sm_$SM_ARCH \
        -O3 \
        -std=c++17 \
        -I/usr/local/cuda/include \
        -I$TORCH_INCLUDE \
        -I$TORCH_INCLUDE_API
    BF16_OBJ="python/flashmoe_science/flash_attention_science_bf16.o"
else
    echo "Skipping BF16 kernel (requires SM80+)"
    BF16_OBJ=""
fi

# Compile bindings
echo "Compiling Python bindings..."
g++ -c python/flashmoe_science/csrc/bindings.cpp \
    -o python/flashmoe_science/bindings.o \
    -fPIC \
    -O3 \
    -std=c++17 \
    -I/usr/local/cuda/include \
    -I$TORCH_INCLUDE \
    -I$TORCH_INCLUDE_API \
    -I$PYBIND_INCLUDE

# Link
echo "Linking shared library..."
g++ -shared \
    python/flashmoe_science/flash_attention_science_fp16.o \
    $BF16_OBJ \
    python/flashmoe_science/bindings.o \
    -o python/flashmoe_science/flash_attention_science.so \
    -L/usr/local/cuda/lib64 \
    -lcudart \
    -L$TORCH_LIB \
    -ltorch -ltorch_cpu -ltorch_python -lc10 -lc10_cuda

# Verify
export PYTHONPATH="$PWD/python:$PYTHONPATH"
python3 -c "import flashmoe_science; print('✅ Library import successful')"

echo "✅ Build complete"

# Run correctness tests
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ Running Correctness Tests"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

cd tests
python3 test_attention_correctness.py || {
    echo "⚠️  Correctness tests failed, but continuing with benchmark"
}
cd ..

echo "✅ Tests complete"

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


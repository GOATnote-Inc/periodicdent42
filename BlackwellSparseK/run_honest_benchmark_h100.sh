#!/bin/bash
# Deploy and run honest head-to-head benchmark on H100
# Author: Brandon Dent, MD
# Date: November 1, 2025

set -e  # Exit on error

echo "========================================"
echo "HONEST HEAD-TO-HEAD BENCHMARK - H100"
echo "========================================"
echo ""
echo "This script will:"
echo "1. Deploy code to H100"
echo "2. Build custom kernel"
echo "3. Run PyTorch sparse vs Custom kernel comparison"
echo "4. Download results"
echo ""

# Configuration (update these for your RunPod)
H100_HOST="${H100_HOST:-root@154.57.34.90}"
H100_PORT="${H100_PORT:-25754}"
H100_DIR="/workspace/BlackwellSparseK_benchmark"

# Check if SSH config provided
if [ -z "$H100_HOST" ]; then
    echo "❌ ERROR: H100_HOST not set"
    echo "Usage: H100_HOST=root@YOUR_IP H100_PORT=YOUR_PORT ./run_honest_benchmark_h100.sh"
    exit 1
fi

echo "📡 Target: $H100_HOST:$H100_PORT"
echo ""

# Step 1: Deploy code
echo "1️⃣  Deploying code to H100..."
scp -P $H100_PORT -r \
    ../BlackwellSparseK \
    $H100_HOST:$H100_DIR/
echo "✅ Code deployed"
echo ""

# Step 2: Build on H100
echo "2️⃣  Building kernel on H100..."
ssh -p $H100_PORT $H100_HOST << 'ENDSSH'
cd /workspace/BlackwellSparseK_benchmark

# Check environment
echo "Checking environment..."
nvcc --version | grep "release"
python3 --version

# Install dependencies
pip3 install torch --index-url https://download.pytorch.org/whl/cu121

# Build custom kernel (if exists)
if [ -f "setup.py" ]; then
    echo "Building custom kernel..."
    python3 setup.py build_ext --inplace
    echo "✅ Kernel built"
else
    echo "⚠️  No setup.py found - will benchmark PyTorch only"
fi

ENDSSH

echo "✅ Build complete"
echo ""

# Step 3: Run benchmark
echo "3️⃣  Running honest benchmark..."
ssh -p $H100_PORT $H100_HOST << 'ENDSSH'
cd /workspace/BlackwellSparseK_benchmark/benchmarks

# Make script executable
chmod +x compare_all_baselines.py

# Run benchmark
python3 compare_all_baselines.py

# Show results
echo ""
echo "📊 Results saved to benchmark_results.json"
ls -lh benchmark_results.json

ENDSSH

echo "✅ Benchmark complete"
echo ""

# Step 4: Download results
echo "4️⃣  Downloading results..."
mkdir -p results/
scp -P $H100_PORT \
    $H100_HOST:$H100_DIR/benchmarks/benchmark_results.json \
    results/benchmark_results_$(date +%Y%m%d_%H%M%S).json
echo "✅ Results downloaded to results/"
echo ""

# Step 5: Display results
echo "========================================"
echo "BENCHMARK COMPLETE"
echo "========================================"
echo ""
echo "Results location:"
ls -lh results/benchmark_results_*.json | tail -1
echo ""
echo "To view results:"
echo "  cat results/benchmark_results_*.json | jq"
echo ""
echo "To run again:"
echo "  H100_HOST=$H100_HOST H100_PORT=$H100_PORT ./run_honest_benchmark_h100.sh"
echo ""


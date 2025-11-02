#!/bin/bash
set -euo pipefail

# Deploy and validate on Brev H100 via SSH

BREV_INSTANCE="${1:-awesome-gpu-name}"

echo "▶ Connecting to $BREV_INSTANCE..."

brev shell "$BREV_INSTANCE" << 'REMOTE_SCRIPT'
set -euo pipefail

echo "▶ Installing essentials..."
apt-get update -qq && apt-get install -y -qq git cmake build-essential 2>&1 | tail -3

echo "▶ Cloning CUTLASS..."
rm -rf /usr/local/cutlass
git clone --depth 1 https://github.com/NVIDIA/cutlass.git /usr/local/cutlass 2>&1 | tail -1

echo "▶ Cloning BlackwellSparseK..."
cd /workspace
rm -rf periodicdent42
git clone --depth 1 https://github.com/GOATnote-Inc/periodicdent42.git 2>&1 | tail -1
cd periodicdent42/BlackwellSparseK

echo "▶ Compiling..."
mkdir -p build
nvcc -O3 -std=c++17 -arch=sm_90a --expt-relaxed-constexpr \
     --maxrregcount=255 -I/usr/local/cutlass/include \
     src/gemm_h100_599tflops_final.cu -o build/blackwell_gemm -lcudart

echo "▶ Running..."
./build/blackwell_gemm

echo ""
echo "▶ NCU validation..."
mkdir -p /workspace/ncu_reports
ncu --set full --export /workspace/ncu_reports/validation.ncu-rep ./build/blackwell_gemm || \
ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed ./build/blackwell_gemm

echo ""
echo "✅ DONE"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
REMOTE_SCRIPT

echo ""
echo "▶ Downloading NCU report..."
brev scp "$BREV_INSTANCE":/workspace/ncu_reports/validation.ncu-rep ./ncu_reports/

echo ""
echo "✅ Validated. Report: ./ncu_reports/validation.ncu-rep"


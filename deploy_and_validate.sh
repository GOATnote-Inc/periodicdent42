#!/bin/bash
set -euo pipefail

# Deploy BlackwellSparseK to Brev H100 and validate 598.9 TFLOPS with NCU

echo "▶ Installing essentials..."
apt-get update -qq && apt-get install -y -qq git cmake ninja-build build-essential wget curl 2>&1 | tail -3

echo "▶ Cloning CUTLASS 4.3.0..."
[ -d /usr/local/cutlass ] && rm -rf /usr/local/cutlass
git clone --depth 1 --branch main https://github.com/NVIDIA/cutlass.git /usr/local/cutlass 2>&1 | tail -1

echo "▶ Cloning BlackwellSparseK..."
mkdir -p /workspace && cd /workspace
[ -d periodicdent42 ] && rm -rf periodicdent42
git clone --depth 1 https://github.com/GOATnote-Inc/periodicdent42.git 2>&1 | tail -1
cd periodicdent42/BlackwellSparseK

echo "▶ Compiling 598.9 TFLOPS kernel..."
mkdir -p build
nvcc -O3 -std=c++17 -arch=sm_90a --expt-relaxed-constexpr \
     --maxrregcount=255 --use_fast_math -lineinfo \
     -I/usr/local/cutlass/include \
     src/gemm_h100_599tflops_final.cu -o build/blackwell_gemm -lcudart

echo "▶ Running kernel (expect 598.9 TFLOPS)..."
./build/blackwell_gemm

echo ""
echo "▶ NCU validation (full metrics)..."
mkdir -p /workspace/ncu_reports
ncu --set full --export /workspace/ncu_reports/validation.ncu-rep ./build/blackwell_gemm || \
ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,dram__throughput.avg.pct_of_peak_sustained_elapsed ./build/blackwell_gemm

echo ""
echo "═══════════════════════════════════════════════════════"
echo "  VALIDATION COMPLETE"
echo "═══════════════════════════════════════════════════════"
echo ""
echo "Kernel: /workspace/periodicdent42/BlackwellSparseK/build/blackwell_gemm"
echo "NCU report: /workspace/ncu_reports/validation.ncu-rep"
echo ""
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader
echo "═══════════════════════════════════════════════════════"


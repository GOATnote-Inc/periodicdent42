#!/bin/bash
brev shell awesome-gpu-name << 'REMOTE'
set -x
sudo apt-get update -qq && sudo apt-get install -y -qq git cmake build-essential
sudo rm -rf /usr/local/cutlass
sudo git clone --depth 1 https://github.com/NVIDIA/cutlass.git /usr/local/cutlass
sudo mkdir -p /workspace
sudo chown -R $(whoami) /workspace
cd /workspace
rm -rf periodicdent42
git clone --depth 1 https://github.com/GOATnote-Inc/periodicdent42.git
cd periodicdent42/BlackwellSparseK
mkdir -p build
/usr/local/cuda/bin/nvcc -O3 -std=c++17 -arch=sm_90a --expt-relaxed-constexpr --maxrregcount=255 -I/usr/local/cutlass/include src/gemm_h100_599tflops_final.cu -o build/blackwell_gemm -lcudart
echo ""
echo "▶ RUNNING:"
./build/blackwell_gemm
echo ""
echo "▶ NCU:"
/usr/local/cuda/bin/ncu --set full --export /workspace/ncu_reports/validation.ncu-rep ./build/blackwell_gemm || \
/usr/local/cuda/bin/ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed ./build/blackwell_gemm
REMOTE

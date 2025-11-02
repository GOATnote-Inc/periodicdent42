#!/bin/bash
set -euo pipefail

# Get SSH command from Brev console, e.g.:
# ssh root@<ip> -p <port>
# 
# Usage: ./deploy_to_brev.sh root@ip -p port

SSH_CMD="$@"

if [ -z "$SSH_CMD" ]; then
    echo "Get SSH command from Brev console 'Connect' button"
    echo "Usage: $0 root@<ip> -p <port>"
    exit 1
fi

echo "▶ Deploying to Brev H100..."

ssh "$@" << 'REMOTE'
set -euo pipefail
apt-get update -qq && apt-get install -y -qq git cmake build-essential 2>&1 | tail -3
rm -rf /usr/local/cutlass
git clone --depth 1 https://github.com/NVIDIA/cutlass.git /usr/local/cutlass 2>&1 | tail -1
cd /workspace
rm -rf periodicdent42
git clone --depth 1 https://github.com/GOATnote-Inc/periodicdent42.git 2>&1 | tail -1
cd periodicdent42/BlackwellSparseK
mkdir -p build
nvcc -O3 -std=c++17 -arch=sm_90a --expt-relaxed-constexpr --maxrregcount=255 -I/usr/local/cutlass/include src/gemm_h100_599tflops_final.cu -o build/blackwell_gemm -lcudart
echo ""
echo "▶ RUNNING KERNEL:"
./build/blackwell_gemm
echo ""
echo "▶ NCU VALIDATION:"
mkdir -p /workspace/ncu_reports
ncu --set full --export /workspace/ncu_reports/validation.ncu-rep ./build/blackwell_gemm 2>&1 | tail -20 || \
ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed ./build/blackwell_gemm 2>&1 | tail -20
REMOTE

echo ""
echo "✅ Done. Get NCU report:"
echo "scp $@ -r :/workspace/ncu_reports ./ncu_reports/"


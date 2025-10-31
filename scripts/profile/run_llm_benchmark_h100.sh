#!/bin/bash
# Run real-world LLM benchmark on H100

set -e

RUNPOD_IP="154.57.34.90"
RUNPOD_PORT="14727"

echo "========================================"
echo "REAL-WORLD LLM BENCHMARK (H100)"
echo "========================================"
echo ""

echo "[1/2] Deploying benchmark..."
scp -P ${RUNPOD_PORT} -o StrictHostKeyChecking=no \
    flashcore/benchmark/llm_metrics_benchmark.py \
    root@${RUNPOD_IP}:/workspace/flashcore_llama/flashcore/benchmark/
echo "✅ Deployed"
echo ""

echo "[2/2] Running comprehensive benchmark..."
echo "Measuring: tokens/sec, sequences/sec, latency, VRAM"
echo ""

ssh -p ${RUNPOD_PORT} -o StrictHostKeyChecking=no root@${RUNPOD_IP} << 'ENDSSH'
cd /workspace/flashcore_llama
export PYTHONPATH=/workspace/flashcore_llama:$PYTHONPATH

echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo ""

python3 flashcore/benchmark/llm_metrics_benchmark.py

ENDSSH

echo ""
echo "========================================"
echo "BENCHMARK COMPLETE"
echo "========================================"


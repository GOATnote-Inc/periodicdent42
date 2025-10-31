#!/bin/bash
# Deploy Split-K kernel to H100 for expert testing

set -e

echo "=== DEPLOYING EXPERT SPLIT-K KERNEL TO H100 ==="

# H100 connection (from memory)
H100_IP="154.57.34.90"
H100_PORT="14727"
H100_USER="root"

echo "Target: ${H100_USER}@${H100_IP}:${H100_PORT}"
echo ""

# Create remote directory
ssh -p ${H100_PORT} ${H100_USER}@${H100_IP} "mkdir -p /workspace/flashcore_hopper/flashcore/fast /workspace/flashcore_hopper/flashcore/cuda /workspace/flashcore_hopper/build"

# Deploy files
echo "[1/4] Deploying Split-K kernel..."
scp -P ${H100_PORT} flashcore/fast/attention_cublaslt_splitk.cu ${H100_USER}@${H100_IP}:/workspace/flashcore_hopper/flashcore/fast/

echo "[2/4] Deploying existing kernels..."
scp -P ${H100_PORT} flashcore/fast/attention_hopper_minimal.cu ${H100_USER}@${H100_IP}:/workspace/flashcore_hopper/flashcore/fast/
scp -P ${H100_PORT} flashcore/fast/attention_cublaslt_sparse.cu ${H100_USER}@${H100_IP}:/workspace/flashcore_hopper/flashcore/fast/

echo "[3/4] Deploying test harness..."
scp -P ${H100_PORT} flashcore/cuda/test_hopper_kernel.cu ${H100_USER}@${H100_IP}:/workspace/flashcore_hopper/flashcore/cuda/

echo "[4/4] Deploying build script..."
scp -P ${H100_PORT} build_cuda_simple.sh ${H100_USER}@${H100_IP}:/workspace/flashcore_hopper/

echo ""
echo "✅ Deployment complete!"
echo ""
echo "To build and run on H100:"
echo "  ssh -p ${H100_PORT} ${H100_USER}@${H100_IP}"
echo "  cd /workspace/flashcore_hopper"
echo "  chmod +x build_cuda_simple.sh"
echo "  ./build_cuda_simple.sh"



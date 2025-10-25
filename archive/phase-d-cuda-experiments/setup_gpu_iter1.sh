#!/bin/bash
# Setup script for L4 GPU instance - EvoEngineer Iteration 1 testing
set -euo pipefail

echo "🚀 Setting up L4 GPU for EvoEngineer Iteration 1 testing..."
echo ""

# 1. Install CUDA 12.2
echo "📦 Installing CUDA 12.2..."
wget https://developer.download.nvidia.com/compute/cuda/12.2.0/local_installers/cuda_12.2.0_535.54.03_linux.run
sudo sh cuda_12.2.0_535.54.03_linux.run --silent --toolkit --override
echo 'export PATH=/usr/local/cuda-12.2/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.2/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# 2. Install Python & PyTorch
echo "🐍 Installing Python 3.10 & PyTorch..."
sudo apt-get update
sudo apt-get install -y python3.10 python3.10-venv python3-pip git
python3 -m venv ~/venv
source ~/venv/bin/activate

pip install --upgrade pip
pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cu121

# 3. Clone repo
echo "📥 Cloning repository..."
cd ~
git clone https://github.com/GOATnote-Inc/periodicdent42.git
cd periodicdent42

# 4. Build Iteration 1 kernel
echo "🔨 Building EvoEngineer Iteration 1 kernel..."
cd cudadent42/bench
python build_v3_release.py

# 5. Correctness test (MUST PASS before benchmark)
echo "✅ Running correctness test..."
cd ../../scripts
python3 << 'EOF'
import torch
import sys
sys.path.insert(0, "../cudadent42/bench")
import flash_attention_s512_v3 as fa_v3

B, H, S, D = 2, 8, 512, 64
q = torch.randn(B, H, S, D, dtype=torch.float16, device='cuda')
k, v = q.clone(), q.clone()

# PyTorch reference
with torch.backends.cuda.sdp_kernel(enable_flash=True):
    ref = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=False)

# Our kernel (config_id=1: 32x64x4x2)
out = fa_v3.flash_attention_s512_v3(q, k, v, config_id=1)

# Correctness
if torch.allclose(out, ref, atol=1e-3, rtol=1e-3):
    print("✅ CORRECTNESS PASSED")
    sys.exit(0)
else:
    max_diff = (out - ref).abs().max().item()
    print(f"❌ CORRECTNESS FAILED: max_diff={max_diff}")
    sys.exit(1)
EOF

if [ $? -ne 0 ]; then
    echo "❌ Correctness test failed! Stopping."
    exit 1
fi

# 6. Benchmark (only if correctness passes)
echo "📊 Running benchmark..."
python3 << 'EOF'
import torch
import time
import sys
sys.path.insert(0, "../cudadent42/bench")
import flash_attention_s512_v3 as fa_v3

B, H, S, D = 2, 8, 512, 64
q = torch.randn(B, H, S, D, dtype=torch.float16, device='cuda')
k, v = q.clone(), q.clone()

# Warm-up
for _ in range(10):
    _ = fa_v3.flash_attention_s512_v3(q, k, v, config_id=1)
torch.cuda.synchronize()

# Benchmark
times = []
for _ in range(100):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    out = fa_v3.flash_attention_s512_v3(q, k, v, config_id=1)
    end.record()
    torch.cuda.synchronize()
    times.append(start.elapsed_time(end) * 1000)  # Convert to μs

times_sorted = sorted(times)
p50 = times_sorted[50]
p90 = times_sorted[90]
p99 = times_sorted[99]

print(f"\n📊 EvoEngineer Iteration 1 Results:")
print(f"  p50: {p50:.2f} μs")
print(f"  p90: {p90:.2f} μs")
print(f"  p99: {p99:.2f} μs")
print(f"\n📈 vs Baseline (38.00 μs):")
if p50 < 38.00:
    speedup = 38.00 / p50
    print(f"  ✅ {speedup:.2f}× faster ({38.00 - p50:.2f} μs improvement)")
elif p50 > 38.00 * 1.05:
    print(f"  ❌ {p50/38.00:.2f}× slower (REGRESSION)")
else:
    print(f"  ⚖️  Neutral ({p50 - 38.00:+.2f} μs)")
EOF

echo ""
echo "✅ Setup complete! Results saved."


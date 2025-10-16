#!/bin/bash
# Install official FlashAttention-2 on L4 GPU
# This gives us a proven working baseline to measure and optimize

set -e

echo "=============================================================================="
echo "Installing FlashAttention-2 (Official, Battle-Tested)"
echo "=============================================================================="

cd ~/periodicdent42/ext/flash-attention-2

# Install with minimal dependencies
echo "Building flash-attn for sm_89 (L4 Ada)..."
MAX_JOBS=4 FLASH_ATTENTION_FORCE_BUILD=TRUE \
  TORCH_CUDA_ARCH_LIST="8.9" \
  pip install -e . --no-build-isolation -v 2>&1 | tee ~/flash_attn_install.log

echo ""
echo "=============================================================================="
echo "Testing FlashAttention-2 Installation"
echo "=============================================================================="

python3 << 'EOF'
import torch
import torch.nn.functional as F

print("Importing flash_attn...")
from flash_attn import flash_attn_func

print("✅ flash_attn imported successfully")

# Test on L4
B, H, S, D = 4, 8, 512, 64
q = torch.randn(B, S, H, D, dtype=torch.float16, device='cuda')  # Note: BS HD format
k, v = q.clone(), q.clone()

print(f"\nTesting with B={B}, H={H}, S={S}, D={D}...")

# Warm-up
for _ in range(5):
    _ = flash_attn_func(q, k, v)
torch.cuda.synchronize()

# Benchmark
times = []
for _ in range(50):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    out_fa = flash_attn_func(q, k, v)
    end.record()
    torch.cuda.synchronize()
    times.append(start.elapsed_time(end) * 1000)  # μs

times.sort()
fa_p50 = times[25]

# Compare with PyTorch SDPA
q_pt = q.transpose(1, 2)  # BHSD format for PyTorch
k_pt, v_pt = k.transpose(1, 2), v.transpose(1, 2)

times_pt = []
for _ in range(50):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    out_pt = F.scaled_dot_product_attention(q_pt, k_pt, v_pt, is_causal=False)
    end.record()
    torch.cuda.synchronize()
    times_pt.append(start.elapsed_time(end) * 1000)

times_pt.sort()
pt_p50 = times_pt[25]

# Correctness
out_fa_check = out_fa.transpose(1, 2)  # Convert to BHSD for comparison
correct = torch.allclose(out_fa_check, out_pt, atol=1e-2, rtol=1e-2)

print(f"\n{'='*80}")
print("BASELINE RESULTS")
print(f"{'='*80}")
print(f"FlashAttention-2:  {fa_p50:7.2f} μs")
print(f"PyTorch SDPA:      {pt_p50:7.2f} μs")
print(f"Speedup:           {pt_p50/fa_p50:7.2f}x")
print(f"Correctness:       {'✅ PASS' if correct else '❌ FAIL'}")
print(f"{'='*80}")

if fa_p50 < pt_p50:
    print(f"\n✅ FlashAttention-2 is {pt_p50/fa_p50:.2f}x FASTER than PyTorch!")
else:
    print(f"\n⚠️  PyTorch is {fa_p50/pt_p50:.2f}x faster (unexpected)")

print("\n✅ Installation and validation complete!")
print("\nNext: Use EvoEngineer-Insight to optimize FlashAttention-2 kernel")
EOF

echo ""
echo "=============================================================================="
echo "✅ FlashAttention-2 ready for optimization"
echo "=============================================================================="


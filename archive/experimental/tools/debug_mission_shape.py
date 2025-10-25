#!/usr/bin/env python3
"""Debug mission shape (B=1, H=8, S=512, D=64) for FP8 WMMA kernel."""

import torch
import math
from torch.utils.cpp_extension import load

# Build extension WITHOUT debug prints for cleaner output
ext = load(
    name="sdpa_fp8_stage_c_wmma",
    sources=[
        "cudadent42/bench/kernels/sdpa_fp8_stage_c_wmma.cu",
        "cudadent42/bench/kernels/sdpa_fp8_stage_c_wmma_bindings.cpp",
    ],
    extra_cuda_cflags=["-O3", "-arch=sm_89", "--use_fast_math", "-lineinfo"],
    verbose=True,
)

# Import quantizer
import sys
sys.path.insert(0, 'cudadent42/bench')
from sdpa_fp8_stage_c_wmma import quantize_sim_fp8_per_head

# Test config
B, H, S, D = 1, 8, 512, 64
torch.manual_seed(42)
device = "cuda"

print(f"\n{'='*80}")
print(f"Testing mission shape: (B={B}, H={H}, S={S}, D={D})")
print(f"{'='*80}\n")

# Generate FP16 inputs
Q = torch.randn(B, H, S, D, device=device, dtype=torch.float16)
K = torch.randn(B, H, S, D, device=device, dtype=torch.float16)
V = torch.randn(B, H, S, D, device=device, dtype=torch.float16)

# Compute reference
softmax_scale = 1.0 / math.sqrt(D)
with torch.inference_mode():
    ref = torch.nn.functional.scaled_dot_product_attention(
        Q, K, V, attn_mask=None, dropout_p=0.0, is_causal=False, scale=softmax_scale
    )

print("✓ Reference computed")

# Quantize
Q_q, Q_s = quantize_sim_fp8_per_head(Q)
K_q, K_s = quantize_sim_fp8_per_head(K)
V_q, V_s = quantize_sim_fp8_per_head(V)

print("✓ Quantization done")
print(f"  Q_s: {Q_s[0].item():.6f}")
print(f"  K_s: {K_s[0].item():.6f}")
print(f"  V_s: {V_s[0].item():.6f}")

# Run kernel
O_fp8 = ext.forward(Q_q, K_q, V_q, Q_s, K_s, V_s, softmax_scale)

print("✓ Kernel executed\n")

# Check results per head
print(f"{'='*80}")
print("Per-head error analysis:")
print(f"{'='*80}\n")

all_pass = True
for h in range(H):
    ref_h = ref[0, h]
    out_h = O_fp8[0, h]
    
    abs_diff = (out_h - ref_h).abs()
    max_err = abs_diff.max().item()
    mean_err = abs_diff.mean().item()
    pct_bad = (abs_diff > 0.05).float().mean().item() * 100
    
    status = "✅ PASS" if max_err < 0.05 and pct_bad < 1.0 else "❌ FAIL"
    print(f"Head {h}: max={max_err:.4f}, mean={mean_err:.4f}, %>0.05={pct_bad:.1f}% {status}")
    
    if status == "❌ FAIL":
        all_pass = False
        # Print first few bad elements
        bad_mask = abs_diff > 0.05
        if bad_mask.any():
            bad_idx = torch.where(bad_mask)
            print(f"  First bad elements:")
            for i in range(min(3, len(bad_idx[0]))):
                s, d = bad_idx[0][i].item(), bad_idx[1][i].item()
                print(f"    [{s},{d}]: out={out_h[s,d].item():.4f}, ref={ref_h[s,d].item():.4f}, err={abs_diff[s,d].item():.4f}")

print(f"\n{'='*80}")
if all_pass:
    print("✅ ALL HEADS PASS!")
else:
    print("❌ SOME HEADS FAILED - needs investigation")
print(f"{'='*80}\n")


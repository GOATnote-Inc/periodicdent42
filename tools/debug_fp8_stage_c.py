#!/usr/bin/env python3
"""
Debug FP8 Stage C WMMA kernel with detailed prints
Compiles with -DDEBUG_PRINT and runs small test shape
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "cudadent42"))

import torch
import torch.nn.functional as F
import math

# Compile with DEBUG_PRINT
print("=" * 80)
print("Compiling FP8 Stage C WMMA kernel with DEBUG_PRINT...")
print("=" * 80)

os.environ['TORCH_CUDA_ARCH_LIST'] = '8.9'

from torch.utils.cpp_extension import load
ext = load(
    name="sdpa_fp8_stage_c_wmma_debug",
    sources=[
        "cudadent42/bench/kernels/sdpa_fp8_stage_c_wmma.cu",
        "cudadent42/bench/kernels/sdpa_fp8_stage_c_wmma_bindings.cpp",
    ],
    extra_cuda_cflags=[
        "-O3",
        "-arch=sm_89",
        "--use_fast_math",
        "-lineinfo",
        "-DDEBUG_PRINT=1",  # Enable debug prints
    ],
    verbose=True,
)

print("\n" + "=" * 80)
print("Kernel compiled successfully!")
print("=" * 80)

# Import Python wrapper for quantization helper
from bench.sdpa_fp8_stage_c_wmma import quantize_sim_fp8_per_head

# Test with small shape
torch.manual_seed(42)
B, H, S, D = 1, 1, 32, 64  # Small for readable debug output
print(f"\nTest shape: (B={B}, H={H}, S={S}, D={D})")

Q = torch.randn(B, H, S, D, device="cuda", dtype=torch.float16)
K = torch.randn_like(Q)
V = torch.randn_like(Q)

# Show input samples
print("\n" + "=" * 80)
print("INPUT SAMPLES (FP16)")
print("=" * 80)
print(f"Q[0,0,0,0:5]: {Q[0,0,0,:5].cpu().tolist()}")
print(f"K[0,0,0,0:5]: {K[0,0,0,:5].cpu().tolist()}")
print(f"V[0,0,0,0:5]: {V[0,0,0,:5].cpu().tolist()}")

# Quantize
Q_q, Q_s = quantize_sim_fp8_per_head(Q)
K_q, K_s = quantize_sim_fp8_per_head(K)
V_q, V_s = quantize_sim_fp8_per_head(V)

print("\n" + "=" * 80)
print("QUANTIZED SAMPLES (FP8)")
print("=" * 80)
print(f"Q_q[0,0,0,0:5]: {Q_q[0,0,0,:5].cpu().tolist()}")
print(f"Q_s[0]: {Q_s[0].item():.6f}")
print(f"K_q[0,0,0,0:5]: {K_q[0,0,0,:5].cpu().tolist()}")
print(f"K_s[0]: {K_s[0].item():.6f}")
print(f"V_q[0,0,0,0:5]: {V_q[0,0,0,:5].cpu().tolist()}")
print(f"V_s[0]: {V_s[0].item():.6f}")

# Compute reference (PyTorch SDPA on FP16)
print("\n" + "=" * 80)
print("COMPUTING REFERENCE (PyTorch SDPA on FP16)...")
print("=" * 80)
ref = F.scaled_dot_product_attention(
    Q.float(), K.float(), V.float(), scale=1.0 / math.sqrt(D)
).to(torch.float16)
print(f"Reference output[0,0,0,0:5]: {ref[0,0,0,:5].cpu().tolist()}")

# Run FP8 kernel (will print debug info)
print("\n" + "=" * 80)
print("RUNNING FP8 KERNEL (check debug prints above)...")
print("=" * 80)
out = ext.forward(Q_q, K_q, V_q, Q_s, K_s, V_s, 1.0 / math.sqrt(D))

print("\n" + "=" * 80)
print("FP8 KERNEL OUTPUT")
print("=" * 80)
print(f"FP8 output[0,0,0,0:5]: {out[0,0,0,:5].cpu().tolist()}")

# Compare
diff = (out - ref).abs()
print("\n" + "=" * 80)
print("COMPARISON")
print("=" * 80)
print(f"Max abs error: {diff.max().item():.4e}")
print(f"Mean abs error: {diff.mean().item():.4e}")
print(f"% elements > 0.05: {(diff > 0.05).float().mean().item() * 100:.1f}%")

# Show element-by-element comparison for first 5 elements
print("\n" + "=" * 80)
print("ELEMENT-BY-ELEMENT (first 5)")
print("=" * 80)
for i in range(5):
    fp8_val = out[0,0,0,i].item()
    ref_val = ref[0,0,0,i].item()
    err = abs(fp8_val - ref_val)
    print(f"  [{i}] FP8: {fp8_val:+.4f}  Ref: {ref_val:+.4f}  Err: {err:.4e}")

# Manual Q@K^T computation for first element
print("\n" + "=" * 80)
print("MANUAL VERIFICATION: Q@K^T[0,0]")
print("=" * 80)
q_row = Q[0,0,0].float()
k_col = K[0,0,0].float()
manual_score = (q_row @ k_col).item() / math.sqrt(D)
print(f"Manual Q[0] @ K[0]^T / sqrt(D) = {manual_score:.4f}")

# Compute full attention manually for row 0
scores = (Q[0,0,0:1].float() @ K[0,0].float().T) / math.sqrt(D)  # [1, S]
probs = torch.softmax(scores, dim=-1)  # [1, S]
manual_out = (probs @ V[0,0].float()).squeeze(0)  # [D]
print(f"\nManual attention output[0:5]: {manual_out[:5].tolist()}")
print(f"PyTorch reference[0:5]:       {ref[0,0,0,:5].cpu().tolist()}")
print(f"Match: {torch.allclose(manual_out, ref[0,0,0].cpu().float(), atol=1e-3)}")

print("\n" + "=" * 80)
print("DEBUG SESSION COMPLETE")
print("=" * 80)
print("\nCheck debug prints above to identify where values diverge!")


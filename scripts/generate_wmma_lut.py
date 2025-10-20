#!/usr/bin/env python3
"""Generate WMMA 16x16 accumulator layout LUT header."""
import os
import sys
import torch
from pathlib import Path

# Add repo root to path
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

def generate_lut():
    from torch.utils.cpp_extension import load
    
    src_cu = repo_root / "cudadent42/bench/kernels/wmma16x16_accum_lut_gen.cu"
    src_cpp = repo_root / "cudadent42/bench/kernels/wmma16x16_accum_lut_bindings.cpp"
    print(f"Building introspection kernel...")
    
    mod = load(
        name="wmma_lut_gen",
        sources=[str(src_cu), str(src_cpp)],
        extra_cuda_cflags=["-O3", "-arch=sm_89"],
        verbose=True
    )
    
    print("Running introspection kernel...")
    out = torch.empty((16, 16), device="cuda", dtype=torch.float32)
    mod.wmma_accum_introspect_kernel(out)
    m = out.cpu().numpy()
    
    # Invert mapping -> for each lane slot (8), find (r,c)
    print("Building LUT...")
    lut = [[] for _ in range(32)]
    for r in range(16):
        for c in range(16):
            val = int(m[r, c] + 0.5)
            lane = val // 8
            slot = val % 8
            if 0 <= lane < 32 and 0 <= slot < 8:
                lut[lane].append((r, c))
    
    # Sanity: each lane must have exactly 8 pairs
    for lane in range(32):
        if len(lut[lane]) != 8:
            raise ValueError(f"LUT build failed: lane {lane} has {len(lut[lane])} elements (expected 8)")
    
    print("✅ LUT validated (32 lanes × 8 elements each)")
    
    # Generate header
    hdr = [
        "#pragma once",
        "// Auto-generated WMMA 16x16 accumulator layout (lane -> (row,col) x8)",
        "// Each of 32 lanes owns 8 elements of the 16×16 accumulator fragment",
        "static __device__ __constant__ int WMMA_ACCUM_LUT[32][8][2] = {"
    ]
    
    for lane in range(32):
        line = "  { " + ", ".join(f"{{{r},{c}}}" for (r, c) in lut[lane]) + " },"
        hdr.append(line)
    
    hdr.append("};")
    
    # Write header
    output_path = repo_root / "cudadent42/bench/kernels/wmma16x16_accum_lut.h"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(hdr) + "\n")
    
    print(f"✅ Generated: {output_path}")
    print(f"   Size: {output_path.stat().st_size} bytes")
    
    # Print sample for verification
    print("\nSample (lane 0):")
    for i, (r, c) in enumerate(lut[0]):
        print(f"  slot {i}: ({r}, {c})")

if __name__ == "__main__":
    generate_lut()


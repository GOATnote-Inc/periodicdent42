#!/usr/bin/env python3
"""
Test P@V with serial (non-atomic) accumulation to check if atomicAdd is the issue
"""

import torch
import subprocess
import sys

print("=" * 80)
print("Testing P@V: Serial vs Parallel Accumulation")
print("=" * 80)

# First, modify kernel to make only ONE warp do P@V (warp 0)
kernel_path = "/home/kiteboard/flashcore/kernels/flashcore_fused_wmma.cu"

# Read kernel
with open(kernel_path, 'r') as f:
    kernel_code = f.read()

# Check if already modified
if "// SERIAL_PV_TEST" in kernel_code:
    print("✅ Kernel already modified for serial test")
else:
    print("Modifying kernel to use single warp for P@V...")
    
    # Find the line "if (warp_valid) {" in P@V section (around line 405)
    # Replace with "if (warp_valid && warp_id == 0) {" to make only warp 0 work
    
    kernel_code = kernel_code.replace(
        "// Partition K (the N dimension) across warp_n to avoid double-counting",
        "// SERIAL_PV_TEST: Use only warp 0 to eliminate atomicAdd races\n        // Partition K (the N dimension) across warp_n to avoid double-counting"
    )
    
    kernel_code = kernel_code.replace(
        "if (warp_valid) {\n            // Each warp computes",
        "if (warp_valid && warp_id == 0) {  // SERIAL_PV_TEST: Only warp 0\n            // Each warp computes"
    )
    
    with open(kernel_path, 'w') as f:
        f.write(kernel_code)
    
    print("✅ Modified kernel")

# Rebuild and test
print("\nRebuilding kernel...")
result = subprocess.run(
    ["rm", "-rf", "~/.cache/torch_extensions"],
    capture_output=True
)

result = subprocess.run(
    ["python3", "test_pv_only.py"],
    capture_output=True,
    text=True
)

print(result.stdout)
if result.stderr:
    print("STDERR:", result.stderr)

sys.exit(result.returncode)


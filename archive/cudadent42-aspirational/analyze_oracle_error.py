#!/usr/bin/env python3
"""
Analyze V3 oracle test error pattern to identify the bug
"""

import numpy as np
import json
from pathlib import Path

# Load numpy arrays
artifacts_dir = Path("artifacts/oracle/noncausal")
O_ref = np.load(artifacts_dir / "v3_config0_O_ref.npy")  # SDPA
O_test = np.load(artifacts_dir / "v3_config0_O_test.npy")  # V3

print("=" * 80)
print("V3 Error Pattern Analysis")
print("=" * 80)

print(f"\nShapes: O_ref={O_ref.shape}, O_test={O_test.shape}")
print(f"O_ref range: [{O_ref.min():.6f}, {O_ref.max():.6f}]")
print(f"O_test range: [{O_test.min():.6f}, {O_test.max():.6f}]")

# Compute differences
abs_diff = np.abs(O_test - O_ref)
rel_diff = abs_diff / (np.abs(O_ref) + 1e-8)

print(f"\nMax abs diff: {abs_diff.max():.6f}")
print(f"Mean abs diff: {abs_diff.mean():.6f}")

# Analyze per-row error
row_max_diff = abs_diff.max(axis=1)  # Max error per row
row_mean_diff = abs_diff.mean(axis=1)  # Mean error per row

print(f"\nPer-row error statistics:")
print(f"  Max of row_max: {row_max_diff.max():.6f}")
print(f"  Mean of row_max: {row_max_diff.mean():.6f}")
print(f"  Rows with error > 0.1: {(row_max_diff > 0.1).sum()} / {len(row_max_diff)}")

# Find worst rows
worst_rows = np.argsort(-row_max_diff)[:10]
print(f"\nTop 10 worst rows:")
for i, row_idx in enumerate(worst_rows):
    print(f"  #{i+1}: Row {row_idx:3d}, max_diff={row_max_diff[row_idx]:.6f}")

# Analyze magnitude ratio
magnitude_ref = np.sqrt((O_ref ** 2).sum(axis=1))  # L2 norm per row
magnitude_test = np.sqrt((O_test ** 2).sum(axis=1))
magnitude_ratio = magnitude_test / (magnitude_ref + 1e-8)

print(f"\nMagnitude ratio (test/ref) per row:")
print(f"  Mean: {magnitude_ratio.mean():.6f}")
print(f"  Std: {magnitude_ratio.std():.6f}")
print(f"  Min: {magnitude_ratio.min():.6f}")
print(f"  Max: {magnitude_ratio.max():.6f}")

# Check if there's a systematic scaling issue
if magnitude_ratio.std() < 0.1:
    print(f"\n⚠️  SYSTEMATIC SCALING: All rows scaled by ~{magnitude_ratio.mean():.3f}×")
    print(f"    Likely bug: Final normalization or online softmax accumulation")
else:
    print(f"\n  No systematic scaling (high variance)")

# Analyze spatial pattern (is error concentrated at beginning/end?)
S = O_ref.shape[0]
first_quarter = row_max_diff[:S//4].mean()
second_quarter = row_max_diff[S//4:S//2].mean()
third_quarter = row_max_diff[S//2:3*S//4].mean()
fourth_quarter = row_max_diff[3*S//4:].mean()

print(f"\nSpatial error pattern (by quarter):")
print(f"  First quarter (0-127): {first_quarter:.6f}")
print(f"  Second quarter (128-255): {second_quarter:.6f}")
print(f"  Third quarter (256-383): {third_quarter:.6f}")
print(f"  Fourth quarter (384-511): {fourth_quarter:.6f}")

if fourth_quarter > 2 * first_quarter:
    print(f"\n⚠️  ERROR GROWS TOWARD END: Likely online softmax accumulation bug")
    print(f"    Bug hypothesis: m_i or l_i not updated correctly across tiles")
elif first_quarter > 2 * fourth_quarter:
    print(f"\n⚠️  ERROR HIGHEST AT START: Likely initialization bug")
else:
    print(f"\n  Error fairly uniform across sequence")

# Check for NaN/Inf
print(f"\nNaN/Inf check:")
print(f"  O_ref: NaN={np.isnan(O_ref).sum()}, Inf={np.isinf(O_ref).sum()}")
print(f"  O_test: NaN={np.isnan(O_test).sum()}, Inf={np.isinf(O_test).sum()}")

# Analyze per-dimension error
dim_max_diff = abs_diff.max(axis=0)  # Max error per dimension
print(f"\nPer-dimension error:")
print(f"  Max of dim_max: {dim_max_diff.max():.6f}")
print(f"  Mean of dim_max: {dim_max_diff.mean():.6f}")
print(f"  Dims with error > 0.1: {(dim_max_diff > 0.1).sum()} / {len(dim_max_diff)}")

# Worst dimensions
worst_dims = np.argsort(-dim_max_diff)[:5]
print(f"\nTop 5 worst dimensions:")
for i, dim_idx in enumerate(worst_dims):
    print(f"  #{i+1}: Dim {dim_idx:2d}, max_diff={dim_max_diff[dim_idx]:.6f}")

print("\n" + "=" * 80)
print("Analysis complete")
print("=" * 80)

# Save analysis
analysis = {
    "max_abs_diff": float(abs_diff.max()),
    "mean_abs_diff": float(abs_diff.mean()),
    "magnitude_ratio": {
        "mean": float(magnitude_ratio.mean()),
        "std": float(magnitude_ratio.std()),
        "min": float(magnitude_ratio.min()),
        "max": float(magnitude_ratio.max()),
    },
    "spatial_pattern": {
        "first_quarter": float(first_quarter),
        "second_quarter": float(second_quarter),
        "third_quarter": float(third_quarter),
        "fourth_quarter": float(fourth_quarter),
    },
    "worst_rows": [int(x) for x in worst_rows[:10]],
    "systematic_scaling": bool(magnitude_ratio.std() < 0.1),
    "error_grows_toward_end": bool(fourth_quarter > 2 * first_quarter),
}

with open(artifacts_dir / "error_analysis.json", "w") as f:
    json.dump(analysis, f, indent=2)

print(f"\nAnalysis saved to: {artifacts_dir / 'error_analysis.json'}")


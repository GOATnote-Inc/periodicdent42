#!/usr/bin/env python3
"""
Test MgB₂ Multi-Band Model (v0.4.4)

Quick validation of multi-band implementation before full calibration.
"""

import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.src.htc.structure_utils import composition_to_structure, estimate_material_properties
from app.src.htc.domain import allen_dynes_tc

print("╔══════════════════════════════════════════════════════════╗")
print("║  MgB₂ Multi-Band Model Test (v0.4.4)                    ║")
print("╚══════════════════════════════════════════════════════════╝\n")

# Known experimental value
TC_EXP = 39.0  # K

# Get structure
structure = composition_to_structure("MgB2")
if structure is None:
    print("❌ Failed to create MgB₂ structure")
    sys.exit(1)

print("✅ Structure created")

# Estimate properties (should use multi-band model)
lambda_ep, omega_log, avg_mass = estimate_material_properties(structure, "MgB2")

print(f"\n📊 ESTIMATED PROPERTIES:")
print(f"   λ_eff: {lambda_ep:.4f}")
print(f"   ω_log: {omega_log:.1f} K")
print(f"   mass: {avg_mass:.2f} amu")

# Predict Tc
tc_pred = allen_dynes_tc(omega_log, lambda_ep, mu_star=0.13)

print(f"\n🎯 RESULTS:")
print(f"   Tc_exp:  {TC_EXP:.1f} K")
print(f"   Tc_pred: {tc_pred:.1f} K")
print(f"   Error:   {tc_pred - TC_EXP:+.1f} K ({(tc_pred - TC_EXP) / TC_EXP * 100:+.1f}%)")

# Target assessment
abs_error_pct = abs((tc_pred - TC_EXP) / TC_EXP * 100)
if abs_error_pct <= 55:
    print(f"\n✅ TARGET MET: {abs_error_pct:.1f}% ≤ 55%")
    sys.exit(0)
elif abs_error_pct <= 70:
    print(f"\n⚠️  NEEDS TUNING: {abs_error_pct:.1f}% (55-70% range)")
    print("   Recommendation: Adjust σ/π weights (try 0.8/0.2)")
    sys.exit(1)
else:
    print(f"\n❌ ABORT: {abs_error_pct:.1f}% > 70%")
    print("   Multi-band model not converging - review physics")
    sys.exit(2)


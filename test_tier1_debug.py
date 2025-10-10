#!/usr/bin/env python3
"""
Quick debug script to test Tier 1 calibration prediction logic.
"""

import sys
sys.path.insert(0, "/Users/kiteboard/periodicdent42")

from app.src.htc.domain import allen_dynes_tc
from app.src.htc.structure_utils import composition_to_structure, estimate_material_properties

# Test materials
test_materials = [
    ("Nb", 9.25),
    ("MgB2", 39.00),
    ("Nb3Sn", 18.05),
]

print("╔══════════════════════════════════════════════════════════╗")
print("║       Tier 1 Calibration Debug Test                    ║")
print("╚══════════════════════════════════════════════════════════╝\n")

for composition, tc_exp in test_materials:
    print(f"Testing: {composition} (Tc_exp = {tc_exp} K)")
    
    # Step 1: Create structure
    structure = composition_to_structure(composition)
    if structure is None:
        print(f"  ❌ Structure creation failed for {composition}\n")
        continue
    
    print(f"  ✅ Structure created: {structure.composition.reduced_formula}")
    
    # Step 2: Estimate properties
    lambda_ep, omega_log, avg_mass = estimate_material_properties(structure, composition)
    print(f"  📊 Estimated properties:")
    print(f"     λ = {lambda_ep:.4f}")
    print(f"     ω = {omega_log:.2f} K")
    print(f"     mass = {avg_mass:.2f} amu")
    
    # Step 3: Predict Tc (NOTE: omega_log comes FIRST in function signature!)
    tc_pred = allen_dynes_tc(omega_log, lambda_ep, mu_star=0.13)
    print(f"  🎯 Allen-Dynes Tc = {tc_pred:.2f} K")
    
    error = tc_pred - tc_exp
    error_pct = (error / tc_exp) * 100
    print(f"  📈 Error: {error:+.2f} K ({error_pct:+.1f}%)\n")

print("\n" + "="*60)


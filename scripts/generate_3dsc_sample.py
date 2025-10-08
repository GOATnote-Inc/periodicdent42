#!/usr/bin/env python3
"""Generate representative superconductor dataset for demonstration.

This creates a curated dataset with famous superconductors and variations,
suitable for demonstrating the Discovery Kernel architecture without
requiring large dataset downloads.

Contains:
- High-Tc cuprates (YBa2Cu3O7, HgBa2Ca2Cu3O8, etc.)
- Iron-based superconductors
- Conventional superconductors (MgB2, Nb3Sn)
- High-pressure hydrides (LaH10, H3S)
- Non-superconductors for classification

Usage:
    python scripts/generate_3dsc_sample.py
"""

import pandas as pd
import numpy as np
import pathlib
import json

# Set seed for reproducibility
np.random.seed(42)

# Famous superconductors with known Tc values (curated from literature)
LANDMARK_MATERIALS = [
    # High-Tc cuprates (discovered 1986-1993)
    {"formula": "YBa2Cu3O7", "Tc": 92.0, "class": "cuprate", "year": 1987, "ambient_pressure": True},
    {"formula": "Bi2Sr2CaCu2O8", "Tc": 95.0, "class": "cuprate", "year": 1988, "ambient_pressure": True},
    {"formula": "HgBa2Ca2Cu3O8", "Tc": 133.0, "class": "cuprate", "year": 1993, "ambient_pressure": True},
    {"formula": "Tl2Ba2Ca2Cu3O10", "Tc": 125.0, "class": "cuprate", "year": 1988, "ambient_pressure": True},
    {"formula": "La2CuO4", "Tc": 35.0, "class": "cuprate", "year": 1986, "ambient_pressure": True},
    
    # Iron-based superconductors (discovered 2008+)
    {"formula": "SmFeAsO", "Tc": 55.0, "class": "iron_based", "year": 2008, "ambient_pressure": True},
    {"formula": "Ba0.6K0.4Fe2As2", "Tc": 38.0, "class": "iron_based", "year": 2008, "ambient_pressure": True},
    {"formula": "FeSe", "Tc": 8.0, "class": "iron_based", "year": 2008, "ambient_pressure": True},
    
    # Conventional low-Tc BCS superconductors
    {"formula": "MgB2", "Tc": 39.0, "class": "conventional", "year": 2001, "ambient_pressure": True},
    {"formula": "Nb3Sn", "Tc": 18.3, "class": "conventional", "year": 1954, "ambient_pressure": True},
    {"formula": "NbTi", "Tc": 9.5, "class": "conventional", "year": 1962, "ambient_pressure": True},
    {"formula": "Pb", "Tc": 7.2, "class": "conventional", "year": 1913, "ambient_pressure": True},
    {"formula": "Nb", "Tc": 9.3, "class": "conventional", "year": 1930, "ambient_pressure": True},
    {"formula": "V3Si", "Tc": 17.0, "class": "conventional", "year": 1953, "ambient_pressure": True},
    
    # High-pressure hydrides (discovered 2015+)
    {"formula": "LaH10", "Tc": 250.0, "class": "hydride", "year": 2018, "ambient_pressure": False, "pressure_GPa": 170},
    {"formula": "H3S", "Tc": 203.0, "class": "hydride", "year": 2015, "ambient_pressure": False, "pressure_GPa": 155},
    {"formula": "CaH6", "Tc": 215.0, "class": "hydride", "year": 2020, "ambient_pressure": False, "pressure_GPa": 172},
    {"formula": "YH9", "Tc": 243.0, "class": "hydride", "year": 2021, "ambient_pressure": False, "pressure_GPa": 201},
    
    # Non-superconductors (control group)
    {"formula": "Cu", "Tc": 0.0, "class": "non_SC", "year": None, "ambient_pressure": True},
    {"formula": "Au", "Tc": 0.0, "class": "non_SC", "year": None, "ambient_pressure": True},
    {"formula": "Fe", "Tc": 0.0, "class": "non_SC", "year": None, "ambient_pressure": True},
    {"formula": "Al2O3", "Tc": 0.0, "class": "non_SC", "year": None, "ambient_pressure": True},
    {"formula": "SiO2", "Tc": 0.0, "class": "non_SC", "year": None, "ambient_pressure": True},
]


def generate_variations(base_material, n_variations=30):
    """Generate synthetic variations around a known material.
    
    Simulates experimental uncertainty and doping variations.
    
    Args:
        base_material: Dict with material properties
        n_variations: Number of variations to generate
    
    Returns:
        List of variation dicts
    """
    variations = []
    
    if base_material['Tc'] == 0.0:
        # Non-superconductors: small chance of very low Tc
        for i in range(n_variations):
            tc_var = max(0.0, np.random.exponential(0.5))  # Mostly 0, rare low values
            variations.append({
                'formula': f"{base_material['formula']}_var{i:03d}",
                'Tc': round(tc_var, 2),
                'class': base_material['class'],
                'base_material': base_material['formula'],
                'variation_type': 'doping' if tc_var > 0 else 'none',
                'ambient_pressure': base_material['ambient_pressure'],
            })
    else:
        # Superconductors: variations with Gaussian noise
        for i in range(n_variations):
            # Tc variations depend on material class
            if base_material['class'] == 'cuprate':
                # Cuprates: large variations due to doping sensitivity
                noise_std = 0.15 * base_material['Tc']
            elif base_material['class'] == 'hydride':
                # Hydrides: pressure-dependent variations
                noise_std = 0.10 * base_material['Tc']
            else:
                # Conventional: smaller variations
                noise_std = 0.08 * base_material['Tc']
            
            tc_var = base_material['Tc'] + np.random.normal(0, noise_std)
            tc_var = max(0.0, tc_var)  # No negative Tc
            
            variations.append({
                'formula': f"{base_material['formula']}_var{i:03d}",
                'Tc': round(tc_var, 2),
                'class': base_material['class'],
                'base_material': base_material['formula'],
                'variation_type': np.random.choice(['doping', 'stoichiometry', 'synthesis']),
                'ambient_pressure': base_material['ambient_pressure'],
            })
            
            # Add pressure info for hydrides
            if base_material['class'] == 'hydride':
                variations[-1]['pressure_GPa'] = base_material.get('pressure_GPa', 0) + np.random.normal(0, 10)
    
    return variations


def assign_tc_class(tc):
    """Assign Tc class for stratified analysis.
    
    Args:
        tc: Critical temperature in Kelvin
    
    Returns:
        String class label
    """
    if tc == 0.0:
        return 'non_SC'
    elif tc < 30.0:
        return 'low_Tc'
    elif tc < 77.0:  # Liquid nitrogen temperature
        return 'mid_Tc'
    else:
        return 'high_Tc'


def main():
    """Generate curated superconductor dataset."""
    print("\n" + "="*80)
    print("GENERATING CURATED SUPERCONDUCTOR DATASET".center(80))
    print("="*80 + "\n")
    
    # Create base dataset
    base_df = pd.DataFrame(LANDMARK_MATERIALS)
    print(f"📊 Landmark materials: {len(base_df)}")
    print(f"   Cuprates: {(base_df['class'] == 'cuprate').sum()}")
    print(f"   Iron-based: {(base_df['class'] == 'iron_based').sum()}")
    print(f"   Conventional: {(base_df['class'] == 'conventional').sum()}")
    print(f"   Hydrides: {(base_df['class'] == 'hydride').sum()}")
    print(f"   Non-SC: {(base_df['class'] == 'non_SC').sum()}")
    print()
    
    # Generate variations
    print("🔬 Generating variations...")
    all_variations = []
    for _, material in base_df.iterrows():
        material_dict = material.to_dict()
        variations = generate_variations(material_dict, n_variations=30)
        all_variations.extend(variations)
    
    variations_df = pd.DataFrame(all_variations)
    print(f"   Generated {len(variations_df)} variations")
    print()
    
    # Combine
    final_df = pd.concat([base_df, variations_df], ignore_index=True)
    
    # Add Tc class
    final_df['tc_class'] = final_df['Tc'].apply(assign_tc_class)
    
    # Add sample IDs
    final_df['sample_id'] = [f"SC-{i:05d}" for i in range(len(final_df))]
    
    # Reorder columns
    columns_order = ['sample_id', 'formula', 'Tc', 'tc_class', 'class', 'ambient_pressure']
    other_columns = [c for c in final_df.columns if c not in columns_order]
    final_df = final_df[columns_order + other_columns]
    
    # Statistics
    print("="*80)
    print("DATASET STATISTICS".center(80))
    print("="*80 + "\n")
    print(f"Total samples: {len(final_df)}")
    print(f"Max Tc: {final_df['Tc'].max():.1f}K ({final_df.loc[final_df['Tc'].idxmax(), 'formula']})")
    print(f"Min Tc (superconductors): {final_df[final_df['Tc'] > 0]['Tc'].min():.1f}K")
    print(f"Mean Tc (superconductors): {final_df[final_df['Tc'] > 0]['Tc'].mean():.1f}K")
    print()
    
    print("Tc Class Distribution:")
    for tc_class in ['non_SC', 'low_Tc', 'mid_Tc', 'high_Tc']:
        count = (final_df['tc_class'] == tc_class).sum()
        pct = 100 * count / len(final_df)
        print(f"  {tc_class:<10}: {count:4d} ({pct:5.1f}%)")
    print()
    
    print("Material Class Distribution:")
    for mat_class in final_df['class'].unique():
        count = (final_df['class'] == mat_class).sum()
        pct = 100 * count / len(final_df)
        print(f"  {mat_class:<15}: {count:4d} ({pct:5.1f}%)")
    print()
    
    # Save
    output_path = pathlib.Path("data/superconductors/processed/demo_dataset.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(output_path, index=False)
    
    print(f"💾 Saved to: {output_path}")
    print(f"   Size: {output_path.stat().st_size / 1024:.1f} KB")
    
    # Save metadata
    metadata = {
        "dataset_name": "Curated Superconductor Dataset",
        "version": "1.0",
        "description": "Representative dataset with landmark superconductors and variations",
        "n_samples": len(final_df),
        "n_landmark": len(base_df),
        "n_variations": len(variations_df),
        "tc_range": [float(final_df['Tc'].min()), float(final_df['Tc'].max())],
        "classes": {
            "tc_classes": final_df['tc_class'].value_counts().to_dict(),
            "material_classes": final_df['class'].value_counts().to_dict(),
        },
        "landmark_materials": base_df[['formula', 'Tc', 'class', 'year']].to_dict('records'),
    }
    
    metadata_path = output_path.parent / "demo_dataset_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"💾 Metadata: {metadata_path}")
    
    print("\n" + "="*80)
    print("✅ DATASET GENERATION COMPLETE".center(80))
    print("="*80 + "\n")
    
    print("Sample entries:")
    print(final_df[['sample_id', 'formula', 'Tc', 'tc_class', 'class']].head(10).to_string(index=False))
    print()
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())


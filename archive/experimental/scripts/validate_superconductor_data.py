#!/usr/bin/env python3
"""Validate UCI Superconductor Dataset integrity and metadata.

Verifies the UCI dataset has expected structure and statistics.
Generates provenance metadata for DTP records.

Dataset: UCI Machine Learning Repository
Source: https://archive.ics.uci.edu/dataset/464/superconductivty+data
Citation: Hamidieh, K. (2018). A data-driven statistical model for predicting 
         the critical temperature of a superconductor. Computational Materials Science.

Usage:
    python scripts/validate_superconductor_data.py
"""

import pandas as pd
import pathlib
import json
import hashlib
from datetime import datetime, timezone

def compute_file_hash(filepath):
    """Compute MD5 hash of file."""
    hasher = hashlib.md5()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            hasher.update(chunk)
    return hasher.hexdigest()


def validate_uci_dataset(data_path):
    """Validate UCI Superconductor dataset.
    
    Args:
        data_path: Path to train.csv
    
    Returns:
        Dict with validation results and metadata
    """
    print("\n" + "="*80)
    print("UCI SUPERCONDUCTOR DATASET VALIDATOR".center(80))
    print("="*80 + "\n")
    
    # Load dataset
    print(f"📂 Loading dataset: {data_path}")
    df = pd.read_csv(data_path)
    print(f"   Loaded {len(df):,} samples × {df.shape[1]} features")
    print()
    
    # Validation checks
    checks = {}
    
    # Check 1: Expected number of samples
    expected_min = 21000
    expected_max = 22000
    checks['sample_count'] = {
        'expected_range': [expected_min, expected_max],
        'actual': int(len(df)),
        'passed': bool(expected_min <= len(df) <= expected_max)
    }
    print(f"✓ Sample count: {len(df):,} (expected {expected_min:,}-{expected_max:,})")
    
    # Check 2: Tc range
    tc_min, tc_max = df['critical_temp'].min(), df['critical_temp'].max()
    checks['tc_range'] = {
        'min_K': float(tc_min),
        'max_K': float(tc_max),
        'passed': bool(0 <= tc_min and tc_max <= 200)  # Reasonable physical range
    }
    print(f"✓ Tc range: {tc_min:.1f}K - {tc_max:.1f}K")
    
    # Check 3: Feature count
    expected_features = 81  # 81 features + 1 target (critical_temp)
    checks['feature_count'] = {
        'expected': expected_features,
        'actual': int(df.shape[1] - 1),  # Exclude critical_temp
        'passed': bool(df.shape[1] == expected_features + 1)
    }
    print(f"✓ Features: {df.shape[1] - 1} (expected {expected_features})")
    
    # Check 4: No missing values
    missing = df.isnull().sum().sum()
    checks['missing_values'] = {
        'count': int(missing),
        'passed': bool(missing == 0)
    }
    print(f"✓ Missing values: {missing}")
    
    # Check 5: Tc distribution
    tc_distribution = {
        'non_SC': int((df['critical_temp'] == 0).sum()),
        'low_Tc_0_30K': int(((df['critical_temp'] > 0) & (df['critical_temp'] <= 30)).sum()),
        'mid_Tc_30_77K': int(((df['critical_temp'] > 30) & (df['critical_temp'] <= 77)).sum()),
        'high_Tc_77K_plus': int((df['critical_temp'] > 77).sum()),
    }
    checks['tc_distribution'] = {
        'distribution': tc_distribution,
        'passed': True  # Informational only
    }
    print(f"✓ Tc distribution:")
    for category, count in tc_distribution.items():
        pct = 100 * count / len(df)
        print(f"    {category:<20}: {count:5,} ({pct:5.1f}%)")
    
    print()
    
    # Compute dataset hash
    dataset_hash = compute_file_hash(data_path)
    print(f"📊 Dataset ID (MD5): {dataset_hash}")
    print()
    
    # Statistics
    stats = {
        'n_samples': int(len(df)),
        'n_features': int(df.shape[1] - 1),
        'tc_statistics': {
            'min': float(df['critical_temp'].min()),
            'max': float(df['critical_temp'].max()),
            'mean': float(df['critical_temp'].mean()),
            'median': float(df['critical_temp'].median()),
            'std': float(df['critical_temp'].std()),
        },
        'tc_distribution': checks['tc_distribution']['distribution'],
        'feature_names': df.columns.tolist(),
    }
    
    # Overall validation
    all_passed = all(check.get('passed', False) for check in checks.values())
    
    print("="*80)
    if all_passed:
        print("✅ ALL VALIDATION CHECKS PASSED".center(80))
    else:
        print("⚠️  SOME VALIDATION CHECKS FAILED".center(80))
    print("="*80)
    print()
    
    # Generate metadata for provenance
    metadata = {
        'dataset_name': 'UCI Superconductor Dataset',
        'dataset_id': dataset_hash,
        'source': 'UCI Machine Learning Repository',
        'url': 'https://archive.ics.uci.edu/dataset/464/superconductivty+data',
        'citation': 'Hamidieh, K. (2018). A data-driven statistical model for predicting the critical temperature of a superconductor. Computational Materials Science, 154, 346-354.',
        'doi': '10.1016/j.commatsci.2018.07.052',
        'validated_at': datetime.now(timezone.utc).isoformat(),
        'validation_checks': checks,
        'statistics': stats,
        'validated': all_passed,
    }
    
    return metadata


def main():
    """Main validator entry point."""
    data_path = pathlib.Path('data/superconductors/raw/train.csv')
    
    if not data_path.exists():
        print(f"❌ Dataset not found: {data_path}")
        print("   Run: python scripts/download_uci_dataset.py")
        return 1
    
    # Validate
    metadata = validate_uci_dataset(data_path)
    
    # Save metadata
    output_path = pathlib.Path('data/superconductors/processed/uci_metadata.json')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"💾 Metadata saved: {output_path}")
    print()
    
    # Save dataset_id separately for easy access
    dataset_id_path = pathlib.Path('data/superconductors/raw/train.csv.md5')
    with open(dataset_id_path, 'w') as f:
        f.write(metadata['dataset_id'])
    
    print(f"📌 Dataset ID saved: {dataset_id_path}")
    print(f"   Use this as dataset_id in DTP records: {metadata['dataset_id']}")
    print()
    
    return 0 if metadata['validated'] else 1


if __name__ == '__main__':
    import sys
    sys.exit(main())


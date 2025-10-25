#!/usr/bin/env python3
"""
Download and split UCI Superconductivity Dataset for validation.

Dataset: UCI Superconductivity Data (21,263 compounds)
Source: https://archive.ics.uci.edu/dataset/464/superconductivty+data
License: CC BY 4.0

Usage:
    python scripts/download_uci_data.py --output data/processed/uci_splits/
"""

import argparse
import json
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

try:
    from ucimlrepo import fetch_ucirepo
    UCI_AVAILABLE = True
except ImportError:
    UCI_AVAILABLE = False
    print("⚠️  ucimlrepo not installed. Install with: pip install ucimlrepo")


def download_uci_superconductivity() -> Tuple[pd.DataFrame, pd.Series]:
    """
    Download UCI Superconductivity Dataset.
    
    Returns:
        X: Features (81 compositional descriptors)
        y: Target (critical temperature in Kelvin)
    """
    if not UCI_AVAILABLE:
        raise ImportError(
            "ucimlrepo package required. Install with: pip install ucimlrepo"
        )
    
    print("📥 Downloading UCI Superconductivity Dataset...")
    
    # Fetch dataset (ID=464)
    dataset = fetch_ucirepo(id=464)
    
    X = dataset.data.features
    y = dataset.data.targets.squeeze()  # Convert to Series
    
    print(f"✅ Downloaded {len(X)} compounds with {X.shape[1]} features")
    print(f"   Tc range: [{y.min():.1f}, {y.max():.1f}] K")
    print(f"   Mean Tc: {y.mean():.1f} K, Std: {y.std():.1f} K")
    
    return X, y


def stratified_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42
) -> dict:
    """
    Create stratified train/val/test splits by Tc quartiles.
    
    Args:
        X: Features
        y: Target (Tc)
        test_size: Fraction for test set
        val_size: Fraction for validation set
        random_state: Random seed for reproducibility
    
    Returns:
        Dictionary with 'train', 'val', 'test' DataFrames
    """
    # Create stratification bins (Tc quartiles)
    y_bins = pd.qcut(y, q=4, labels=False, duplicates='drop')
    
    # Split: train+val vs test
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y_bins, random_state=random_state
    )
    
    # Recompute bins for train+val
    y_trainval_bins = pd.qcut(y_trainval, q=4, labels=False, duplicates='drop')
    
    # Split: train vs val
    val_size_adjusted = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=val_size_adjusted,
        stratify=y_trainval_bins, random_state=random_state
    )
    
    # Combine into DataFrames
    train_df = X_train.copy()
    train_df['critical_temp'] = y_train.values
    
    val_df = X_val.copy()
    val_df['critical_temp'] = y_val.values
    
    test_df = X_test.copy()
    test_df['critical_temp'] = y_test.values
    
    print(f"📊 Split sizes:")
    print(f"   Train: {len(train_df)} ({len(train_df)/len(X)*100:.1f}%)")
    print(f"   Val:   {len(val_df)} ({len(val_df)/len(X)*100:.1f}%)")
    print(f"   Test:  {len(test_df)} ({len(test_df)/len(X)*100:.1f}%)")
    
    # Verify no overlap
    train_idx = set(train_df.index)
    val_idx = set(val_df.index)
    test_idx = set(test_df.index)
    
    assert len(train_idx & val_idx) == 0, "Train-val overlap detected!"
    assert len(train_idx & test_idx) == 0, "Train-test overlap detected!"
    assert len(val_idx & test_idx) == 0, "Val-test overlap detected!"
    
    print("✅ No data leakage detected (zero index overlap)")
    
    return {
        'train': train_df,
        'val': val_df,
        'test': test_df
    }


def save_splits(splits: dict, output_dir: Path):
    """
    Save train/val/test splits to CSV files.
    
    Args:
        splits: Dictionary with 'train', 'val', 'test' DataFrames
        output_dir: Output directory
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for split_name, df in splits.items():
        output_path = output_dir / f"{split_name}.csv"
        df.to_csv(output_path, index=False)
        print(f"💾 Saved {split_name} split: {output_path} ({len(df)} samples)")
    
    # Save metadata
    metadata = {
        'dataset': 'UCI Superconductivity',
        'source': 'https://archive.ics.uci.edu/dataset/464/superconductivty+data',
        'license': 'CC BY 4.0',
        'n_total': sum(len(df) for df in splits.values()),
        'n_train': len(splits['train']),
        'n_val': len(splits['val']),
        'n_test': len(splits['test']),
        'n_features': len(splits['train'].columns) - 1,  # Exclude target
        'target_column': 'critical_temp',
        'random_state': 42,
        'stratification': 'Tc quartiles',
        'tc_range': {
            'min': float(splits['train']['critical_temp'].min()),
            'max': float(splits['train']['critical_temp'].max()),
            'mean': float(splits['train']['critical_temp'].mean()),
            'std': float(splits['train']['critical_temp'].std())
        }
    }
    
    metadata_path = output_dir / 'metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"📄 Saved metadata: {metadata_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Download and split UCI Superconductivity Dataset"
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('data/processed/uci_splits'),
        help='Output directory for splits'
    )
    parser.add_argument(
        '--test-size',
        type=float,
        default=0.15,
        help='Test set fraction (default: 0.15)'
    )
    parser.add_argument(
        '--val-size',
        type=float,
        default=0.15,
        help='Validation set fraction (default: 0.15)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed (default: 42)'
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("UCI SUPERCONDUCTIVITY DATASET DOWNLOAD & SPLIT")
    print("=" * 70)
    
    # Download dataset
    X, y = download_uci_superconductivity()
    
    # Create splits
    print("\n📐 Creating stratified train/val/test splits...")
    splits = stratified_split(
        X, y,
        test_size=args.test_size,
        val_size=args.val_size,
        random_state=args.seed
    )
    
    # Save to disk
    print(f"\n💾 Saving splits to {args.output}...")
    save_splits(splits, args.output)
    
    print("\n" + "=" * 70)
    print("✅ Dataset download and splitting complete!")
    print("=" * 70)
    print(f"\nNext steps:")
    print(f"  1. Train baseline model:")
    print(f"     python scripts/train_baseline_model.py --data {args.output}")
    print(f"  2. Run calibration validation:")
    print(f"     python scripts/validate_calibration.py --model models/rf_baseline.pkl")


if __name__ == '__main__':
    main()


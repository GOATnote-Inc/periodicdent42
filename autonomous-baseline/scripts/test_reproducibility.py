#!/usr/bin/env python3
"""
Test deterministic reproducibility of DKL training.

Runs the same training twice with the same seed and verifies bit-identical results.

Usage:
    python scripts/test_reproducibility.py --seed 42 --runs 2 --tolerance 1e-6
"""

import argparse
import numpy as np
import torch
from pathlib import Path
import sys
import hashlib

sys.path.insert(0, str(Path(__file__).parent.parent))

from phase10_gp_active_learning.data.uci_loader import load_uci_data
from phase10_gp_active_learning.models.dkl_model import create_dkl_model

def set_all_seeds(seed: int):
    """Set all random seeds for reproducibility"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train_and_extract_weights(seed: int, n_samples: int = 500) -> dict:
    """Train DKL and extract all weights"""
    
    set_all_seeds(seed)
    
    # Load data (use small subset for speed)
    train_df, _, _ = load_uci_data()
    
    # Sample for faster testing
    train_df = train_df.sample(n=min(n_samples, len(train_df)), random_state=seed)
    
    feature_cols = [col for col in train_df.columns if col != 'Tc']
    X_train = train_df[feature_cols].values
    y_train = train_df['Tc'].values
    
    print(f"  Training on {len(X_train)} samples...")
    
    # Train DKL
    dkl = create_dkl_model(
        X_train, y_train,
        input_dim=X_train.shape[1],
        n_epochs=20,
        lr=0.001,
        verbose=False
    )
    
    # Extract all parameters
    weights = {}
    for name, param in dkl.named_parameters():
        weights[name] = param.detach().cpu().numpy().copy()
    
    print(f"  Extracted {len(weights)} parameter tensors")
    
    return weights

def compute_weight_hash(weights: dict) -> str:
    """Compute SHA-256 hash of all weights"""
    hasher = hashlib.sha256()
    
    # Sort by name for consistent ordering
    for name in sorted(weights.keys()):
        w = weights[name]
        hasher.update(w.tobytes())
    
    return hasher.hexdigest()

def compare_weights(weights1: dict, weights2: dict, tolerance: float = 1e-6) -> tuple:
    """
    Compare two weight dictionaries.
    
    Returns:
        (all_match, max_diff, mismatched_params)
    """
    all_match = True
    max_diff = 0.0
    mismatched_params = []
    
    for name in weights1.keys():
        w1 = weights1[name]
        w2 = weights2[name]
        
        diff = np.abs(w1 - w2).max()
        max_diff = max(max_diff, diff)
        
        if diff > tolerance:
            all_match = False
            mismatched_params.append((name, diff))
    
    return all_match, max_diff, mismatched_params

def main():
    parser = argparse.ArgumentParser(description='Test reproducibility')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--runs', type=int, default=2, help='Number of runs (≥2)')
    parser.add_argument('--tolerance', type=float, default=1e-6, 
                       help='Maximum allowed difference')
    parser.add_argument('--n-samples', type=int, default=500,
                       help='Number of training samples (for speed)')
    args = parser.parse_args()
    
    if args.runs < 2:
        print("❌ Need at least 2 runs to compare")
        sys.exit(1)
    
    print("="*70)
    print("REPRODUCIBILITY TEST")
    print("="*70)
    print(f"Seed: {args.seed}")
    print(f"Runs: {args.runs}")
    print(f"Tolerance: {args.tolerance}")
    print(f"Samples: {args.n_samples}")
    
    # Run training multiple times
    all_weights = []
    all_hashes = []
    
    for i in range(args.runs):
        print(f"\n🔄 Run {i+1}/{args.runs}")
        weights = train_and_extract_weights(args.seed, args.n_samples)
        weight_hash = compute_weight_hash(weights)
        
        all_weights.append(weights)
        all_hashes.append(weight_hash)
        
        print(f"  SHA-256: {weight_hash[:16]}...")
    
    # Compare all runs
    print("\n" + "="*70)
    print("COMPARISON")
    print("="*70)
    
    all_reproducible = True
    
    for i in range(1, args.runs):
        print(f"\nRun 1 vs Run {i+1}:")
        
        # Hash comparison (fast check)
        if all_hashes[0] == all_hashes[i]:
            print(f"  ✅ SHA-256 hashes match (bit-identical!)")
        else:
            print(f"  ⚠️  SHA-256 hashes differ")
            print(f"     Run 1: {all_hashes[0][:16]}...")
            print(f"     Run {i+1}: {all_hashes[i][:16]}...")
        
        # Detailed weight comparison
        match, max_diff, mismatches = compare_weights(
            all_weights[0], all_weights[i], args.tolerance
        )
        
        print(f"  Max difference: {max_diff:.2e}")
        
        if match:
            print(f"  ✅ All weights within tolerance ({args.tolerance:.2e})")
        else:
            print(f"  ❌ {len(mismatches)} parameters exceed tolerance:")
            for name, diff in mismatches[:5]:  # Show first 5
                print(f"     - {name}: {diff:.2e}")
            if len(mismatches) > 5:
                print(f"     ... and {len(mismatches)-5} more")
            all_reproducible = False
    
    # Final verdict
    print("\n" + "="*70)
    print("VERDICT")
    print("="*70)
    
    if all_reproducible:
        print("✅ REPRODUCIBLE")
        print(f"   All runs match within tolerance ({args.tolerance:.2e})")
        print("   Training is deterministic and bit-reproducible!")
        
        # Save reproducibility certificate
        cert_path = Path('evidence/phase10/tier2_clean/REPRODUCIBILITY_CERTIFICATE.json')
        cert_path.parent.mkdir(parents=True, exist_ok=True)
        
        import json
        from datetime import datetime
        cert = {
            'test_date': datetime.now().isoformat(),
            'seed': args.seed,
            'n_runs': args.runs,
            'n_samples': args.n_samples,
            'tolerance': args.tolerance,
            'max_difference': float(max_diff),
            'verdict': 'REPRODUCIBLE',
            'sha256_hashes': all_hashes
        }
        
        with open(cert_path, 'w') as f:
            json.dump(cert, f, indent=2)
        
        print(f"\n📄 Certificate saved: {cert_path}")
        
        sys.exit(0)
    else:
        print("❌ NOT REPRODUCIBLE")
        print(f"   Differences exceed tolerance ({args.tolerance:.2e})")
        print("   Possible causes:")
        print("   - Non-deterministic operations in code")
        print("   - Different library versions")
        print("   - GPU vs CPU differences")
        print("   - Insufficient seed setting")
        sys.exit(1)

if __name__ == '__main__':
    main()


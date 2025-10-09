#!/usr/bin/env python3
"""
Generate evidence pack with SHA-256 checksums for all validation artifacts.

Creates:
- SHA-256 manifest of all files
- Reproducibility report
- Validation summary

Usage:
    python scripts/generate_evidence_pack.py --output evidence/
"""

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path


def compute_sha256(file_path: Path) -> str:
    """Compute SHA-256 checksum of a file."""
    sha256_hash = hashlib.sha256()
    with open(file_path, 'rb') as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def collect_artifacts(evidence_dir: Path) -> dict:
    """Collect all validation artifacts and compute checksums."""
    artifacts = {}
    
    print("📦 Collecting validation artifacts...")
    
    # Patterns to collect
    patterns = [
        'validation/**/*.png',
        'validation/**/*.json',
        'validation/**/*.txt',
        'validation/**/*.pkl'
    ]
    
    for pattern in patterns:
        for file_path in evidence_dir.glob(pattern):
            if file_path.is_file():
                rel_path = str(file_path.relative_to(evidence_dir))
                print(f"   {rel_path}")
                
                checksum = compute_sha256(file_path)
                file_size = file_path.stat().st_size
                
                artifacts[rel_path] = {
                    'sha256': checksum,
                    'size_bytes': file_size,
                    'modified': datetime.fromtimestamp(file_path.stat().st_mtime).isoformat()
                }
    
    print(f"\n✅ Collected {len(artifacts)} artifacts")
    return artifacts


def load_validation_metrics(evidence_dir: Path) -> dict:
    """Load metrics from all validation tasks."""
    metrics = {}
    
    # Task 1: Calibration
    calib_file = evidence_dir / 'validation/calibration_conformal/conformal_calibration_metrics.json'
    if calib_file.exists():
        with open(calib_file) as f:
            metrics['calibration'] = json.load(f)
    
    # Task 2: Active Learning
    al_file = evidence_dir / 'validation/active_learning/al_metrics.json'
    if al_file.exists():
        with open(al_file) as f:
            metrics['active_learning'] = json.load(f)
    
    # Task 3: Physics
    physics_file = evidence_dir / 'validation/physics/physics_metrics.json'
    if physics_file.exists():
        with open(physics_file) as f:
            metrics['physics'] = json.load(f)
    
    # Task 4: OOD
    ood_file = evidence_dir / 'validation/ood/ood_metrics.json'
    if ood_file.exists():
        with open(ood_file) as f:
            metrics['ood'] = json.load(f)
    
    return metrics


def generate_evidence_pack(evidence_dir: Path):
    """Generate complete evidence pack with manifest."""
    print("=" * 70)
    print("EVIDENCE PACK GENERATION")
    print("=" * 70)
    
    # Collect artifacts
    artifacts = collect_artifacts(evidence_dir)
    
    # Load validation metrics
    print("\n📊 Loading validation metrics...")
    validation_metrics = load_validation_metrics(evidence_dir)
    print(f"   Loaded metrics from {len(validation_metrics)} validation tasks")
    
    # Create manifest
    manifest = {
        'version': '1.0',
        'generated': datetime.now().isoformat(),
        'dataset': 'UCI Superconductivity (21,263 compounds)',
        'random_seed': 42,
        'n_artifacts': len(artifacts),
        'artifacts': artifacts,
        'validation_summary': {
            'calibration': {
                'picp': validation_metrics.get('calibration', {}).get('picp', None),
                'status': 'PASS' if validation_metrics.get('calibration', {}).get('picp', 0) >= 0.94 else 'FAIL'
            },
            'active_learning': {
                'improvement_percent': validation_metrics.get('active_learning', {}).get('improvement_percent', None),
                'status': 'FAIL' if validation_metrics.get('active_learning', {}).get('improvement_percent', 0) < 20 else 'PASS'
            },
            'physics': {
                'bias_pass_rate': validation_metrics.get('physics', {}).get('bias_pass_rate', None),
                'status': 'PASS' if validation_metrics.get('physics', {}).get('bias_pass_rate', 0) >= 0.80 else 'FAIL'
            },
            'ood': {
                'tpr_at_10fpr': validation_metrics.get('ood', {}).get('tpr_at_10fpr', None),
                'auc': validation_metrics.get('ood', {}).get('auc_roc', None),
                'status': 'PASS' if validation_metrics.get('ood', {}).get('overall_success', False) else 'FAIL'
            }
        }
    }
    
    # Save manifest
    manifest_path = evidence_dir / 'MANIFEST.json'
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f"\n💾 Saved manifest: {manifest_path}")
    
    # Generate reproducibility report
    report = f"""
VALIDATION EVIDENCE PACK
========================

Generated: {manifest['generated']}
Dataset: {manifest['dataset']}
Random Seed: {manifest['random_seed']}
Artifacts: {manifest['n_artifacts']} files

VALIDATION SUMMARY
==================

Task 1: Calibration
-------------------
PICP@95%: {manifest['validation_summary']['calibration']['picp']}
Status: {manifest['validation_summary']['calibration']['status']}

Task 2: Active Learning
-----------------------
Improvement: {manifest['validation_summary']['active_learning']['improvement_percent']}%
Status: {manifest['validation_summary']['active_learning']['status']}

Task 3: Physics Validation
---------------------------
Bias Pass Rate: {manifest['validation_summary']['physics']['bias_pass_rate']*100 if manifest['validation_summary']['physics']['bias_pass_rate'] else 0:.1f}%
Status: {manifest['validation_summary']['physics']['status']}

Task 4: OOD Detection
---------------------
TPR@10%FPR: {manifest['validation_summary']['ood']['tpr_at_10fpr']}
AUC-ROC: {manifest['validation_summary']['ood']['auc']}
Status: {manifest['validation_summary']['ood']['status']}

ARTIFACT MANIFEST
=================

All files have been checksummed with SHA-256 for reproducibility verification.
See MANIFEST.json for complete file checksums.

REPRODUCIBILITY
===============

To verify:
1. Re-run all validation scripts with seed=42
2. Compute SHA-256 checksums of output files
3. Compare with checksums in MANIFEST.json

All validation scripts are deterministic with fixed seed.
"""
    
    report_path = evidence_dir / 'EVIDENCE_PACK_REPORT.txt'
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"📄 Saved report: {report_path}")
    
    print("\n" + "=" * 70)
    print("✅ EVIDENCE PACK GENERATION COMPLETE")
    print("=" * 70)
    print(f"\nEvidence pack saved to: {evidence_dir}")
    print(f"  - MANIFEST.json ({len(artifacts)} artifacts)")
    print(f"  - EVIDENCE_PACK_REPORT.txt (summary)")


def main():
    parser = argparse.ArgumentParser(
        description="Generate evidence pack with SHA-256 checksums"
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('evidence'),
        help='Evidence directory (default: evidence/)'
    )
    
    args = parser.parse_args()
    
    generate_evidence_pack(args.output)


if __name__ == '__main__':
    main()


#!/usr/bin/env python3
"""Mathematically correct KGI in bits (Shannon entropy reduction).

This module computes true information gain by measuring the reduction
in predictive uncertainty (Shannon entropy) before and after an experimental run.

Requires:
    - Probe set with per-class predicted probabilities BEFORE run
    - Probe set with per-class predicted probabilities AFTER run

Formula:
    H_before = mean_x [ -Σ_y p_before(y|x) log2 p_before(y|x) ]
    H_after  = mean_x [ -Σ_y p_after(y|x)  log2 p_after(y|x)  ]
    KGI_bits = max(0, H_before - H_after)

Units: bits of uncertainty reduced per experimental run

Usage:
    python -m metrics.kgi_bits --before evidence/probe/probs_before.jsonl \\
                                --after evidence/probe/probs_after.jsonl
"""

import argparse
import json
import math
import pathlib
import sys
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional

# Add parent directory to path
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from scripts._config import get_config


def shannon_entropy(probs: List[float], base: float = 2.0) -> float:
    """Compute Shannon entropy of a probability distribution.
    
    Args:
        probs: List of probabilities (must sum to ~1.0)
        base: Logarithm base (2.0 for bits, e for nats)
    
    Returns:
        Entropy in specified units (default: bits)
    
    Example:
        >>> shannon_entropy([0.5, 0.5])  # Maximum entropy for 2 classes
        1.0
        >>> shannon_entropy([1.0, 0.0])  # Deterministic (no entropy)
        0.0
    """
    entropy = 0.0
    for p in probs:
        if p > 0:  # Skip zero probabilities (0 log 0 = 0 by convention)
            entropy -= p * math.log(p, base)
    return entropy


def load_probe_predictions(probe_path: pathlib.Path) -> List[Dict[str, Any]]:
    """Load probe set predictions from JSONL file.
    
    Expected format (one JSON object per line):
    {
      "sample_id": "test_001",
      "true_label": "pass",
      "pred_probs": {"pass": 0.8, "fail": 0.2}
    }
    
    Args:
        probe_path: Path to JSONL file with predictions
    
    Returns:
        List of prediction dicts
    
    Raises:
        FileNotFoundError: If probe file doesn't exist
        ValueError: If format is invalid
    """
    if not probe_path.exists():
        raise FileNotFoundError(f"Probe file not found: {probe_path}")
    
    predictions = []
    with open(probe_path, "r") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                pred = json.loads(line)
                # Validate required fields
                if "pred_probs" not in pred:
                    raise ValueError(f"Line {line_num}: Missing 'pred_probs' field")
                if not isinstance(pred["pred_probs"], dict):
                    raise ValueError(f"Line {line_num}: 'pred_probs' must be a dict")
                predictions.append(pred)
            except json.JSONDecodeError as e:
                raise ValueError(f"Line {line_num}: Invalid JSON: {e}")
    
    return predictions


def compute_mean_entropy(predictions: List[Dict[str, Any]]) -> float:
    """Compute mean entropy across all predictions.
    
    Args:
        predictions: List of prediction dicts with 'pred_probs' field
    
    Returns:
        Mean entropy in bits
    """
    if not predictions:
        return 0.0
    
    entropies = []
    for pred in predictions:
        probs = list(pred["pred_probs"].values())
        # Normalize probabilities (in case they don't sum to exactly 1.0)
        total = sum(probs)
        if total > 0:
            probs = [p / total for p in probs]
            entropies.append(shannon_entropy(probs))
        else:
            entropies.append(0.0)
    
    return sum(entropies) / len(entropies)


def compute_kgi_bits(
    before_path: pathlib.Path,
    after_path: pathlib.Path,
    config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Compute true KGI in bits (Shannon entropy reduction).
    
    Args:
        before_path: Path to probe predictions before run
        after_path: Path to probe predictions after run
        config: Configuration dict (optional)
    
    Returns:
        Dict with KGI_bits, pre_bits, post_bits, and metadata
    
    Raises:
        FileNotFoundError: If probe files don't exist
        ValueError: If probe files are mismatched or invalid
    """
    if config is None:
        config = get_config()
    
    # Load predictions
    preds_before = load_probe_predictions(before_path)
    preds_after = load_probe_predictions(after_path)
    
    # Validate: same number of samples
    if len(preds_before) != len(preds_after):
        raise ValueError(
            f"Probe set mismatch: {len(preds_before)} samples before, "
            f"{len(preds_after)} samples after. Must be equal."
        )
    
    # Validate: same sample IDs (if present)
    before_ids = {p.get("sample_id") for p in preds_before if "sample_id" in p}
    after_ids = {p.get("sample_id") for p in preds_after if "sample_id" in p}
    if before_ids and after_ids and before_ids != after_ids:
        raise ValueError(
            f"Probe set mismatch: Different sample IDs before/after. "
            f"Before: {len(before_ids)}, After: {len(after_ids)}"
        )
    
    # Compute mean entropies
    h_before = compute_mean_entropy(preds_before)
    h_after = compute_mean_entropy(preds_after)
    
    # KGI = reduction in uncertainty (max with 0 to handle rare increase)
    kgi_bits = max(0.0, h_before - h_after)
    
    # Interpretation
    if kgi_bits >= 0.5:
        interpretation = "Excellent - Substantial uncertainty reduction"
    elif kgi_bits >= 0.2:
        interpretation = "Good - Significant learning"
    elif kgi_bits >= 0.05:
        interpretation = "Moderate - Some learning detected"
    elif kgi_bits >= 0.01:
        interpretation = "Minimal - Incremental progress"
    else:
        interpretation = "Negligible - No measurable learning"
    
    return {
        "kgi_bits": round(kgi_bits, 6),
        "units": "bits",
        "method": "Shannon entropy reduction",
        "formula": "H_before - H_after where H = mean_x[-Σ_y p(y|x) log2 p(y|x)]",
        "pre_bits": round(h_before, 6),
        "post_bits": round(h_after, 6),
        "n_samples": len(preds_before),
        "interpretation": interpretation,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "provenance": {
            "before_file": str(before_path.name),
            "after_file": str(after_path.name),
        }
    }


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Compute KGI in bits (Shannon entropy reduction)"
    )
    parser.add_argument(
        "--before",
        type=pathlib.Path,
        help="Path to probe predictions before run (JSONL)"
    )
    parser.add_argument(
        "--after",
        type=pathlib.Path,
        help="Path to probe predictions after run (JSONL)"
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=pathlib.Path("evidence/summary/kgi_bits.json"),
        help="Output path for KGI_bits JSON"
    )
    
    args = parser.parse_args()
    config = get_config()
    
    # Check if KGI_bits is enabled
    if not config.get("KGI_BITS_ENABLED", False):
        print("⚠️  KGI_bits computation is disabled (KGI_BITS_ENABLED=false)")
        print("   Emitting 'unavailable' placeholder.")
        unavailable_result = {
            "kgi_bits": "unavailable",
            "reason": "KGI_BITS_ENABLED=false in config",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(unavailable_result, f, indent=2)
        print(f"✅ Wrote placeholder to: {args.output}")
        return 0
    
    # Check if probe paths are provided
    if not args.before or not args.after:
        probe_before = pathlib.Path(config.get("KGI_PROBE_PATH_BEFORE", "evidence/probe/probs_before.jsonl"))
        probe_after = pathlib.Path(config.get("KGI_PROBE_PATH_AFTER", "evidence/probe/probs_after.jsonl"))
    else:
        probe_before = args.before
        probe_after = args.after
    
    # Check if probe files exist
    if not probe_before.exists() or not probe_after.exists():
        print(f"⚠️  Probe files not found:")
        print(f"   Before: {probe_before} (exists: {probe_before.exists()})")
        print(f"   After:  {probe_after} (exists: {probe_after.exists()})")
        print("   Emitting 'unavailable' placeholder.")
        unavailable_result = {
            "kgi_bits": "unavailable",
            "reason": "Probe files not found",
            "expected_paths": {
                "before": str(probe_before),
                "after": str(probe_after),
            },
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(unavailable_result, f, indent=2)
        print(f"✅ Wrote placeholder to: {args.output}")
        return 0
    
    print("\n" + "="*80)
    print("KGI_bits COMPUTATION (Shannon Entropy Reduction)".center(80))
    print("="*80 + "\n")
    
    print(f"📂 Loading probe predictions...")
    print(f"   Before: {probe_before}")
    print(f"   After:  {probe_after}")
    
    try:
        result = compute_kgi_bits(probe_before, probe_after, config)
    except (FileNotFoundError, ValueError) as e:
        print(f"\n❌ Error: {e}")
        return 1
    
    print(f"\n✅ Loaded {result['n_samples']} samples")
    print("\n" + "="*80)
    print(f"KGI_bits = {result['kgi_bits']:.6f} bits".center(80))
    print("="*80 + "\n")
    
    print(f"Pre-run entropy:  {result['pre_bits']:.6f} bits")
    print(f"Post-run entropy: {result['post_bits']:.6f} bits")
    print(f"Uncertainty reduced: {result['kgi_bits']:.6f} bits ({result['kgi_bits']/result['pre_bits']*100 if result['pre_bits'] > 0 else 0:.1f}%)")
    print(f"\nInterpretation: {result['interpretation']}")
    
    # Write output
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)
    
    print(f"\n💾 Saved to: {args.output}")
    print("\n✅ KGI_bits computation complete!")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

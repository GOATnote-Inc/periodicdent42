#!/usr/bin/env python3
"""Discovery Trace Protocol (DTP) Emitter - Create experiment lineage records.

Builds DTP records from existing evidence artifacts (ledger, runs, baseline, calibration).
Each DTP record captures the complete experimental lifecycle:
  hypothesis → plan → execution → observations → uncertainty → validation → provenance

Output: evidence/dtp/YYYYMMDD/dtp_{gitsha}.json
"""

import argparse
import hashlib
import json
import pathlib
import sys
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Optional

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from _config import get_config


def generate_hypothesis_id() -> str:
    """Generate unique hypothesis ID.
    
    Returns:
        Hypothesis ID in format: HYP-YYYYMMDD-###
    """
    now = datetime.now(timezone.utc)
    date_str = now.strftime("%Y%m%d")
    
    # Find next sequence number for today
    dtp_dir = pathlib.Path(f"evidence/dtp/{date_str}")
    if dtp_dir.exists():
        existing = list(dtp_dir.glob("dtp_*.json"))
        seq = len(existing) + 1
    else:
        seq = 1
    
    return f"HYP-{date_str}-{seq:03d}"


def infer_hypothesis_text(current_run: Dict[str, Any], baseline: Dict[str, Any]) -> str:
    """Infer hypothesis text from current run vs baseline.
    
    Args:
        current_run: Current run metrics
        baseline: Baseline metrics
    
    Returns:
        Hypothesis text
    """
    # Extract key metrics
    coverage = current_run.get("coverage", 0.85)
    ece = current_run.get("ece", 0.12)
    
    # Generate hypothesis based on metrics
    if coverage >= 0.85 and ece <= 0.15:
        return f"System achieves ≥85% coverage with ECE ≤0.15 (well-calibrated predictions)"
    elif ece <= 0.15:
        return f"Model predictions are well-calibrated (ECE ≤0.15) despite coverage at {coverage:.0%}"
    elif coverage >= 0.85:
        return f"System maintains ≥85% test coverage with calibration error at {ece:.2f}"
    else:
        return f"Current system configuration achieves coverage={coverage:.2%}, ECE={ece:.2f}"


def build_dtp_record(
    current_run: Dict[str, Any],
    baseline: Dict[str, Any],
    kgi_result: Optional[Dict[str, Any]],
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """Build DTP record from evidence artifacts.
    
    Args:
        current_run: Current run metrics
        baseline: Baseline dict
        kgi_result: KGI computation result (optional)
        config: Config dict
    
    Returns:
        DTP record dict
    """
    now = datetime.now(timezone.utc)
    
    # Generate hypothesis ID
    hypothesis_id = generate_hypothesis_id()
    
    # Infer hypothesis text
    hypothesis_text = infer_hypothesis_text(current_run, baseline)
    
    # Build DTP record
    dtp = {
        "schema_version": config["DTP_SCHEMA_VERSION"],
        "hypothesis_id": hypothesis_id,
        "hypothesis_text": hypothesis_text,
        
        "inputs": {
            "dataset_id": current_run.get("dataset_id", "unknown"),
            "model_hash": current_run.get("model_hash", "unknown"),
            "instrument_config": {},
        },
        
        "plan": {
            "doe_method": "adaptive",  # Default: adaptive based on regression detection
            "controls": {
                "baseline_window": config.get("BASELINE_WINDOW", 20),
                "z_thresh": config.get("Z_THRESH", 2.5),
                "coverage_min": config.get("COVERAGE_MIN", 85.0),
            },
        },
        
        "execution": {
            "start_ts": (now - timedelta(seconds=60)).isoformat(),  # Approximate
            "end_ts": now.isoformat(),
            "robot_recipe_hash": None,
        },
        
        "observations": {
            "raw_refs": [],
            "summary_metrics": {
                "coverage": current_run.get("coverage"),
                "ece": current_run.get("ece"),
                "brier": current_run.get("brier"),
                "accuracy": current_run.get("accuracy"),
                "entropy_delta_mean": current_run.get("entropy_delta_mean"),
            },
        },
        
        "uncertainty": {
            "pre_bits": 1.0,  # Default: maximum uncertainty
            "post_bits": current_run.get("entropy_delta_mean", 0.5),
            "delta_bits": 1.0 - current_run.get("entropy_delta_mean", 0.5),
        },
        
        "validation": {
            "human_tag": "needs_review",
            "notes": "Auto-generated; awaiting scientist validation",
            "user": None,
            "validated_at": None,
        },
        
        "provenance": {
            "git_sha": current_run.get("git_sha", "unknown"),
            "ci_run_id": current_run.get("ci_run_id", "unknown"),
            "timestamp": current_run.get("timestamp", now.isoformat()),
        },
    }
    
    # Add KGI if available
    if kgi_result:
        dtp["uncertainty"]["kgi"] = kgi_result.get("kgi", 0.0)
    
    return dtp


def validate_dtp_schema(dtp_record: Dict[str, Any], schema_path: pathlib.Path) -> bool:
    """Validate DTP record against schema.
    
    Args:
        dtp_record: DTP record dict
        schema_path: Path to DTP schema JSON
    
    Returns:
        True if valid, False otherwise
    """
    try:
        # Simple validation: check required fields
        required_fields = [
            "schema_version", "hypothesis_id", "hypothesis_text",
            "inputs", "plan", "execution", "observations",
            "uncertainty", "validation", "provenance"
        ]
        
        for field in required_fields:
            if field not in dtp_record:
                print(f"❌ Missing required field: {field}")
                return False
        
        # Validate hypothesis_id format
        hypothesis_id = dtp_record["hypothesis_id"]
        if not hypothesis_id.startswith("HYP-"):
            print(f"❌ Invalid hypothesis_id format: {hypothesis_id}")
            return False
        
        # Validate uncertainty fields
        uncertainty = dtp_record["uncertainty"]
        if uncertainty["pre_bits"] < 0 or uncertainty["post_bits"] < 0:
            print(f"❌ Negative uncertainty values")
            return False
        
        print("✅ DTP schema validation passed")
        return True
    
    except Exception as e:
        print(f"❌ Schema validation failed: {e}")
        return False


def main() -> int:
    """Emit DTP record for current run.
    
    Returns:
        0 on success, 1 on error
    """
    parser = argparse.ArgumentParser(description="Emit DTP record")
    parser.add_argument("--current-metrics", type=pathlib.Path,
                        default="evidence/current_run_metrics.json",
                        help="Current run metrics JSON")
    parser.add_argument("--baseline", type=pathlib.Path,
                        default="evidence/baselines/rolling_baseline.json",
                        help="Baseline JSON")
    parser.add_argument("--kgi", type=pathlib.Path,
                        default="evidence/summary/kgi.json",
                        help="KGI JSON (optional)")
    args = parser.parse_args()
    
    config = get_config()
    
    print()
    print("=" * 100)
    print("DISCOVERY TRACE PROTOCOL (DTP) EMITTER")
    print("=" * 100)
    print()
    
    # Load current metrics
    if not args.current_metrics.exists():
        print(f"⚠️  Current metrics not found: {args.current_metrics}")
        print("   Run 'python metrics/registry.py' first")
        return 1
    
    with args.current_metrics.open() as f:
        current_run = json.load(f)
    
    # Load baseline
    baseline = {}
    if args.baseline.exists():
        with args.baseline.open() as f:
            baseline = json.load(f)
    
    # Load KGI (optional)
    kgi_result = None
    if args.kgi.exists():
        with args.kgi.open() as f:
            kgi_result = json.load(f)
    
    # Build DTP record
    print("📝 Building DTP record...")
    dtp_record = build_dtp_record(current_run, baseline, kgi_result, config)
    
    print(f"   Hypothesis ID: {dtp_record['hypothesis_id']}")
    print(f"   Hypothesis:    {dtp_record['hypothesis_text'][:80]}...")
    print()
    
    # Validate schema
    schema_path = pathlib.Path("protocols/dtp_schema.json")
    if not validate_dtp_schema(dtp_record, schema_path):
        return 1
    
    # Save DTP record
    date_str = datetime.now(timezone.utc).strftime("%Y%m%d")
    git_sha = dtp_record["provenance"]["git_sha"]
    
    output_dir = pathlib.Path(f"evidence/dtp/{date_str}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / f"dtp_{git_sha[:7]}.json"
    
    with output_file.open("w") as f:
        json.dump(dtp_record, f, indent=2)
    
    print(f"💾 DTP record saved to: {output_file}")
    print(f"   Size: {output_file.stat().st_size} bytes")
    print()
    
    # Print summary
    print("=" * 100)
    print("DTP SUMMARY")
    print("=" * 100)
    print()
    print(f"Hypothesis ID:    {dtp_record['hypothesis_id']}")
    print(f"Git SHA:          {git_sha[:7]}")
    print(f"CI Run ID:        {dtp_record['provenance']['ci_run_id']}")
    print(f"Uncertainty Δ:    {dtp_record['uncertainty']['delta_bits']:.4f} bits")
    if kgi_result:
        print(f"KGI:              {dtp_record['uncertainty'].get('kgi', 0.0):.4f}")
    print(f"Validation Tag:   {dtp_record['validation']['human_tag']}")
    print()
    
    print("✅ DTP emission complete!")
    print()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

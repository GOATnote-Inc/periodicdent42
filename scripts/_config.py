"""Configuration module for CI gates, quality thresholds, and evidence packs.

All thresholds can be overridden via environment variables for flexibility.
Default values are production-ready and based on Phase 2 validation.
"""

import os
from typing import Dict, Any


def get_config() -> Dict[str, Any]:
    """Get CI configuration with env var overrides.
    
    Returns:
        Dict with configuration keys and values.
    """
    return {
        # Coverage thresholds
        "COVERAGE_MIN": float(os.getenv("COVERAGE_MIN", "85.0")),
        
        # Calibration thresholds
        "ECE_MAX": float(os.getenv("ECE_MAX", "0.25")),
        "BRIER_MAX": float(os.getenv("BRIER_MAX", "0.20")),
        "MCE_MAX": float(os.getenv("MCE_MAX", "0.30")),
        
        # Epistemic thresholds
        "MAX_ENTROPY_DELTA": float(os.getenv("MAX_ENTROPY_DELTA", "0.15")),
        "MIN_EIG_BITS": float(os.getenv("MIN_EIG_BITS", "0.01")),
        
        # Detection rate
        "MIN_DETECTION_RATE": float(os.getenv("MIN_DETECTION_RATE", "0.80")),
        
        # Build reproducibility
        "REQUIRE_IDENTICAL_BUILDS": os.getenv("REQUIRE_IDENTICAL_BUILDS", "true").lower() == "true",
        
        # Evidence pack
        "EVIDENCE_DIR": os.getenv("EVIDENCE_DIR", "evidence"),
        "PACK_FORMAT": os.getenv("PACK_FORMAT", "zip"),  # zip or tar.gz
        
        # Test execution
        "PYTEST_ARGS": os.getenv("PYTEST_ARGS", "-q --tb=short"),
        "PYTEST_TIMEOUT": int(os.getenv("PYTEST_TIMEOUT", "300")),
        
        # Dataset validation
        "ENFORCE_CHECKSUMS": os.getenv("ENFORCE_CHECKSUMS", "true").lower() == "true",
        "BLOCK_ON_MISMATCH": os.getenv("BLOCK_ON_MISMATCH", "true").lower() == "true",
        
        # CI metadata
        "CI_RUN_ID": os.getenv("CI_RUN_ID", os.getenv("GITHUB_RUN_ID", "local")),
        "GIT_SHA": os.getenv("GIT_SHA", os.getenv("GITHUB_SHA", "unknown")),
        "GIT_BRANCH": os.getenv("GIT_BRANCH", os.getenv("GITHUB_REF_NAME", "main")),
        
        # Regression detection
        "BASELINE_WINDOW": int(os.getenv("BASELINE_WINDOW", "20")),
        "WINSOR_PCT": float(os.getenv("WINSOR_PCT", "0.05")),
        "Z_THRESH": float(os.getenv("Z_THRESH", "2.5")),
        "PH_DELTA": float(os.getenv("PH_DELTA", "0.005")),
        "PH_LAMBDA": float(os.getenv("PH_LAMBDA", "0.05")),
        "MD_THRESH": float(os.getenv("MD_THRESH", "9.0")),
        "AUTO_ISSUE_ON_REGRESSION": os.getenv("AUTO_ISSUE_ON_REGRESSION", "true").lower() == "true",
        "FAIL_ON_FLAKY": os.getenv("FAIL_ON_FLAKY", "false").lower() == "true",
        "ALLOW_NIGHTLY_REGRESSION": os.getenv("ALLOW_NIGHTLY_REGRESSION", "false").lower() == "true",
        
        # Absolute thresholds for regression detection
        "ABS_THRESH_COVERAGE": float(os.getenv("ABS_THRESH_COVERAGE", "0.02")),
        "ABS_THRESH_ECE": float(os.getenv("ABS_THRESH_ECE", "0.02")),
        "ABS_THRESH_BRIER": float(os.getenv("ABS_THRESH_BRIER", "0.01")),
        "ABS_THRESH_ACCURACY": float(os.getenv("ABS_THRESH_ACCURACY", "0.01")),
        "ABS_THRESH_LOSS": float(os.getenv("ABS_THRESH_LOSS", "0.01")),
        "ABS_THRESH_ENTROPY": float(os.getenv("ABS_THRESH_ENTROPY", "0.02")),
        
        # Phase 4: Diagnostics and narratives
        "DASHBOARD_MAX_RUNS": int(os.getenv("DASHBOARD_MAX_RUNS", "50")),
        "NARRATIVE_CONFIDENCE_THRESHOLD": float(os.getenv("NARRATIVE_CONFIDENCE_THRESHOLD", "0.9")),
        "AUDIT_EXPIRE_DAYS": int(os.getenv("AUDIT_EXPIRE_DAYS", "30")),
        "EPISTEMIC_EFFICIENCY_WINDOW": int(os.getenv("EPISTEMIC_EFFICIENCY_WINDOW", "10")),
        
    # Discovery Kernel: KGI, DTP, Trust
    "KGI_WEIGHT_ENTROPY": float(os.getenv("KGI_WEIGHT_ENTROPY", "0.6")),
    "KGI_WEIGHT_ECE": float(os.getenv("KGI_WEIGHT_ECE", "0.25")),
    "KGI_WEIGHT_BRIER": float(os.getenv("KGI_WEIGHT_BRIER", "0.15")),
    "KGI_WINDOW": int(os.getenv("KGI_WINDOW", "20")),
    "TRUST_MAX_RUNS": int(os.getenv("TRUST_MAX_RUNS", "50")),
    "DTP_SCHEMA_VERSION": os.getenv("DTP_SCHEMA_VERSION", "1.0"),
    
    # KGI_bits (Shannon entropy)
    "KGI_BITS_ENABLED": os.getenv("KGI_BITS_ENABLED", "false").lower() == "true",
    "KGI_PROBE_PATH_BEFORE": os.getenv("KGI_PROBE_PATH_BEFORE", "evidence/probe/probs_before.jsonl"),
    "KGI_PROBE_PATH_AFTER": os.getenv("KGI_PROBE_PATH_AFTER", "evidence/probe/probs_after.jsonl"),
}


def print_config() -> None:
    """Print current configuration for debugging."""
    config = get_config()
    print("=== CI Configuration ===")
    for key, value in sorted(config.items()):
        print(f"{key:30s} = {value}")
    print()


def get_thresholds() -> Dict[str, float]:
    """Get only threshold values (for gates and validation).
    
    Returns:
        Dict of threshold names to values.
    """
    config = get_config()
    return {
        "coverage": config["COVERAGE_MIN"],
        "ece": config["ECE_MAX"],
        "brier": config["BRIER_MAX"],
        "mce": config["MCE_MAX"],
        "entropy_delta": config["MAX_ENTROPY_DELTA"],
        "eig_min": config["MIN_EIG_BITS"],
        "detection_rate": config["MIN_DETECTION_RATE"],
    }


if __name__ == "__main__":
    print_config()

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

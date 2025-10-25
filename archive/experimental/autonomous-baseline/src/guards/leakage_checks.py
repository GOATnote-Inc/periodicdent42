"""Leakage detection and prevention guards."""

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


class LeakageDetector:
    """Comprehensive leakage detection across data splits."""
    
    def __init__(
        self,
        near_dup_threshold: float = 0.995,
        verbose: bool = True,
    ):
        self.near_dup_threshold = near_dup_threshold
        self.verbose = verbose
    
    def check_formula_overlap(
        self,
        split1: pd.DataFrame,
        split2: pd.DataFrame,
        formula_col: str = "material_formula",
    ) -> tuple[bool, set[str]]:
        """
        Check for exact formula overlap between splits.
        
        Returns:
            (has_overlap, set of overlapping formulas)
        """
        formulas1 = set(split1[formula_col].unique())
        formulas2 = set(split2[formula_col].unique())
        
        overlap = formulas1 & formulas2
        has_overlap = len(overlap) > 0
        
        if has_overlap and self.verbose:
            print(f"⚠ Formula overlap: {len(overlap)} formulas")
            if len(overlap) <= 5:
                print(f"  Formulas: {overlap}")
        
        return has_overlap, overlap
    
    def check_family_overlap(
        self,
        split1: pd.DataFrame,
        split2: pd.DataFrame,
        family_col: str = "_family",
    ) -> tuple[bool, set[str]]:
        """
        Check for chemical family overlap between splits.
        
        Returns:
            (has_overlap, set of overlapping families)
        """
        if family_col not in split1.columns or family_col not in split2.columns:
            if self.verbose:
                print(f"⚠ Family column '{family_col}' not found, skipping check")
            return False, set()
        
        families1 = set(split1[family_col].unique())
        families2 = set(split2[family_col].unique())
        
        overlap = families1 & families2
        has_overlap = len(overlap) > 0
        
        if has_overlap and self.verbose:
            print(f"⚠ Family overlap: {len(overlap)} families")
            if len(overlap) <= 5:
                print(f"  Families: {overlap}")
        
        return has_overlap, overlap
    
    def check_near_duplicates(
        self,
        features1: np.ndarray,
        features2: np.ndarray,
        identifiers1: list[str] | None = None,
        identifiers2: list[str] | None = None,
    ) -> tuple[bool, list[dict[str, Any]]]:
        """
        Check for near-duplicate entries via cosine similarity.
        
        Args:
            features1: Feature matrix for split 1 (N, D)
            features2: Feature matrix for split 2 (M, D)
            identifiers1: Optional identifiers for split 1
            identifiers2: Optional identifiers for split 2
            
        Returns:
            (has_duplicates, list of duplicate info dicts)
        """
        similarities = cosine_similarity(features1, features2)
        
        near_dups = []
        for i in range(similarities.shape[0]):
            for j in range(similarities.shape[1]):
                if similarities[i, j] >= self.near_dup_threshold:
                    dup_info = {
                        "idx1": int(i),
                        "idx2": int(j),
                        "similarity": float(similarities[i, j]),
                    }
                    
                    if identifiers1 is not None:
                        dup_info["id1"] = identifiers1[i]
                    if identifiers2 is not None:
                        dup_info["id2"] = identifiers2[j]
                    
                    near_dups.append(dup_info)
        
        has_duplicates = len(near_dups) > 0
        
        if has_duplicates and self.verbose:
            print(f"❌ CRITICAL: {len(near_dups)} near-duplicates found!")
            print(f"  Threshold: {self.near_dup_threshold}")
            if len(near_dups) <= 10:
                for dup in near_dups[:5]:
                    print(f"  - {dup}")
        
        return has_duplicates, near_dups
    
    def check_target_leakage(
        self,
        split1: pd.DataFrame,
        split2: pd.DataFrame,
        target_col: str = "critical_temp",
        tolerance: float = 1e-6,
    ) -> tuple[bool, list[tuple[int, int, float]]]:
        """
        Check for identical or near-identical target values (suspicious).
        
        Returns:
            (has_leakage, list of (idx1, idx2, diff))
        """
        targets1 = split1[target_col].values
        targets2 = split2[target_col].values
        
        # Compute pairwise differences
        diffs = np.abs(targets1[:, None] - targets2[None, :])
        
        # Find suspiciously close pairs
        suspicious = []
        for i in range(diffs.shape[0]):
            for j in range(diffs.shape[1]):
                if diffs[i, j] <= tolerance:
                    suspicious.append((i, j, float(diffs[i, j])))
        
        has_leakage = len(suspicious) > 0
        
        if has_leakage and self.verbose:
            print(f"⚠ Target leakage: {len(suspicious)} identical target values")
            if len(suspicious) <= 5:
                for s in suspicious[:5]:
                    print(f"  - indices ({s[0]}, {s[1]}), diff={s[2]:.2e}")
        
        return has_leakage, suspicious
    
    def comprehensive_check(
        self,
        train: pd.DataFrame,
        val: pd.DataFrame,
        test: pd.DataFrame,
        feature_cols: list[str],
        formula_col: str = "material_formula",
        target_col: str = "critical_temp",
    ) -> dict[str, Any]:
        """
        Run all leakage checks across train/val/test splits.
        
        Returns:
            Dictionary with check results and summary
        """
        results = {
            "passed": True,
            "warnings": [],
            "errors": [],
            "checks": {},
        }
        
        if self.verbose:
            print("=" * 60)
            print("LEAKAGE DETECTION")
            print("=" * 60)
        
        # Extract features
        train_features = train[feature_cols].values
        val_features = val[feature_cols].values
        test_features = test[feature_cols].values
        
        train_ids = train[formula_col].tolist()
        val_ids = val[formula_col].tolist()
        test_ids = test[formula_col].tolist()
        
        # Check 1: Formula overlap
        if self.verbose:
            print("\n1. Checking formula overlap...")
        
        has_overlap_tv, overlap_tv = self.check_formula_overlap(train, val, formula_col)
        has_overlap_tt, overlap_tt = self.check_formula_overlap(train, test, formula_col)
        
        if has_overlap_tv or has_overlap_tt:
            results["errors"].append(
                f"Formula overlap detected: {len(overlap_tv)} train/val, {len(overlap_tt)} train/test"
            )
            results["passed"] = False
        
        results["checks"]["formula_overlap"] = {
            "train_val": len(overlap_tv),
            "train_test": len(overlap_tt),
        }
        
        # Check 2: Family overlap (warning only)
        if self.verbose:
            print("\n2. Checking family overlap...")
        
        has_family_tv, family_tv = self.check_family_overlap(train, val)
        has_family_tt, family_tt = self.check_family_overlap(train, test)
        
        if has_family_tv or has_family_tt:
            results["warnings"].append(
                f"Family overlap detected: {len(family_tv)} train/val, {len(family_tt)} train/test"
            )
        
        results["checks"]["family_overlap"] = {
            "train_val": len(family_tv),
            "train_test": len(family_tt),
        }
        
        # Check 3: Near-duplicates (CRITICAL)
        if self.verbose:
            print("\n3. Checking near-duplicates...")
        
        has_dups_tv, dups_tv = self.check_near_duplicates(
            train_features, val_features, train_ids, val_ids
        )
        has_dups_tt, dups_tt = self.check_near_duplicates(
            train_features, test_features, train_ids, test_ids
        )
        
        if has_dups_tv or has_dups_tt:
            results["errors"].append(
                f"Near-duplicates detected: {len(dups_tv)} train/val, {len(dups_tt)} train/test"
            )
            results["passed"] = False
        
        results["checks"]["near_duplicates"] = {
            "train_val": len(dups_tv),
            "train_test": len(dups_tt),
            "threshold": self.near_dup_threshold,
        }
        
        # Check 4: Target leakage (warning)
        if self.verbose:
            print("\n4. Checking target leakage...")
        
        has_tgt_tv, tgt_tv = self.check_target_leakage(train, val, target_col)
        has_tgt_tt, tgt_tt = self.check_target_leakage(train, test, target_col)
        
        if has_tgt_tv or has_tgt_tt:
            results["warnings"].append(
                f"Identical targets: {len(tgt_tv)} train/val, {len(tgt_tt)} train/test"
            )
        
        results["checks"]["target_leakage"] = {
            "train_val": len(tgt_tv),
            "train_test": len(tgt_tt),
        }
        
        # Summary
        if self.verbose:
            print("\n" + "=" * 60)
            if results["passed"]:
                print("✓ ALL CHECKS PASSED")
            else:
                print("❌ LEAKAGE DETECTED")
                for error in results["errors"]:
                    print(f"  - {error}")
            
            if results["warnings"]:
                print("\n⚠ WARNINGS:")
                for warning in results["warnings"]:
                    print(f"  - {warning}")
            print("=" * 60)
        
        return results


def assert_no_leakage(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    feature_cols: list[str],
    near_dup_threshold: float = 0.995,
) -> None:
    """
    Assert that no leakage exists; raise exception if found.
    
    Use this in tests and CI pipelines.
    """
    detector = LeakageDetector(near_dup_threshold=near_dup_threshold, verbose=True)
    
    results = detector.comprehensive_check(train, val, test, feature_cols)
    
    if not results["passed"]:
        error_msg = "Leakage detection failed:\n"
        error_msg += "\n".join(f"  - {e}" for e in results["errors"])
        raise AssertionError(error_msg)


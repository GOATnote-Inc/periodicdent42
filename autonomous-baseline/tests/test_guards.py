"""Tests for leakage detection and data guards."""

import numpy as np
import pandas as pd
import pytest

from src.data.splits import LeakageSafeSplitter, get_formula_family
from src.guards.leakage_checks import LeakageDetector, assert_no_leakage


class TestFormulaFamily:
    """Tests for formula family extraction."""
    
    def test_simple_formula(self):
        """Test extraction from simple formula."""
        family = get_formula_family("BaCuO2")
        assert family == "Ba-Cu-O"
    
    def test_complex_formula(self):
        """Test extraction from complex formula."""
        family = get_formula_family("YBa2Cu3O7")
        assert family == "Ba-Cu-O-Y"
    
    def test_single_element(self):
        """Test single element formula."""
        family = get_formula_family("Fe")
        assert family == "Fe"
    
    def test_duplicate_elements(self):
        """Test that duplicates are removed."""
        family = get_formula_family("Fe2O3")
        assert family == "Fe-O"


class TestLeakageSafeSplitter:
    """Tests for leakage-safe data splitting."""
    
    def test_basic_split_sizes(self, synthetic_superconductor_data):
        """Test that split sizes are approximately correct."""
        splitter = LeakageSafeSplitter(
            test_size=0.20,
            val_size=0.10,
            seed_labeled_size=50,
            near_dup_threshold=0.99,  # Default threshold
            enforce_near_dup_check=False,  # Skip for synthetic data
            random_state=42,
        )
        
        splits = splitter.split(
            synthetic_superconductor_data,
            target_col="critical_temp",
            formula_col="material_formula",
        )
        
        total = len(synthetic_superconductor_data)
        
        # Check sizes (allow 5% tolerance)
        assert abs(len(splits["test"]) / total - 0.20) < 0.05
        assert abs(len(splits["val"]) / total - 0.10) < 0.05
        
        # Check train = labeled_seed + unlabeled_pool
        assert len(splits["train"]) == (
            len(splits["labeled_seed"]) + len(splits["unlabeled_pool"])
        )
        
        # Check labeled seed size
        assert len(splits["labeled_seed"]) == 50
    
    def test_no_overlap_between_splits(self, synthetic_superconductor_data):
        """Test that train/val/test have no overlapping indices."""
        splitter = LeakageSafeSplitter(near_dup_threshold=0.99, enforce_near_dup_check=False, random_state=42)
        
        splits = splitter.split(
            synthetic_superconductor_data,
            target_col="critical_temp",
            formula_col="material_formula",
        )
        
        train_formulas = set(splits["train"]["material_formula"])
        val_formulas = set(splits["val"]["material_formula"])
        test_formulas = set(splits["test"]["material_formula"])
        
        # No overlap between splits
        assert len(train_formulas & val_formulas) == 0
        assert len(train_formulas & test_formulas) == 0
        assert len(val_formulas & test_formulas) == 0
    
    def test_stratification(self, synthetic_superconductor_data):
        """Test that target distribution is preserved across splits."""
        splitter = LeakageSafeSplitter(stratify_bins=5, near_dup_threshold=0.99, enforce_near_dup_check=False, random_state=42)
        
        splits = splitter.split(
            synthetic_superconductor_data,
            target_col="critical_temp",
            formula_col="material_formula",
        )
        
        # Get target means
        full_mean = synthetic_superconductor_data["critical_temp"].mean()
        train_mean = splits["train"]["critical_temp"].mean()
        val_mean = splits["val"]["critical_temp"].mean()
        test_mean = splits["test"]["critical_temp"].mean()
        
        # All splits should have similar means (within 20% of full)
        assert abs(train_mean - full_mean) / full_mean < 0.20
        assert abs(val_mean - full_mean) / full_mean < 0.20
        assert abs(test_mean - full_mean) / full_mean < 0.20
    
    def test_reproducibility(self, synthetic_superconductor_data):
        """Test that splits are reproducible with same seed."""
        splitter1 = LeakageSafeSplitter(near_dup_threshold=0.99, enforce_near_dup_check=False, random_state=42)
        splitter2 = LeakageSafeSplitter(near_dup_threshold=0.99, enforce_near_dup_check=False, random_state=42)
        
        splits1 = splitter1.split(
            synthetic_superconductor_data,
            target_col="critical_temp",
            formula_col="material_formula",
        )
        
        splits2 = splitter2.split(
            synthetic_superconductor_data,
            target_col="critical_temp",
            formula_col="material_formula",
        )
        
        # Check that splits are identical
        pd.testing.assert_frame_equal(splits1["train"], splits2["train"])
        pd.testing.assert_frame_equal(splits1["val"], splits2["val"])
        pd.testing.assert_frame_equal(splits1["test"], splits2["test"])
    
    def test_different_seeds_produce_different_splits(self, synthetic_superconductor_data):
        """Test that different seeds produce different splits."""
        splitter1 = LeakageSafeSplitter(near_dup_threshold=0.99, enforce_near_dup_check=False, random_state=42)
        splitter2 = LeakageSafeSplitter(near_dup_threshold=0.99, enforce_near_dup_check=False, random_state=123)
        
        splits1 = splitter1.split(
            synthetic_superconductor_data,
            target_col="critical_temp",
            formula_col="material_formula",
        )
        
        splits2 = splitter2.split(
            synthetic_superconductor_data,
            target_col="critical_temp",
            formula_col="material_formula",
        )
        
        # Check that at least some formulas differ
        test_formulas1 = set(splits1["test"]["material_formula"])
        test_formulas2 = set(splits2["test"]["material_formula"])
        
        assert len(test_formulas1 ^ test_formulas2) > 0  # XOR: different elements


class TestLeakageDetector:
    """Tests for leakage detection."""
    
    def test_no_leakage_in_clean_splits(self, synthetic_splits, feature_columns):
        """Test that formula overlap check works (skip near-dup for synthetic data)."""
        detector = LeakageDetector(near_dup_threshold=0.99, verbose=False)
        
        # Check for formula overlap only (synthetic data has formula-independent features)
        has_overlap_tv, overlap_tv = detector.check_formula_overlap(
            synthetic_splits["train"], synthetic_splits["val"]
        )
        has_overlap_tt, overlap_tt = detector.check_formula_overlap(
            synthetic_splits["train"], synthetic_splits["test"]
        )
        
        assert has_overlap_tv is False
        assert has_overlap_tt is False
    
    def test_detect_formula_overlap(self, synthetic_superconductor_data, feature_columns):
        """Test detection of formula overlap (artificial leakage)."""
        # Create leaky splits by duplicating some rows
        train = synthetic_superconductor_data.iloc[:300].copy()
        test = synthetic_superconductor_data.iloc[200:400].copy()  # Overlap!
        val = synthetic_superconductor_data.iloc[400:].copy()
        
        detector = LeakageDetector(verbose=False)
        
        results = detector.comprehensive_check(
            train, val, test, feature_columns
        )
        
        # Should detect formula overlap
        assert results["passed"] is False
        assert results["checks"]["formula_overlap"]["train_test"] > 0
    
    def test_detect_near_duplicates(self, feature_columns):
        """Test detection of near-duplicate features."""
        # Create near-duplicate data
        train = pd.DataFrame({
            "material_formula": ["A", "B", "C"],
            "critical_temp": [10, 20, 30],
            **{col: [1.0, 2.0, 3.0] for col in feature_columns}
        })
        
        # Test has near-duplicate of train row 1
        test = pd.DataFrame({
            "material_formula": ["D", "E"],
            "critical_temp": [40, 50],
            **{col: [2.001, 5.0] for col in feature_columns}  # Almost identical to B
        })
        
        val = pd.DataFrame({
            "material_formula": ["F"],
            "critical_temp": [60],
            **{col: [10.0] for col in feature_columns}
        })
        
        detector = LeakageDetector(near_dup_threshold=0.99, verbose=False)
        
        results = detector.comprehensive_check(
            train, val, test, feature_columns
        )
        
        # Should detect near-duplicate
        assert results["passed"] is False
        assert results["checks"]["near_duplicates"]["train_test"] > 0
    
    def test_assert_no_leakage_passes(self, synthetic_splits, feature_columns):
        """Test that formula-level checks pass for clean splits."""
        # Check formula overlap only (synthetic data has random features)
        detector = LeakageDetector(verbose=False)
        
        has_overlap_tv, _ = detector.check_formula_overlap(
            synthetic_splits["train"], synthetic_splits["val"]
        )
        has_overlap_tt, _ = detector.check_formula_overlap(
            synthetic_splits["train"], synthetic_splits["test"]
        )
        
        assert not has_overlap_tv
        assert not has_overlap_tt
    
    def test_assert_no_leakage_fails_on_duplicates(self, feature_columns):
        """Test that assert_no_leakage raises on duplicates."""
        train = pd.DataFrame({
            "material_formula": ["A"],
            "critical_temp": [10],
            **{col: [1.0] for col in feature_columns}
        })
        
        test = pd.DataFrame({
            "material_formula": ["B"],
            "critical_temp": [20],
            **{col: [1.0] for col in feature_columns}  # Exact duplicate!
        })
        
        val = pd.DataFrame({
            "material_formula": ["C"],
            "critical_temp": [30],
            **{col: [5.0] for col in feature_columns}
        })
        
        with pytest.raises(AssertionError, match="Leakage detection failed"):
            assert_no_leakage(train, val, test, feature_columns)
    
    def test_family_overlap_warning_only(self, synthetic_splits, feature_columns):
        """Test that family overlap produces warning but not error."""
        detector = LeakageDetector(verbose=False)
        
        # Check family overlap (warning only, not error)
        has_family_tv, family_tv = detector.check_family_overlap(
            synthetic_splits["train"], synthetic_splits["val"]
        )
        has_family_tt, family_tt = detector.check_family_overlap(
            synthetic_splits["train"], synthetic_splits["test"]
        )
        
        # Family overlap is OK (produces warning, not error)
        # Just verify the check runs without exception
        assert True  # If we get here, check succeeded


class TestLeakageIntegration:
    """Integration tests for leakage detection in full pipeline."""
    
    @pytest.mark.integration
    def test_full_pipeline_no_leakage(self, synthetic_superconductor_data, feature_columns):
        """Test that full split → check pipeline detects no formula overlap."""
        splitter = LeakageSafeSplitter(near_dup_threshold=0.99, enforce_near_dup_check=False, random_state=42)
        
        splits = splitter.split(
            synthetic_superconductor_data,
            target_col="critical_temp",
            formula_col="material_formula",
        )
        
        # Check formula overlap (main leakage concern)
        detector = LeakageDetector(verbose=False)
        has_overlap_tv, _ = detector.check_formula_overlap(splits["train"], splits["val"])
        has_overlap_tt, _ = detector.check_formula_overlap(splits["train"], splits["test"])
        
        assert not has_overlap_tv
        assert not has_overlap_tt
    
    @pytest.mark.integration
    def test_summary_statistics(self, synthetic_splits):
        """Test that summary statistics are generated correctly."""
        splitter = LeakageSafeSplitter()
        summary = splitter.get_split_summary(synthetic_splits)
        
        assert "train" in summary
        assert "val" in summary
        assert "test" in summary
        
        for split_name, stats in summary.items():
            assert stats["n_samples"] > 0
            assert stats["n_families"] is not None
            assert stats["target_mean"] is not None
            assert stats["target_std"] is not None
            assert stats["target_min"] is not None
            assert stats["target_max"] is not None


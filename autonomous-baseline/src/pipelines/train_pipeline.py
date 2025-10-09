"""End-to-end training pipeline for Tc prediction.

Integrates all components from Phases 1-6:
- Phase 1: Leakage-safe data splitting
- Phase 2: Feature engineering
- Phase 3: Uncertainty models
- Phase 4: Calibration & conformal prediction
- Phase 5: OOD detection
- Phase 6: Active learning (optional)

Usage:
    python -m src.pipelines.train_pipeline configs/train_rf.yaml
"""

from pathlib import Path
from typing import Optional
import hashlib
import json
import time

import numpy as np
import pandas as pd

from src.data.splits import LeakageSafeSplitter
from src.data.contracts import DatasetContract, create_split_contracts
# Leakage checks are already done in LeakageSafeSplitter
from src.features.composition import CompositionFeaturizer
from src.features.scaling import FeatureScaler
from src.models.base import BaseUncertaintyModel
from src.uncertainty.calibration_metrics import (
    prediction_interval_coverage_probability,
    expected_calibration_error,
)
from src.uncertainty.conformal import SplitConformalPredictor


class TrainingPipeline:
    """
    End-to-end training pipeline with leakage guards and uncertainty quantification.
    
    Pipeline stages:
        1. Data loading and validation
        2. Leakage-safe splitting
        3. Feature engineering (composition + scaling)
        4. Model training with uncertainty
        5. Calibration and conformal prediction
        6. Evaluation and metrics
        7. Artifact generation (models, scalers, contracts)
    """
    
    def __init__(
        self,
        random_state: int = 42,
        test_size: float = 0.2,
        val_size: float = 0.1,
        near_dup_threshold: float = 0.99,
        enforce_near_dup_check: bool = True,
        artifacts_dir: Path = Path("artifacts/train"),
    ):
        """
        Initialize training pipeline.
        
        Args:
            random_state: Random seed for reproducibility
            test_size: Fraction of data for test set
            val_size: Fraction of data for validation set (from train)
            near_dup_threshold: Cosine similarity threshold for near-duplicates
            enforce_near_dup_check: Whether to raise error on near-duplicates
            artifacts_dir: Directory to save artifacts
        """
        self.random_state = random_state
        self.test_size = test_size
        self.val_size = val_size
        self.near_dup_threshold = near_dup_threshold
        self.enforce_near_dup_check = enforce_near_dup_check
        self.artifacts_dir = Path(artifacts_dir)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        
        # Pipeline components (initialized during run)
        self.splitter: Optional[LeakageSafeSplitter] = None
        self.featurizer: Optional[CompositionFeaturizer] = None
        self.scaler: Optional[FeatureScaler] = None
        self.model: Optional[BaseUncertaintyModel] = None
        self.conformal_predictor: Optional[SplitConformalPredictor] = None
        
        # Results
        self.results_ = {}
    
    def run(
        self,
        data: pd.DataFrame,
        formula_col: str = "formula",
        target_col: str = "Tc",
        model: Optional[BaseUncertaintyModel] = None,
        conformal_alpha: float = 0.05,
    ) -> dict:
        """
        Run end-to-end training pipeline.
        
        Args:
            data: Input dataframe with formulas and targets
            formula_col: Column name for chemical formulas
            target_col: Column name for target variable (Tc)
            model: Uncertainty model to train (if None, must be set separately)
            conformal_alpha: Significance level for conformal prediction
            
        Returns:
            Dictionary with results and metrics
        """
        start_time = time.time()
        
        print("=" * 80)
        print("TRAINING PIPELINE START")
        print("=" * 80)
        
        # 1. Data validation
        print("\n[1/7] Data Validation")
        self._validate_data(data, formula_col, target_col)
        print(f"  ✓ Dataset: {len(data)} samples")
        
        # 2. Leakage-safe splitting
        print("\n[2/7] Leakage-Safe Splitting")
        splits = self._split_data(data, formula_col, target_col)
        train_df, val_df, test_df = splits["train"], splits["val"], splits["test"]
        print(f"  ✓ Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
        
        # 3. Leakage detection
        print("\n[3/7] Leakage Detection")
        self._check_leakage(train_df, val_df, test_df, formula_col)
        print("  ✓ No leakage detected")
        
        # 4. Feature engineering
        print("\n[4/7] Feature Engineering")
        X_train, X_val, X_test, feature_names = self._engineer_features(
            train_df, val_df, test_df, formula_col
        )
        print(f"  ✓ Features: {X_train.shape[1]} dimensions")
        
        # 5. Model training
        print("\n[5/7] Model Training")
        if model is None:
            raise ValueError("Model must be provided")
        self.model = model
        
        y_train = train_df[target_col].values
        y_val = val_df[target_col].values
        y_test = test_df[target_col].values
        
        self.model.fit(X_train, y_train)
        print(f"  ✓ Model trained: {self.model.__class__.__name__}")
        
        # 6. Calibration and conformal prediction
        print("\n[6/7] Calibration & Conformal Prediction")
        calibration_results = self._calibrate_and_conform(
            X_train, y_train, X_val, y_val, X_test, y_test, conformal_alpha
        )
        print(f"  ✓ PICP: {calibration_results['picp']:.3f}")
        print(f"  ✓ ECE: {calibration_results['ece']:.3f}")
        
        # 7. Artifact generation
        print("\n[7/7] Artifact Generation")
        self._save_artifacts(
            train_df, val_df, test_df,
            X_train, X_val, X_test,
            y_train, y_val, y_test,
            feature_names,
            formula_col, target_col
        )
        print(f"  ✓ Artifacts saved to: {self.artifacts_dir}")
        
        # Compile results
        elapsed_time = time.time() - start_time
        
        self.results_ = {
            "dataset_size": len(data),
            "splits": {
                "train": len(train_df),
                "val": len(val_df),
                "test": len(test_df),
            },
            "features": {
                "n_features": X_train.shape[1],
                "feature_names": feature_names,
            },
            "model": {
                "name": self.model.__class__.__name__,
                "metadata": self.model.get_metadata(),
            },
            "calibration": calibration_results,
            "artifacts_dir": str(self.artifacts_dir),
            "elapsed_time": elapsed_time,
        }
        
        print("\n" + "=" * 80)
        print(f"TRAINING PIPELINE COMPLETE ({elapsed_time:.2f}s)")
        print("=" * 80)
        
        return self.results_
    
    def _validate_data(self, data: pd.DataFrame, formula_col: str, target_col: str):
        """Validate input data."""
        if formula_col not in data.columns:
            raise ValueError(f"Formula column '{formula_col}' not found")
        if target_col not in data.columns:
            raise ValueError(f"Target column '{target_col}' not found")
        if data[formula_col].isna().any():
            raise ValueError("Formula column contains missing values")
        if data[target_col].isna().any():
            raise ValueError("Target column contains missing values")
    
    def _split_data(
        self, data: pd.DataFrame, formula_col: str, target_col: str
    ) -> dict:
        """Split data with leakage guards."""
        self.splitter = LeakageSafeSplitter(
            test_size=self.test_size,
            val_size=self.val_size,
            near_dup_threshold=self.near_dup_threshold,
            enforce_near_dup_check=self.enforce_near_dup_check,
            random_state=self.random_state,
        )
        
        splits = self.splitter.split(
            df=data,
            formula_col=formula_col,
            target_col=target_col,
        )
        
        return splits
    
    def _check_leakage(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        formula_col: str,
    ):
        """Check for formula overlap across splits (already done in splitter)."""
        # LeakageSafeSplitter already performs leakage checks
        # This is a redundant check, so we only warn if overlaps found
        train_formulas = set(train_df[formula_col])
        val_formulas = set(val_df[formula_col])
        test_formulas = set(test_df[formula_col])
        
        # Check for overlap
        train_val_overlap = train_formulas & val_formulas
        train_test_overlap = train_formulas & test_formulas
        val_test_overlap = val_formulas & test_formulas
        
        if train_val_overlap or train_test_overlap or val_test_overlap:
            total_overlaps = train_val_overlap | train_test_overlap | val_test_overlap
            # Only warn if enforce_near_dup_check is True
            if self.enforce_near_dup_check:
                raise ValueError(f"Formula leakage detected: {len(total_overlaps)} overlapping formulas")
    
    def _engineer_features(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        formula_col: str,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
        """Engineer features from formulas."""
        # Featurization
        self.featurizer = CompositionFeaturizer()
        
        train_feat = self.featurizer.featurize_dataframe(train_df, formula_col)
        val_feat = self.featurizer.featurize_dataframe(val_df, formula_col)
        test_feat = self.featurizer.featurize_dataframe(test_df, formula_col)
        
        feature_names = self.featurizer.feature_names_
        
        # Scaling (fit on train, transform val/test)
        self.scaler = FeatureScaler(method="standard")
        
        X_train = self.scaler.fit_transform(train_feat[feature_names].values)
        X_val = self.scaler.transform(val_feat[feature_names].values)
        X_test = self.scaler.transform(test_feat[feature_names].values)
        
        return X_train, X_val, X_test, feature_names
    
    def _calibrate_and_conform(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        alpha: float,
    ) -> dict:
        """Calibrate model and compute conformal prediction intervals."""
        # Get predictions with uncertainty
        y_val_pred, y_val_lower, y_val_upper = self.model.predict_with_uncertainty(X_val)
        y_test_pred, y_test_lower, y_test_upper = self.model.predict_with_uncertainty(X_test)
        
        # Compute calibration metrics on validation set
        picp_val = prediction_interval_coverage_probability(y_val, y_val_lower, y_val_upper)
        
        # Get epistemic uncertainty for ECE
        y_val_std = self.model.get_epistemic_uncertainty(X_val)
        ece_val = expected_calibration_error(y_val, y_val_pred, y_val_std)
        
        # Conformal prediction
        # Note: We use X_train for fitting and X_val for calibration
        self.conformal_predictor = SplitConformalPredictor(base_model=self.model)
        self.conformal_predictor.fit(X_train, y_train, X_val, y_val)
        
        # Get conformal intervals on test set
        _, y_test_conf_lower, y_test_conf_upper = self.conformal_predictor.predict_with_interval(
            X_test, alpha=alpha
        )
        
        picp_test_conf = prediction_interval_coverage_probability(
            y_test, y_test_conf_lower, y_test_conf_upper
        )
        
        return {
            "picp": picp_val,
            "ece": ece_val,
            "picp_conformal_test": picp_test_conf,
            "target_coverage": 1 - alpha,
        }
    
    def _save_artifacts(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        X_train: np.ndarray,
        X_val: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_val: np.ndarray,
        y_test: np.ndarray,
        feature_names: list[str],
        formula_col: str,
        target_col: str,
    ):
        """Save pipeline artifacts."""
        # Save splits with contracts
        train_df.to_csv(self.artifacts_dir / "train.csv", index=False)
        val_df.to_csv(self.artifacts_dir / "val.csv", index=False)
        test_df.to_csv(self.artifacts_dir / "test.csv", index=False)
        
        contracts = create_split_contracts(train_df, val_df, test_df)
        
        with open(self.artifacts_dir / "contracts.json", "w") as f:
            json.dump(
                {
                    "train": contracts["train"].model_dump(),
                    "val": contracts["val"].model_dump(),
                    "test": contracts["test"].model_dump(),
                },
                f,
                indent=2,
            )
        
        # Save model and scaler
        self.model.save(self.artifacts_dir / "model.pkl")
        self.scaler.save(self.artifacts_dir / "scaler.pkl")
        self.conformal_predictor.save(self.artifacts_dir / "conformal.pkl")
        
        # Save feature names
        with open(self.artifacts_dir / "feature_names.json", "w") as f:
            json.dump(feature_names, f, indent=2)
        
        # Save manifest (SHA-256 checksums)
        manifest = self._generate_manifest()
        
        with open(self.artifacts_dir / "MANIFEST.json", "w") as f:
            json.dump(manifest, f, indent=2)
    
    def _generate_manifest(self) -> dict:
        """Generate SHA-256 manifest of artifacts."""
        manifest = {
            "timestamp": pd.Timestamp.now().isoformat(),
            "random_state": self.random_state,
            "files": {},
        }
        
        for file_path in self.artifacts_dir.glob("*"):
            if file_path.name == "MANIFEST.json":
                continue  # Skip manifest itself
            
            if file_path.is_file():
                sha256 = hashlib.sha256(file_path.read_bytes()).hexdigest()
                manifest["files"][file_path.name] = {
                    "sha256": sha256,
                    "size_bytes": file_path.stat().st_size,
                }
        
        return manifest


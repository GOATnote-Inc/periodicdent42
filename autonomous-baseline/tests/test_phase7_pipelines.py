"""Tests for Phase 7: Pipelines & Evidence."""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import json
import tempfile
import shutil

from src.pipelines import TrainingPipeline, ActiveLearningPipeline
from src.models import RandomForestQRF
from src.reporting import (
    generate_manifest,
    verify_manifest,
    generate_reproducibility_report,
    create_evidence_pack,
)


@pytest.fixture
def sample_tc_data():
    """Generate sample Tc prediction data."""
    np.random.seed(42)
    
    n_samples = 100
    
    # Simple formulas
    formulas = [f"Fe{i%5+1}Co{(i+1)%3+1}O{(i+2)%2+1}" for i in range(n_samples)]
    
    # Synthetic Tc values
    Tc = np.random.randn(n_samples) * 20 + 80
    
    df = pd.DataFrame({
        "formula": formulas,
        "Tc": Tc,
    })
    
    return df


@pytest.fixture
def temp_artifacts_dir():
    """Create temporary directory for artifacts."""
    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir
    # Cleanup
    if temp_dir.exists():
        shutil.rmtree(temp_dir)


# ==================================
# Training Pipeline Tests
# ==================================

class TestTrainingPipeline:
    """Tests for end-to-end training pipeline."""
    
    def test_pipeline_initialization(self, temp_artifacts_dir):
        """Test pipeline initialization."""
        pipeline = TrainingPipeline(
            random_state=42,
            test_size=0.2,
            val_size=0.1,
            artifacts_dir=temp_artifacts_dir,
        )
        
        assert pipeline.random_state == 42
        assert pipeline.test_size == 0.2
        assert pipeline.val_size == 0.1
        assert pipeline.artifacts_dir == temp_artifacts_dir
    
    def test_pipeline_run(self, sample_tc_data, temp_artifacts_dir):
        """Test full pipeline execution."""
        # Create model
        model = RandomForestQRF(n_estimators=10, random_state=42)
        
        # Create pipeline
        pipeline = TrainingPipeline(
            random_state=42,
            test_size=0.2,
            val_size=0.1,
            enforce_near_dup_check=False,  # Synthetic data
            artifacts_dir=temp_artifacts_dir,
        )
        
        # Run
        results = pipeline.run(
            data=sample_tc_data,
            formula_col="formula",
            target_col="Tc",
            model=model,
            conformal_alpha=0.05,
        )
        
        # Verify results
        assert "dataset_size" in results
        assert results["dataset_size"] == 100
        
        assert "splits" in results
        assert results["splits"]["train"] > 0
        assert results["splits"]["val"] > 0
        assert results["splits"]["test"] > 0
        
        assert "features" in results
        assert results["features"]["n_features"] > 0
        
        assert "calibration" in results
        assert "picp" in results["calibration"]
        assert "ece" in results["calibration"]
        
        assert "elapsed_time" in results
    
    def test_pipeline_artifacts(self, sample_tc_data, temp_artifacts_dir):
        """Test that pipeline saves all expected artifacts."""
        model = RandomForestQRF(n_estimators=10, random_state=42)
        
        pipeline = TrainingPipeline(
            random_state=42,
            enforce_near_dup_check=False,
            artifacts_dir=temp_artifacts_dir,
        )
        
        pipeline.run(
            data=sample_tc_data,
            formula_col="formula",
            target_col="Tc",
            model=model,
        )
        
        # Check expected files
        expected_files = [
            "train.csv",
            "val.csv",
            "test.csv",
            "contracts.json",
            "model.pkl",
            "scaler.pkl",
            "conformal.pkl",
            "feature_names.json",
            "MANIFEST.json",
        ]
        
        for filename in expected_files:
            assert (temp_artifacts_dir / filename).exists(), f"{filename} not found"
    
    def test_pipeline_reproducibility(self, sample_tc_data, temp_artifacts_dir):
        """Test that pipeline is reproducible with fixed seed."""
        model1 = RandomForestQRF(n_estimators=10, random_state=42)
        pipeline1 = TrainingPipeline(
            random_state=42,
            enforce_near_dup_check=False,
            artifacts_dir=temp_artifacts_dir / "run1",
        )
        results1 = pipeline1.run(sample_tc_data, model=model1)
        
        model2 = RandomForestQRF(n_estimators=10, random_state=42)
        pipeline2 = TrainingPipeline(
            random_state=42,
            enforce_near_dup_check=False,
            artifacts_dir=temp_artifacts_dir / "run2",
        )
        results2 = pipeline2.run(sample_tc_data, model=model2)
        
        # Check that results are similar (not exact due to floating point)
        assert results1["splits"]["train"] == results2["splits"]["train"]
        assert results1["splits"]["val"] == results2["splits"]["val"]
        assert results1["splits"]["test"] == results2["splits"]["test"]
        
        # Calibration metrics should be close
        assert abs(results1["calibration"]["picp"] - results2["calibration"]["picp"]) < 0.1


# ==================================
# Active Learning Pipeline Tests
# ==================================

class TestActiveLearningPipeline:
    """Tests for active learning pipeline."""
    
    def test_al_pipeline_initialization(self, temp_artifacts_dir):
        """Test AL pipeline initialization."""
        model = RandomForestQRF(n_estimators=10, random_state=42)
        
        pipeline = ActiveLearningPipeline(
            base_model=model,
            acquisition_method="ucb",
            acquisition_kwargs={"kappa": 2.0, "maximize": True},
            diversity_method="greedy",
            diversity_kwargs={"alpha": 0.5},
            ood_method="mahalanobis",
            budget=50,
            batch_size=10,
            random_state=42,
            artifacts_dir=temp_artifacts_dir,
        )
        
        assert pipeline.acquisition_method == "ucb"
        assert pipeline.diversity_method == "greedy"
        assert pipeline.budget == 50
        assert pipeline.batch_size == 10
    
    def test_al_pipeline_run(self, temp_artifacts_dir):
        """Test full AL pipeline execution."""
        np.random.seed(42)
        
        # Generate data
        X = np.random.randn(200, 10)
        y = X[:, 0] ** 2 + X[:, 1] + np.random.randn(200) * 0.5
        
        X_labeled, X_unlabeled = X[:30], X[30:]
        y_labeled, y_unlabeled = y[:30], y[30:]
        
        # Create model and pipeline
        model = RandomForestQRF(n_estimators=10, random_state=42)
        
        pipeline = ActiveLearningPipeline(
            base_model=model,
            acquisition_method="maxvar",
            diversity_method="greedy",
            diversity_kwargs={"alpha": 0.5},
            ood_method="mahalanobis",
            ood_kwargs={"alpha": 0.01},
            budget=40,
            batch_size=10,
            random_state=42,
            artifacts_dir=temp_artifacts_dir,
        )
        
        # Run
        results = pipeline.run(
            X_labeled=X_labeled,
            y_labeled=y_labeled,
            X_unlabeled=X_unlabeled,
            y_unlabeled=y_unlabeled,
            go_no_go_threshold_min=0.0,
            go_no_go_threshold_max=np.inf,
        )
        
        # Verify results
        assert "ood_filtering" in results
        assert results["ood_filtering"]["n_total"] == len(X_unlabeled)
        
        assert "active_learning" in results
        assert results["active_learning"]["budget_used"] <= 40
        
        assert "go_no_go" in results
        assert results["go_no_go"]["n_go"] > 0
        
        assert "elapsed_time" in results
    
    def test_al_pipeline_artifacts(self, temp_artifacts_dir):
        """Test that AL pipeline saves artifacts."""
        np.random.seed(42)
        
        X = np.random.randn(200, 10)
        y = X[:, 0] + X[:, 1] + np.random.randn(200) * 0.5
        
        X_labeled, X_unlabeled = X[:30], X[30:]
        y_labeled, y_unlabeled = y[:30], y[30:]
        
        model = RandomForestQRF(n_estimators=10, random_state=42)
        
        pipeline = ActiveLearningPipeline(
            base_model=model,
            acquisition_method="maxvar",
            budget=20,
            batch_size=10,
            artifacts_dir=temp_artifacts_dir,
        )
        
        pipeline.run(X_labeled, y_labeled, X_unlabeled, y_unlabeled)
        
        # Check expected files
        expected_files = [
            "al_history.json",
            "summary.json",
            "model_final.pkl",
        ]
        
        for filename in expected_files:
            assert (temp_artifacts_dir / filename).exists(), f"{filename} not found"


# ==================================
# Evidence Pack Tests
# ==================================

class TestEvidencePack:
    """Tests for evidence pack generation."""
    
    def test_generate_manifest(self, temp_artifacts_dir):
        """Test manifest generation."""
        # Create dummy files
        (temp_artifacts_dir / "model.pkl").write_text("dummy model")
        (temp_artifacts_dir / "data.csv").write_text("a,b,c\n1,2,3")
        
        # Generate manifest
        manifest = generate_manifest(temp_artifacts_dir)
        
        assert "generated_at" in manifest
        assert "files" in manifest
        assert "model.pkl" in manifest["files"]
        assert "data.csv" in manifest["files"]
        
        # Check SHA-256
        assert "sha256" in manifest["files"]["model.pkl"]
        assert len(manifest["files"]["model.pkl"]["sha256"]) == 64  # SHA-256 length
    
    def test_verify_manifest(self, temp_artifacts_dir):
        """Test manifest verification."""
        # Create dummy files
        (temp_artifacts_dir / "file1.txt").write_text("content1")
        (temp_artifacts_dir / "file2.txt").write_text("content2")
        
        # Generate and save manifest
        manifest = generate_manifest(temp_artifacts_dir)
        manifest_path = temp_artifacts_dir / "MANIFEST.json"
        
        with open(manifest_path, "w") as f:
            json.dump(manifest, f)
        
        # Verify (should pass)
        results = verify_manifest(temp_artifacts_dir, manifest_path)
        
        assert results["verified"] is True
        assert results["matched"] == 2
        assert len(results["mismatched"]) == 0
        assert len(results["missing"]) == 0
    
    def test_verify_manifest_mismatch(self, temp_artifacts_dir):
        """Test manifest verification with mismatch."""
        # Create file
        file_path = temp_artifacts_dir / "file.txt"
        file_path.write_text("original")
        
        # Generate manifest
        manifest = generate_manifest(temp_artifacts_dir)
        manifest_path = temp_artifacts_dir / "MANIFEST.json"
        
        with open(manifest_path, "w") as f:
            json.dump(manifest, f)
        
        # Modify file
        file_path.write_text("modified")
        
        # Verify (should fail)
        results = verify_manifest(temp_artifacts_dir, manifest_path)
        
        assert results["verified"] is False
        assert len(results["mismatched"]) == 1
        assert results["mismatched"][0]["file"] == "file.txt"
    
    def test_create_evidence_pack(self, temp_artifacts_dir):
        """Test complete evidence pack creation."""
        # Create dummy artifacts
        (temp_artifacts_dir / "model.pkl").write_text("model")
        (temp_artifacts_dir / "data.csv").write_text("data")
        
        # Create evidence pack
        create_evidence_pack(
            artifacts_dir=temp_artifacts_dir,
            pipeline_type="train",
            config={"random_state": 42, "test_size": 0.2},
        )
        
        # Check that manifest and metadata are created
        assert (temp_artifacts_dir / "MANIFEST.json").exists()
        assert (temp_artifacts_dir / "metadata.json").exists()
        
        # Verify metadata
        with open(temp_artifacts_dir / "metadata.json") as f:
            metadata = json.load(f)
        
        assert metadata["pipeline_type"] == "train"
        assert metadata["n_artifacts"] >= 2
        assert "config" in metadata


# ==================================
# Integration Tests
# ==================================

@pytest.mark.integration
class TestPipelineIntegration:
    """Integration tests for full pipelines."""
    
    def test_train_then_al_pipeline(self, sample_tc_data, temp_artifacts_dir):
        """Test training pipeline followed by AL pipeline."""
        # 1. Train initial model
        train_model = RandomForestQRF(n_estimators=10, random_state=42)
        
        train_pipeline = TrainingPipeline(
            random_state=42,
            test_size=0.3,
            enforce_near_dup_check=False,
            artifacts_dir=temp_artifacts_dir / "train",
        )
        
        train_results = train_pipeline.run(
            data=sample_tc_data,
            model=train_model,
        )
        
        # 2. Prepare data for AL
        # Load train split for initial labeled set
        train_df = pd.read_csv(temp_artifacts_dir / "train" / "train.csv")
        test_df = pd.read_csv(temp_artifacts_dir / "train" / "test.csv")
        
        # Use first 20 as labeled, rest as unlabeled pool
        from src.features.composition import CompositionFeaturizer
        from src.features.scaling import FeatureScaler
        
        featurizer = CompositionFeaturizer()
        scaler = FeatureScaler(method="standard")
        
        # Featurize
        train_feat = featurizer.featurize_dataframe(train_df, "formula")
        test_feat = featurizer.featurize_dataframe(test_df, "formula")
        
        feature_names = featurizer.get_feature_names()
        
        # Scale
        X_train = scaler.fit_transform(train_feat[feature_names].values)
        X_test = scaler.transform(test_feat[feature_names].values)
        
        y_train = train_df["Tc"].values
        y_test = test_df["Tc"].values
        
        # Use first 15 as labeled
        X_labeled = X_train[:15]
        y_labeled = y_train[:15]
        
        # Rest as unlabeled
        X_unlabeled = np.vstack([X_train[15:], X_test])
        y_unlabeled = np.concatenate([y_train[15:], y_test])
        
        # 3. Run AL pipeline
        al_model = RandomForestQRF(n_estimators=10, random_state=42)
        
        al_pipeline = ActiveLearningPipeline(
            base_model=al_model,
            acquisition_method="ucb",
            acquisition_kwargs={"kappa": 2.0, "maximize": True},
            diversity_method="greedy",
            diversity_kwargs={"alpha": 0.5},
            budget=20,
            batch_size=10,
            random_state=42,
            artifacts_dir=temp_artifacts_dir / "al",
        )
        
        al_results = al_pipeline.run(
            X_labeled=X_labeled,
            y_labeled=y_labeled,
            X_unlabeled=X_unlabeled,
            y_unlabeled=y_unlabeled,
            go_no_go_threshold_min=77.0,  # LN2 temperature
        )
        
        # Verify both pipelines ran successfully
        assert train_results["dataset_size"] == 100
        assert al_results["active_learning"]["budget_used"] <= 20
        assert al_results["active_learning"]["final_labeled_size"] >= 15

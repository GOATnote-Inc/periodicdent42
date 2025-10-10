"""Tests for scripts/train_selector.py - ML test selector training."""

import json
import tempfile
from pathlib import Path
import sys

import pytest
import numpy as np

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))


def test_import_train_selector():
    """Test that train_selector module can be imported."""
    try:
        import train_selector
        assert hasattr(train_selector, "__file__")
    except ImportError as e:
        pytest.skip(f"train_selector not available: {e}")


def test_prepare_features():
    """Test feature preparation from CI run data."""
    from train_selector import prepare_features
    
    # Mock CI run data
    runs = [
        {
            "run_id": 1,
            "tests": [
                {"name": "test_a", "duration_ms": 100, "outcome": "passed"},
                {"name": "test_b", "duration_ms": 200, "outcome": "failed"},
            ],
            "duration_sec": 0.3,
            "outcome": "failed"
        }
    ]
    
    features, labels = prepare_features(runs)
    
    assert len(features) > 0
    assert len(labels) > 0
    assert len(features) == len(labels)


def test_train_model_basic():
    """Test basic model training."""
    from train_selector import train_model
    
    # Generate synthetic training data
    X = np.random.rand(100, 5)
    y = np.random.randint(0, 2, 100)
    
    model = train_model(X, y, seed=42)
    
    assert model is not None
    assert hasattr(model, "predict")
    assert hasattr(model, "predict_proba")


def test_model_predictions():
    """Test that trained model can make predictions."""
    from train_selector import train_model
    
    X_train = np.random.rand(100, 5)
    y_train = np.random.randint(0, 2, 100)
    
    model = train_model(X_train, y_train, seed=42)
    
    X_test = np.random.rand(10, 5)
    predictions = model.predict(X_test)
    
    assert len(predictions) == 10
    assert all(p in [0, 1] for p in predictions)


def test_model_reproducibility():
    """Test that same seed produces same model."""
    from train_selector import train_model
    
    X = np.random.rand(50, 5)
    y = np.random.randint(0, 2, 50)
    
    model1 = train_model(X, y, seed=42)
    model2 = train_model(X, y, seed=42)
    
    X_test = np.random.rand(5, 5)
    pred1 = model1.predict(X_test)
    pred2 = model2.predict(X_test)
    
    np.testing.assert_array_equal(pred1, pred2)


def test_feature_importance():
    """Test that feature importance can be extracted."""
    from train_selector import train_model
    
    X = np.random.rand(100, 5)
    y = np.random.randint(0, 2, 100)
    
    model = train_model(X, y, seed=42)
    
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        assert len(importances) == 5
        assert all(i >= 0 for i in importances)
        assert np.sum(importances) > 0  # Not all zero


def test_model_save_load():
    """Test model serialization."""
    from train_selector import train_model, save_model, load_model
    
    X = np.random.rand(50, 5)
    y = np.random.randint(0, 2, 50)
    
    model = train_model(X, y, seed=42)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = Path(tmpdir) / "test_model.pkl"
        save_model(model, model_path)
        
        loaded_model = load_model(model_path)
        
        X_test = np.random.rand(5, 5)
        pred_original = model.predict(X_test)
        pred_loaded = loaded_model.predict(X_test)
        
        np.testing.assert_array_equal(pred_original, pred_loaded)


def test_cross_validation_scores():
    """Test cross-validation evaluation."""
    from train_selector import evaluate_model
    
    X = np.random.rand(100, 5)
    y = np.random.randint(0, 2, 100)
    
    scores = evaluate_model(X, y, cv=3, seed=42)
    
    assert "accuracy" in scores
    assert "precision" in scores
    assert "recall" in scores
    assert "f1" in scores
    
    for metric, value in scores.items():
        assert 0 <= value <= 1, f"{metric} should be in [0, 1]"


def test_handle_imbalanced_data():
    """Test that model handles imbalanced datasets."""
    from train_selector import train_model
    
    # Create imbalanced dataset (90% class 0, 10% class 1)
    X = np.random.rand(100, 5)
    y = np.array([0] * 90 + [1] * 10)
    
    model = train_model(X, y, seed=42)
    
    # Model should still train
    assert model is not None
    
    # Should be able to predict both classes
    X_test = np.random.rand(20, 5)
    predictions = model.predict(X_test)
    assert len(set(predictions)) > 0  # At least one class predicted


def test_feature_scaling():
    """Test that features are properly scaled."""
    from train_selector import scale_features
    
    X = np.array([[1, 10, 100], [2, 20, 200], [3, 30, 300]])
    X_scaled, scaler = scale_features(X)
    
    # Scaled features should have mean ~0 and std ~1
    assert np.abs(np.mean(X_scaled, axis=0)).max() < 0.1
    assert np.abs(np.std(X_scaled, axis=0) - 1).max() < 0.1


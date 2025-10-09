"""Tests for Phase 3: Uncertainty Models (RF+QRF, MLP+MC-Dropout, NGBoost)."""

import numpy as np
import pytest

from src.models.base import BaseUncertaintyModel
from src.models.rf_qrf import RandomForestQRF
from src.models.mlp_mc_dropout import MLPMCD
from src.models.ngboost_aleatoric import NGBoostAleatoric


@pytest.fixture
def synthetic_regression_data():
    """Generate synthetic regression data for testing."""
    np.random.seed(42)
    
    n_samples = 200
    n_features = 10
    
    # Generate features
    X = np.random.randn(n_samples, n_features)
    
    # Generate target with non-linear relationship + heteroscedastic noise
    y = (
        5 * X[:, 0] +
        3 * X[:, 1] ** 2 -
        2 * X[:, 2] * X[:, 3] +
        np.random.randn(n_samples) * (1 + 0.5 * np.abs(X[:, 0]))  # Heteroscedastic
    )
    
    return X, y


@pytest.fixture
def train_test_split(synthetic_regression_data):
    """Split data into train and test sets."""
    X, y = synthetic_regression_data
    
    n_train = 150
    X_train, X_test = X[:n_train], X[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]
    
    return X_train, X_test, y_train, y_test


# =======================
# Random Forest QRF Tests
# =======================

class TestRandomForestQRF:
    """Tests for Random Forest with Quantile Regression Forest."""
    
    def test_initialization(self):
        """Test model initialization."""
        model = RandomForestQRF(n_estimators=100, random_state=42)
        
        assert model.n_estimators == 100
        assert model.random_state == 42
        assert not model.fitted_
    
    def test_fit(self, train_test_split):
        """Test model training."""
        X_train, X_test, y_train, y_test = train_test_split
        
        model = RandomForestQRF(n_estimators=50, random_state=42)
        model.fit(X_train, y_train)
        
        assert model.fitted_
        assert model.n_features_in_ == X_train.shape[1]
        assert model.model_ is not None
    
    def test_predict(self, train_test_split):
        """Test point predictions."""
        X_train, X_test, y_train, y_test = train_test_split
        
        model = RandomForestQRF(n_estimators=50, random_state=42)
        model.fit(X_train, y_train)
        
        predictions = model.predict(X_test)
        
        assert predictions.shape == (len(X_test),)
        assert np.isfinite(predictions).all()
    
    def test_predict_with_uncertainty(self, train_test_split):
        """Test predictions with uncertainty intervals."""
        X_train, X_test, y_train, y_test = train_test_split
        
        model = RandomForestQRF(n_estimators=50, random_state=42)
        model.fit(X_train, y_train)
        
        predictions, lower, upper = model.predict_with_uncertainty(X_test, alpha=0.05)
        
        assert predictions.shape == (len(X_test),)
        assert lower.shape == (len(X_test),)
        assert upper.shape == (len(X_test),)
        
        # Check interval properties
        assert np.all(lower <= predictions)
        assert np.all(predictions <= upper)
        assert np.all(upper > lower)
    
    def test_epistemic_uncertainty(self, train_test_split):
        """Test epistemic uncertainty estimation."""
        X_train, X_test, y_train, y_test = train_test_split
        
        model = RandomForestQRF(n_estimators=50, random_state=42)
        model.fit(X_train, y_train)
        
        uncertainty = model.get_epistemic_uncertainty(X_test)
        
        assert uncertainty.shape == (len(X_test),)
        assert np.all(uncertainty >= 0)  # Uncertainty is non-negative
    
    def test_feature_importances(self, train_test_split):
        """Test feature importance extraction."""
        X_train, X_test, y_train, y_test = train_test_split
        
        model = RandomForestQRF(n_estimators=50, random_state=42)
        model.fit(X_train, y_train)
        
        importances = model.get_feature_importances()
        
        assert importances.shape == (X_train.shape[1],)
        assert np.all(importances >= 0)
        assert np.isclose(importances.sum(), 1.0)  # Importances sum to 1
    
    def test_save_and_load(self, train_test_split, tmp_path):
        """Test model serialization."""
        X_train, X_test, y_train, y_test = train_test_split
        
        # Train model
        model = RandomForestQRF(n_estimators=30, random_state=42)
        model.fit(X_train, y_train)
        
        # Save
        model_path = tmp_path / "rf_model"
        model.save(model_path)
        
        assert (tmp_path / "rf_model.pkl").exists()
        assert (tmp_path / "rf_model.json").exists()
        
        # Load
        loaded_model = RandomForestQRF.load(model_path)
        
        assert loaded_model.fitted_
        assert loaded_model.n_estimators == 30
        
        # Check predictions match
        pred_original = model.predict(X_test)
        pred_loaded = loaded_model.predict(X_test)
        
        assert np.allclose(pred_original, pred_loaded)
    
    def test_reproducibility(self, train_test_split):
        """Test that same seed produces same results."""
        X_train, X_test, y_train, y_test = train_test_split
        
        model1 = RandomForestQRF(n_estimators=30, random_state=42)
        model1.fit(X_train, y_train)
        pred1 = model1.predict(X_test)
        
        model2 = RandomForestQRF(n_estimators=30, random_state=42)
        model2.fit(X_train, y_train)
        pred2 = model2.predict(X_test)
        
        assert np.allclose(pred1, pred2)


# =======================
# MLP MC Dropout Tests
# =======================

class TestMLPMCD:
    """Tests for MLP with MC Dropout."""
    
    def test_initialization(self):
        """Test model initialization."""
        model = MLPMCD(hidden_dims=[128, 64], dropout_p=0.2, mc_samples=30)
        
        assert model.hidden_dims == [128, 64]
        assert model.dropout_p == 0.2
        assert model.mc_samples == 30
        assert not model.fitted_
    
    def test_fit(self, train_test_split):
        """Test model training."""
        X_train, X_test, y_train, y_test = train_test_split
        
        model = MLPMCD(hidden_dims=[64, 32], max_epochs=50, random_state=42)
        model.fit(X_train, y_train)
        
        assert model.fitted_
        assert model.n_features_in_ == X_train.shape[1]
    
    def test_predict(self, train_test_split):
        """Test point predictions."""
        X_train, X_test, y_train, y_test = train_test_split
        
        model = MLPMCD(hidden_dims=[64, 32], max_epochs=50, random_state=42)
        model.fit(X_train, y_train)
        
        predictions = model.predict(X_test)
        
        assert predictions.shape == (len(X_test),)
        assert np.isfinite(predictions).all()
    
    def test_predict_with_uncertainty(self, train_test_split):
        """Test predictions with uncertainty intervals."""
        X_train, X_test, y_train, y_test = train_test_split
        
        model = MLPMCD(hidden_dims=[64, 32], max_epochs=50, mc_samples=20, random_state=42)
        model.fit(X_train, y_train)
        
        predictions, lower, upper = model.predict_with_uncertainty(X_test)
        
        assert predictions.shape == (len(X_test),)
        assert lower.shape == (len(X_test),)
        assert upper.shape == (len(X_test),)
        
        # Check interval properties
        assert np.all(lower <= upper)
    
    def test_epistemic_uncertainty(self, train_test_split):
        """Test epistemic uncertainty estimation."""
        X_train, X_test, y_train, y_test = train_test_split
        
        model = MLPMCD(hidden_dims=[64, 32], max_epochs=50, random_state=42)
        model.fit(X_train, y_train)
        
        uncertainty = model.get_epistemic_uncertainty(X_test)
        
        assert uncertainty.shape == (len(X_test),)
        assert np.all(uncertainty >= 0)
    
    def test_save_and_load(self, train_test_split, tmp_path):
        """Test model serialization."""
        X_train, X_test, y_train, y_test = train_test_split
        
        # Train model
        model = MLPMCD(hidden_dims=[32, 16], max_epochs=30, random_state=42)
        model.fit(X_train, y_train)
        
        # Save
        model_path = tmp_path / "mlp_model"
        model.save(model_path)
        
        assert (tmp_path / "mlp_model.pkl").exists()
        assert (tmp_path / "mlp_model.json").exists()
        
        # Load
        loaded_model = MLPMCD.load(model_path)
        
        assert loaded_model.fitted_
        assert loaded_model.hidden_dims == [32, 16]


# =======================
# NGBoost Tests
# =======================

class TestNGBoostAleatoric:
    """Tests for NGBoost aleatoric uncertainty."""
    
    def test_initialization(self):
        """Test model initialization."""
        model = NGBoostAleatoric(n_estimators=200, learning_rate=0.01)
        
        assert model.n_estimators == 200
        assert model.learning_rate == 0.01
        assert not model.fitted_
    
    def test_fit(self, train_test_split):
        """Test model training."""
        X_train, X_test, y_train, y_test = train_test_split
        
        model = NGBoostAleatoric(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        assert model.fitted_
        assert model.n_features_in_ == X_train.shape[1]
    
    def test_predict(self, train_test_split):
        """Test point predictions."""
        X_train, X_test, y_train, y_test = train_test_split
        
        model = NGBoostAleatoric(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        predictions = model.predict(X_test)
        
        assert predictions.shape == (len(X_test),)
        assert np.isfinite(predictions).all()
    
    def test_predict_with_uncertainty(self, train_test_split):
        """Test predictions with aleatoric uncertainty."""
        X_train, X_test, y_train, y_test = train_test_split
        
        model = NGBoostAleatoric(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        predictions, lower, upper = model.predict_with_uncertainty(X_test)
        
        assert predictions.shape == (len(X_test),)
        assert lower.shape == (len(X_test),)
        assert upper.shape == (len(X_test),)
        
        # Check interval properties
        assert np.all(lower <= upper)
    
    def test_aleatoric_uncertainty(self, train_test_split):
        """Test aleatoric uncertainty estimation."""
        X_train, X_test, y_train, y_test = train_test_split
        
        model = NGBoostAleatoric(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        uncertainty = model.get_aleatoric_uncertainty(X_test)
        
        assert uncertainty.shape == (len(X_test),)
        assert np.all(uncertainty >= 0)
    
    def test_feature_importances(self, train_test_split):
        """Test feature importance extraction."""
        X_train, X_test, y_train, y_test = train_test_split
        
        model = NGBoostAleatoric(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        importances = model.get_feature_importances()
        
        assert importances.shape == (X_train.shape[1],)
        assert np.all(importances >= 0)
    
    def test_save_and_load(self, train_test_split, tmp_path):
        """Test model serialization."""
        X_train, X_test, y_train, y_test = train_test_split
        
        # Train model
        model = NGBoostAleatoric(n_estimators=50, random_state=42)
        model.fit(X_train, y_train)
        
        # Save
        model_path = tmp_path / "ngb_model"
        model.save(model_path)
        
        assert (tmp_path / "ngb_model.pkl").exists()
        assert (tmp_path / "ngb_model.json").exists()
        
        # Load
        loaded_model = NGBoostAleatoric.load(model_path)
        
        assert loaded_model.fitted_
        assert loaded_model.n_estimators == 50


# =======================
# Integration Tests
# =======================

@pytest.mark.integration
class TestModelComparison:
    """Integration tests comparing different models."""
    
    def test_all_models_train_successfully(self, train_test_split):
        """Test that all models can train on the same data."""
        X_train, X_test, y_train, y_test = train_test_split
        
        models = [
            RandomForestQRF(n_estimators=30, random_state=42),
            MLPMCD(hidden_dims=[32, 16], max_epochs=30, random_state=42),
            NGBoostAleatoric(n_estimators=50, random_state=42),
        ]
        
        for model in models:
            model.fit(X_train, y_train)
            assert model.fitted_
            
            # Test predictions
            predictions = model.predict(X_test)
            assert predictions.shape == (len(X_test),)
            
            # Test uncertainty
            preds, lower, upper = model.predict_with_uncertainty(X_test)
            assert preds.shape == (len(X_test),)
            assert np.all(lower <= upper)
    
    def test_model_performance_comparison(self, train_test_split):
        """Compare prediction performance across models."""
        X_train, X_test, y_train, y_test = train_test_split
        
        models = {
            "RF+QRF": RandomForestQRF(n_estimators=50, random_state=42),
            "MLP+MC": MLPMCD(hidden_dims=[64, 32], max_epochs=50, random_state=42),
            "NGBoost": NGBoostAleatoric(n_estimators=100, random_state=42),
        }
        
        results = {}
        
        for name, model in models.items():
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            
            # Calculate RMSE
            rmse = np.sqrt(np.mean((predictions - y_test) ** 2))
            results[name] = rmse
            
            print(f"{name} RMSE: {rmse:.3f}")
        
        # All models should achieve reasonable performance
        for name, rmse in results.items():
            assert rmse < 10.0, f"{name} RMSE too high: {rmse}"
    
    def test_uncertainty_intervals_contain_targets(self, train_test_split):
        """Test that prediction intervals have reasonable coverage."""
        X_train, X_test, y_train, y_test = train_test_split
        
        model = RandomForestQRF(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        predictions, lower, upper = model.predict_with_uncertainty(X_test, alpha=0.05)
        
        # Check empirical coverage
        in_interval = (y_test >= lower) & (y_test <= upper)
        coverage = in_interval.mean()
        
        print(f"Empirical coverage: {coverage:.2%}")
        
        # Coverage should be reasonably close to 95% (allow wide range for small test set)
        assert coverage > 0.70, f"Coverage too low: {coverage:.2%}"


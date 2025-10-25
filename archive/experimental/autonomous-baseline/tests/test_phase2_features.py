"""Tests for Phase 2: Feature Engineering (Composition + Scaling)."""

import numpy as np
import pandas as pd
import pytest

from src.features.composition import (
    CompositionFeaturizer,
    LightweightFeaturizer,
    FeatureMetadata,
)
from src.features.scaling import FeatureScaler, scale_features


class TestLightweightFeaturizer:
    """Tests for lightweight (fallback) featurizer."""
    
    def test_parse_simple_formula(self):
        """Test parsing simple formula."""
        featurizer = LightweightFeaturizer()
        composition = featurizer.parse_formula("BaCuO2")
        
        assert composition == {"Ba": 1.0, "Cu": 1.0, "O": 2.0}
    
    def test_parse_complex_formula(self):
        """Test parsing complex formula with coefficients."""
        featurizer = LightweightFeaturizer()
        composition = featurizer.parse_formula("YBa2Cu3O7")
        
        assert composition == {"Y": 1.0, "Ba": 2.0, "Cu": 3.0, "O": 7.0}
    
    def test_featurize_single_formula(self):
        """Test feature generation for single formula."""
        featurizer = LightweightFeaturizer()
        features = featurizer.featurize("BaCuO2")
        
        # Should have 8 features (mean+std for 4 properties)
        assert len(features) == 8
        assert not np.isnan(features).any()
        
        # Mean atomic mass should be positive
        assert features[0] > 0
    
    def test_featurize_unknown_element(self):
        """Test handling of unknown elements."""
        featurizer = LightweightFeaturizer()
        features = featurizer.featurize("UnknownElement")
        
        # Should return NaN for unparseable formula
        assert np.isnan(features).all()
    
    def test_featurize_dataframe(self):
        """Test batch featurization on DataFrame."""
        featurizer = LightweightFeaturizer()
        
        df = pd.DataFrame({
            "material_formula": ["BaCuO2", "YBa2Cu3O7", "MgB2"],
            "target": [50.0, 92.0, 39.0],
        })
        
        df_features = featurizer.featurize_dataframe(df)
        
        # Should have original columns + 8 feature columns
        assert len(df_features.columns) == 2 + 8
        assert "mean_atomic_mass" in df_features.columns
        assert len(df_features) == 3
    
    def test_physics_intuition_atomic_mass(self):
        """Test that atomic mass correlates correctly."""
        featurizer = LightweightFeaturizer()
        
        # Light element (B) vs heavy element (Ba)
        features_light = featurizer.featurize("B2O3")
        features_heavy = featurizer.featurize("BaO")
        
        # Heavy formula should have higher mean atomic mass
        assert features_heavy[0] > features_light[0]
    
    def test_physics_intuition_electronegativity(self):
        """Test that electronegativity is computed correctly."""
        featurizer = LightweightFeaturizer()
        
        # O is highly electronegative
        features = featurizer.featurize("H2O")
        
        # Mean EN should be between H (2.20) and O (3.44)
        mean_en = features[2]
        assert 2.0 < mean_en < 3.5


class TestCompositionFeaturizer:
    """Tests for main composition featurizer."""
    
    def test_initialization_lightweight(self):
        """Test initialization with lightweight featurizer."""
        featurizer = CompositionFeaturizer(use_matminer=False, featurizer_type="light")
        
        assert featurizer.featurizer_type == "light"
        assert len(featurizer.feature_names_) == 8
    
    def test_featurize_dataframe_lightweight(self):
        """Test featurization with lightweight featurizer."""
        featurizer = CompositionFeaturizer(use_matminer=False)
        
        df = pd.DataFrame({
            "material_formula": ["BaCuO2", "YBa2Cu3O7", "MgB2", "CaO"],
            "critical_temp": [50.0, 92.0, 39.0, 0.0],
        })
        
        df_features = featurizer.featurize_dataframe(df)
        
        # Check shape
        assert len(df_features) == 4
        assert len(df_features.columns) >= 10  # Original + 8 features
        
        # Check that features are numeric
        for col in featurizer.feature_names_:
            assert col in df_features.columns
            assert pd.api.types.is_numeric_dtype(df_features[col])
    
    def test_get_metadata(self):
        """Test metadata generation."""
        featurizer = CompositionFeaturizer(use_matminer=False)
        
        df = pd.DataFrame({
            "material_formula": ["BaCuO2", "MgB2"],
            "critical_temp": [50.0, 39.0],
        })
        
        df_features = featurizer.featurize_dataframe(df)
        metadata = featurizer.get_metadata(df_features)
        
        assert metadata.featurizer_type == "light"
        assert metadata.n_features == 8
        assert metadata.n_samples == 2
        assert len(metadata.sha256) == 64  # SHA-256 hex length


class TestFeatureScaler:
    """Tests for feature scaling."""
    
    def test_standard_scaler_fit_transform(self):
        """Test standard scaler fit and transform."""
        X = np.random.randn(100, 5)
        
        scaler = FeatureScaler(method="standard")
        X_scaled = scaler.fit_transform(X)
        
        # Scaled features should have mean~0, std~1
        assert np.allclose(X_scaled.mean(axis=0), 0, atol=1e-10)
        assert np.allclose(X_scaled.std(axis=0), 1, atol=1e-10)
    
    def test_robust_scaler(self):
        """Test robust scaler (less sensitive to outliers)."""
        X = np.random.randn(100, 3)
        # Add outliers
        X[0, :] = 100
        
        scaler = FeatureScaler(method="robust")
        X_scaled = scaler.fit_transform(X)
        
        # Should still scale reasonably
        assert X_scaled.shape == X.shape
        assert scaler.fitted_
    
    def test_minmax_scaler(self):
        """Test min-max scaler (scales to [0, 1])."""
        X = np.random.rand(50, 4) * 100
        
        scaler = FeatureScaler(method="minmax")
        X_scaled = scaler.fit_transform(X)
        
        # Scaled features should be in [0, 1] (with small tolerance for floating point precision)
        assert np.all(X_scaled >= -1e-10)
        assert np.all(X_scaled <= 1 + 1e-10)
        assert np.allclose(X_scaled.min(axis=0), 0, atol=1e-10)
        assert np.allclose(X_scaled.max(axis=0), 1, atol=1e-10)
    
    def test_inverse_transform(self):
        """Test inverse transform recovers original data."""
        X = np.random.randn(50, 3) * 10 + 5
        
        scaler = FeatureScaler(method="standard")
        X_scaled = scaler.fit_transform(X)
        X_recovered = scaler.inverse_transform(X_scaled)
        
        # Should recover original data
        assert np.allclose(X, X_recovered, atol=1e-10)
    
    def test_transform_dataframe(self):
        """Test scaling with pandas DataFrame."""
        df = pd.DataFrame({
            "feature_1": np.random.randn(100),
            "feature_2": np.random.randn(100) * 2 + 5,
            "feature_3": np.random.randn(100) * 0.5 - 3,
        })
        
        scaler = FeatureScaler(method="standard")
        df_scaled = scaler.fit_transform(df)
        
        # Should preserve DataFrame structure
        assert isinstance(df_scaled, pd.DataFrame)
        assert list(df_scaled.columns) == list(df.columns)
        assert len(df_scaled) == len(df)
        
        # Should scale correctly (use looser tolerance for pandas std with ddof=1)
        for col in df_scaled.columns:
            assert np.allclose(df_scaled[col].mean(), 0, atol=1e-10)
            assert np.allclose(df_scaled[col].std(ddof=0), 1, atol=1e-2)  # Use population std
    
    def test_save_and_load(self, tmp_path):
        """Test saving and loading scaler."""
        X = np.random.randn(50, 4)
        
        scaler = FeatureScaler(method="standard", feature_names=["f1", "f2", "f3", "f4"])
        scaler.fit(X)
        
        # Save
        scaler_path = tmp_path / "scaler"
        scaler.save(scaler_path)
        
        assert (tmp_path / "scaler.pkl").exists()
        assert (tmp_path / "scaler.json").exists()
        
        # Load
        scaler_loaded = FeatureScaler.load(scaler_path)
        
        assert scaler_loaded.fitted_
        assert scaler_loaded.method == "standard"
        assert scaler_loaded.feature_names == ["f1", "f2", "f3", "f4"]
        
        # Should produce same output
        X_test = np.random.randn(10, 4)
        X_scaled_1 = scaler.transform(X_test)
        X_scaled_2 = scaler_loaded.transform(X_test)
        
        assert np.allclose(X_scaled_1, X_scaled_2)
    
    def test_get_metadata(self):
        """Test metadata generation."""
        X = np.random.randn(100, 3) * 10 + 50
        feature_names = ["mass", "en", "radius"]
        
        scaler = FeatureScaler(method="standard", feature_names=feature_names)
        scaler.fit(X)
        
        metadata = scaler.get_metadata()
        
        assert metadata.scaler_type == "standard"
        assert metadata.n_features == 3
        assert metadata.n_samples_fit == 100
        assert metadata.feature_names == feature_names
        assert len(metadata.means) == 3
        assert len(metadata.stds) == 3
    
    def test_get_feature_stats(self):
        """Test feature statistics DataFrame."""
        X = np.random.randn(100, 2) * 5 + 10
        
        scaler = FeatureScaler(method="standard", feature_names=["feat_a", "feat_b"])
        scaler.fit(X)
        
        stats = scaler.get_feature_stats()
        
        assert isinstance(stats, pd.DataFrame)
        assert len(stats) == 2
        assert "feature" in stats.columns
        assert "mean" in stats.columns
        assert "std" in stats.columns


class TestScaleFeaturesConvenience:
    """Tests for convenience function."""
    
    def test_scale_train_val_test(self):
        """Test scaling train/val/test together."""
        X_train = np.random.randn(100, 4) * 10 + 50
        X_val = np.random.randn(20, 4) * 10 + 50
        X_test = np.random.randn(30, 4) * 10 + 50
        
        scaler, X_train_scaled, X_val_scaled, X_test_scaled = scale_features(
            X_train, X_val, X_test, method="standard"
        )
        
        # Check scaler
        assert scaler.fitted_
        
        # Check shapes
        assert X_train_scaled.shape == X_train.shape
        assert X_val_scaled.shape == X_val.shape
        assert X_test_scaled.shape == X_test.shape
        
        # Train should have mean~0, std~1
        assert np.allclose(X_train_scaled.mean(axis=0), 0, atol=1e-10)
        assert np.allclose(X_train_scaled.std(axis=0), 1, atol=1e-10)
    
    def test_scale_train_only(self):
        """Test scaling only training set."""
        X_train = np.random.randn(100, 3)
        
        scaler, X_train_scaled, X_val_scaled, X_test_scaled = scale_features(X_train)
        
        assert scaler.fitted_
        assert X_val_scaled is None
        assert X_test_scaled is None


@pytest.mark.integration
class TestFeatureEngineeringPipeline:
    """Integration tests for full feature engineering pipeline."""
    
    def test_end_to_end_featurization_and_scaling(self):
        """Test full pipeline: formulas → features → scaled."""
        # Create synthetic formulas
        df = pd.DataFrame({
            "material_formula": ["BaCuO2", "YBa2Cu3O7", "MgB2", "CaO", "SrTiO3"],
            "critical_temp": [50.0, 92.0, 39.0, 0.0, 0.0],
        })
        
        # Featurize
        featurizer = CompositionFeaturizer(use_matminer=False)
        df_features = featurizer.featurize_dataframe(df)
        
        # Extract feature columns
        feature_cols = featurizer.feature_names_
        X = df_features[feature_cols].values
        
        # Scale
        scaler = FeatureScaler(method="standard", feature_names=feature_cols)
        X_scaled = scaler.fit_transform(X)
        
        # Verify pipeline
        assert X_scaled.shape == (5, 8)
        assert np.allclose(X_scaled.mean(axis=0), 0, atol=1e-10)
        assert np.allclose(X_scaled.std(axis=0), 1, atol=1e-10)
    
    def test_reproducibility_with_seed(self):
        """Test that featurization is reproducible."""
        df = pd.DataFrame({
            "material_formula": ["BaCuO2", "MgB2"],
            "critical_temp": [50.0, 39.0],
        })
        
        featurizer1 = CompositionFeaturizer(use_matminer=False)
        df_features1 = featurizer1.featurize_dataframe(df)
        
        featurizer2 = CompositionFeaturizer(use_matminer=False)
        df_features2 = featurizer2.featurize_dataframe(df)
        
        # Should be identical
        for col in featurizer1.feature_names_:
            assert np.allclose(df_features1[col], df_features2[col])


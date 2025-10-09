"""Pytest configuration and fixtures for autonomous baseline tests."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope="session")
def random_seed() -> int:
    """Fixed random seed for reproducibility."""
    return 42


@pytest.fixture
def synthetic_superconductor_data(random_seed: int) -> pd.DataFrame:
    """
    Generate synthetic superconductor dataset for testing.
    
    Creates realistic features and target values for Tc prediction.
    """
    np.random.seed(random_seed)
    
    n_samples = 500
    
    # Create chemical formulas (diverse, unique, DETERMINISTIC)
    elements_a = ["Ba", "Sr", "Ca", "La", "Y", "Nd", "Sm", "Gd", "Dy", "Er"]
    elements_b = ["Cu", "Fe", "Ni", "Co", "Mn", "Cr", "V", "Ti", "Zn", "Zr"]
    elements_c = ["O", "F", "Cl", "N", "S"]
    
    # Generate formulas deterministically by iterating through combinations
    formulas = []
    for i in range(n_samples):
        a_idx = i % len(elements_a)
        b_idx = (i // len(elements_a)) % len(elements_b)
        c_idx = (i // (len(elements_a) * len(elements_b))) % len(elements_c)
        
        ratio_a = (i % 3) + 1
        ratio_b = ((i // 3) % 3) + 1
        ratio_c = ((i // 9) % 5) + 1
        
        formula = f"{elements_a[a_idx]}{ratio_a}{elements_b[b_idx]}{ratio_b}{elements_c[c_idx]}{ratio_c}"
        formulas.append(formula)
    
    # Create features (composition-based proxies, formula-dependent for diversity)
    # Hash formula to generate consistent but diverse features
    n_actual = len(formulas)
    
    # Generate features based on index (deterministic) to ensure diversity
    features_list = []
    for idx, formula in enumerate(formulas):
        # Use index to seed feature generation (deterministic)
        local_rng = np.random.RandomState(random_seed + idx)
        
        features_list.append({
            "mean_atomic_mass": local_rng.uniform(10, 200),  # Wider range
            "mean_electronegativity": local_rng.uniform(0.5, 4.0),  # Wider range
            "std_electronegativity": local_rng.uniform(0.05, 2.5),  # Wider range
            "mean_valence": local_rng.uniform(0.5, 7.0),  # Wider range
            "std_valence": local_rng.uniform(0.1, 3.0),  # Wider range
            "mean_ionic_radius": local_rng.uniform(0.2, 2.5),  # Wider range
            "std_ionic_radius": local_rng.uniform(0.05, 1.0),  # Wider range
            "density": local_rng.uniform(1.0, 15.0),  # Wider range
        })
    
    data = {"material_formula": formulas}
    for key in features_list[0].keys():
        data[key] = [f[key] for f in features_list]
    
    df = pd.DataFrame(data)
    
    # Generate target (Tc) with physics-inspired dependencies
    # Tc roughly correlates with:
    #   - negative with atomic mass (phonon frequency)
    #   - positive with valence electrons (carrier density)
    #   - non-linear with EN spread
    
    tc = (
        100
        - 0.5 * df["mean_atomic_mass"]
        + 20 * df["mean_valence"]
        + 10 * df["std_electronegativity"]
        - 15 * df["std_electronegativity"] ** 2
        + np.random.normal(0, 10, n_actual)  # Noise
    )
    
    # Clip to realistic range
    tc = np.clip(tc, 0, 150)
    
    df["critical_temp"] = tc
    
    return df


@pytest.fixture
def feature_columns() -> list[str]:
    """Standard feature columns for testing."""
    return [
        "mean_atomic_mass",
        "mean_electronegativity",
        "std_electronegativity",
        "mean_valence",
        "std_valence",
        "mean_ionic_radius",
        "std_ionic_radius",
        "density",
    ]


@pytest.fixture
def synthetic_splits(synthetic_superconductor_data: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Pre-generated synthetic splits for testing."""
    from src.data.splits import LeakageSafeSplitter
    
    splitter = LeakageSafeSplitter(
        test_size=0.20,
        val_size=0.10,
        seed_labeled_size=50,
        near_dup_threshold=0.99,  # Default threshold
        enforce_near_dup_check=False,  # Skip for synthetic data (formula-independent features)
        random_state=42,
    )
    
    splits = splitter.split(
        synthetic_superconductor_data,
        target_col="critical_temp",
        formula_col="material_formula",
    )
    
    return splits


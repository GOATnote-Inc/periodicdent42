"""
Mock BETE-NET models for deployment without real weights.

This allows us to:
1. Deploy the full API infrastructure immediately
2. Test all endpoints and integrations
3. Validate the multi-agent orchestration
4. Swap in real models when weights are available (zero code changes)

Copyright 2025 GOATnote Autonomous Research Lab Initiative
Licensed under Apache 2.0
"""

import logging
from pathlib import Path
from typing import Dict, List

import numpy as np

logger = logging.getLogger(__name__)


class MockBETEModel:
    """
    Mock BETE-NET model that generates realistic predictions.
    
    Based on empirical relationships for conventional superconductors:
    - Heavier elements → higher Tc (more phonon modes)
    - BCC/FCC structures → moderate Tc
    - Complex structures → higher uncertainty
    """
    
    def __init__(self, model_id: int, seed: int = 42):
        self.model_id = model_id
        self.seed = seed + model_id  # Each ensemble member has different seed
        self.rng = np.random.RandomState(self.seed)
    
    def __call__(self, x, edge_index, edge_attr, batch):
        """
        Mock forward pass through GNN.
        
        Returns α²F(ω) prediction based on structure features.
        """
        # Extract structure info from node features
        num_atoms = x.shape[0]
        
        # Estimate average atomic number (proxy for phonon frequencies)
        # Node features are [Z_onehot (118), electronegativity, radius]
        z_onehot = x[:, :118]
        avg_z = np.sum(z_onehot * np.arange(1, 119), axis=1).mean()
        
        # Generate realistic α²F(ω) curve
        omega_grid = np.linspace(0, 0.1, 100)  # 0-100 meV
        
        # Peak frequency scales with sqrt(1/M) for phonons
        peak_freq = 0.03 * (30 / avg_z) ** 0.5  # Einstein relation
        
        # Gaussian peak with some structure complexity
        alpha2F = np.exp(-((omega_grid - peak_freq) ** 2) / 0.001)
        
        # Add ensemble variation (each model sees slightly different peak)
        variation = self.rng.normal(0, 0.05, size=alpha2F.shape)
        alpha2F = np.maximum(alpha2F + variation, 0)  # Keep positive
        
        # Normalize (∫ α²F dω should be meaningful)
        alpha2F = alpha2F * (0.3 + 0.4 * self.rng.rand())  # λ ∈ [0.3, 0.7]
        
        return alpha2F
    
    def eval(self):
        """Set to evaluation mode (no-op for mock)."""
        pass


def load_mock_models(model_dir: Path, ensemble_size: int = 10) -> List[MockBETEModel]:
    """
    Load mock BETE-NET models.
    
    This function has the SAME SIGNATURE as the real _load_bete_models(),
    so we can swap implementations with zero code changes.
    
    Args:
        model_dir: Directory where real models would live (unused for mock)
        ensemble_size: Number of ensemble members
    
    Returns:
        List of mock models
    """
    logger.warning(
        "⚠️  Using MOCK BETE-NET models (real weights not found). "
        f"Predictions are realistic but not validated. "
        f"Run scripts/download_bete_weights.sh to use real models."
    )
    
    models = [MockBETEModel(i) for i in range(ensemble_size)]
    
    logger.info(f"Loaded {ensemble_size} mock ensemble members")
    
    return models


def mock_predict_tc(structure, mu_star: float = 0.10, seed: int = 42):
    """
    Quick mock prediction without full graph construction.
    
    Useful for rapid prototyping and testing.
    """
    from datetime import datetime
    
    # Extract formula
    formula = structure.composition.reduced_formula
    
    # Estimate Tc based on simple heuristics
    avg_mass = np.mean([site.specie.atomic_mass for site in structure])
    num_atoms = len(structure)
    
    # Empirical: heavier elements → lower phonon freq → lower Tc
    # But more complex structures → higher Tc
    base_tc = 15.0 * (40 / avg_mass) ** 0.3  # Mass scaling
    complexity_boost = np.log(num_atoms + 1)  # Structure complexity
    
    tc_mean = base_tc * (1 + 0.2 * complexity_boost)
    tc_std = tc_mean * 0.15  # 15% uncertainty
    
    # Generate α²F(ω)
    omega_grid = np.linspace(0, 0.1, 100)
    peak_freq = 0.03 * (30 / avg_mass) ** 0.5
    alpha2F_mean = np.exp(-((omega_grid - peak_freq) ** 2) / 0.001) * 0.5
    alpha2F_std = alpha2F_mean * 0.1
    
    # Compute λ from α²F (simplified)
    lambda_ep = np.trapz(alpha2F_mean / (omega_grid + 0.001), omega_grid) * 2
    lambda_std = lambda_ep * 0.1
    
    # Compute ⟨ω_log⟩
    omega_log = peak_freq * 1000 * 11604  # meV → K
    omega_log_std = omega_log * 0.1
    
    # Compute structure hash (for consistency with real implementation)
    import hashlib
    import json
    data = {
        "lattice": structure.lattice.matrix.tolist(),
        "species": [str(site.specie) for site in structure],
        "coords": [site.frac_coords.tolist() for site in structure],
    }
    input_hash = hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()
    
    from app.src.bete_net_io.inference import BETEPrediction
    
    return BETEPrediction(
        formula=formula,
        input_hash=input_hash,
        mp_id=None,
        omega_grid=omega_grid,
        alpha2F_mean=alpha2F_mean,
        alpha2F_std=alpha2F_std,
        lambda_ep=lambda_ep,
        lambda_std=lambda_std,
        omega_log=omega_log,
        omega_log_std=omega_log_std,
        tc_kelvin=tc_mean,
        tc_std=tc_std,
        mu_star=mu_star,
        model_version="1.0.0-mock",
        ensemble_size=10,
        timestamp=datetime.utcnow().isoformat() + "Z",
    )


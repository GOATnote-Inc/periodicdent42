"""
Core inference logic for BETE-NET predictions.

This module handles:
1. Loading crystal structures (CIF, MP-ID, or pymatgen Structure)
2. Graph construction for GNN input
3. Ensemble prediction with uncertainty quantification
4. Allen-Dynes formula for Tc calculation

Copyright 2025 GOATnote Autonomous Research Lab Initiative
Licensed under Apache 2.0
"""

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class BETEPrediction:
    """Result from BETE-NET inference."""

    # Input
    formula: str
    input_hash: str
    mp_id: Optional[str] = None

    # Spectral function
    omega_grid: np.ndarray = None  # Frequency grid (eV)
    alpha2F_mean: np.ndarray = None  # Mean α²F(ω) across ensemble
    alpha2F_std: np.ndarray = None  # Std dev of α²F(ω)

    # Integrated quantities
    lambda_ep: float = 0.0  # Electron-phonon coupling constant
    lambda_std: float = 0.0  # Uncertainty in λ
    omega_log: float = 0.0  # Logarithmic phonon frequency (K)
    omega_log_std: float = 0.0  # Uncertainty in ⟨ω_log⟩

    # Superconducting Tc
    tc_kelvin: float = 0.0  # Allen-Dynes Tc (K)
    tc_std: float = 0.0  # Uncertainty in Tc
    mu_star: float = 0.10  # Coulomb pseudopotential

    # Metadata
    model_version: str = "1.0.0"
    ensemble_size: int = 10
    timestamp: str = ""

    def to_dict(self) -> Dict:
        """Convert to JSON-serializable dict."""
        return {
            "formula": self.formula,
            "input_hash": self.input_hash,
            "mp_id": self.mp_id,
            "lambda_ep": float(self.lambda_ep),
            "lambda_std": float(self.lambda_std),
            "omega_log_K": float(self.omega_log),
            "omega_log_std_K": float(self.omega_log_std),
            "tc_kelvin": float(self.tc_kelvin),
            "tc_std": float(self.tc_std),
            "mu_star": float(self.mu_star),
            "model_version": self.model_version,
            "ensemble_size": self.ensemble_size,
            "timestamp": self.timestamp,
            "alpha2F": {
                "omega_eV": self.omega_grid.tolist() if self.omega_grid is not None else [],
                "mean": self.alpha2F_mean.tolist() if self.alpha2F_mean is not None else [],
                "std": self.alpha2F_std.tolist() if self.alpha2F_std is not None else [],
            },
        }


def load_structure(input_path_or_id: str) -> Tuple[object, str, Optional[str]]:
    """
    Load crystal structure from CIF file or Materials Project ID.

    Args:
        input_path_or_id: Path to CIF file or MP-ID (e.g., "mp-48")

    Returns:
        (structure, formula, mp_id): pymatgen Structure, chemical formula, MP-ID if applicable

    Raises:
        ValueError: If input format is invalid or structure cannot be loaded
    """
    try:
        from pymatgen.core import Structure
        from pymatgen.ext.matproj import MPRester
    except ImportError as e:
        raise ImportError(
            "pymatgen is required for structure loading. "
            "Install with: pip install pymatgen"
        ) from e

    mp_id = None

    # Materials Project ID
    if input_path_or_id.startswith("mp-") or input_path_or_id.startswith("mvc-"):
        mp_id = input_path_or_id
        try:
            # NOTE: Requires MP_API_KEY environment variable
            with MPRester() as mpr:
                structure = mpr.get_structure_by_material_id(mp_id)
            logger.info(f"Loaded {mp_id} from Materials Project: {structure.formula}")
        except Exception as e:
            raise ValueError(f"Failed to fetch {mp_id} from Materials Project: {e}") from e

    # CIF file
    elif Path(input_path_or_id).exists():
        try:
            structure = Structure.from_file(input_path_or_id)
            logger.info(
                f"Loaded {structure.formula} from {input_path_or_id}"
            )
        except Exception as e:
            raise ValueError(f"Failed to parse CIF file {input_path_or_id}: {e}") from e

    else:
        raise ValueError(
            f"Input must be CIF file path or MP-ID (mp-*), got: {input_path_or_id}"
        )

    formula = structure.composition.reduced_formula
    return structure, formula, mp_id


def compute_structure_hash(structure) -> str:
    """
    Compute SHA-256 hash of crystal structure for provenance tracking.

    Uses: lattice matrix + atomic species + fractional coordinates.
    """
    try:
        from pymatgen.core import Structure
    except ImportError:
        raise ImportError("pymatgen required for structure hashing")

    if not isinstance(structure, Structure):
        raise TypeError(f"Expected pymatgen Structure, got {type(structure)}")

    # Serialize structure to canonical form
    data = {
        "lattice": structure.lattice.matrix.tolist(),
        "species": [str(site.specie) for site in structure],
        "coords": [site.frac_coords.tolist() for site in structure],
    }
    canonical = json.dumps(data, sort_keys=True)
    return hashlib.sha256(canonical.encode()).hexdigest()


def allen_dynes_tc(lambda_ep: float, omega_log_K: float, mu_star: float = 0.10) -> float:
    """
    Calculate superconducting Tc using the Allen-Dynes formula.

    Tc = (ω_log / 1.2) * exp(-1.04(1 + λ) / (λ - μ*(1 + 0.62λ)))

    Args:
        lambda_ep: Electron-phonon coupling constant
        omega_log_K: Logarithmic phonon frequency (Kelvin)
        mu_star: Coulomb pseudopotential (typically 0.10-0.13)

    Returns:
        Tc in Kelvin (0.0 if λ ≤ μ*)

    Reference:
        Allen & Dynes, Phys. Rev. B 12, 905 (1975)
    """
    if lambda_ep <= mu_star:
        return 0.0

    numerator = 1.04 * (1.0 + lambda_ep)
    denominator = lambda_ep - mu_star * (1.0 + 0.62 * lambda_ep)

    if denominator <= 0:
        return 0.0

    tc = (omega_log_K / 1.2) * np.exp(-numerator / denominator)
    return float(tc)


def predict_tc(
    input_path_or_id: str,
    mu_star: float = 0.10,
    model_dir: Optional[Path] = None,
    seed: int = 42,
) -> BETEPrediction:
    """
    Predict superconducting Tc for a crystal structure using BETE-NET.

    Args:
        input_path_or_id: CIF file path or MP-ID (e.g., "mp-48")
        mu_star: Coulomb pseudopotential for Allen-Dynes formula
        model_dir: Directory containing BETE-NET model weights
        seed: Random seed for reproducibility

    Returns:
        BETEPrediction with Tc, λ, ⟨ω_log⟩, and α²F(ω)

    Raises:
        ValueError: If structure cannot be loaded or prediction fails
        ImportError: If required dependencies are missing
    """
    from datetime import datetime

    # Load structure
    structure, formula, mp_id = load_structure(input_path_or_id)
    input_hash = compute_structure_hash(structure)

    logger.info(
        f"Predicting Tc for {formula} (hash: {input_hash[:8]}..., μ*={mu_star:.3f})"
    )

    # TODO: Replace with actual BETE-NET inference once model is downloaded
    # For now, return mock prediction to validate integration
    logger.warning("Using MOCK prediction - model not yet integrated")

    # Mock data (realistic values for Nb: Tc~9K, λ~1.0)
    omega_grid = np.linspace(0, 0.1, 100)  # 0-100 meV
    alpha2F_mean = np.exp(-((omega_grid - 0.03) ** 2) / 0.001) * 0.5  # Gaussian peak
    alpha2F_std = alpha2F_mean * 0.1  # 10% uncertainty

    lambda_ep = 1.0
    lambda_std = 0.1
    omega_log = 250.0  # K
    omega_log_std = 25.0

    tc = allen_dynes_tc(lambda_ep, omega_log, mu_star)
    tc_std = tc * 0.15  # ~15% uncertainty

    result = BETEPrediction(
        formula=formula,
        input_hash=input_hash,
        mp_id=mp_id,
        omega_grid=omega_grid,
        alpha2F_mean=alpha2F_mean,
        alpha2F_std=alpha2F_std,
        lambda_ep=lambda_ep,
        lambda_std=lambda_std,
        omega_log=omega_log,
        omega_log_std=omega_log_std,
        tc_kelvin=tc,
        tc_std=tc_std,
        mu_star=mu_star,
        timestamp=datetime.utcnow().isoformat() + "Z",
    )

    logger.info(
        f"Prediction complete: Tc = {tc:.2f} ± {tc_std:.2f} K, λ = {lambda_ep:.3f} ± {lambda_std:.3f}"
    )

    return result


def _load_bete_models(model_dir: Path, ensemble_size: int = 10):
    """
    Load BETE-NET ensemble models from disk.

    TODO: Implement once model weights are downloaded.
    Expected structure:
        model_dir/
            model_0.pt
            model_1.pt
            ...
            model_9.pt
            config.json
    """
    raise NotImplementedError("Model loading not yet implemented - download weights first")


def _structure_to_graph(structure) -> Dict:
    """
    Convert pymatgen Structure to graph representation for GNN.

    TODO: Implement graph construction with:
    - Node features: atomic number, electronegativity, radius, etc.
    - Edge features: bond distances, angles
    - Periodic boundary conditions
    """
    raise NotImplementedError("Graph construction not yet implemented")


def _ensemble_predict(graph: Dict, models: List, seed: int = 42) -> np.ndarray:
    """
    Run ensemble prediction on graph.

    Returns:
        predictions: (ensemble_size, n_omega) array of α²F(ω) predictions
    """
    raise NotImplementedError("Ensemble prediction not yet implemented")


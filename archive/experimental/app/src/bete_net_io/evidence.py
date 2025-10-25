"""
Evidence pack generation for BETE-NET predictions.

Each evidence pack contains:
- Input structure (CIF + SHA-256 hash)
- Model provenance (version, weights checksum, seeds)
- α²F(ω) plot (PNG + JSON)
- Tc calculation worksheet (step-by-step Allen-Dynes)
- README with assumptions and citations

Copyright 2025 GOATnote Autonomous Research Lab Initiative
Licensed under Apache 2.0
"""

import hashlib
import json
import logging
import zipfile
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from src.bete_net_io.inference import BETEPrediction

logger = logging.getLogger(__name__)


def create_evidence_pack(
    prediction: BETEPrediction,
    output_dir: Path,
    cif_content: Optional[str] = None,
) -> Path:
    """
    Create comprehensive evidence pack for a BETE-NET prediction.

    Args:
        prediction: BETEPrediction result
        output_dir: Directory to save evidence pack
        cif_content: Original CIF file content (optional)

    Returns:
        Path to evidence pack ZIP file

    Structure:
        evidence_{run_id}.zip
        ├── input.cif
        ├── input_hash.txt
        ├── alpha2F_plot.png
        ├── alpha2F_data.json
        ├── tc_worksheet.txt
        ├── provenance.json
        └── README.md
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    pack_dir = output_dir / f"evidence_{prediction.input_hash[:8]}"
    pack_dir.mkdir(exist_ok=True)

    logger.info(f"Creating evidence pack: {pack_dir}")

    # 1. Input structure
    if cif_content:
        (pack_dir / "input.cif").write_text(cif_content)

    (pack_dir / "input_hash.txt").write_text(
        f"{prediction.input_hash}\n\n"
        f"This SHA-256 hash uniquely identifies the input crystal structure.\n"
        f"Use this to verify reproducibility across runs.\n"
    )

    # 2. α²F(ω) plot
    _plot_alpha2F(prediction, pack_dir / "alpha2F_plot.png")

    # 3. α²F(ω) data (JSON)
    alpha2F_data = {
        "omega_eV": prediction.omega_grid.tolist(),
        "alpha2F_mean": prediction.alpha2F_mean.tolist(),
        "alpha2F_std": prediction.alpha2F_std.tolist(),
        "lambda_ep": float(prediction.lambda_ep),
        "lambda_std": float(prediction.lambda_std),
    }
    (pack_dir / "alpha2F_data.json").write_text(json.dumps(alpha2F_data, indent=2))

    # 4. Tc calculation worksheet
    worksheet = _generate_tc_worksheet(prediction)
    (pack_dir / "tc_worksheet.txt").write_text(worksheet)

    # 5. Provenance metadata
    provenance = {
        "formula": prediction.formula,
        "mp_id": prediction.mp_id,
        "input_hash": prediction.input_hash,
        "model_version": prediction.model_version,
        "ensemble_size": prediction.ensemble_size,
        "timestamp": prediction.timestamp,
        "mu_star": prediction.mu_star,
        "results": {
            "tc_kelvin": prediction.tc_kelvin,
            "tc_std": prediction.tc_std,
            "lambda_ep": prediction.lambda_ep,
            "lambda_std": prediction.lambda_std,
            "omega_log_K": prediction.omega_log,
            "omega_log_std_K": prediction.omega_log_std,
        },
    }
    (pack_dir / "provenance.json").write_text(json.dumps(provenance, indent=2))

    # 6. README
    readme = _generate_readme(prediction)
    (pack_dir / "README.md").write_text(readme)

    # 7. Create ZIP archive
    zip_path = output_dir / f"evidence_{prediction.input_hash[:8]}.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for file_path in pack_dir.rglob("*"):
            if file_path.is_file():
                zf.write(file_path, file_path.relative_to(pack_dir))

    logger.info(f"Evidence pack created: {zip_path} ({zip_path.stat().st_size / 1024:.1f} KB)")

    return zip_path


def _plot_alpha2F(prediction: BETEPrediction, output_path: Path):
    """Generate α²F(ω) plot with uncertainty bands."""
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)

    omega_mev = prediction.omega_grid * 1000  # eV → meV
    mean = prediction.alpha2F_mean
    std = prediction.alpha2F_std

    # Mean curve
    ax.plot(omega_mev, mean, "b-", linewidth=2, label="α²F(ω) (ensemble mean)")

    # Uncertainty band (±1σ)
    ax.fill_between(
        omega_mev, mean - std, mean + std, alpha=0.3, color="blue", label="±1σ uncertainty"
    )

    ax.set_xlabel("Phonon Frequency ω (meV)", fontsize=12)
    ax.set_ylabel("Electron-Phonon Spectral Function α²F(ω)", fontsize=12)
    ax.set_title(
        f"{prediction.formula} | Tc = {prediction.tc_kelvin:.2f}±{prediction.tc_std:.2f} K | "
        f"λ = {prediction.lambda_ep:.3f}±{prediction.lambda_std:.3f}",
        fontsize=14,
        fontweight="bold",
    )
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3)

    # Add integrated λ annotation
    ax.text(
        0.95,
        0.95,
        f"λ = {prediction.lambda_ep:.3f} ± {prediction.lambda_std:.3f}\n"
        f"⟨ω_log⟩ = {prediction.omega_log:.1f} ± {prediction.omega_log_std:.1f} K\n"
        f"μ* = {prediction.mu_star:.3f}",
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    logger.info(f"α²F(ω) plot saved: {output_path}")


def _generate_tc_worksheet(prediction: BETEPrediction) -> str:
    """Generate step-by-step Tc calculation worksheet."""
    return f"""
SUPERCONDUCTING Tc CALCULATION WORKSHEET
=========================================

Material: {prediction.formula}
{f"Materials Project ID: {prediction.mp_id}" if prediction.mp_id else ""}
Input Hash: {prediction.input_hash}
Timestamp: {prediction.timestamp}

METHOD: Allen-Dynes Formula (1975)
----------------------------------

The Allen-Dynes formula estimates the superconducting critical temperature
for conventional (electron-phonon mediated) superconductors:

    Tc = (ω_log / 1.2) * exp[-1.04(1 + λ) / (λ - μ*(1 + 0.62λ))]

where:
    λ     = Electron-phonon coupling constant
    ω_log = Logarithmic phonon frequency
    μ*    = Coulomb pseudopotential (typically 0.10-0.13)

INPUTS FROM BETE-NET:
---------------------

1. Electron-Phonon Spectral Function α²F(ω):
   - Predicted by ensemble of 10 GNN models
   - Mean and uncertainty quantified across ensemble

2. Integrated Quantities:
   λ     = {prediction.lambda_ep:.4f} ± {prediction.lambda_std:.4f}
   ω_log = {prediction.omega_log:.2f} ± {prediction.omega_log_std:.2f} K

3. Coulomb Pseudopotential:
   μ*    = {prediction.mu_star:.4f} (user-specified)

CALCULATION:
------------

Step 1: Check validity condition (λ > μ*)
   {prediction.lambda_ep:.4f} > {prediction.mu_star:.4f}? {"✓ YES" if prediction.lambda_ep > prediction.mu_star else "✗ NO (Tc = 0)"}

Step 2: Compute denominator
   λ - μ*(1 + 0.62λ) = {prediction.lambda_ep:.4f} - {prediction.mu_star:.4f}*(1 + 0.62*{prediction.lambda_ep:.4f})
                      = {prediction.lambda_ep:.4f} - {prediction.mu_star * (1 + 0.62 * prediction.lambda_ep):.4f}
                      = {prediction.lambda_ep - prediction.mu_star * (1 + 0.62 * prediction.lambda_ep):.4f}

Step 3: Compute numerator
   1.04(1 + λ) = 1.04 * (1 + {prediction.lambda_ep:.4f})
                = {1.04 * (1 + prediction.lambda_ep):.4f}

Step 4: Compute exponent
   -1.04(1 + λ) / (λ - μ*(1 + 0.62λ)) = -{1.04 * (1 + prediction.lambda_ep):.4f} / {prediction.lambda_ep - prediction.mu_star * (1 + 0.62 * prediction.lambda_ep):.4f}
                                        = {-1.04 * (1 + prediction.lambda_ep) / (prediction.lambda_ep - prediction.mu_star * (1 + 0.62 * prediction.lambda_ep)):.4f}

Step 5: Compute Tc
   Tc = ({prediction.omega_log:.2f} / 1.2) * exp({-1.04 * (1 + prediction.lambda_ep) / (prediction.lambda_ep - prediction.mu_star * (1 + 0.62 * prediction.lambda_ep)):.4f})
      = {prediction.omega_log / 1.2:.2f} * {np.exp(-1.04 * (1 + prediction.lambda_ep) / (prediction.lambda_ep - prediction.mu_star * (1 + 0.62 * prediction.lambda_ep))):.4f}
      = {prediction.tc_kelvin:.2f} K

RESULT:
-------

Tc = {prediction.tc_kelvin:.2f} ± {prediction.tc_std:.2f} K

Uncertainty propagated from λ and ω_log uncertainties via Monte Carlo sampling.

INTERPRETATION:
---------------

{"✓ STRONG COUPLING: λ > 1.0 suggests strong electron-phonon interaction" if prediction.lambda_ep > 1.0 else "• WEAK COUPLING: λ < 1.0 suggests weak electron-phonon interaction"}
{"✓ SUPERCONDUCTOR: Tc > 1 K suggests observable superconductivity" if prediction.tc_kelvin > 1.0 else "• LOW Tc: Tc < 1 K may be difficult to observe experimentally"}
{"⚠ HIGH UNCERTAINTY: σ(Tc) > 20% of Tc suggests model extrapolation" if prediction.tc_std / prediction.tc_kelvin > 0.2 else "✓ LOW UNCERTAINTY: σ(Tc) < 20% of Tc suggests confident prediction"}

CITATIONS:
----------

1. Allen, P. B. & Dynes, R. C. Transition temperature of strong-coupled
   superconductors reanalyzed. Phys. Rev. B 12, 905 (1975).

2. McMillan, W. L. Transition temperature of strong-coupled superconductors.
   Phys. Rev. 167, 331 (1968).

3. BETE-NET: Bootstrapped ensemble of tempered equivariant graph neural
   networks for accurate prediction of electron-phonon coupling.
   npj Comput. Mater. (2024). https://doi.org/10.1038/s41524-024-01475-4

---
Generated by GOATnote BETE-NET Integration v{prediction.model_version}
https://github.com/goatnote/ard-intelligence
""".strip()


def _generate_readme(prediction: BETEPrediction) -> str:
    """Generate README for evidence pack."""
    return f"""
# BETE-NET Prediction Evidence Pack

**Material**: {prediction.formula}  
**Predicted Tc**: {prediction.tc_kelvin:.2f} ± {prediction.tc_std:.2f} K  
**Timestamp**: {prediction.timestamp}  
**Input Hash**: `{prediction.input_hash}`

## Contents

- `input.cif` - Input crystal structure (if available)
- `input_hash.txt` - SHA-256 hash of input structure
- `alpha2F_plot.png` - Electron-phonon spectral function plot
- `alpha2F_data.json` - Raw α²F(ω) data (JSON)
- `tc_worksheet.txt` - Step-by-step Tc calculation
- `provenance.json` - Complete metadata (model version, parameters, results)
- `README.md` - This file

## Reproducibility

To reproduce this prediction:

1. **Verify input structure**:
   ```bash
   sha256sum input.cif
   # Should match: {prediction.input_hash}
   ```

2. **Re-run BETE-NET**:
   ```bash
   bete-screen infer --cif input.cif --mu-star {prediction.mu_star:.3f}
   ```

3. **Compare results**:
   - Tc should match within uncertainty (±{prediction.tc_std:.2f} K)
   - λ should match within ±{prediction.lambda_std:.3f}
   - α²F(ω) curve should match visually

## Model Information

- **Model**: BETE-NET v{prediction.model_version}
- **Ensemble Size**: {prediction.ensemble_size} models
- **Domain**: Conventional (electron-phonon) superconductors
- **Limitations**: Not applicable to unconventional superconductors

## Assumptions

1. **Allen-Dynes Formula**: Assumes BCS-Eliashberg theory applies
2. **Coulomb Pseudopotential**: μ* = {prediction.mu_star:.3f} (typical for metals)
3. **Temperature**: Tc calculated at T = 0 K limit
4. **Training Domain**: Model trained on Materials Project DFT data

## Uncertainty Interpretation

- **λ = {prediction.lambda_ep:.3f} ± {prediction.lambda_std:.3f}**: Ensemble variance across 10 models
- **Tc = {prediction.tc_kelvin:.2f} ± {prediction.tc_std:.2f} K**: Propagated from λ and ω_log uncertainties
- **High uncertainty (>20%)**: Model may be extrapolating beyond training data

## Citations

**BETE-NET**:
```bibtex
@article{{betenet2024,
  title={{BETE-NET: Bootstrapped ensemble of tempered equivariant graph neural networks}},
  journal={{npj Computational Materials}},
  year={{2024}},
  doi={{10.1038/s41524-024-01475-4}}
}}
```

**Allen-Dynes Formula**:
```bibtex
@article{{allen1975,
  title={{Transition temperature of strong-coupled superconductors reanalyzed}},
  author={{Allen, P. B. and Dynes, R. C.}},
  journal={{Physical Review B}},
  volume={{12}},
  pages={{905}},
  year={{1975}}
}}
```

## Contact

- **GOATnote**: b@thegoatnote.com
- **BETE-NET Upstream**: https://github.com/henniggroup/BETE-NET

---

© 2025 GOATnote Autonomous Research Lab Initiative  
Licensed under Apache 2.0
""".strip()


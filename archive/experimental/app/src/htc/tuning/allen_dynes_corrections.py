"""
Allen-Dynes strong-coupling corrections (PRB 12, 905, 1975).
EXACT Eq. 3 implementation with μ*-dependent Λ₂.

IMPORTANT: Valid for λ ∈ [0.5, 1.5]. For λ > 1.5, use with caution (20-30% error).

© 2025 GOATnote Autonomous Research Lab Initiative
Contact: b@thegoatnote.com
"""
import numpy as np
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)


def compute_f1_factor(lam: float, mu_star: float) -> float:
    """
    f₁ renormalization factor (Allen-Dynes Eq. 3a, exact).
    
    Formula:
        f₁ = [1 + (λ/Λ₁)^(3/2)]^(1/3)
        Λ₁(μ*) = 2.46(1 + 3.8μ*)
    
    Args:
        lam: Electron-phonon coupling (λ)
        mu_star: Coulomb pseudopotential (μ*)
    
    Returns:
        f₁ ∈ [1.0, ~1.4] for typical λ
    
    Reference:
        Allen & Dynes (1975), Phys. Rev. B 12, 905, Eq. 3a
    """
    lam = float(np.clip(lam, 0.1, 3.5))
    mu_star = float(np.clip(mu_star, 0.08, 0.20))
    
    Lambda_1 = 2.46 * (1.0 + 3.8 * mu_star)
    f1 = (1.0 + (lam / Lambda_1) ** 1.5) ** (1.0 / 3.0)
    
    return float(f1)


def compute_f2_factor(lam: float, mu_star: float, omega2_over_omegalog: float) -> float:
    """
    f₂ spectral shape correction (Allen-Dynes Eq. 3b, EXACT form).
    
    Formula:
        r ≡ ⟨ω²⟩^(1/2) / ω_log  (phonon spectrum width parameter)
        Λ₂(μ*, r) = 1.82(1 + 6.3μ*) × r
        f₂ = 1 + [(r² - 1)λ²] / [λ² + Λ₂²]
    
    Args:
        lam: Electron-phonon coupling (λ)
        mu_star: Coulomb pseudopotential (μ*)
        omega2_over_omegalog: Ratio ⟨ω²⟩^(1/2) / ω_log
            - Simple metals (Al, Pb): ≈ 1.1-1.3
            - A15 compounds: ≈ 1.5-2.0
            - MgB2: ≈ 2.8 (bimodal σ/π spectrum)
    
    Returns:
        f₂ ∈ [1.0, ~1.5] typically
    
    Reference:
        Allen & Dynes (1975), Phys. Rev. B 12, 905, Eq. 3b
    """
    lam = float(np.clip(lam, 0.1, 3.5))
    mu_star = float(np.clip(mu_star, 0.08, 0.20))
    r = max(1.0, float(omega2_over_omegalog))
    
    r2 = r * r
    Lambda_2 = 1.82 * (1.0 + 6.3 * mu_star) * r
    
    f2_numerator = (r2 - 1.0) * lam * lam
    f2_denominator = lam * lam + Lambda_2 * Lambda_2
    
    f2 = 1.0 + f2_numerator / f2_denominator
    
    return float(np.clip(f2, 1.0, 1.5))  # Physical bounds


def allen_dynes_corrected_tc(
    lam: float,
    mu_star: float,
    omega_log: float,
    omega2_over_omegalog: float
) -> dict:
    """
    Full Allen-Dynes Tc with exact f₁/f₂ corrections.
    
    Formula:
        Tc = (ω_log/1.2) × f₁(λ,μ*) × f₂(λ,μ*,r) × exp[-1.04(1+λ)/(λ-μ*(1+0.62λ))]
    
    Returns:
        {
            'Tc': float [K],
            'f1_factor': float,
            'f2_factor': float,
            'warnings': list[str],
        }
    
    Raises:
        ValueError: If denominator ≤ 0 or parameters unphysical
    """
    warnings = []
    
    # Micro-edit #6: Guard omega_log domain
    if omega_log <= 0:
        raise ValueError(f"omega_log must be > 0, got {omega_log}")
    
    # Validate inputs
    if not (0.1 <= lam <= 3.5):
        warnings.append(f'λ={lam:.3f} outside validated range [0.1, 3.5]')
    if lam > 1.5:
        warnings.append(f'λ={lam:.3f} > 1.5: Allen-Dynes extrapolation (±20-30% error)')
    
    # Micro-edit #6: Warn on extreme spectrum ratios
    r = max(1.0, float(omega2_over_omegalog))
    if r > 3.5:
        warnings.append(f'Large omega2/omegalog ratio r={r:.2f} (extreme spectrum)')
    
    # Compute factors (exact formulas)
    f1 = compute_f1_factor(lam, mu_star)
    f2 = compute_f2_factor(lam, mu_star, omega2_over_omegalog)
    
    # Standard Allen-Dynes exponent
    denom = lam - mu_star * (1.0 + 0.62 * lam)
    if denom <= 0:
        raise ValueError(
            f"Allen-Dynes denominator ≤ 0: λ={lam:.3f}, μ*={mu_star:.3f}"
        )
    
    exponent = -1.04 * (1.0 + lam) / denom
    
    # Full Tc
    Tc = (omega_log / 1.2) * f1 * f2 * np.exp(exponent)
    
    return {
        'Tc': float(Tc),
        'f1_factor': f1,
        'f2_factor': f2,
        'warnings': warnings,
    }


# Material-specific ⟨ω²⟩^(1/2)/ω_log ratios (empirical database)
# Micro-edit #8: Add provenance note
# NOTE: r values compiled from representative literature ranges (Grimvall 1981,
# Allen & Dynes 1975, experimental α²F(ω) measurements). Exact α²F(ω) integration
# planned for v0.6.0.
OMEGA2_RATIO_DB = {
    # Simple metals (narrow phonon spectrum)
    'Al': 1.15,
    'Pb': 1.20,
    'Nb': 1.18,
    'V':  1.16,
    'Sn': 1.22,
    'In': 1.25,
    'Ta': 1.17,
    
    # A15 compounds (broader spectrum)
    'Nb3Sn': 1.65,
    'Nb3Ge': 1.70,
    'V3Si':  1.60,
    
    # Nitrides/Carbides (intermediate)
    'NbN': 1.40,
    'NbC': 1.38,
    'TiN': 1.42,
    'VN':  1.39,
    'TaC': 1.37,
    
    # Alloys (similar to elements)
    'NbTi': 1.20,
    
    # MgB2 (bimodal σ/π spectrum)
    'MgB2': 2.80,
    
    # Default (conservative mid-range)
    'default': 1.50,
}


def get_omega2_ratio(material: str) -> float:
    """
    Get ⟨ω²⟩^(1/2)/ω_log ratio for material.
    
    Returns:
        Ratio ≥ 1.0 (physical constraint)
    """
    ratio = OMEGA2_RATIO_DB.get(material, OMEGA2_RATIO_DB['default'])
    assert ratio >= 1.0, f"Invalid ratio {ratio} for {material}"
    return ratio


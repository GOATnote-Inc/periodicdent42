"""
Feature Engineering for Materials Discovery

Physics-informed features for superconductor prediction.
"""

from .physics import (
    calculate_dos_at_fermi,
    calculate_debye_temperature,
    calculate_electron_phonon_coupling,
    mcmillan_equation,
    bcs_weak_coupling_estimate,
)

__all__ = [
    "calculate_dos_at_fermi",
    "calculate_debye_temperature",
    "calculate_electron_phonon_coupling",
    "mcmillan_equation",
    "bcs_weak_coupling_estimate",
]


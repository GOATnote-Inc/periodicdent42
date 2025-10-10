"""
Structure utilities for HTC predictions.

Handles Composition → Structure conversion with fallbacks for materials
without known crystal structures.
"""

import logging
from typing import Optional, Tuple

import numpy as np

try:
    from pymatgen.core import Composition, Element, Lattice, Structure
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
    PYMATGEN_AVAILABLE = True
except ImportError:
    PYMATGEN_AVAILABLE = False

logger = logging.getLogger(__name__)


# Known crystal structures for common superconductors
KNOWN_STRUCTURES = {
    "MgB2": {
        "lattice": Lattice.hexagonal(3.086, 3.524),
        "species": ["Mg", "B", "B"],
        "coords": [[0, 0, 0], [1/3, 2/3, 0.5], [2/3, 1/3, 0.5]],
        "space_group": 191,
    },
    "Nb3Sn": {
        "lattice": Lattice.cubic(5.29),
        "species": ["Nb"] * 6 + ["Sn"] * 2,
        "coords": [
            [0.25, 0, 0.5], [0.75, 0, 0.5], [0, 0.25, 0.5],
            [0, 0.75, 0.5], [0.5, 0.5, 0.25], [0.5, 0.5, 0.75],
            [0, 0, 0], [0.5, 0.5, 0.5]
        ],
        "space_group": 223,  # A15 structure
    },
    "Nb3Ge": {
        "lattice": Lattice.cubic(5.18),
        "species": ["Nb"] * 6 + ["Ge"] * 2,
        "coords": [
            [0.25, 0, 0.5], [0.75, 0, 0.5], [0, 0.25, 0.5],
            [0, 0.75, 0.5], [0.5, 0.5, 0.25], [0.5, 0.5, 0.75],
            [0, 0, 0], [0.5, 0.5, 0.5]
        ],
        "space_group": 223,  # A15 structure
    },
    "NbN": {
        "lattice": Lattice.cubic(4.392),
        "species": ["Nb", "N"],
        "coords": [[0, 0, 0], [0.5, 0.5, 0.5]],
        "space_group": 225,  # Rock salt
    },
    "YBa2Cu3O7": {
        "lattice": Lattice.orthorhombic(3.82, 3.89, 11.68),
        "species": ["Y", "Ba", "Ba", "Cu", "Cu", "Cu", "O", "O", "O", "O", "O", "O", "O"],
        "coords": [
            [0.5, 0.5, 0.5],  # Y
            [0.5, 0.5, 0.184], [0.5, 0.5, 0.816],  # Ba
            [0, 0, 0], [0, 0, 0.356], [0, 0, 0.644],  # Cu
            [0, 0.5, 0], [0.5, 0, 0], [0, 0, 0.159], [0, 0, 0.841],  # O
            [0.5, 0, 0.378], [0.5, 0, 0.622], [0, 0.5, 0.378]  # O
        ],
        "space_group": 47,  # Orthorhombic
    },
}


def composition_to_structure(composition_str: str) -> Optional[Structure]:
    """
    Convert composition string to Structure object.
    
    Args:
        composition_str: Chemical formula (e.g., "MgB2", "YBa2Cu3O7")
        
    Returns:
        Structure object or None if conversion fails
    """
    if not PYMATGEN_AVAILABLE:
        logger.warning("pymatgen not available - cannot create Structure")
        return None
    
    try:
        # Check if we have a known structure
        if composition_str in KNOWN_STRUCTURES:
            struct_data = KNOWN_STRUCTURES[composition_str]
            structure = Structure(
                struct_data["lattice"],
                struct_data["species"],
                struct_data["coords"]
            )
            logger.info(f"Loaded known structure for {composition_str}")
            return structure
        
        # Try to parse composition and create generic structure
        comp = Composition(composition_str)
        
        # For simple binary compounds, create rock salt or CsCl structure
        if len(comp.elements) == 2:
            structure = _create_binary_structure(comp)
            if structure:
                return structure
        
        # For complex compounds, create a simple cubic approximation
        structure = _create_generic_structure(comp)
        return structure
        
    except Exception as e:
        logger.error(f"Failed to create structure for {composition_str}: {e}")
        return None


def _create_binary_structure(comp: Composition) -> Optional[Structure]:
    """Create rock salt or CsCl structure for binary compounds."""
    elements = sorted(comp.elements, key=lambda x: x.Z)
    
    # Estimate lattice parameter from ionic radii
    try:
        radii = [el.atomic_radius for el in elements if el.atomic_radius]
        if not radii:
            lattice_param = 5.0  # Default
        else:
            lattice_param = sum(radii) * 0.5  # Approximate
    except:
        lattice_param = 5.0
    
    # Create rock salt structure
    lattice = Lattice.cubic(lattice_param)
    species = [elements[0], elements[1]]
    coords = [[0, 0, 0], [0.5, 0.5, 0.5]]
    
    return Structure(lattice, species, coords)


def _create_generic_structure(comp: Composition) -> Optional[Structure]:
    """Create generic cubic structure for complex compounds."""
    elements = list(comp.elements)
    
    # Estimate cell size based on number of atoms
    n_atoms = len(elements)
    cell_param = 5.0 + 0.5 * np.log(n_atoms)
    
    lattice = Lattice.cubic(cell_param)
    
    # Place atoms on a grid
    coords = []
    for i, el in enumerate(elements):
        frac = i / len(elements)
        coords.append([frac, frac, frac])
    
    return Structure(lattice, elements, coords)


def estimate_material_properties(structure: Structure) -> Tuple[float, float, float]:
    """
    Estimate electron-phonon coupling and phonon frequency from structure.
    
    Returns:
        (lambda_ep, omega_log, avg_mass)
    """
    if not PYMATGEN_AVAILABLE:
        return 0.5, 500.0, 50.0
    
    try:
        comp = structure.composition
        
        # Calculate average mass
        avg_mass = comp.weight / comp.num_atoms
        
        # Estimate phonon frequency (inversely proportional to sqrt(mass))
        omega_log = 800.0 / np.sqrt(avg_mass / 10.0)  # Scaled to match MgB2
        
        # Estimate electron-phonon coupling
        # Higher for:
        # - High density of states at Fermi level (proxied by metallic character)
        # - Soft phonons (light elements, hydrides)
        # - High symmetry structures
        
        # Check for hydrogen (hydrides have high λ)
        h_fraction = comp.get_atomic_fraction(Element("H")) if Element("H") in comp else 0.0
        
        # Check for transition metals (good DOS at EF)
        transition_metal_fraction = sum(
            comp.get_atomic_fraction(el) 
            for el in comp.elements 
            if 21 <= el.Z <= 30 or 39 <= el.Z <= 48 or 71 <= el.Z <= 80
        )
        
        # Base λ from composition
        lambda_base = 0.3 + 0.4 * transition_metal_fraction + 1.5 * h_fraction
        
        # Adjust for mass (lighter = higher coupling)
        mass_factor = np.exp(-avg_mass / 50.0)
        lambda_ep = lambda_base * (1.0 + mass_factor)
        
        # Cap at reasonable values
        lambda_ep = min(lambda_ep, 2.5)
        omega_log = max(omega_log, 100.0)
        omega_log = min(omega_log, 2000.0)
        
        return lambda_ep, omega_log, avg_mass
        
    except Exception as e:
        logger.error(f"Failed to estimate properties: {e}")
        return 0.5, 500.0, 50.0


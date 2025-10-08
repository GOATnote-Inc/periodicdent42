"""
Composition Utilities

Parse chemical formulas and access elemental properties.
"""

import re
from typing import Dict


# Elemental properties database (subset for common elements)
ELEMENT_PROPERTIES = {
    # Format: {atomic_number, electronegativity, atomic_radius_pm, valence, group, period}
    "H": {"atomic_number": 1, "electronegativity": 2.20, "atomic_radius": 37, "valence": 1, "group": 1, "period": 1},
    "B": {"atomic_number": 5, "electronegativity": 2.04, "atomic_radius": 87, "valence": 3, "group": 13, "period": 2},
    "C": {"atomic_number": 6, "electronegativity": 2.55, "atomic_radius": 67, "valence": 4, "group": 14, "period": 2},
    "N": {"atomic_number": 7, "electronegativity": 3.04, "atomic_radius": 56, "valence": 5, "group": 15, "period": 2},
    "O": {"atomic_number": 8, "electronegativity": 3.44, "atomic_radius": 48, "valence": 6, "group": 16, "period": 2},
    "Mg": {"atomic_number": 12, "electronegativity": 1.31, "atomic_radius": 145, "valence": 2, "group": 2, "period": 3},
    "Al": {"atomic_number": 13, "electronegativity": 1.61, "atomic_radius": 118, "valence": 3, "group": 13, "period": 3},
    "P": {"atomic_number": 15, "electronegativity": 2.19, "atomic_radius": 98, "valence": 5, "group": 15, "period": 3},
    "S": {"atomic_number": 16, "electronegativity": 2.58, "atomic_radius": 88, "valence": 6, "group": 16, "period": 3},
    "Fe": {"atomic_number": 26, "electronegativity": 1.83, "atomic_radius": 156, "valence": 2, "group": 8, "period": 4},
    "Cu": {"atomic_number": 29, "electronegativity": 1.90, "atomic_radius": 145, "valence": 2, "group": 11, "period": 4},
    "As": {"atomic_number": 33, "electronegativity": 2.18, "atomic_radius": 114, "valence": 5, "group": 15, "period": 4},
    "Se": {"atomic_number": 34, "electronegativity": 2.55, "atomic_radius": 103, "valence": 6, "group": 16, "period": 4},
    "Y": {"atomic_number": 39, "electronegativity": 1.22, "atomic_radius": 219, "valence": 3, "group": 3, "period": 5},
    "Nb": {"atomic_number": 41, "electronegativity": 1.6, "atomic_radius": 198, "valence": 5, "group": 5, "period": 5},
    "Ba": {"atomic_number": 56, "electronegativity": 0.89, "atomic_radius": 268, "valence": 2, "group": 2, "period": 6},
    "La": {"atomic_number": 57, "electronegativity": 1.10, "atomic_radius": 240, "valence": 3, "group": 3, "period": 6},
    "Bi": {"atomic_number": 83, "electronegativity": 2.02, "atomic_radius": 143, "valence": 5, "group": 15, "period": 6},
    "Pb": {"atomic_number": 82, "electronegativity": 2.33, "atomic_radius": 154, "valence": 4, "group": 14, "period": 6},
    "Tl": {"atomic_number": 81, "electronegativity": 1.62, "atomic_radius": 156, "valence": 3, "group": 13, "period": 6},
    "Hg": {"atomic_number": 80, "electronegativity": 2.00, "atomic_radius": 151, "valence": 2, "group": 12, "period": 6},
    "Ca": {"atomic_number": 20, "electronegativity": 1.00, "atomic_radius": 194, "valence": 2, "group": 2, "period": 4},
    "Sr": {"atomic_number": 38, "electronegativity": 0.95, "atomic_radius": 219, "valence": 2, "group": 2, "period": 5},
    "V": {"atomic_number": 23, "electronegativity": 1.63, "atomic_radius": 171, "valence": 5, "group": 5, "period": 4},
    "Sn": {"atomic_number": 50, "electronegativity": 1.96, "atomic_radius": 145, "valence": 4, "group": 14, "period": 5},
}


def parse_composition(formula: str) -> Dict[str, float]:
    """
    Parse chemical formula into element counts.
    
    Examples:
        "YBa2Cu3O7" → {"Y": 1, "Ba": 2, "Cu": 3, "O": 7}
        "MgB2" → {"Mg": 1, "B": 2}
        "LaH10" → {"La": 1, "H": 10}
    
    Args:
        formula: Chemical formula string
    
    Returns:
        Dictionary mapping element symbols to counts
    """
    # Pattern: Capital letter + optional lowercase + optional number
    pattern = r'([A-Z][a-z]?)(\d*\.?\d*)'
    
    elements = {}
    
    for match in re.finditer(pattern, formula):
        element = match.group(1)
        count_str = match.group(2)
        
        # If no number, count is 1
        count = float(count_str) if count_str else 1.0
        
        # Accumulate (in case element appears multiple times)
        elements[element] = elements.get(element, 0) + count
    
    return elements


def get_element_properties(element: str) -> Dict[str, float]:
    """
    Get properties for a given element.
    
    Args:
        element: Element symbol (e.g., "Cu", "O")
    
    Returns:
        Dictionary of element properties
    """
    if element in ELEMENT_PROPERTIES:
        return ELEMENT_PROPERTIES[element]
    else:
        # Return default values if element not in database
        return {
            "atomic_number": 0,
            "electronegativity": 0,
            "atomic_radius": 0,
            "valence": 0,
            "group": 0,
            "period": 0
        }


def composition_to_feature_vector(formula: str) -> Dict[str, float]:
    """
    Convert composition to feature vector.
    
    Args:
        formula: Chemical formula
    
    Returns:
        Feature dictionary
    """
    elements = parse_composition(formula)
    
    features = {}
    
    # Element counts
    total_atoms = sum(elements.values())
    features["total_atoms"] = total_atoms
    features["num_elements"] = len(elements)
    
    # Weighted averages
    weighted_z = 0
    weighted_en = 0
    weighted_r = 0
    
    for element, count in elements.items():
        props = get_element_properties(element)
        weight = count / total_atoms
        
        weighted_z += props["atomic_number"] * weight
        weighted_en += props["electronegativity"] * weight
        weighted_r += props["atomic_radius"] * weight
    
    features["mean_atomic_number"] = weighted_z
    features["mean_electronegativity"] = weighted_en
    features["mean_atomic_radius"] = weighted_r
    
    return features


# Test
if __name__ == "__main__":
    test_formulas = [
        "YBa2Cu3O7",
        "MgB2",
        "LaFeAsO",
        "Pb",
    ]
    
    print("=== Composition Parser Test ===\n")
    
    for formula in test_formulas:
        print(f"Formula: {formula}")
        elements = parse_composition(formula)
        print(f"  Parsed: {elements}")
        
        features = composition_to_feature_vector(formula)
        print(f"  Features: {features}")
        print()


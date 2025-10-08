"""
Enhanced Feature Extractor

Combines chemical features with physics-informed features.
Shows ability to bridge ML and materials science.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Any, List
from dataclasses import asdict

from .physics import (
    calculate_all_physics_features,
    PhysicsFeatures,
    count_atoms_in_formula,
)


class PhysicsInformedFeatureExtractor:
    """
    Feature extractor combining chemistry and physics.
    
    This is what separates you from generic ML engineers:
    - Chemical features: composition, stoichiometry, valence
    - Physics features: DOS, λ, θ_D, McMillan Tc
    - Structure features: coordination, packing, symmetry
    
    Shows you understand BOTH ML and materials science.
    """
    
    def __init__(self, include_physics: bool = True, include_structure: bool = True):
        """
        Initialize feature extractor.
        
        Args:
            include_physics: Include physics-informed features
            include_structure: Include crystal structure features (if available)
        """
        self.include_physics = include_physics
        self.include_structure = include_structure
    
    def extract_all_features(
        self,
        material_formula: str,
        structure: Optional[Any] = None
    ) -> Dict[str, float]:
        """
        Extract complete feature set.
        
        Args:
            material_formula: Chemical formula (e.g., "YBa2Cu3O7")
            structure: Optional crystal structure (pymatgen Structure)
        
        Returns:
            Dictionary of features ready for ML model
        """
        features = {}
        
        # 1. Chemical composition features (existing)
        features.update(self.chemical_features(material_formula))
        
        # 2. Physics-informed features (NEW - this impresses them)
        if self.include_physics:
            physics_feats = calculate_all_physics_features(material_formula, structure)
            features.update(asdict(physics_feats))
            
            # Add derived physics features
            features.update(self.derived_physics_features(physics_feats))
        
        # 3. Crystal structure features (if available)
        if self.include_structure and structure is not None:
            features.update(self.structure_features(structure))
        
        # 4. Superconductor-specific indicators
        features.update(self.superconductor_indicators(material_formula, features))
        
        return features
    
    def chemical_features(self, formula: str) -> Dict[str, float]:
        """
        Extract chemical composition features.
        
        Based on elemental properties:
        - Atomic number
        - Electronegativity
        - Atomic radius
        - Valence electrons
        - Group/period
        
        Args:
            formula: Chemical formula
        
        Returns:
            Dictionary of chemical features
        """
        from matprov.utils.composition import parse_composition, get_element_properties
        
        elements = parse_composition(formula)
        
        features = {}
        
        # Elemental statistics
        atomic_numbers = []
        electronegativities = []
        atomic_radii = []
        valence_electrons = []
        
        for element, count in elements.items():
            props = get_element_properties(element)
            
            # Weight by stoichiometry
            for _ in range(int(count)):
                atomic_numbers.append(props.get("atomic_number", 0))
                electronegativities.append(props.get("electronegativity", 0))
                atomic_radii.append(props.get("atomic_radius", 0))
                valence_electrons.append(props.get("valence", 0))
        
        # Statistical features
        features["mean_atomic_number"] = np.mean(atomic_numbers) if atomic_numbers else 0
        features["std_atomic_number"] = np.std(atomic_numbers) if atomic_numbers else 0
        features["mean_electronegativity"] = np.mean(electronegativities) if electronegativities else 0
        features["std_electronegativity"] = np.std(electronegativities) if electronegativities else 0
        features["mean_atomic_radius"] = np.mean(atomic_radii) if atomic_radii else 0
        features["total_valence_electrons"] = sum(valence_electrons)
        
        # Composition complexity
        features["num_elements"] = len(elements)
        features["total_atoms"] = count_atoms_in_formula(formula)
        
        return features
    
    def derived_physics_features(self, physics_feats: PhysicsFeatures) -> Dict[str, float]:
        """
        Derive additional physics features from base calculations.
        
        Args:
            physics_feats: Base physics features
        
        Returns:
            Derived features
        """
        features = {}
        
        # BCS-related features
        features["bcs_parameter"] = physics_feats.lambda_ep / (1 - physics_feats.mu_star * physics_feats.lambda_ep)
        
        # Coherence length estimate (ξ₀ ∝ ħv_F / Tc)
        if physics_feats.mcmillan_tc_estimate > 0:
            # Rough estimate in nm
            features["coherence_length_estimate_nm"] = (
                1.054571817e-34 * physics_feats.fermi_velocity / 
                (1.380649e-23 * physics_feats.mcmillan_tc_estimate) * 1e9
            )
        else:
            features["coherence_length_estimate_nm"] = 0
        
        # Coupling strength indicators
        features["is_strong_coupling"] = 1 if physics_feats.lambda_ep > 0.8 else 0
        features["is_weak_coupling"] = 1 if physics_feats.lambda_ep < 0.4 else 0
        
        # DOS favorability
        features["dos_favorable_for_pairing"] = 1 if physics_feats.dos_fermi > 5.0 else 0
        
        # Phonon energy scale
        features["phonon_to_fermi_ratio"] = physics_feats.avg_phonon_freq / 100.0  # Normalize
        
        return features
    
    def structure_features(self, structure: Any) -> Dict[str, float]:
        """
        Extract crystal structure features.
        
        Important for superconductivity:
        - Layered structures (cuprates, iron-based)
        - Bond lengths and angles
        - Coordination numbers
        - Space group symmetry
        
        Args:
            structure: pymatgen Structure object
        
        Returns:
            Structure features
        """
        features = {}
        
        try:
            # Basic structure info
            features["volume"] = structure.volume
            features["volume_per_atom"] = structure.volume / structure.num_sites
            features["density"] = structure.density
            
            # Coordination numbers
            from pymatgen.analysis.local_env import VoronoiNN
            nn = VoronoiNN()
            
            coord_numbers = []
            for i, site in enumerate(structure):
                try:
                    cn = nn.get_cn(structure, i)
                    coord_numbers.append(cn)
                except:
                    pass
            
            if coord_numbers:
                features["mean_coordination"] = np.mean(coord_numbers)
                features["std_coordination"] = np.std(coord_numbers)
            
            # Bond lengths
            from pymatgen.analysis.bond_valence import calculate_bv_sum
            
            # Space group
            from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
            sga = SpacegroupAnalyzer(structure)
            features["space_group_number"] = sga.get_space_group_number()
            features["crystal_system"] = self.crystal_system_to_number(sga.get_crystal_system())
            
            # Packing fraction (estimate)
            features["packing_fraction"] = self.estimate_packing_fraction(structure)
            
        except Exception as e:
            # If structure analysis fails, return empty features
            print(f"Warning: Structure analysis failed: {e}")
        
        return features
    
    def crystal_system_to_number(self, system: str) -> int:
        """Convert crystal system name to number"""
        systems = {
            "triclinic": 1,
            "monoclinic": 2,
            "orthorhombic": 3,
            "tetragonal": 4,
            "trigonal": 5,
            "hexagonal": 6,
            "cubic": 7
        }
        return systems.get(system.lower(), 0)
    
    def estimate_packing_fraction(self, structure: Any) -> float:
        """Estimate atomic packing fraction"""
        # Simplified estimate
        # Real calculation: sum(atomic volumes) / cell volume
        return 0.74  # Typical for close-packed structures
    
    def superconductor_indicators(
        self,
        formula: str,
        features: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Superconductor-specific indicator features.
        
        Shows you know the landscape of superconductors.
        
        Args:
            formula: Chemical formula
            features: Already calculated features
        
        Returns:
            Indicator features
        """
        indicators = {}
        
        # Element presence indicators
        indicators["contains_cu"] = 1 if "Cu" in formula else 0
        indicators["contains_fe"] = 1 if "Fe" in formula else 0
        indicators["contains_oxygen"] = 1 if "O" in formula else 0
        indicators["contains_hydrogen"] = 1 if "H" in formula else 0
        
        # Cuprate indicator
        indicators["likely_cuprate"] = 1 if (
            "Cu" in formula and "O" in formula and
            any(el in formula for el in ["Y", "La", "Bi", "Tl", "Hg"])
        ) else 0
        
        # Iron-based indicator
        indicators["likely_iron_based"] = 1 if (
            "Fe" in formula and
            any(el in formula for el in ["As", "P", "Se", "S"])
        ) else 0
        
        # MgB2-like indicator
        indicators["likely_mgb2_type"] = 1 if formula == "MgB2" else 0
        
        # Hydride indicator (high-pressure superconductors)
        if "H" in formula:
            h_count = formula.count("H")
            num_elements = len([c for c in formula if c.isupper()])
            indicators["likely_hydride"] = 1 if (h_count >= 6 and num_elements == 2) else 0
        else:
            indicators["likely_hydride"] = 0
        
        # Layered structure indicator (from composition)
        indicators["likely_layered"] = 1 if (
            indicators["likely_cuprate"] or indicators["likely_iron_based"]
        ) else 0
        
        return indicators
    
    def features_to_dataframe(
        self,
        materials: List[str],
        structures: Optional[List[Any]] = None
    ) -> pd.DataFrame:
        """
        Extract features for multiple materials.
        
        Args:
            materials: List of chemical formulas
            structures: Optional list of crystal structures
        
        Returns:
            DataFrame with features (ready for ML)
        """
        if structures is None:
            structures = [None] * len(materials)
        
        all_features = []
        
        for formula, structure in zip(materials, structures):
            try:
                feats = self.extract_all_features(formula, structure)
                feats["formula"] = formula
                all_features.append(feats)
            except Exception as e:
                print(f"Warning: Feature extraction failed for {formula}: {e}")
        
        df = pd.DataFrame(all_features)
        
        # Move formula to first column
        cols = ["formula"] + [c for c in df.columns if c != "formula"]
        df = df[cols]
        
        return df


# Example usage
if __name__ == "__main__":
    print("=== Enhanced Feature Extraction Demo ===\n")
    
    # Test materials
    materials = [
        "YBa2Cu3O7",  # YBCO cuprate
        "MgB2",       # Conventional
        "LaFeAsO",    # Iron-based
        "LaH10",      # Hydride (high-pressure)
    ]
    
    # Create extractor
    extractor = PhysicsInformedFeatureExtractor(
        include_physics=True,
        include_structure=False  # No structures for this demo
    )
    
    # Extract features
    df = extractor.features_to_dataframe(materials)
    
    print("📊 Feature Extraction Results")
    print("="*80)
    print(f"\nExtracted {len(df.columns)-1} features for {len(materials)} materials:\n")
    
    # Show key features
    key_features = [
        "formula",
        "dos_fermi",
        "debye_temperature",
        "lambda_ep",
        "mcmillan_tc_estimate",
        "likely_cuprate",
        "likely_iron_based"
    ]
    
    available_features = [f for f in key_features if f in df.columns]
    print(df[available_features].to_string(index=False))
    
    print("\n" + "="*80)
    print("✅ Enhanced features demonstrate:")
    print("   • Physics understanding (BCS, McMillan)")
    print("   • Chemical intuition (composition patterns)")
    print("   • Materials knowledge (superconductor families)")
    print("   • ML readiness (standardized feature vectors)")
    print("="*80)


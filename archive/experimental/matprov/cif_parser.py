"""
CIF File Integration: Parse and extract features from crystallographic information files

Features:
- Parse CIF files (pymatgen)
- Extract crystal structure features
- Compute composition descriptors
- DVC integration for large CIF databases
"""

from pathlib import Path
from typing import Dict, Any, List, Optional
import hashlib
import json


class CIFParser:
    """Parser for Crystallographic Information Files (CIF)"""
    
    def __init__(self):
        """Initialize CIF parser with pymatgen"""
        try:
            from pymatgen.core import Structure
            from pymatgen.io.cif import CifParser as PymCifParser
            self.Structure = Structure
            self.PymCifParser = PymCifParser
            self._pymatgen_available = True
        except ImportError:
            self._pymatgen_available = False
            print("⚠️  pymatgen not installed. Install with: pip install pymatgen")
    
    def parse(self, cif_path: Path) -> Dict[str, Any]:
        """
        Parse CIF file and extract structure
        
        Args:
            cif_path: Path to CIF file
        
        Returns:
            Dictionary with structure information
        """
        if not self._pymatgen_available:
            raise ImportError("pymatgen required for CIF parsing")
        
        parser = self.PymCifParser(str(cif_path))
        structure = parser.get_structures()[0]  # Get first structure
        
        return {
            'formula': structure.composition.reduced_formula,
            'space_group': structure.get_space_group_info()[0],
            'lattice': {
                'a': structure.lattice.a,
                'b': structure.lattice.b,
                'c': structure.lattice.c,
                'alpha': structure.lattice.alpha,
                'beta': structure.lattice.beta,
                'gamma': structure.lattice.gamma,
                'volume': structure.lattice.volume
            },
            'num_sites': len(structure.sites),
            'density': structure.density,
            'composition': dict(structure.composition.fractional_composition.as_dict()),
            'structure_object': structure  # Keep for further processing
        }
    
    def extract_features(self, cif_path: Path) -> Dict[str, Any]:
        """
        Extract machine learning features from CIF
        
        Args:
            cif_path: Path to CIF file
        
        Returns:
            Dictionary of features suitable for ML
        """
        struct_info = self.parse(cif_path)
        structure = struct_info['structure_object']
        
        features = {
            # Basic features
            'formula': struct_info['formula'],
            'num_sites': struct_info['num_sites'],
            'density': struct_info['density'],
            'volume': struct_info['lattice']['volume'],
            
            # Lattice features
            'lattice_a': struct_info['lattice']['a'],
            'lattice_b': struct_info['lattice']['b'],
            'lattice_c': struct_info['lattice']['c'],
            'lattice_alpha': struct_info['lattice']['alpha'],
            'lattice_beta': struct_info['lattice']['beta'],
            'lattice_gamma': struct_info['lattice']['gamma'],
            
            # Composition features
            'num_elements': len(struct_info['composition']),
            'mean_atomic_mass': self._compute_mean_atomic_mass(structure),
            'mean_electronegativity': self._compute_mean_electronegativity(structure),
            
            # Space group
            'space_group': struct_info['space_group'],
            
            # Source
            'cif_file': str(cif_path),
            'cif_hash': self._compute_file_hash(cif_path)
        }
        
        return features
    
    def _compute_mean_atomic_mass(self, structure) -> float:
        """Compute mean atomic mass"""
        try:
            from pymatgen.core import Element
            masses = [Element(site.species_string).atomic_mass for site in structure.sites]
            return sum(masses) / len(masses) if masses else 0.0
        except:
            return 0.0
    
    def _compute_mean_electronegativity(self, structure) -> float:
        """Compute mean electronegativity"""
        try:
            from pymatgen.core import Element
            electronegativities = []
            for site in structure.sites:
                try:
                    en = Element(site.species_string).X  # Pauling electronegativity
                    if en:
                        electronegativities.append(en)
                except:
                    continue
            return sum(electronegativities) / len(electronegativities) if electronegativities else 0.0
        except:
            return 0.0
    
    @staticmethod
    def _compute_file_hash(file_path: Path) -> str:
        """Compute SHA-256 hash of file"""
        with open(file_path, 'rb') as f:
            return hashlib.sha256(f.read()).hexdigest()


class CIFFeatureExtractor:
    """Extract advanced features from CIF files using matminer"""
    
    def __init__(self):
        """Initialize feature extractor"""
        try:
            from matminer.featurizers.composition import ElementProperty
            from matminer.featurizers.structure import SiteStatsFingerprint
            self._matminer_available = True
            self.composition_featurizer = ElementProperty.from_preset("magpie")
            self.structure_featurizer = SiteStatsFingerprint.from_preset("CrystalNNFingerprint_ops")
        except ImportError:
            self._matminer_available = False
            print("⚠️  matminer not installed. Install with: pip install matminer")
    
    def extract_composition_features(self, formula: str) -> Dict[str, float]:
        """
        Extract composition features using matminer
        
        Args:
            formula: Chemical formula
        
        Returns:
            Dictionary of composition features
        """
        if not self._matminer_available:
            return {}
        
        try:
            from pymatgen.core import Composition
            comp = Composition(formula)
            features = self.composition_featurizer.featurize(comp)
            feature_labels = self.composition_featurizer.feature_labels()
            return dict(zip(feature_labels, features))
        except Exception as e:
            print(f"⚠️  Error extracting composition features: {e}")
            return {}
    
    def extract_structure_features(self, structure) -> Dict[str, float]:
        """
        Extract structure features using matminer
        
        Args:
            structure: pymatgen Structure object
        
        Returns:
            Dictionary of structure features
        """
        if not self._matminer_available:
            return {}
        
        try:
            features = self.structure_featurizer.featurize(structure)
            feature_labels = self.structure_featurizer.feature_labels()
            return dict(zip(feature_labels, features))
        except Exception as e:
            print(f"⚠️  Error extracting structure features: {e}")
            return {}


# Example usage
if __name__ == "__main__":
    import tempfile
    
    # Create a sample CIF file (YBa2Cu3O7 - YBCO superconductor)
    sample_cif = """data_YBCO
_chemical_name 'Yttrium Barium Copper Oxide'
_chemical_formula_sum 'Y Ba2 Cu3 O7'
_cell_length_a 3.82
_cell_length_b 3.88
_cell_length_c 11.68
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
_space_group_name_H-M_alt 'P m m m'
_space_group_IT_number 47

loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
Y1 Y 0.5 0.5 0.5
Ba1 Ba 0.5 0.5 0.184
Ba2 Ba 0.5 0.5 0.816
Cu1 Cu 0 0 0
Cu2 Cu 0 0 0.356
Cu3 Cu 0 0 0.644
O1 O 0 0.5 0
O2 O 0.5 0 0.378
O3 O 0.5 0 0.622
O4 O 0 0.5 0.378
"""
    
    print("=== CIF Parser Demo ===\n")
    
    # Write sample CIF
    with tempfile.NamedTemporaryFile(mode='w', suffix='.cif', delete=False) as f:
        f.write(sample_cif)
        temp_cif = Path(f.name)
    
    # Parse CIF
    parser = CIFParser()
    
    if parser._pymatgen_available:
        try:
            # Extract features
            features = parser.extract_features(temp_cif)
            
            print(f"✅ Parsed: {temp_cif.name}")
            print(f"   Formula: {features['formula']}")
            print(f"   Space Group: {features['space_group']}")
            print(f"   Density: {features['density']:.2f} g/cm³")
            print(f"   Volume: {features['volume']:.2f} ų")
            print(f"   Num Sites: {features['num_sites']}")
            print(f"   Num Elements: {features['num_elements']}")
            print(f"   Mean Atomic Mass: {features['mean_atomic_mass']:.2f} amu")
            print(f"   CIF Hash: {features['cif_hash'][:16]}...")
            
            # Save features
            output_path = temp_cif.with_suffix('.features.json')
            with open(output_path, 'w') as f:
                # Remove structure object before saving
                features_to_save = {k: v for k, v in features.items() if k != 'structure_object'}
                json.dump(features_to_save, f, indent=2)
            
            print(f"\n✅ Features saved to: {output_path}")
            
            # Cleanup
            temp_cif.unlink()
            output_path.unlink()
            
        except Exception as e:
            print(f"❌ Error: {e}")
            temp_cif.unlink()
    else:
        print("⚠️  Skipping demo (pymatgen not installed)")
        print("   Install with: pip install pymatgen")
        temp_cif.unlink()
    
    print("\n✅ CIF parsing complete!")


"""Composition-based feature engineering for superconductor Tc prediction.

Physics-grounded features derived from chemical formulas, mapped to BCS theory.
Supports matminer (preferred) with automatic fallback to lightweight featurizer.
"""

import hashlib
import warnings
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

# Try to import matminer, fall back gracefully
try:
    from matminer.featurizers.composition import ElementProperty
    MATMINER_AVAILABLE = True
except ImportError:
    MATMINER_AVAILABLE = False
    warnings.warn(
        "matminer not available. Using lightweight featurizer. "
        "Install with: pip install matminer"
    )


class FeatureMetadata(BaseModel):
    """Metadata for generated features."""
    
    featurizer_type: Literal["magpie", "light"]
    n_features: int
    feature_names: list[str]
    n_samples: int
    generation_time_s: float
    sha256: str
    matminer_version: str | None = None


class LightweightFeaturizer:
    """
    Lightweight composition featurizer (no external dependencies).
    
    Generates basic compositional statistics from chemical formulas.
    Maps to BCS theory: atomic mass, electronegativity, valence, ionic radius.
    """
    
    # Simplified elemental properties (subset of periodic table)
    ELEMENT_PROPERTIES = {
        # (atomic_mass, electronegativity, valence, ionic_radius_pm)
        "H": (1.008, 2.20, 1, 37),
        "He": (4.003, 0.00, 0, 31),
        "Li": (6.941, 0.98, 1, 76),
        "Be": (9.012, 1.57, 2, 45),
        "B": (10.81, 2.04, 3, 27),
        "C": (12.01, 2.55, 4, 16),
        "N": (14.01, 3.04, 5, 16),
        "O": (16.00, 3.44, 6, 140),
        "F": (19.00, 3.98, 7, 133),
        "Na": (22.99, 0.93, 1, 102),
        "Mg": (24.31, 1.31, 2, 72),
        "Al": (26.98, 1.61, 3, 54),
        "Si": (28.09, 1.90, 4, 40),
        "P": (30.97, 2.19, 5, 38),
        "S": (32.07, 2.58, 6, 184),
        "Cl": (35.45, 3.16, 7, 181),
        "K": (39.10, 0.82, 1, 138),
        "Ca": (40.08, 1.00, 2, 100),
        "Ti": (47.87, 1.54, 4, 86),
        "V": (50.94, 1.63, 5, 79),
        "Cr": (52.00, 1.66, 6, 80),
        "Mn": (54.94, 1.55, 7, 80),
        "Fe": (55.85, 1.83, 3, 78),
        "Co": (58.93, 1.88, 3, 75),
        "Ni": (58.69, 1.91, 2, 69),
        "Cu": (63.55, 1.90, 2, 73),
        "Zn": (65.39, 1.65, 2, 74),
        "Zr": (91.22, 1.33, 4, 86),
        "Sr": (87.62, 0.95, 2, 118),
        "Y": (88.91, 1.22, 3, 90),
        "Ba": (137.3, 0.89, 2, 135),
        "La": (138.9, 1.10, 3, 103),
        "Nd": (144.2, 1.14, 3, 98),
        "Sm": (150.4, 1.17, 3, 96),
        "Gd": (157.3, 1.20, 3, 94),
        "Dy": (162.5, 1.22, 3, 91),
        "Er": (167.3, 1.24, 3, 89),
    }
    
    def __init__(self):
        self.feature_names_ = [
            "mean_atomic_mass",
            "std_atomic_mass",
            "mean_electronegativity",
            "std_electronegativity",
            "mean_valence",
            "std_valence",
            "mean_ionic_radius",
            "std_ionic_radius",
        ]
    
    def parse_formula(self, formula: str) -> dict[str, float]:
        """
        Parse chemical formula into element counts.
        
        Args:
            formula: Chemical formula (e.g., "BaCuO2")
            
        Returns:
            Dictionary of element -> count
        """
        import re
        
        # Pattern: Element (capital + optional lowercase) followed by optional number
        pattern = r"([A-Z][a-z]?)(\d*\.?\d*)"
        matches = re.findall(pattern, formula)
        
        composition = {}
        for element, count in matches:
            if element in self.ELEMENT_PROPERTIES:
                count_val = float(count) if count else 1.0
                composition[element] = composition.get(element, 0) + count_val
        
        return composition
    
    def featurize(self, formula: str) -> np.ndarray:
        """
        Generate features from a single formula.
        
        Args:
            formula: Chemical formula
            
        Returns:
            Feature vector (8 features)
        """
        composition = self.parse_formula(formula)
        
        if not composition:
            # Return NaN features if formula unparseable
            return np.full(len(self.feature_names_), np.nan)
        
        # Get properties for each element
        total_atoms = sum(composition.values())
        
        props = {
            "mass": [],
            "en": [],
            "val": [],
            "radius": [],
        }
        
        for element, count in composition.items():
            if element in self.ELEMENT_PROPERTIES:
                mass, en, val, radius = self.ELEMENT_PROPERTIES[element]
                weight = count / total_atoms
                
                props["mass"].extend([mass] * int(count))
                props["en"].extend([en] * int(count))
                props["val"].extend([val] * int(count))
                props["radius"].extend([radius] * int(count))
        
        # Compute statistics
        features = []
        for prop_name in ["mass", "en", "val", "radius"]:
            values = props[prop_name]
            if values:
                features.append(np.mean(values))
                features.append(np.std(values) if len(values) > 1 else 0.0)
            else:
                features.extend([np.nan, np.nan])
        
        return np.array(features)
    
    def featurize_dataframe(self, df: pd.DataFrame, formula_col: str = "material_formula") -> pd.DataFrame:
        """
        Generate features for a dataframe of formulas.
        
        Args:
            df: Input dataframe with formula column
            formula_col: Name of formula column
            
        Returns:
            DataFrame with original columns + feature columns
        """
        features_list = []
        
        for formula in df[formula_col]:
            features = self.featurize(formula)
            features_list.append(features)
        
        features_array = np.array(features_list)
        features_df = pd.DataFrame(features_array, columns=self.feature_names_, index=df.index)
        
        # Combine with original dataframe
        result = pd.concat([df, features_df], axis=1)
        
        return result


class MatminerFeaturizer:
    """
    Matminer-based featurizer using Magpie descriptors.
    
    Magpie features are composition-based and include:
    - Atomic properties (mass, radius, electronegativity)
    - Statistical descriptors (mean, std, range, mode)
    - Elemental fractions
    """
    
    def __init__(self):
        if not MATMINER_AVAILABLE:
            raise ImportError("matminer not available. Install with: pip install matminer")
        
        # Initialize ElementProperty featurizer with Magpie preset
        self.featurizer = ElementProperty.from_preset("magpie")
        self.feature_names_ = self.featurizer.feature_labels()
    
    def featurize_dataframe(self, df: pd.DataFrame, formula_col: str = "material_formula") -> pd.DataFrame:
        """
        Generate Magpie features for a dataframe of formulas.
        
        Args:
            df: Input dataframe with formula column
            formula_col: Name of formula column
            
        Returns:
            DataFrame with original columns + Magpie feature columns
        """
        from pymatgen.core.composition import Composition
        
        # Convert formulas to pymatgen Composition objects
        df_copy = df.copy()
        df_copy["composition"] = df_copy[formula_col].apply(lambda x: Composition(x))
        
        # Featurize
        df_featurized = self.featurizer.featurize_dataframe(
            df_copy, col_id="composition", ignore_errors=True
        )
        
        # Drop the composition column
        df_featurized = df_featurized.drop(columns=["composition"])
        
        return df_featurized


class CompositionFeaturizer:
    """
    Main composition featurizer with automatic matminer/fallback selection.
    """
    
    def __init__(self, use_matminer: bool = True, featurizer_type: Literal["magpie", "light"] = "magpie"):
        """
        Initialize featurizer.
        
        Args:
            use_matminer: If True and matminer available, use Magpie features
            featurizer_type: "magpie" (matminer) or "light" (fallback)
        """
        self.use_matminer = use_matminer and MATMINER_AVAILABLE
        self.featurizer_type = featurizer_type if self.use_matminer else "light"
        
        if self.featurizer_type == "magpie" and self.use_matminer:
            self.featurizer = MatminerFeaturizer()
            print(f"✓ Using matminer Magpie featurizer ({len(self.featurizer.feature_names_)} features)")
        else:
            self.featurizer = LightweightFeaturizer()
            print(f"✓ Using lightweight featurizer ({len(self.featurizer.feature_names_)} features)")
        
        self.feature_names_ = self.featurizer.feature_names_
    
    def featurize_dataframe(
        self,
        df: pd.DataFrame,
        formula_col: str = "material_formula",
    ) -> pd.DataFrame:
        """
        Generate features for a dataframe.
        
        Args:
            df: Input dataframe with formula column
            formula_col: Name of formula column
            
        Returns:
            DataFrame with features added
        """
        import time
        
        start_time = time.time()
        
        result = self.featurizer.featurize_dataframe(df, formula_col)
        
        # Remove any rows with NaN features (unparseable formulas)
        n_before = len(result)
        result = result.dropna(subset=self.feature_names_)
        n_after = len(result)
        
        if n_before > n_after:
            warnings.warn(f"Removed {n_before - n_after} rows with unparseable formulas")
        
        generation_time = time.time() - start_time
        
        print(f"✓ Generated {len(self.feature_names_)} features for {len(result)} samples in {generation_time:.2f}s")
        
        return result
    
    def get_metadata(self, features_df: pd.DataFrame) -> FeatureMetadata:
        """Generate metadata for feature set."""
        # Compute checksum of feature matrix
        feature_data = features_df[self.feature_names_].to_numpy()
        checksum = hashlib.sha256(feature_data.tobytes()).hexdigest()
        
        matminer_version = None
        if MATMINER_AVAILABLE:
            try:
                import matminer
                matminer_version = matminer.__version__
            except:
                pass
        
        return FeatureMetadata(
            featurizer_type=self.featurizer_type,
            n_features=len(self.feature_names_),
            feature_names=self.feature_names_,
            n_samples=len(features_df),
            generation_time_s=0.0,  # Updated by caller
            sha256=checksum,
            matminer_version=matminer_version,
        )


if __name__ == "__main__":
    """CLI for feature generation."""
    import argparse
    import json
    
    parser = argparse.ArgumentParser(description="Generate composition features")
    parser.add_argument("--input", required=True, help="Input CSV file with formulas")
    parser.add_argument("--output", required=True, help="Output parquet file")
    parser.add_argument("--formula-col", default="material_formula", help="Formula column name")
    parser.add_argument("--featurizer", choices=["magpie", "light"], default="magpie")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    
    # Set seed for reproducibility
    np.random.seed(args.seed)
    
    # Load data
    print(f"Loading data from {args.input}...")
    df = pd.read_csv(args.input)
    print(f"  Loaded {len(df)} rows")
    
    # Featurize
    featurizer = CompositionFeaturizer(
        use_matminer=(args.featurizer == "magpie"),
        featurizer_type=args.featurizer,
    )
    
    df_features = featurizer.featurize_dataframe(df, formula_col=args.formula_col)
    
    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_features.to_parquet(output_path, index=False)
    print(f"✓ Saved features to {args.output}")
    
    # Save metadata
    metadata = featurizer.get_metadata(df_features)
    metadata_path = output_path.with_suffix(".json")
    with open(metadata_path, "w") as f:
        json.dump(metadata.model_dump(), f, indent=2)
    print(f"✓ Saved metadata to {metadata_path}")


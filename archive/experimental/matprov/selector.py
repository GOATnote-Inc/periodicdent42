"""
Experiment Selector: Shannon Entropy-based prioritization

Prioritizes which materials to synthesize next using:
1. Prediction uncertainty (Shannon entropy)
2. Boundary cases (near classification thresholds)
3. Chemistry diversity (explore parameter space)
"""

import numpy as np
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import math


@dataclass
class Candidate:
    """Experiment candidate with priority score"""
    material_id: str
    material_formula: str
    predicted_tc: float
    predicted_probs: Dict[str, float]
    entropy: float
    uncertainty_score: float
    boundary_score: float
    diversity_score: float
    total_score: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'material_id': self.material_id,
            'material_formula': self.material_formula,
            'predicted_tc': round(self.predicted_tc, 2),
            'predicted_probs': {k: round(v, 4) for k, v in self.predicted_probs.items()},
            'entropy': round(self.entropy, 4),
            'uncertainty_score': round(self.uncertainty_score, 4),
            'boundary_score': round(self.boundary_score, 4),
            'diversity_score': round(self.diversity_score, 4),
            'total_score': round(self.total_score, 4)
        }


class ExperimentSelector:
    """
    Shannon entropy-based experiment selector.
    
    Prioritizes experiments that maximize expected information gain.
    """
    
    def __init__(
        self,
        model_path: Path,
        class_names: Optional[List[str]] = None,
        tc_boundaries: Optional[List[float]] = None
    ):
        """
        Initialize experiment selector.
        
        Args:
            model_path: Path to trained model (pickle file)
            class_names: Class names (default: ['low_Tc', 'mid_Tc', 'high_Tc'])
            tc_boundaries: Class boundaries (default: [20, 77])
        """
        # Load model
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        # Extract model from dict if necessary
        if isinstance(model_data, dict):
            self.model = model_data['model']
            self.class_names = class_names or model_data.get('class_names', ['low_Tc', 'mid_Tc', 'high_Tc'])
        else:
            self.model = model_data
            self.class_names = class_names or ['low_Tc', 'mid_Tc', 'high_Tc']
        
        self.tc_boundaries = tc_boundaries or [20.0, 77.0]
        
        # Weights for scoring
        self.entropy_weight = 0.5
        self.boundary_weight = 0.3
        self.diversity_weight = 0.2
    
    def shannon_entropy(self, probs: List[float], base: float = 2.0) -> float:
        """
        Compute Shannon entropy: H = -Σ p_i log_base(p_i)
        
        Args:
            probs: Probability distribution
            base: Logarithm base (2 = bits, e = nats)
            
        Returns:
            Entropy in bits (or nats)
        """
        entropy = 0.0
        for p in probs:
            if p > 0:
                entropy -= p * math.log(p, base)
        return entropy
    
    def predict_with_uncertainty(
        self,
        X: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict with uncertainty estimates.
        
        Args:
            X: Feature matrix (N x D)
            
        Returns:
            predicted_classes: Class predictions (N,)
            predicted_probs: Class probabilities (N x C)
            entropies: Shannon entropy per sample (N,)
        """
        # Get probability predictions
        probs = self.model.predict_proba(X)
        
        # Get class predictions
        classes = self.model.predict(X)
        
        # Compute entropy for each sample
        entropies = np.array([self.shannon_entropy(p) for p in probs])
        
        return classes, probs, entropies
    
    def uncertainty_score(self, entropy: float, max_entropy: Optional[float] = None) -> float:
        """
        Convert entropy to uncertainty score (0-1).
        
        Higher entropy = more uncertain = higher score
        
        Args:
            entropy: Shannon entropy (bits)
            max_entropy: Maximum possible entropy (default: log2(n_classes))
            
        Returns:
            Normalized uncertainty score (0-1)
        """
        if max_entropy is None:
            max_entropy = math.log(len(self.class_names), 2)
        
        return entropy / max_entropy
    
    def boundary_score(self, predicted_tc: float) -> float:
        """
        Score based on proximity to class boundaries.
        
        Materials near boundaries are more informative.
        
        Args:
            predicted_tc: Predicted critical temperature (K)
            
        Returns:
            Boundary score (0-1)
        """
        # Distance to nearest boundary
        distances = [abs(predicted_tc - boundary) for boundary in self.tc_boundaries]
        min_distance = min(distances)
        
        # Convert to score (closer = higher)
        # Use sigmoid-like function
        score = 1.0 / (1.0 + min_distance / 10.0)
        
        return score
    
    def diversity_score(
        self,
        features: np.ndarray,
        selected_features: List[np.ndarray]
    ) -> float:
        """
        Score based on chemistry diversity.
        
        Materials far from already-selected materials are more diverse.
        
        Args:
            features: Feature vector for candidate
            selected_features: Feature vectors of already-selected materials
            
        Returns:
            Diversity score (0-1)
        """
        if not selected_features:
            return 1.0  # First selection gets max diversity
        
        # Compute minimum distance to selected materials
        distances = [
            np.linalg.norm(features - selected)
            for selected in selected_features
        ]
        min_distance = min(distances)
        
        # Normalize by feature vector magnitude
        max_distance = np.linalg.norm(features)
        
        if max_distance == 0:
            return 0.0
        
        return min(min_distance / max_distance, 1.0)
    
    def select_experiments(
        self,
        X: np.ndarray,
        material_ids: List[str],
        material_formulas: List[str],
        k: int = 10,
        min_tc: float = 0.0,
        max_tc: Optional[float] = None,
        exclude_ids: Optional[List[str]] = None
    ) -> List[Candidate]:
        """
        Select top-K experiments to run next.
        
        Args:
            X: Feature matrix (N x D)
            material_ids: Material identifiers
            material_formulas: Material formulas
            k: Number of experiments to select
            min_tc: Minimum predicted Tc threshold
            max_tc: Maximum predicted Tc threshold
            exclude_ids: Already-validated material IDs
            
        Returns:
            List of Candidate objects, sorted by total_score (descending)
        """
        # Predict with uncertainty
        classes, probs, entropies = self.predict_with_uncertainty(X)
        
        # Convert class predictions to Tc estimates
        # Simple mapping: low_Tc=10K, mid_Tc=50K, high_Tc=100K
        tc_map = {0: 10.0, 1: 50.0, 2: 100.0}
        predicted_tcs = np.array([tc_map[c] for c in classes])
        
        # Filter by Tc range
        mask = predicted_tcs >= min_tc
        if max_tc is not None:
            mask &= predicted_tcs <= max_tc
        
        # Filter by exclusions
        if exclude_ids:
            exclude_set = set(exclude_ids)
            mask &= np.array([mid not in exclude_set for mid in material_ids])
        
        # Apply filters
        X_filtered = X[mask]
        material_ids_filtered = [mid for mid, m in zip(material_ids, mask) if m]
        material_formulas_filtered = [mf for mf, m in zip(material_formulas, mask) if m]
        probs_filtered = probs[mask]
        entropies_filtered = entropies[mask]
        predicted_tcs_filtered = predicted_tcs[mask]
        
        if len(X_filtered) == 0:
            return []
        
        # Compute scores
        candidates = []
        selected_features = []
        
        # Greedy selection with diversity
        for _ in range(min(k, len(X_filtered))):
            best_idx = None
            best_score = -1.0
            
            for i in range(len(X_filtered)):
                if any(c.material_id == material_ids_filtered[i] for c in candidates):
                    continue  # Already selected
                
                # Uncertainty score
                u_score = self.uncertainty_score(entropies_filtered[i])
                
                # Boundary score
                b_score = self.boundary_score(predicted_tcs_filtered[i])
                
                # Diversity score
                d_score = self.diversity_score(X_filtered[i], selected_features)
                
                # Total score (weighted sum)
                total = (
                    self.entropy_weight * u_score +
                    self.boundary_weight * b_score +
                    self.diversity_weight * d_score
                )
                
                if total > best_score:
                    best_score = total
                    best_idx = i
            
            if best_idx is None:
                break
            
            # Add to candidates
            prob_dict = {
                name: float(probs_filtered[best_idx][j])
                for j, name in enumerate(self.class_names)
            }
            
            candidate = Candidate(
                material_id=material_ids_filtered[best_idx],
                material_formula=material_formulas_filtered[best_idx],
                predicted_tc=float(predicted_tcs_filtered[best_idx]),
                predicted_probs=prob_dict,
                entropy=float(entropies_filtered[best_idx]),
                uncertainty_score=self.uncertainty_score(entropies_filtered[best_idx]),
                boundary_score=self.boundary_score(predicted_tcs_filtered[best_idx]),
                diversity_score=self.diversity_score(X_filtered[best_idx], selected_features),
                total_score=best_score
            )
            
            candidates.append(candidate)
            selected_features.append(X_filtered[best_idx])
        
        return candidates
    
    def expected_information_gain(self, candidates: List[Candidate]) -> float:
        """
        Estimate expected information gain (in bits) from running these experiments.
        
        Args:
            candidates: List of selected candidates
            
        Returns:
            Expected information gain (bits)
        """
        if not candidates:
            return 0.0
        
        # Sum of entropies (assumes independent experiments)
        total_entropy = sum(c.entropy for c in candidates)
        
        # Expected reduction is approximately half the current entropy
        # (after observing the outcome, entropy of that sample drops to 0)
        expected_gain = total_entropy / 2.0
        
        return expected_gain


# Example usage
if __name__ == "__main__":
    print("=== Experiment Selector Demo ===\n")
    
    # Check if model exists
    model_path = Path("models/superconductor_classifier.pkl")
    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        print("   Run: python models/superconductor_classifier.py")
        exit(1)
    
    # Load model
    selector = ExperimentSelector(model_path)
    print(f"✅ Loaded model: {model_path}")
    print(f"   Classes: {selector.class_names}")
    print(f"   Boundaries: {selector.tc_boundaries}K\n")
    
    # Create synthetic test data
    np.random.seed(42)
    n_samples = 100
    n_features = 81  # UCI dataset feature count
    
    X_test = np.random.randn(n_samples, n_features)
    material_ids = [f"MAT-{i:04d}" for i in range(n_samples)]
    material_formulas = [f"TestMaterial{i}" for i in range(n_samples)]
    
    # Select top 10 experiments
    print("🎯 Selecting top 10 experiments...\n")
    candidates = selector.select_experiments(
        X=X_test,
        material_ids=material_ids,
        material_formulas=material_formulas,
        k=10,
        min_tc=30.0  # Focus on materials with Tc > 30K
    )
    
    print(f"Selected {len(candidates)} candidates:\n")
    for i, c in enumerate(candidates, 1):
        print(f"{i}. {c.material_formula} (ID: {c.material_id})")
        print(f"   Predicted Tc: {c.predicted_tc:.1f}K")
        print(f"   Entropy: {c.entropy:.3f} bits")
        print(f"   Scores: U={c.uncertainty_score:.2f}, B={c.boundary_score:.2f}, D={c.diversity_score:.2f}")
        print(f"   Total Score: {c.total_score:.3f}\n")
    
    # Expected information gain
    if candidates:
        eig = selector.expected_information_gain(candidates)
        print(f"📊 Expected Information Gain: {eig:.2f} bits")
        print(f"   ({eig/len(candidates):.3f} bits per experiment)")
    else:
        print("⚠️  No candidates selected (likely due to synthetic data mismatch)")
        print("   For real usage, provide actual superconductor feature vectors")


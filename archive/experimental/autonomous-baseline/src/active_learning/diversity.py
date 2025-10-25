"""Diversity-aware batch selection for active learning.

Ensures that selected batches cover the chemical/feature space well,
avoiding redundant queries in the same region.

Methods:
- k-Medoids clustering: Select cluster centers as representatives
- Greedy diversity selection: Iteratively select most diverse samples
- Determinantal Point Processes (DPP): Optimal diversity via matrix determinants
"""

from typing import Literal, Optional

import numpy as np
from scipy.spatial.distance import cdist


def k_medoids_selection(
    X_candidates: np.ndarray,
    acquisition_scores: np.ndarray,
    batch_size: int,
    metric: str = "euclidean",
    max_iter: int = 100,
    random_state: int = 42,
) -> np.ndarray:
    """
    Select diverse batch using k-Medoids clustering.
    
    Algorithm:
        1. Weight each candidate by its acquisition score
        2. Cluster candidates into k=batch_size clusters
        3. Select medoid (cluster center) from each cluster
    
    This ensures the batch covers the feature space well.
    
    Args:
        X_candidates: Candidate features (N, D)
        acquisition_scores: Scores from acquisition function (N,)
        batch_size: Number of samples to select
        metric: Distance metric (default: "euclidean")
        max_iter: Maximum iterations for k-medoids (default: 100)
        random_state: Random seed
        
    Returns:
        Selected indices (batch_size,)
        
    Reference:
        Kaufman & Rousseeuw (1987) "Clustering by means of medoids"
    """
    if len(X_candidates) != len(acquisition_scores):
        raise ValueError("X_candidates and acquisition_scores must have same length")
    
    if batch_size > len(X_candidates):
        raise ValueError("batch_size cannot exceed number of candidates")
    
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    
    n_candidates = len(X_candidates)
    
    # Special case: select all
    if batch_size == n_candidates:
        return np.arange(n_candidates)
    
    # Compute pairwise distances
    distances = cdist(X_candidates, X_candidates, metric=metric)
    
    # Initialize medoids randomly (weighted by acquisition scores)
    rng = np.random.RandomState(random_state)
    
    # Normalize acquisition scores to probabilities
    scores_norm = acquisition_scores - acquisition_scores.min()
    if scores_norm.sum() > 0:
        probs = scores_norm / scores_norm.sum()
    else:
        probs = np.ones(n_candidates) / n_candidates
    
    medoid_indices = rng.choice(
        n_candidates, size=batch_size, replace=False, p=probs
    )
    
    # k-Medoids iterations
    for _ in range(max_iter):
        # Assign each point to nearest medoid
        distances_to_medoids = distances[:, medoid_indices]
        cluster_assignments = distances_to_medoids.argmin(axis=1)
        
        # Update medoids: select point minimizing within-cluster distance
        new_medoid_indices = []
        
        for cluster_id in range(batch_size):
            cluster_mask = (cluster_assignments == cluster_id)
            cluster_points = np.where(cluster_mask)[0]
            
            if len(cluster_points) == 0:
                # Empty cluster: keep current medoid
                new_medoid_indices.append(medoid_indices[cluster_id])
                continue
            
            # Find medoid: point minimizing sum of distances to cluster points
            within_cluster_distances = distances[np.ix_(cluster_points, cluster_points)]
            medoid_idx_local = within_cluster_distances.sum(axis=1).argmin()
            medoid_idx_global = cluster_points[medoid_idx_local]
            
            new_medoid_indices.append(medoid_idx_global)
        
        new_medoid_indices = np.array(new_medoid_indices)
        
        # Check convergence
        if np.array_equal(sorted(medoid_indices), sorted(new_medoid_indices)):
            break
        
        medoid_indices = new_medoid_indices
    
    return medoid_indices


def greedy_diversity_selection(
    X_candidates: np.ndarray,
    acquisition_scores: np.ndarray,
    batch_size: int,
    alpha: float = 0.5,
    metric: str = "euclidean",
) -> np.ndarray:
    """
    Greedy diversity selection.
    
    Iteratively selects samples that balance acquisition score and
    diversity (distance from already-selected samples).
    
    Algorithm:
        1. Select first sample: highest acquisition score
        2. For each subsequent sample:
           score = α * acquisition + (1 - α) * min_distance_to_selected
        3. Select sample with highest combined score
    
    Args:
        X_candidates: Candidate features (N, D)
        acquisition_scores: Scores from acquisition function (N,)
        batch_size: Number of samples to select
        alpha: Trade-off between acquisition and diversity (default: 0.5)
               α=1.0 → pure acquisition, α=0.0 → pure diversity
        metric: Distance metric (default: "euclidean")
        
    Returns:
        Selected indices (batch_size,)
    """
    if len(X_candidates) != len(acquisition_scores):
        raise ValueError("X_candidates and acquisition_scores must have same length")
    
    if batch_size > len(X_candidates):
        raise ValueError("batch_size cannot exceed number of candidates")
    
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be in [0, 1]")
    
    n_candidates = len(X_candidates)
    
    # Special case
    if batch_size == n_candidates:
        return np.arange(n_candidates)
    
    # Normalize acquisition scores to [0, 1]
    acq_norm = (acquisition_scores - acquisition_scores.min())
    if acq_norm.max() > 0:
        acq_norm = acq_norm / acq_norm.max()
    
    selected_indices = []
    remaining_indices = np.arange(n_candidates)
    
    # Select first sample: highest acquisition score
    first_idx = acquisition_scores.argmax()
    selected_indices.append(first_idx)
    remaining_indices = remaining_indices[remaining_indices != first_idx]
    
    # Iteratively select remaining samples
    for _ in range(batch_size - 1):
        if len(remaining_indices) == 0:
            break
        
        # Compute distances to selected samples
        distances_to_selected = cdist(
            X_candidates[remaining_indices],
            X_candidates[selected_indices],
            metric=metric
        )  # Shape: (n_remaining, n_selected)
        
        # Diversity score: minimum distance to any selected sample
        min_distances = distances_to_selected.min(axis=1)
        
        # Normalize diversity scores to [0, 1]
        diversity_norm = min_distances
        if diversity_norm.max() > 0:
            diversity_norm = diversity_norm / diversity_norm.max()
        
        # Combined score
        combined_score = (
            alpha * acq_norm[remaining_indices] +
            (1 - alpha) * diversity_norm
        )
        
        # Select sample with highest combined score
        best_idx_local = combined_score.argmax()
        best_idx_global = remaining_indices[best_idx_local]
        
        selected_indices.append(best_idx_global)
        remaining_indices = remaining_indices[remaining_indices != best_idx_global]
    
    return np.array(selected_indices)


def dpp_selection(
    X_candidates: np.ndarray,
    acquisition_scores: np.ndarray,
    batch_size: int,
    lambda_diversity: float = 1.0,
    metric: str = "euclidean",
) -> np.ndarray:
    """
    Determinantal Point Process (DPP) selection.
    
    Selects a diverse batch by maximizing the determinant of a quality-diversity
    kernel matrix. This provides a principled balance between quality (acquisition)
    and diversity (coverage).
    
    Kernel:
        K_ij = quality_i * quality_j * diversity(x_i, x_j)
    
    Args:
        X_candidates: Candidate features (N, D)
        acquisition_scores: Quality scores (N,)
        batch_size: Number of samples to select
        lambda_diversity: Diversity weight (default: 1.0)
        metric: Distance metric (default: "euclidean")
        
    Returns:
        Selected indices (batch_size,)
        
    Note:
        Exact DPP sampling is expensive (O(N³)). We use greedy approximation.
        
    Reference:
        Kulesza & Taskar (2012) "Determinantal Point Processes for
        Machine Learning"
    """
    if len(X_candidates) != len(acquisition_scores):
        raise ValueError("X_candidates and acquisition_scores must have same length")
    
    if batch_size > len(X_candidates):
        raise ValueError("batch_size cannot exceed number of candidates")
    
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    
    n_candidates = len(X_candidates)
    
    # Special case
    if batch_size == n_candidates:
        return np.arange(n_candidates)
    
    # Normalize acquisition scores (quality)
    quality = acquisition_scores - acquisition_scores.min()
    if quality.max() > 0:
        quality = quality / quality.max()
    quality = quality + 1e-6  # Avoid zeros
    
    # Compute pairwise distances
    distances = cdist(X_candidates, X_candidates, metric=metric)
    
    # Convert distances to similarity: sim = exp(-λ * d²)
    similarity = np.exp(-lambda_diversity * distances ** 2)
    
    # Construct kernel matrix: K_ij = quality_i * quality_j * similarity_ij
    quality_matrix = quality[:, np.newaxis] * quality[np.newaxis, :]
    kernel = quality_matrix * similarity
    
    # Greedy DPP: iteratively select points maximizing determinant
    selected_indices = []
    remaining_indices = np.arange(n_candidates)
    
    for _ in range(batch_size):
        if len(remaining_indices) == 0:
            break
        
        if len(selected_indices) == 0:
            # First sample: highest quality
            best_idx = remaining_indices[quality[remaining_indices].argmax()]
        else:
            # Subsequent samples: maximize determinant gain
            # det(K_selected ∪ {i}) / det(K_selected)
            # = K_ii - K_i,selected @ K_selected^{-1} @ K_selected,i
            
            K_selected = kernel[np.ix_(selected_indices, selected_indices)]
            
            # For numerical stability, use pseudo-inverse
            try:
                K_selected_inv = np.linalg.inv(K_selected)
            except np.linalg.LinAlgError:
                K_selected_inv = np.linalg.pinv(K_selected)
            
            # Compute determinant gain for each remaining candidate
            det_gains = []
            
            for idx in remaining_indices:
                K_ii = kernel[idx, idx]
                K_i_selected = kernel[idx, selected_indices]
                K_selected_i = kernel[selected_indices, idx]
                
                # Determinant gain (Schur complement)
                det_gain = K_ii - K_i_selected @ K_selected_inv @ K_selected_i
                det_gains.append(det_gain)
            
            det_gains = np.array(det_gains)
            
            # Select candidate with highest determinant gain
            best_idx_local = det_gains.argmax()
            best_idx = remaining_indices[best_idx_local]
        
        selected_indices.append(best_idx)
        remaining_indices = remaining_indices[remaining_indices != best_idx]
    
    return np.array(selected_indices)


def create_diversity_selector(
    method: Literal["k_medoids", "greedy", "dpp"],
    **kwargs,
):
    """
    Factory function to create diversity selectors.
    
    Args:
        method: Diversity selection method
        **kwargs: Method-specific arguments
        
    Returns:
        Callable diversity selector
        
    Example:
        >>> selector = create_diversity_selector("k_medoids", metric="euclidean")
        >>> selected = selector(X_candidates, acquisition_scores, batch_size=10)
    """
    if method == "k_medoids":
        def selector(X_candidates, acquisition_scores, batch_size, **extra_kwargs):
            merged_kwargs = {**kwargs, **extra_kwargs}
            return k_medoids_selection(
                X_candidates, acquisition_scores, batch_size, **merged_kwargs
            )
        return selector
    
    elif method == "greedy":
        def selector(X_candidates, acquisition_scores, batch_size, **extra_kwargs):
            merged_kwargs = {**kwargs, **extra_kwargs}
            return greedy_diversity_selection(
                X_candidates, acquisition_scores, batch_size, **merged_kwargs
            )
        return selector
    
    elif method == "dpp":
        def selector(X_candidates, acquisition_scores, batch_size, **extra_kwargs):
            merged_kwargs = {**kwargs, **extra_kwargs}
            return dpp_selection(
                X_candidates, acquisition_scores, batch_size, **merged_kwargs
            )
        return selector
    
    else:
        raise ValueError(
            f"Unknown diversity method: {method}. "
            "Choose from: 'k_medoids', 'greedy', 'dpp'"
        )


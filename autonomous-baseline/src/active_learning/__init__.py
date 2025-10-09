"""Active learning for efficient data acquisition."""

from src.active_learning.acquisition import (
    upper_confidence_bound,
    expected_improvement,
    maximum_variance,
    expected_information_gain_proxy,
    thompson_sampling,
    create_acquisition_function,
)
from src.active_learning.diversity import (
    k_medoids_selection,
    greedy_diversity_selection,
    dpp_selection,
    create_diversity_selector,
)
from src.active_learning.loop import (
    ActiveLearningLoop,
    go_no_go_gate,
)

__all__ = [
    # Acquisition functions
    "upper_confidence_bound",
    "expected_improvement",
    "maximum_variance",
    "expected_information_gain_proxy",
    "thompson_sampling",
    "create_acquisition_function",
    # Diversity selection
    "k_medoids_selection",
    "greedy_diversity_selection",
    "dpp_selection",
    "create_diversity_selector",
    # Active learning loop
    "ActiveLearningLoop",
    "go_no_go_gate",
]


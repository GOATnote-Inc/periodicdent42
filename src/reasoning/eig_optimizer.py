"""EIG-Driven Planning: Bayesian experimental design for maximum information gain per cost.

This module implements Expected Information Gain (EIG) calculations and active learning
strategies to prioritize experiments that reduce uncertainty most efficiently.

Moat: TIME - Maximize learning velocity through intelligent experiment selection.
"""

import copy
import math
import random
import numpy as np
from typing import List, Tuple, Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from scipy.stats import entropy, norm
from scipy.optimize import minimize
import structlog

from configs.data_schema import Experiment, Protocol, Sample, Decision

logger = structlog.get_logger()


def _to_hashable(value: Any) -> Any:
    """Convert arbitrary objects to a hashable representation."""

    if isinstance(value, dict):
        return tuple(sorted((key, _to_hashable(val)) for key, val in value.items()))
    if isinstance(value, (list, tuple, set)):
        return tuple(_to_hashable(item) for item in value)
    if hasattr(value, "__dict__"):
        return tuple(sorted((key, _to_hashable(val)) for key, val in vars(value).items()))
    return value


@dataclass
class Hypothesis:
    """Representation of a discrete or continuous hypothesis."""

    name: str
    prior: float
    hypothesis_type: str = "discrete"
    mean: Optional[float] = None
    variance: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    likelihood_fn: Optional[Callable[[Any, Any, "Hypothesis"], float]] = None

    def compute_likelihood(self, test: Any, outcome: Any) -> float:
        """Compute likelihood of an outcome under the hypothesis."""
        if self.likelihood_fn is not None:
            likelihood = self.likelihood_fn(test, outcome, self)
            return max(float(likelihood), 1e-12)

        if self.hypothesis_type == "continuous":
            return self._continuous_likelihood(test, outcome)

        likelihoods = self.metadata.get("likelihoods", {})
        test_key = self._resolve_test_key(test)
        outcome_key = self._hashable(outcome)
        likelihood: Optional[float] = None

        if isinstance(likelihoods, dict):
            tuple_key = (self._hashable(test), outcome_key)
            if tuple_key in likelihoods:
                likelihood = likelihoods.get(tuple_key)
            else:
                mapping = likelihoods.get(test_key)
                if isinstance(mapping, dict):
                    likelihood = mapping.get(outcome)
                    if likelihood is None:
                        likelihood = mapping.get(outcome_key)
                elif isinstance(mapping, (int, float)):
                    likelihood = mapping

        if likelihood is None:
            default = self.metadata.get("default_likelihood")
            if default is not None:
                likelihood = float(default)
            else:
                outcomes = self.metadata.get("test_outcomes", {}).get(test_key)
                if outcomes:
                    likelihood = 1.0 / max(len(outcomes), 1)
                else:
                    likelihood = 1e-6

        return max(float(likelihood), 1e-12)

    def _continuous_likelihood(self, test: Any, outcome: Any) -> float:
        """Default Gaussian likelihood for continuous hypotheses."""
        observation_variance = self._resolve_observation_variance(test)
        expected_value = self._resolve_expected_value(test)
        effective_variance = max((self.variance or 1.0) + observation_variance, 1e-12)
        likelihood = norm.pdf(float(outcome), loc=expected_value, scale=np.sqrt(effective_variance))
        self._update_gaussian_posterior(outcome, observation_variance)
        return max(float(likelihood), 1e-12)

    def _resolve_observation_variance(self, test: Any) -> float:
        if isinstance(test, dict):
            return float(test.get("observation_variance", self.metadata.get("observation_variance", 1.0)))
        return float(getattr(test, "observation_variance", self.metadata.get("observation_variance", 1.0)))

    def _resolve_expected_value(self, test: Any) -> float:
        predict_fn = self.metadata.get("predict_fn")
        if callable(predict_fn):
            return float(predict_fn(test, self))
        if self.mean is not None:
            return float(self.mean)
        return float(self.metadata.get("expected_value", 0.0))

    def _update_gaussian_posterior(self, outcome: Any, observation_variance: float) -> None:
        if self.mean is None or self.variance is None:
            self.mean = float(outcome)
            self.variance = float(observation_variance)
            return

        prior_variance = max(float(self.variance), 1e-12)
        posterior_precision = 1.0 / prior_variance + 1.0 / max(float(observation_variance), 1e-12)
        posterior_variance = 1.0 / posterior_precision
        posterior_mean = posterior_variance * (
            self.mean / prior_variance + float(outcome) / max(float(observation_variance), 1e-12)
        )

        self.mean = float(posterior_mean)
        self.variance = float(posterior_variance)

    @staticmethod
    def _hashable(value: Any) -> Any:
        return _to_hashable(value)

    @staticmethod
    def _resolve_test_key(test: Any) -> Any:
        if isinstance(test, dict):
            return test.get("name") or _to_hashable(test)
        if hasattr(test, "name"):
            return getattr(test, "name")
        return test


@dataclass
class BeliefState:
    """Maintains a collection of hypotheses and supports Bayesian updates."""

    hypotheses: List[Hypothesis]
    min_posterior_threshold: float = 1e-3
    eliminated_hypotheses: List[Dict[str, Any]] = field(default_factory=list)
    archived_hypotheses: List[Hypothesis] = field(default_factory=list)

    def __post_init__(self):
        self.logger = structlog.get_logger(__name__)
        self.normalize()

    def normalize(self) -> None:
        total = sum(h.prior for h in self.hypotheses)
        if total <= 0:
            if not self.hypotheses:
                return
            uniform = 1.0 / len(self.hypotheses)
            for hypothesis in self.hypotheses:
                hypothesis.prior = uniform
            return

        for hypothesis in self.hypotheses:
            hypothesis.prior = hypothesis.prior / total

    def update_beliefs(self, test: Any, outcome: Any) -> None:
        if not self.hypotheses:
            self.logger.warning("belief_update_skipped", reason="no_hypotheses")
            return

        prior_snapshot = {h.name: h.prior for h in self.hypotheses}
        likelihoods = []

        for hypothesis in self.hypotheses:
            likelihood = hypothesis.compute_likelihood(test, outcome)
            likelihoods.append(max(likelihood, 1e-12))

        posteriors = [h.prior * l for h, l in zip(self.hypotheses, likelihoods)]
        total_posterior = sum(posteriors)

        if total_posterior <= 0:
            self.logger.warning("belief_update_degenerate", reason="zero_total", test=test, outcome=outcome)
            self.normalize()
            return

        for hypothesis, posterior in zip(self.hypotheses, posteriors):
            hypothesis.prior = posterior / total_posterior

        self._handle_thresholds(test, outcome)
        self._log_negative_outcomes(test, outcome, prior_snapshot)
        self.normalize()

    def _handle_thresholds(self, test: Any, outcome: Any) -> None:
        surviving_hypotheses: List[Hypothesis] = []

        for hypothesis in self.hypotheses:
            if hypothesis.prior < self.min_posterior_threshold:
                self.eliminated_hypotheses.append(
                    {
                        "hypothesis": hypothesis.name,
                        "posterior": hypothesis.prior,
                        "test": test,
                        "outcome": outcome,
                        "reason": "posterior_below_threshold",
                    }
                )
                self.archived_hypotheses.append(hypothesis)
            else:
                surviving_hypotheses.append(hypothesis)

        self.hypotheses = surviving_hypotheses

    def _log_negative_outcomes(self, test: Any, outcome: Any, prior_snapshot: Dict[str, float]) -> None:
        if not self._is_negative_outcome(outcome):
            return

        for hypothesis in self.hypotheses:
            prior_value = prior_snapshot.get(hypothesis.name, hypothesis.prior)
            if hypothesis.prior < prior_value:
                self.eliminated_hypotheses.append(
                    {
                        "hypothesis": hypothesis.name,
                        "prior": prior_value,
                        "posterior": hypothesis.prior,
                        "test": test,
                        "outcome": outcome,
                        "reason": "negative_outcome",
                    }
                )

    @staticmethod
    def _is_negative_outcome(outcome: Any) -> bool:
        if isinstance(outcome, bool):
            return not outcome
        if isinstance(outcome, (int, float)):
            return outcome < 0
        if isinstance(outcome, str):
            return outcome.strip().lower() in {"negative", "fail", "failure", "reject"}
        return False

    def current_entropy(self) -> float:
        probabilities = [h.prior for h in self.hypotheses]
        if not probabilities:
            return 0.0
        return float(entropy(probabilities))

    def current_variance(self) -> float:
        weighted_components = [
            (h.prior, h.mean, h.variance)
            for h in self.hypotheses
            if h.mean is not None and h.variance is not None
        ]

        if not weighted_components:
            return 0.0

        total_weight = sum(w for w, _, _ in weighted_components)
        if total_weight <= 0:
            return 0.0

        weighted_mean = sum(w * m for w, m, _ in weighted_components) / total_weight
        total_variance = sum(
            w * ((m - weighted_mean) ** 2 + v) for w, m, v in weighted_components
        ) / total_weight
        return float(total_variance)


@dataclass
class EIGResult:
    """Result of EIG calculation."""
    candidate: np.ndarray
    eig: float
    eig_per_cost: float
    cost_hours: float
    cost_usd: float
    rationale: str


class GaussianProcessSurrogate:
    """Simple GP surrogate model for demonstration.
    
    In production, use GPyTorch or BoTorch for more sophisticated models.
    """
    
    def __init__(self, kernel_length_scale: float = 1.0, noise: float = 0.1):
        self.X_train: Optional[np.ndarray] = None
        self.y_train: Optional[np.ndarray] = None
        self.length_scale = kernel_length_scale
        self.noise = noise
        self.logger = structlog.get_logger()
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit GP to training data."""
        self.X_train = X.copy()
        self.y_train = y.copy()
        self.logger.info("gp_fitted", n_train=len(X))
    
    def predict(self, X_test: np.ndarray, return_std: bool = False) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Predict mean and std at test points."""
        if self.X_train is None:
            raise ValueError("Model not fitted")
        
        # RBF kernel: k(x, x') = exp(-||x - x'||^2 / (2 * l^2))
        def rbf_kernel(X1, X2):
            dists = np.sum(X1**2, axis=1, keepdims=True) + \
                    np.sum(X2**2, axis=1, keepdims=True).T - \
                    2 * np.dot(X1, X2.T)
            return np.exp(-dists / (2 * self.length_scale**2))
        
        K = rbf_kernel(self.X_train, self.X_train) + self.noise * np.eye(len(self.X_train))
        K_star = rbf_kernel(self.X_train, X_test)
        K_star_star = rbf_kernel(X_test, X_test)
        
        # Posterior mean
        K_inv = np.linalg.inv(K)
        mean = K_star.T @ K_inv @ self.y_train
        
        if return_std:
            # Posterior variance
            var = np.diag(K_star_star - K_star.T @ K_inv @ K_star)
            std = np.sqrt(np.maximum(var, 0))  # Ensure non-negative
            return mean, std
        
        return mean, None
    
    def posterior_variance(self, X_test: np.ndarray) -> np.ndarray:
        """Get posterior variance at test points."""
        _, std = self.predict(X_test, return_std=True)
        return std**2 if std is not None else np.zeros(len(X_test))


class EIGOptimizer:
    """EIG-driven experiment planner.
    
    Moat: TIME - Bayesian optimization for maximum information/cost.
    """
    
    def __init__(
        self,
        gp_model: GaussianProcessSurrogate,
        cost_model: Optional[Dict[str, float]] = None,
        belief_state: Optional[BeliefState] = None,
    ):
        self.gp = gp_model
        self.cost_model = cost_model or {}
        self.logger = structlog.get_logger()
        self.belief_state = belief_state or BeliefState(hypotheses=[])

    def update_beliefs(self, test: Any, outcome: Any) -> None:
        """Update belief state using the outcome of a test."""
        self.belief_state.update_beliefs(test, outcome)
        self.logger.info(
            "belief_state_updated",
            entropy=self.belief_state.current_entropy(),
            variance=self.belief_state.current_variance(),
            n_active=len(self.belief_state.hypotheses),
            eliminated=len(self.belief_state.eliminated_hypotheses),
        )
    
    def _test_to_array(self, test: Any) -> Optional[np.ndarray]:
        """Extract a feature vector from a test specification."""

        if isinstance(test, np.ndarray):
            return test.reshape(1, -1)

        if isinstance(test, (list, tuple)):
            return np.asarray(test, dtype=float).reshape(1, -1)

        if isinstance(test, dict):
            if "params" in test:
                return np.asarray(test["params"], dtype=float).reshape(1, -1)
            if "candidate" in test:
                return np.asarray(test["candidate"], dtype=float).reshape(1, -1)

        if hasattr(test, "parameters"):
            return np.asarray(getattr(test, "parameters"), dtype=float).reshape(1, -1)

        return None

    def _candidate_outcomes_and_weights(
        self,
        test: Any,
        belief_state: BeliefState,
        n_outcomes: int,
    ) -> List[Tuple[Any, float]]:
        """Enumerate plausible outcomes and their support weights."""

        discrete_outcomes: List[Any] = []
        test_key = Hypothesis._resolve_test_key(test)

        for hypothesis in belief_state.hypotheses:
            outcomes_map = hypothesis.metadata.get("test_outcomes")
            if outcomes_map and test_key in outcomes_map:
                discrete_outcomes.extend(outcomes_map[test_key])
            else:
                outcomes = hypothesis.metadata.get("possible_outcomes") or hypothesis.metadata.get("outcomes")
                if outcomes:
                    discrete_outcomes.extend(outcomes)

        if not discrete_outcomes:
            if isinstance(test, dict):
                discrete_outcomes.extend(test.get("possible_outcomes", []))
            elif hasattr(test, "possible_outcomes"):
                discrete_outcomes.extend(getattr(test, "possible_outcomes"))

        if discrete_outcomes:
            unique_outcomes = list(dict.fromkeys(discrete_outcomes))
            return [(outcome, 1.0) for outcome in unique_outcomes]

        feature_vector = self._test_to_array(test)

        if feature_vector is None:
            return []

        mean, std = self.gp.predict(feature_vector, return_std=True)

        if std is None or std[0] < 1e-9:
            return [(float(mean[0]), 1.0)]

        quantile_edges = np.linspace(0.0, 1.0, n_outcomes + 1)
        centers = 0.5 * (quantile_edges[:-1] + quantile_edges[1:])
        centers = np.clip(centers, 1e-3, 1 - 1e-3)
        outcomes = norm.ppf(centers, loc=float(mean[0]), scale=float(std[0]))
        widths = np.diff(quantile_edges)

        return list(zip(outcomes, widths))

    def _clone_belief_state(self, belief_state: BeliefState) -> BeliefState:
        """Create a copy of a belief state without sharing mutable members."""

        cloned_hypotheses = [
            Hypothesis(
                name=hypothesis.name,
                prior=hypothesis.prior,
                hypothesis_type=hypothesis.hypothesis_type,
                mean=hypothesis.mean,
                variance=hypothesis.variance,
                metadata=copy.deepcopy(hypothesis.metadata),
                likelihood_fn=hypothesis.likelihood_fn,
            )
            for hypothesis in belief_state.hypotheses
        ]

        clone = BeliefState(
            hypotheses=cloned_hypotheses,
            min_posterior_threshold=belief_state.min_posterior_threshold,
        )
        clone.eliminated_hypotheses = copy.deepcopy(belief_state.eliminated_hypotheses)
        clone.archived_hypotheses = copy.deepcopy(belief_state.archived_hypotheses)
        return clone

    def compute_information_gain(
        self,
        test: Any,
        belief_state: Optional[BeliefState] = None,
        n_outcomes: int = 50,
    ) -> float:
        """Compute Expected Information Gain for a test against a belief state."""

        state = belief_state or self.belief_state

        if state is None or not state.hypotheses:
            return 0.0

        current_entropy = state.current_entropy()

        if current_entropy <= 0:
            return 0.0

        candidate_outcomes = self._candidate_outcomes_and_weights(test, state, n_outcomes)

        if not candidate_outcomes:
            return 0.0

        priors = np.asarray([h.prior for h in state.hypotheses], dtype=float)
        expected_entropy = 0.0
        total_weight = 0.0

        for outcome, support_weight in candidate_outcomes:
            likelihood_state = self._clone_belief_state(state)
            likelihoods = np.asarray(
                [h.compute_likelihood(test, outcome) for h in likelihood_state.hypotheses],
                dtype=float,
            )

            evidence = float(np.dot(priors, likelihoods))

            if evidence <= 0 or support_weight <= 0:
                continue

            outcome_weight = evidence * support_weight

            posterior_state = self._clone_belief_state(state)
            posterior_state.update_beliefs(test, outcome)
            posterior_entropy = posterior_state.current_entropy()

            expected_entropy += outcome_weight * posterior_entropy
            total_weight += outcome_weight

        if total_weight <= 0:
            return 0.0

        expected_entropy /= total_weight
        eig = current_entropy - expected_entropy

        return max(float(eig), 0.0)

    def calculate_eig(self, X_candidate: np.ndarray, n_samples: int = 50) -> float:
        """Calculate Expected Information Gain for a candidate parameter vector."""

        test = {"params": np.asarray(X_candidate, dtype=float)}
        return self.compute_information_gain(test, self.belief_state, n_outcomes=n_samples)
    
    def estimate_cost(self, X_candidate: np.ndarray, instrument_id: str) -> Tuple[float, float]:
        """Estimate experiment cost in hours and USD.
        
        Args:
            X_candidate: Experiment parameters
            instrument_id: Target instrument
        
        Returns:
            (cost_hours, cost_usd)
        """
        # Simple cost model (replace with actual estimates)
        base_time = self.cost_model.get(f"{instrument_id}_time", 2.0)
        base_cost = self.cost_model.get(f"{instrument_id}_cost", 100.0)
        
        # Add complexity factor based on parameters
        complexity = 1.0 + 0.1 * np.sum(np.abs(X_candidate))
        
        return base_time * complexity, base_cost * complexity
    
    def calculate_eig_per_cost(self, X_candidate: np.ndarray, instrument_id: str) -> EIGResult:
        """Calculate EIG per unit cost.
        
        This is the key metric: information gained per dollar/hour spent.
        
        Moat: TIME - Optimize learning velocity, not just information.
        """
        eig = self.calculate_eig(X_candidate)
        cost_hours, cost_usd = self.estimate_cost(X_candidate, instrument_id)
        
        # Combined cost metric (normalize to similar scale)
        total_cost = cost_hours + cost_usd / 100.0
        
        eig_per_cost = eig / total_cost if total_cost > 0 else 0.0
        
        rationale = f"EIG={eig:.3f}, cost={cost_hours:.1f}h/${cost_usd:.0f}, ratio={eig_per_cost:.3f}"
        
        return EIGResult(
            candidate=X_candidate,
            eig=eig,
            eig_per_cost=eig_per_cost,
            cost_hours=cost_hours,
            cost_usd=cost_usd,
            rationale=rationale
        )
    
    def select_batch_greedy(
        self,
        X_pool: np.ndarray,
        batch_size: int,
        instrument_id: str
    ) -> List[EIGResult]:
        """Greedily select batch of high-EIG experiments.
        
        Args:
            X_pool: Pool of candidate experiments (n_candidates, n_dim)
            batch_size: Number of experiments to select
            instrument_id: Target instrument
        
        Returns:
            List of selected experiments with EIG scores
        
        Moat: TIME - Batch selection for parallel execution.
        """
        selected = []
        remaining_indices = list(range(len(X_pool)))
        
        for i in range(min(batch_size, len(X_pool))):
            # Calculate EIG for all remaining candidates
            eig_results = []

            for idx in remaining_indices:
                result = self.calculate_eig_per_cost(X_pool[idx], instrument_id)
                eig_results.append((idx, result))
                self.logger.info(
                    "candidate_eig",
                    candidate_index=idx,
                    eig=result.eig,
                    eig_per_cost=result.eig_per_cost,
                )

            # Select highest EIG/cost
            best_idx, best_result = max(eig_results, key=lambda x: x[1].eig_per_cost)

            selected.append(best_result)
            remaining_indices.remove(best_idx)
            
            self.logger.info(
                "experiment_selected",
                iteration=i,
                eig=best_result.eig,
                eig_per_cost=best_result.eig_per_cost
            )
            
            # Simulate GP update for next iteration (fantasize)
            # In production, use GPyTorch's condition_on_observations
            # For now, just continue with current GP
        
        return selected
    
    def uncertainty_sampling(self, X_pool: np.ndarray, top_k: int = 5) -> np.ndarray:
        """Select points with highest predictive variance.
        
        Simple baseline strategy for comparison.
        """
        _, std = self.gp.predict(X_pool, return_std=True)
        
        if std is None:
            return X_pool[:top_k]
        
        top_indices = np.argsort(std)[-top_k:]
        return X_pool[top_indices]
    
    def upper_confidence_bound(
        self,
        X_pool: np.ndarray,
        beta: float = 2.0,
        top_k: int = 5
    ) -> np.ndarray:
        """UCB acquisition function.
        
        UCB = mean + beta * std
        """
        mean, std = self.gp.predict(X_pool, return_std=True)
        
        if std is None:
            return X_pool[:top_k]
        
        ucb = mean + beta * std
        top_indices = np.argsort(ucb)[-top_k:]

        return X_pool[top_indices]


@dataclass
class DiagnosticTest:
    """Definition of a diagnostic test including operational cost and risk."""

    name: str
    rationale: str
    possible_outcomes: List[Any]
    cost: float
    risk: float


@dataclass
class ExperimentRecord:
    """Record of a single executed diagnostic test."""

    test_name: str
    rationale: str
    outcome: Any
    information_gain: float
    value_of_information: float
    entropy_before: float
    entropy_after: float
    entropy_change: float
    top_hypotheses: List[Tuple[str, float]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "test": self.test_name,
            "rationale": self.rationale,
            "outcome": self.outcome,
            "information_gain": self.information_gain,
            "value_of_information": self.value_of_information,
            "entropy_before": self.entropy_before,
            "entropy_after": self.entropy_after,
            "entropy_change": self.entropy_change,
            "top_hypotheses": self.top_hypotheses,
        }


@dataclass
class AgentState:
    """Mutable state of the diagnostic agent during sequential testing."""

    belief_state: BeliefState
    total_cost_allowed: float
    remaining_budget: float
    max_risk_level: float
    experiment_history: List[ExperimentRecord] = field(default_factory=list)


def generate_decision_log(
    selected: List[EIGResult],
    alternatives: List[EIGResult],
    agent_id: str = "eig_optimizer"
) -> Decision:
    """Generate glass-box decision log.
    
    Moat: INTERPRETABILITY - Every planning decision is explainable.
    """
    # Format rationale
    rationale_parts = [
        f"Selected {len(selected)} experiments to maximize EIG/cost.",
        f"Total expected EIG: {sum(r.eig for r in selected):.2f}",
        f"Total cost: {sum(r.cost_hours for r in selected):.1f} hours, ${sum(r.cost_usd for r in selected):.0f}",
        "Selection prioritizes high-uncertainty regions while minimizing cost."
    ]
    
    rationale = " ".join(rationale_parts)
    
    # Format alternatives
    alternatives_list = [
        {
            "eig": alt.eig,
            "eig_per_cost": alt.eig_per_cost,
            "reason": f"Lower EIG/cost ratio ({alt.eig_per_cost:.3f} vs {selected[0].eig_per_cost:.3f})"
        }
        for alt in alternatives[:5]  # Top 5 alternatives
    ]
    
    return Decision(
        agent_id=agent_id,
        action=f"select_batch_{len(selected)}",
        rationale=rationale,
        confidence=0.85,  # Could be based on GP uncertainty
        alternatives_considered=alternatives_list,
        input_state={"n_pool": len(alternatives) + len(selected)},
        expected_outcome={"total_eig": sum(r.eig for r in selected)}
    )


def _outcome_distribution(hypothesis: Hypothesis, test_name: str) -> Dict[Any, float]:
    """Return outcome probabilities for a test under a hypothesis."""

    likelihoods = hypothesis.metadata.get("likelihoods", {})
    distribution: Dict[Any, float] = {}

    if isinstance(likelihoods, dict):
        mapping = likelihoods.get(test_name)
        if isinstance(mapping, dict):
            distribution = {key: float(value) for key, value in mapping.items()}
        else:
            hashed_key = Hypothesis._hashable({"name": test_name})
            mapping = likelihoods.get(hashed_key)
            if isinstance(mapping, dict):
                distribution = {key: float(value) for key, value in mapping.items()}

        if not distribution:
            hashed_test = Hypothesis._hashable({"name": test_name})
            for key, value in likelihoods.items():
                if isinstance(key, tuple) and len(key) == 2 and key[0] == hashed_test:
                    distribution[key[1]] = float(value)

    return distribution


def _simulate_outcome(
    test: DiagnosticTest,
    ground_truth: str,
    hypotheses: List[Hypothesis],
    rng: random.Random,
) -> Any:
    """Sample an outcome for the given test using the ground-truth hypothesis."""

    hypothesis = next((h for h in hypotheses if h.name == ground_truth), None)
    if hypothesis is None:
        return rng.choice(test.possible_outcomes)

    distribution = _outcome_distribution(hypothesis, test.name)
    if not distribution:
        return rng.choice(test.possible_outcomes)

    total = sum(distribution.values())
    if total <= 0:
        return rng.choice(test.possible_outcomes)

    threshold = rng.random() * total
    cumulative = 0.0
    for outcome in test.possible_outcomes:
        probability = float(
            distribution.get(outcome)
            or distribution.get(Hypothesis._hashable(outcome), 0.0)
        )
        cumulative += probability
        if threshold <= cumulative:
            return outcome

    return max(distribution.items(), key=lambda item: item[1])[0]


def _select_next_test(
    mode: str,
    candidates: List[DiagnosticTest],
    optimizer: EIGOptimizer,
    belief_state: BeliefState,
) -> Optional[Tuple[DiagnosticTest, float, float]]:
    """Score candidates and return the next test plus its metrics."""

    scores: List[Tuple[DiagnosticTest, float, float]] = []
    for test in candidates:
        payload = {
            "name": test.name,
            "possible_outcomes": list(test.possible_outcomes),
            "cost": test.cost,
            "risk": test.risk,
        }
        info_gain = optimizer.compute_information_gain(payload, belief_state=belief_state)
        voi = info_gain / max(test.cost, 1e-6)
        scores.append((test, info_gain, voi))

    if not scores:
        return None

    mode_key = mode.lower()
    if mode_key == "greedy":
        return max(scores, key=lambda item: (item[1], -item[0].cost))
    if mode_key == "infogain":
        return max(scores, key=lambda item: (item[2], item[1]))

    raise ValueError(f"Unknown diagnostic mode: {mode}")


def build_diagnostic_scenario() -> Tuple[List[DiagnosticTest], List[Hypothesis]]:
    """Create a synthetic diagnostic scenario with priors and likelihoods."""

    tests = [
        DiagnosticTest(
            name="VacuumIntegrity",
            rationale="Mass-spec leak check to validate chamber integrity",
            possible_outcomes=["leak", "clear"],
            cost=4.0,
            risk=0.2,
        ),
        DiagnosticTest(
            name="AlignmentDriftScan",
            rationale="Precision scan to measure goniometer drift",
            possible_outcomes=["drift", "stable"],
            cost=3.0,
            risk=0.3,
        ),
        DiagnosticTest(
            name="DetectorHealthCheck",
            rationale="Power cycle detector and run built-in self test",
            possible_outcomes=["recovered", "still_faulty"],
            cost=5.5,
            risk=0.55,
        ),
        DiagnosticTest(
            name="SampleChargeProbe",
            rationale="Kelvin probe to quantify surface charging",
            possible_outcomes=["charging", "neutral"],
            cost=2.5,
            risk=0.1,
        ),
    ]

    outcomes_map = {test.name: list(test.possible_outcomes) for test in tests}
    all_outcomes: List[Any] = []
    for test in tests:
        all_outcomes.extend(test.possible_outcomes)
    unique_outcomes = list(dict.fromkeys(all_outcomes))

    hypotheses = [
        Hypothesis(
            name="Vacuum Leak",
            prior=0.35,
            metadata={
                "likelihoods": {
                    "VacuumIntegrity": {"leak": 0.9, "clear": 0.1},
                    "AlignmentDriftScan": {"drift": 0.7, "stable": 0.3},
                    "DetectorHealthCheck": {"recovered": 0.2, "still_faulty": 0.8},
                    "SampleChargeProbe": {"charging": 0.6, "neutral": 0.4},
                },
                "test_outcomes": copy.deepcopy(outcomes_map),
                "possible_outcomes": unique_outcomes,
            },
        ),
        Hypothesis(
            name="Alignment Drift",
            prior=0.4,
            metadata={
                "likelihoods": {
                    "VacuumIntegrity": {"leak": 0.2, "clear": 0.8},
                    "AlignmentDriftScan": {"drift": 0.85, "stable": 0.15},
                    "DetectorHealthCheck": {"recovered": 0.4, "still_faulty": 0.6},
                    "SampleChargeProbe": {"charging": 0.3, "neutral": 0.7},
                },
                "test_outcomes": copy.deepcopy(outcomes_map),
                "possible_outcomes": unique_outcomes,
            },
        ),
        Hypothesis(
            name="Sample Charging",
            prior=0.25,
            metadata={
                "likelihoods": {
                    "VacuumIntegrity": {"leak": 0.1, "clear": 0.9},
                    "AlignmentDriftScan": {"drift": 0.25, "stable": 0.75},
                    "DetectorHealthCheck": {"recovered": 0.6, "still_faulty": 0.4},
                    "SampleChargeProbe": {"charging": 0.85, "neutral": 0.15},
                },
                "test_outcomes": copy.deepcopy(outcomes_map),
                "possible_outcomes": unique_outcomes,
            },
        ),
    ]

    return tests, hypotheses


def run_diagnostic_session(
    mode: str = "infogain",
    ground_truth: Optional[str] = None,
    *,
    tests: Optional[List[DiagnosticTest]] = None,
    hypotheses: Optional[List[Hypothesis]] = None,
    total_budget: float = 12.0,
    max_risk: float = 0.5,
    rng: Optional[random.Random] = None,
) -> Dict[str, Any]:
    """Run a sequential diagnostic session and return summary metrics."""

    scenario_tests, scenario_hypotheses = tests, hypotheses
    if scenario_tests is None or scenario_hypotheses is None:
        scenario_tests, scenario_hypotheses = build_diagnostic_scenario()

    reference_hypotheses = scenario_hypotheses
    belief_hypotheses = [
        Hypothesis(
            name=h.name,
            prior=h.prior,
            hypothesis_type=h.hypothesis_type,
            mean=h.mean,
            variance=h.variance,
            metadata=copy.deepcopy(h.metadata),
            likelihood_fn=h.likelihood_fn,
        )
        for h in reference_hypotheses
    ]

    belief_state = BeliefState(hypotheses=belief_hypotheses, min_posterior_threshold=1e-3)
    agent_state = AgentState(
        belief_state=belief_state,
        total_cost_allowed=total_budget,
        remaining_budget=total_budget,
        max_risk_level=max_risk,
    )

    gp_stub = GaussianProcessSurrogate()
    optimizer = EIGOptimizer(gp_stub, cost_model={}, belief_state=belief_state)

    rng = rng or random.Random()
    available_names = [h.name for h in reference_hypotheses]
    if ground_truth and ground_truth not in available_names:
        raise ValueError(f"Unknown ground_truth '{ground_truth}'. Expected one of {available_names}")
    ground_truth_name = ground_truth or rng.choice(available_names)

    available_tests = list(scenario_tests)
    total_info_gain = 0.0
    tests_run = 0
    mode_key = mode.lower()

    while available_tests:
        selection = _select_next_test(mode_key, available_tests, optimizer, belief_state)
        if selection is None:
            break

        test, info_gain, voi = selection
        if test.risk > agent_state.max_risk_level:
            logger.info(
                "test_skipped_risk",
                test=test.name,
                risk=test.risk,
                max_risk=agent_state.max_risk_level,
            )
            available_tests.remove(test)
            continue

        if test.cost > agent_state.remaining_budget:
            logger.info(
                "budget_exhausted",
                test=test.name,
                cost=test.cost,
                remaining=agent_state.remaining_budget,
            )
            break

        payload = {
            "name": test.name,
            "possible_outcomes": list(test.possible_outcomes),
            "cost": test.cost,
            "risk": test.risk,
        }
        entropy_before = belief_state.current_entropy()
        outcome = _simulate_outcome(test, ground_truth_name, reference_hypotheses, rng)
        belief_state.update_beliefs(payload, outcome)
        entropy_after = belief_state.current_entropy()
        entropy_change = entropy_before - entropy_after
        top_hypotheses = sorted(
            ((h.name, float(h.prior)) for h in belief_state.hypotheses),
            key=lambda item: item[1],
            reverse=True,
        )[:3]

        rationale = (
            f"{mode_key} selection: IG={info_gain:.3f} bits, "
            f"cost={test.cost:.2f}, VoI={voi:.3f}"
        )
        record = ExperimentRecord(
            test_name=test.name,
            rationale=rationale,
            outcome=outcome,
            information_gain=info_gain,
            value_of_information=voi,
            entropy_before=entropy_before,
            entropy_after=entropy_after,
            entropy_change=entropy_change,
            top_hypotheses=top_hypotheses,
        )
        agent_state.experiment_history.append(record)

        agent_state.remaining_budget -= test.cost
        available_tests.remove(test)
        total_info_gain += info_gain
        tests_run += 1

        logger.info(
            "test_executed",
            test=test.name,
            outcome=outcome,
            information_gain=info_gain,
            voi=voi,
            remaining_budget=agent_state.remaining_budget,
        )

        if entropy_after <= 1e-3 or len(belief_state.hypotheses) <= 1:
            logger.info(
                "entropy_threshold_reached",
                entropy=entropy_after,
                tests_run=tests_run,
            )
            break

    total_cost_spent = agent_state.total_cost_allowed - agent_state.remaining_budget
    cost_per_bit = (
        total_cost_spent / total_info_gain if total_info_gain > 0 else float("inf")
    )
    final_top = None
    if belief_state.hypotheses:
        final_top = max(
            ((h.name, float(h.prior)) for h in belief_state.hypotheses),
            key=lambda item: item[1],
        )
    success = bool(final_top and final_top[0] == ground_truth_name)

    return {
        "mode": mode_key,
        "ground_truth": ground_truth_name,
        "tests_run": tests_run,
        "total_cost": total_cost_spent,
        "remaining_budget": agent_state.remaining_budget,
        "total_information_gain": total_info_gain,
        "cost_per_bit": cost_per_bit,
        "success": success,
        "history": [record.to_dict() for record in agent_state.experiment_history],
        "final_top_hypothesis": final_top,
    }


def run_comparison_driver(n_cases: int = 25, seed: int = 0) -> Dict[str, Dict[str, float]]:
    """Run multiple synthetic sessions and print greedy vs info-gain comparison."""

    rng = random.Random(seed)
    results: Dict[str, List[Dict[str, Any]]] = {"greedy": [], "infogain": []}

    for _ in range(n_cases):
        tests, hypotheses = build_diagnostic_scenario()
        ground_truth = rng.choice([h.name for h in hypotheses])

        results["greedy"].append(
            run_diagnostic_session(
                mode="greedy",
                ground_truth=ground_truth,
                tests=tests,
                hypotheses=hypotheses,
                rng=random.Random(rng.random()),
            )
        )

        tests_inf, hypotheses_inf = build_diagnostic_scenario()
        results["infogain"].append(
            run_diagnostic_session(
                mode="infogain",
                ground_truth=ground_truth,
                tests=tests_inf,
                hypotheses=hypotheses_inf,
                rng=random.Random(rng.random()),
            )
        )

    summary: Dict[str, Dict[str, float]] = {}
    for mode, sessions in results.items():
        if not sessions:
            continue
        avg_tests = sum(session["tests_run"] for session in sessions) / len(sessions)
        avg_cost = sum(session["total_cost"] for session in sessions) / len(sessions)
        success_rate = sum(1 for session in sessions if session["success"]) / len(sessions)
        finite_costs = [
            session["cost_per_bit"]
            for session in sessions
            if math.isfinite(session["cost_per_bit"])
        ]
        avg_cost_per_bit = (
            sum(finite_costs) / len(finite_costs) if finite_costs else float("inf")
        )
        summary[mode] = {
            "avg_tests_used": avg_tests,
            "avg_total_cost": avg_cost,
            "success_rate": success_rate,
            "avg_cost_per_bit": avg_cost_per_bit,
        }

    print(f"\nDiagnostic strategy comparison over {n_cases} synthetic runs")
    print("Mode       | Avg Tests | Avg Cost | Success | Cost/bit")
    for mode in ("greedy", "infogain"):
        stats = summary.get(mode)
        if not stats:
            continue
        cost_bit = stats["avg_cost_per_bit"]
        cost_bit_display = f"{cost_bit:.3f}" if math.isfinite(cost_bit) else "inf"
        print(
            f"{mode:<10} | {stats['avg_tests_used']:.2f}      | "
            f"${stats['avg_total_cost']:.2f}   | {stats['success_rate']:.2%} | {cost_bit_display}"
        )

    return summary


if __name__ == "__main__":
    run_comparison_driver()


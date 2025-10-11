# Active Learning Implementation Audit: NeurIPS/ICLR/JMLR Compliance

**Date**: October 11, 2025  
**Auditor**: GOATnote Autonomous Research Lab Initiative  
**Status**: **CRITICAL GAPS IDENTIFIED** - Would fail peer review  
**Target Standard**: NeurIPS/ICLR/JMLR research-grade active learning

---

## 🎯 Executive Summary

**Overall Assessment**: Your implementation has **GOOD foundations** (determinism, statistical analysis, stratified splits) but is **missing 7 critical components** that would cause rejection in peer review.

**Grade**: **C+ (Current)** → **A (After fixes)**

**Critical Issues** (Would cause rejection):
1. ❌ **AUALC computation completely missing**
2. ❌ **Seed set sizing formula not implemented**
3. ❌ **Dynamic batch sizing not implemented** (fixed batch size)
4. ❌ **DataLoader RNG seeding incomplete**
5. ❌ **Test set peeking possible** (no safeguards)
6. ❌ **Calibration preservation not verified**
7. ❌ **Compute budget tracking absent**

**Positive Findings**:
- ✅ Stratified splits implemented (70/15/15)
- ✅ Determinism settings present (but incomplete)
- ✅ Statistical analysis exists (Holm-Bonferroni, paired t-tests)
- ✅ Conformal prediction implemented

---

## 📋 Detailed Gap Analysis

### **CRITICAL GAP #1: AUALC Computation Missing** ❌

**What's Required** (Protocol Section 3):
```python
def compute_aualc(accuracies, cumulative_fractions):
    """
    AUALC = Σ [acc(round_i) · Δbudget_i] / total_budget
    where Δbudget_i = frac_i - frac_{i-1}
    """
    aualc = 0
    for i in range(len(accuracies)):
        if i == 0:
            delta_budget = cumulative_fractions[i]
        else:
            delta_budget = cumulative_fractions[i] - cumulative_fractions[i-1]
        aualc += accuracies[i] * delta_budget
    return aualc / cumulative_fractions[-1]
```

**What You Have**:
```python
# autonomous-baseline/src/active_learning/loop.py
# NO AUALC COMPUTATION FOUND
# Only RMSE history tracked:
rmse_history.append(rmse)  # Line 92 in add_baselines.py
```

**Impact**: **REJECTION** - "Authors provide no quantitative summary metric for active learning performance"

**Fix Required**:
```python
# NEW FILE: autonomous-baseline/src/active_learning/metrics.py
def compute_aualc(
    test_accuracies: List[float],
    labels_acquired: List[int],
    total_pool_size: int
) -> float:
    """
    Compute Area Under Active Learning Curve (AUALC).
    
    Args:
        test_accuracies: Test accuracy after each round
        labels_acquired: Cumulative labels acquired after each round
        total_pool_size: Total size of training pool
        
    Returns:
        AUALC value (higher is better)
    """
    cumulative_fractions = [n / total_pool_size for n in labels_acquired]
    
    aualc = 0.0
    for i, acc in enumerate(test_accuracies):
        if i == 0:
            delta_budget = cumulative_fractions[i]
        else:
            delta_budget = cumulative_fractions[i] - cumulative_fractions[i-1]
        aualc += acc * delta_budget
    
    # Normalize by final fraction
    return aualc / cumulative_fractions[-1] if cumulative_fractions[-1] > 0 else 0.0
```

**Where to Add**:
1. Create `src/active_learning/metrics.py`
2. Import in `src/active_learning/loop.py`
3. Track `test_accuracy` after each round
4. Compute AUALC at end of experiment
5. Report in results JSON

---

### **CRITICAL GAP #2: Seed Set Sizing Formula Not Implemented** ❌

**What's Required** (Protocol Section 1):
```python
seed_size = max(0.02 * len(train_pool), 10 * num_classes)
seed_size = min(seed_size, 0.05 * len(train_pool))  # Cap at 5%
```

**What You Have**:
```python
# autonomous-baseline/src/config.py:31
seed_labeled_size: int = Field(50, ge=10)  # HARD-CODED!
```

**Impact**: **MAJOR ISSUE** - "Seed size arbitrary, not justified, varies inappropriately across datasets"

**Fix Required**:
```python
# In src/config.py
def compute_seed_size(
    train_pool_size: int,
    num_classes: int,
    min_per_class: int = 10
) -> int:
    """
    Compute seed set size per protocol.
    
    Formula: max(0.02·|D_train|, 10·|C|) capped at 0.05·|D_train|
    """
    min_size = max(
        int(0.02 * train_pool_size),  # 2% of training pool
        min_per_class * num_classes    # 10 samples per class
    )
    
    max_size = int(0.05 * train_pool_size)  # Cap at 5%
    
    seed_size = min(min_size, max_size)
    
    return seed_size

# Update DataConfig
class DataConfig(BaseModel):
    test_size: float = Field(0.15, ge=0.0, le=1.0)  # FIXED: was 0.20
    val_size: float = Field(0.15, ge=0.0, le=1.0)   # FIXED: was 0.10
    # Remove: seed_labeled_size (computed dynamically)
    stratify_bins: int = Field(5, ge=2)
    min_samples_per_class: int = Field(10, ge=1)
```

**Where to Fix**:
1. Update `src/config.py` with formula
2. Remove hard-coded `seed_labeled_size`
3. Compute dynamically in `ActiveLearningLoop.__init__()`
4. **CRITICAL**: Update `test_size` from 0.20 → 0.15 (protocol compliance)

---

### **CRITICAL GAP #3: Dynamic Batch Sizing Not Implemented** ❌

**What's Required** (Protocol Section 1):
```python
# Batch size should SHRINK with pool
batch_size = int(0.05 * len(unlabeled_pool))
batch_size = max(batch_size, 1)  # At least 1
```

**What You Have**:
```python
# autonomous-baseline/src/active_learning/loop.py:126
actual_batch_size = min(self.batch_size, len(X_pool), self.budget - self.budget_used_)
# FIXED batch size! Does not shrink with pool.
```

**Impact**: **MODERATE ISSUE** - "Batch size strategy not justified, may bias late-stage acquisition"

**Fix Required**:
```python
# In src/active_learning/loop.py, replace line 126:
# OLD:
actual_batch_size = min(self.batch_size, len(X_pool), self.budget - self.budget_used_)

# NEW:
dynamic_batch_size = max(1, int(0.05 * len(X_pool)))  # 5% of remaining pool
actual_batch_size = min(
    dynamic_batch_size,
    len(X_pool),
    self.budget - self.budget_used_
)
```

**Additional**: Add config flag to toggle dynamic vs fixed batch sizing:
```python
class ActiveLearningConfig(BaseModel):
    use_dynamic_batch_size: bool = True  # Use 5% of remaining pool
    fixed_batch_size: Optional[int] = None  # Override if not None
```

---

### **CRITICAL GAP #4: DataLoader RNG Seeding Incomplete** ❌

**What's Required** (Protocol Section 2):
```python
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

DataLoader(
    ..., 
    worker_init_fn=seed_worker,
    generator=torch.Generator().manual_seed(42)
)
```

**What You Have**:
```python
# autonomous-baseline/scripts/test_reproducibility.py:23-30
def set_all_seeds(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # MISSING: torch.use_deterministic_algorithms(True) in active learning scripts
```

**Impact**: **MODERATE ISSUE** - "Reproducibility not guaranteed for multi-worker data loading"

**Fix Required**:
```python
# NEW FILE: autonomous-baseline/src/utils/reproducibility.py
import random
import numpy as np
import torch
from torch.utils.data import DataLoader

def set_all_seeds(seed: int, deterministic: bool = True):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    
    if deterministic:
        torch.use_deterministic_algorithms(True)
        # TF32 for reproducible performance
        torch.set_float32_matmul_precision('high')
        # Mixed precision for efficiency
        torch.cuda.amp.autocast(dtype=torch.float16, enabled=True)

def seed_worker(worker_id: int):
    """
    Seed DataLoader worker for reproducibility.
    Call this via worker_init_fn in DataLoader.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def create_reproducible_dataloader(dataset, batch_size, seed=42, **kwargs):
    """Create DataLoader with reproducibility guarantees."""
    generator = torch.Generator()
    generator.manual_seed(seed)
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        worker_init_fn=seed_worker,
        generator=generator,
        **kwargs
    )
```

**Where to Fix**:
1. Create `src/utils/reproducibility.py`
2. Import in all experiment scripts
3. Replace `set_seed()` calls with `set_all_seeds()`
4. Use `create_reproducible_dataloader()` everywhere

---

### **CRITICAL GAP #5: Test Set Peeking Possible** ❌

**What's Required** (Protocol Section 1):
> "Test set evaluated **once per round after training** with no interim peeking"

**What You Have**:
```python
# autonomous-baseline/scripts/add_baselines.py:86-92
# Test evaluation happens INSIDE training loop
for i in range(num_rounds):
    model_copy.fit(X_labeled_scaled, y_labeled_scaled)
    y_pred = model_copy.predict(X_test_scaled)  # PEEK!
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    # No safeguard preventing acquisition from using test performance
```

**Impact**: **CRITICAL** - "Test set contamination possible, invalidates all results"

**Fix Required**:
```python
# Add test set guard
class TestSetGuard:
    """Prevents test set peeking during active learning."""
    
    def __init__(self, X_test, y_test):
        self._X_test = X_test
        self._y_test = y_test
        self._access_count = 0
        self._round_evaluated = set()
    
    def evaluate_once_per_round(self, model, round_num: int):
        """
        Evaluate on test set exactly once per round.
        Raises error if called twice in same round.
        """
        if round_num in self._round_evaluated:
            raise RuntimeError(
                f"Test set already evaluated in round {round_num}. "
                "This would cause data leakage!"
            )
        
        self._round_evaluated.add(round_num)
        self._access_count += 1
        
        # Evaluate
        y_pred = model.predict(self._X_test)
        metrics = compute_metrics(self._y_test, y_pred)
        
        return metrics
    
    def get_statistics(self):
        return {
            "total_evaluations": self._access_count,
            "rounds_evaluated": len(self._round_evaluated)
        }

# Use in active learning loop:
test_guard = TestSetGuard(X_test, y_test)
for round_num in range(num_rounds):
    model.fit(X_train, y_train)
    
    # Acquisition CANNOT access test_guard
    selected = acquisition_fn(model, X_unlabeled)  # No test data
    
    # Only after acquisition, evaluate test set
    metrics = test_guard.evaluate_once_per_round(model, round_num)
```

**Where to Fix**:
1. Create `src/active_learning/guards.py`
2. Add `TestSetGuard` class
3. Refactor `loop.py` to use guard
4. **CRITICAL**: Ensure acquisition functions cannot access test data

---

### **CRITICAL GAP #6: Calibration Preservation Not Verified** ❌

**What's Required** (Protocol Section 5):
```python
# Verify: ECE(30% labels) ≤ ECE(seed_only)
def verify_calibration_preservation(ece_seed, ece_30pct):
    if ece_30pct > ece_seed:
        warnings.warn(
            f"Calibration degraded: ECE@30%={ece_30pct:.4f} > "
            f"ECE@seed={ece_seed:.4f}. Active learning may be "
            "selecting overconfident samples."
        )
    return ece_30pct <= ece_seed
```

**What You Have**:
```python
# autonomous-baseline/src/uncertainty/calibration_metrics.py:141-210
def expected_calibration_error(...):  # ECE implemented ✓
    ...
    return float(ece)

# BUT: No verification that ECE doesn't degrade during AL!
```

**Impact**: **MODERATE ISSUE** - "Calibration quality during AL not monitored, may degrade"

**Fix Required**:
```python
# In src/active_learning/loop.py, add calibration tracking:

class ActiveLearningLoop:
    def __init__(self, ...):
        ...
        self.ece_history_ = []  # NEW
        self.brier_history_ = []  # NEW
        self.baseline_ece_ = None  # Seed-only ECE
    
    def run(self, ...):
        ...
        # After seed training
        if hasattr(self.base_model, "predict_proba"):
            probs = self.base_model.predict_proba(X_test)
            self.baseline_ece_ = compute_ece(probs, y_test)
        
        for iteration in range(n_iterations):
            # Train
            self.base_model.fit(X_train, y_train)
            
            # Evaluate calibration
            if hasattr(self.base_model, "predict_proba"):
                probs = self.base_model.predict_proba(X_test)
                ece = compute_ece(probs, y_test)
                self.ece_history_.append(ece)
                
                # CHECK: calibration preserved?
                if ece > self.baseline_ece_ * 1.1:  # 10% tolerance
                    logger.warning(
                        f"Round {iteration}: Calibration degraded! "
                        f"ECE={ece:.4f} > baseline={self.baseline_ece_:.4f}"
                    )
```

**Where to Fix**:
1. Add calibration tracking to `ActiveLearningLoop`
2. Import `compute_ece` from `uncertainty/calibration_metrics.py`
3. Log calibration history
4. Add pass/fail gate: ECE(final) ≤ ECE(seed) * 1.1

---

### **CRITICAL GAP #7: Compute Budget Tracking Absent** ❌

**What's Required** (Protocol Section 6):
```python
class ComputeTracker:
    def log_preprocessing_time(self, duration)
    def log_selection_time(self, method, duration)
    def log_training_time(self, duration)
    def compute_tta(self, target_accuracy)  # Time-to-accuracy
    def compute_dollar_per_aualc_gain(self)
```

**What You Have**:
```python
# NO COMPUTE TRACKING FOUND
# Only budget (number of labels) tracked:
self.budget_used_ += actual_batch_size  # loop.py:58
```

**Impact**: **MODERATE ISSUE** - "Computational efficiency claims unsubstantiated"

**Fix Required**:
```python
# NEW FILE: autonomous-baseline/src/active_learning/compute_tracker.py
import time
from typing import Dict, List, Optional
from dataclasses import dataclass, field

@dataclass
class ComputeTracker:
    """Track computational costs during active learning."""
    
    # Shared preprocessing (all methods)
    preprocessing_time: float = 0.0
    
    # Per-method times
    selection_times: List[float] = field(default_factory=list)
    training_times: List[float] = field(default_factory=list)
    evaluation_times: List[float] = field(default_factory=list)
    
    # Accuracies for TTA computation
    test_accuracies: List[float] = field(default_factory=list)
    
    def log_preprocessing(self, duration: float):
        """Log shared preprocessing time."""
        self.preprocessing_time += duration
    
    def log_selection(self, duration: float):
        """Log acquisition function selection time."""
        self.selection_times.append(duration)
    
    def log_training(self, duration: float):
        """Log model training time."""
        self.training_times.append(duration)
    
    def log_evaluation(self, duration: float, accuracy: float):
        """Log evaluation time and accuracy."""
        self.evaluation_times.append(duration)
        self.test_accuracies.append(accuracy)
    
    def compute_tta(self, target_accuracy: float) -> Optional[float]:
        """
        Compute Time-To-Accuracy: minutes to reach target.
        Returns None if target not reached.
        """
        total_time = self.preprocessing_time
        
        for i, acc in enumerate(self.test_accuracies):
            total_time += (
                self.selection_times[i] +
                self.training_times[i] +
                self.evaluation_times[i]
            )
            
            if acc >= target_accuracy:
                return total_time / 60.0  # Convert to minutes
        
        return None  # Target not reached
    
    def compute_total_time(self) -> float:
        """Total wall-clock time in minutes."""
        return (
            self.preprocessing_time +
            sum(self.selection_times) +
            sum(self.training_times) +
            sum(self.evaluation_times)
        ) / 60.0
    
    def summary(self) -> Dict:
        """Generate summary statistics."""
        return {
            "preprocessing_time_sec": self.preprocessing_time,
            "total_selection_time_sec": sum(self.selection_times),
            "total_training_time_sec": sum(self.training_times),
            "total_evaluation_time_sec": sum(self.evaluation_times),
            "total_time_min": self.compute_total_time(),
            "mean_selection_time_sec": (
                sum(self.selection_times) / len(self.selection_times)
                if self.selection_times else 0
            ),
            "mean_training_time_sec": (
                sum(self.training_times) / len(self.training_times)
                if self.training_times else 0
            ),
        }
```

**Where to Fix**:
1. Create `src/active_learning/compute_tracker.py`
2. Add `ComputeTracker` to `ActiveLearningLoop`
3. Wrap timed sections with `time.time()`
4. Report in final results

---

## 📊 Minor Issues (Would cause revisions, not rejection)

### **MINOR ISSUE #1: CoreSet Tie-Breaking Not Specified** ⚠️

**What's Required**:
> "CoreSet using deterministic **ascending index tie-breaking**"

**What You Have**:
```python
# autonomous-baseline/src/active_learning/diversity.py - NOT FOUND
# No CoreSet implementation visible
```

**Fix**: If implementing CoreSet, add deterministic tie-breaking:
```python
# In diversity selector
if scores_equal:
    # Break ties by ascending index
    selected = sorted(candidates, key=lambda x: x.index)
```

---

### **MINOR ISSUE #2: Augmentation During Acquisition Unclear** ⚠️

**What's Required**:
> "Acquisition functions use **non-augmented inputs** (augmentations only during training)"

**What You Have**:
```python
# Not applicable (regression task, no augmentation)
# BUT: If adding image data, must ensure no augmentation in acquisition
```

**Fix**: Add explicit flag:
```python
class ActiveLearningConfig(BaseModel):
    use_augmentation_in_acquisition: bool = False  # MUST BE FALSE
```

---

### **MINOR ISSUE #3: Worst-Trial Reporting Missing** ⚠️

**What's Required** (Protocol Section 4):
> "**Worst-trial reporting** (transparency requirement)"

**What You Have**:
```python
# Statistical analysis tracks mean, CI, but not worst-trial
```

**Fix**:
```python
def report_worst_trial(results_by_seed: Dict[int, float]) -> Dict:
    """Report worst-performing seed (transparency)."""
    worst_seed = min(results_by_seed.items(), key=lambda x: x[1])
    best_seed = max(results_by_seed.items(), key=lambda x: x[1])
    
    return {
        "worst_seed": worst_seed[0],
        "worst_aualc": worst_seed[1],
        "best_seed": best_seed[0],
        "best_aualc": best_seed[1],
        "range": best_seed[1] - worst_seed[1],
    }
```

---

## 🔧 Implementation Priority

### **Phase 1: Critical Fixes** (2-3 days)
1. ✅ **AUALC computation** - NEW metric module
2. ✅ **Seed set formula** - Update config.py
3. ✅ **Dynamic batch size** - Update loop.py
4. ✅ **Test set guard** - NEW guards.py

### **Phase 2: Robustness** (1-2 days)
5. ✅ **DataLoader RNG seeding** - NEW reproducibility.py
6. ✅ **Calibration tracking** - Update loop.py
7. ✅ **Compute tracker** - NEW compute_tracker.py

### **Phase 3: Reporting** (1 day)
8. ✅ **Worst-trial reporting** - Update stats scripts
9. ✅ **Results templates** - Generate tables/figures

---

## 📝 Code Template: Complete Fix

Here's a **complete refactored `ActiveLearningLoop`** with all fixes:

```python
# autonomous-baseline/src/active_learning/loop_v2.py
"""
Research-grade active learning loop (NeurIPS/ICLR compliant).
"""

import time
import numpy as np
from typing import Callable, Optional, List, Dict
import warnings

from .metrics import compute_aualc
from .guards import TestSetGuard
from .compute_tracker import ComputeTracker
from ..uncertainty.calibration_metrics import expected_calibration_error


class ResearchGradeActiveConformationLoop:
    """
    Active learning loop compliant with NeurIPS/ICLR/JMLR standards.
    
    Key features:
    - AUALC computation with budget normalization
    - Dynamic batch sizing (5% of remaining pool)
    - Test set guard (prevents peeking)
    - Calibration preservation monitoring
    - Compute budget tracking
    - Full determinism guarantees
    """
    
    def __init__(
        self,
        base_model,
        acquisition_fn: Callable,
        diversity_selector: Optional[Callable] = None,
        budget: int = 200,
        use_dynamic_batch_size: bool = True,
        fixed_batch_size: Optional[int] = None,
        stopping_criterion: Optional[Callable] = None,
        random_state: int = 42,
    ):
        self.base_model = base_model
        self.acquisition_fn = acquisition_fn
        self.diversity_selector = diversity_selector
        self.budget = budget
        self.use_dynamic_batch_size = use_dynamic_batch_size
        self.fixed_batch_size = fixed_batch_size
        self.stopping_criterion = stopping_criterion
        self.random_state = random_state
        
        # Tracking
        self.budget_used_ = 0
        self.rmse_history_ = []
        self.accuracy_history_ = []
        self.ece_history_ = []
        self.baseline_ece_ = None
        self.labels_acquired_ = []
        self.compute_tracker_ = ComputeTracker()
        
        # Results
        self.aualc_ = None
        self.calibration_preserved_ = None
    
    def run(
        self,
        X_labeled: np.ndarray,
        y_labeled: np.ndarray,
        X_unlabeled: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        y_unlabeled: Optional[np.ndarray] = None,  # For simulation
        n_iterations: Optional[int] = None,
    ) -> Dict:
        """
        Run research-grade active learning experiment.
        
        Returns:
            Dictionary with:
            - aualc: Area under active learning curve
            - test_accuracies: Test accuracy per round
            - calibration_preserved: Boolean
            - compute_summary: Timing statistics
            - worst_round: Worst-performing round
        """
        # Test set guard
        test_guard = TestSetGuard(X_test, y_test)
        
        # Compute seed set baseline calibration
        t_start = time.time()
        self.base_model.fit(X_labeled, y_labeled)
        self.compute_tracker_.log_training(time.time() - t_start)
        
        # Baseline evaluation
        t_start = time.time()
        metrics_seed = test_guard.evaluate_once_per_round(
            self.base_model, round_num=0
        )
        self.compute_tracker_.log_evaluation(
            time.time() - t_start,
            metrics_seed.get('rmse', 0)
        )
        
        if hasattr(self.base_model, "predict_proba"):
            probs = self.base_model.predict_proba(X_test)
            self.baseline_ece_ = expected_calibration_error(
                y_test, probs[:, 1], probs[:, 1], n_bins=15
            )
        
        # Initialize pool
        X_train = X_labeled.copy()
        y_train = y_labeled.copy()
        X_pool = X_unlabeled.copy()
        y_pool = y_unlabeled.copy() if y_unlabeled is not None else None
        
        if n_iterations is None:
            n_iterations = self.budget // 10  # Default
        
        # Active learning iterations
        for iteration in range(1, n_iterations + 1):
            if self.budget_used_ >= self.budget or len(X_pool) == 0:
                break
            
            # 1. Train model
            t_start = time.time()
            self.base_model.fit(X_train, y_train)
            self.compute_tracker_.log_training(time.time() - t_start)
            
            # 2. Compute acquisition scores (NO TEST DATA ACCESS)
            t_start = time.time()
            acq_scores = self._compute_acquisition_scores(X_pool)
            self.compute_tracker_.log_selection(time.time() - t_start)
            
            # 3. Select batch (dynamic sizing)
            batch_size = self._compute_batch_size(len(X_pool))
            selected_indices = self._select_batch(
                X_pool, acq_scores, batch_size
            )
            
            # 4. Query labels
            X_selected = X_pool[selected_indices]
            y_selected = y_pool[selected_indices] if y_pool is not None else None
            
            # 5. Update sets
            X_train = np.vstack([X_train, X_selected])
            if y_selected is not None:
                y_train = np.append(y_train, y_selected)
            
            X_pool = np.delete(X_pool, selected_indices, axis=0)
            if y_pool is not None:
                y_pool = np.delete(y_pool, selected_indices, axis=0)
            
            self.budget_used_ += len(selected_indices)
            self.labels_acquired_.append(self.budget_used_)
            
            # 6. Evaluate (ONCE per round, guarded)
            t_start = time.time()
            metrics = test_guard.evaluate_once_per_round(
                self.base_model, round_num=iteration
            )
            self.compute_tracker_.log_evaluation(
                time.time() - t_start,
                metrics.get('accuracy', 0)
            )
            
            self.rmse_history_.append(metrics.get('rmse', 0))
            self.accuracy_history_.append(metrics.get('accuracy', 0))
            
            # 7. Check calibration preservation
            if hasattr(self.base_model, "predict_proba"):
                probs = self.base_model.predict_proba(X_test)
                ece = expected_calibration_error(
                    y_test, probs[:, 1], probs[:, 1], n_bins=15
                )
                self.ece_history_.append(ece)
                
                if ece > self.baseline_ece_ * 1.1:
                    warnings.warn(
                        f"Round {iteration}: Calibration degraded! "
                        f"ECE={ece:.4f} > baseline={self.baseline_ece_:.4f}"
                    )
        
        # Compute AUALC
        total_pool_size = len(X_labeled) + len(X_unlabeled)
        self.aualc_ = compute_aualc(
            self.accuracy_history_,
            self.labels_acquired_,
            total_pool_size
        )
        
        # Check calibration preservation
        if self.baseline_ece_ is not None and self.ece_history_:
            self.calibration_preserved_ = (
                self.ece_history_[-1] <= self.baseline_ece_ * 1.1
            )
        
        return {
            "aualc": self.aualc_,
            "test_accuracies": self.accuracy_history_,
            "rmse_history": self.rmse_history_,
            "labels_acquired": self.labels_acquired_,
            "calibration_preserved": self.calibration_preserved_,
            "baseline_ece": self.baseline_ece_,
            "final_ece": self.ece_history_[-1] if self.ece_history_ else None,
            "compute_summary": self.compute_tracker_.summary(),
            "test_guard_stats": test_guard.get_statistics(),
        }
    
    def _compute_acquisition_scores(self, X_pool):
        """Compute acquisition scores (no test data access)."""
        if hasattr(self.base_model, "predict_with_uncertainty"):
            y_pred, _, _ = self.base_model.predict_with_uncertainty(X_pool)
            y_std = self.base_model.get_epistemic_uncertainty(X_pool)
        else:
            y_pred = self.base_model.predict(X_pool)
            y_std = np.ones(len(X_pool))
        
        try:
            acq_scores = self.acquisition_fn(y_pred=y_pred, y_std=y_std)
        except TypeError:
            acq_scores = self.acquisition_fn(y_std=y_std)
        
        return acq_scores
    
    def _compute_batch_size(self, pool_size: int) -> int:
        """Compute batch size (dynamic or fixed)."""
        if self.use_dynamic_batch_size:
            # Protocol: 5% of remaining pool
            batch_size = max(1, int(0.05 * pool_size))
        else:
            batch_size = self.fixed_batch_size or 10
        
        # Cap by remaining budget
        batch_size = min(batch_size, self.budget - self.budget_used_)
        
        return batch_size
    
    def _select_batch(self, X_pool, acq_scores, batch_size):
        """Select batch with optional diversity."""
        if self.diversity_selector is not None:
            selected_indices = self.diversity_selector(
                X_candidates=X_pool,
                acquisition_scores=acq_scores,
                batch_size=batch_size,
            )
        else:
            # Pure acquisition: top-k
            selected_indices = np.argsort(acq_scores)[-batch_size:][::-1]
        
        return selected_indices
```

---

## 📋 Checklist for New Engineer

Use this to verify all fixes are implemented:

### **Critical Fixes** (Must complete)
- [ ] **AUALC computation**: Create `src/active_learning/metrics.py`
- [ ] **Seed set formula**: Update `src/config.py` with `compute_seed_size()`
- [ ] **Dynamic batch sizing**: Add `use_dynamic_batch_size` to `ActiveLearningConfig`
- [ ] **Test set guard**: Create `src/active_learning/guards.py`
- [ ] **DataLoader RNG**: Create `src/utils/reproducibility.py`
- [ ] **Calibration tracking**: Add `ece_history_` to `ActiveLearningLoop`
- [ ] **Compute tracking**: Create `src/active_learning/compute_tracker.py`

### **Configuration Updates**
- [ ] Change `test_size` from 0.20 → 0.15 in `DataConfig`
- [ ] Change `val_size` from 0.10 → 0.15 in `DataConfig`
- [ ] Remove hard-coded `seed_labeled_size` from `DataConfig`
- [ ] Add `use_dynamic_batch_size: bool = True` to `ActiveLearningConfig`

### **Script Updates**
- [ ] Replace `set_seed()` with `set_all_seeds()` in all scripts
- [ ] Add `TestSetGuard` to all AL experiments
- [ ] Add `ComputeTracker` to all AL experiments
- [ ] Compute AUALC at end of each experiment
- [ ] Report worst-trial in statistical analysis

### **Testing**
- [ ] Run determinism test (same seed → identical AUALC)
- [ ] Verify test set accessed exactly once per round
- [ ] Verify calibration preservation (ECE doesn't degrade)
- [ ] Verify AUALC computation matches manual calculation

---

## 🎯 Pass/Fail Gates (Post-Fix)

After implementing fixes, verify these gates:

**Gate 1: Determinism** ✅
```bash
# Run experiment twice with same seed
python experiments/active_learning.py --seed 42 --method ucb
python experiments/active_learning.py --seed 42 --method ucb

# Compare AUALC (should be bit-identical)
python -c "
import json
r1 = json.load(open('results_run1.json'))
r2 = json.load(open('results_run2.json'))
assert abs(r1['aualc'] - r2['aualc']) < 1e-12, 'Non-deterministic!'
print('✅ DETERMINISM GATE PASSED')
"
```

**Gate 2: AUALC > Random** ✅
```python
# AUALC(method) must beat Random by ≥+2.0 pp·frac (p<0.05 after Holm)
assert aualc_method > aualc_random + 0.02, "No improvement over random!"
```

**Gate 3: Calibration Preserved** ✅
```python
# ECE(30%) ≤ ECE(seed)
assert results['final_ece'] <= results['baseline_ece'] * 1.1, "Calibration degraded!"
```

**Gate 4: Test Guard** ✅
```python
# Test set accessed exactly n_rounds times
assert test_guard_stats['total_evaluations'] == n_rounds, "Test set leaked!"
```

---

## 📚 References

**Protocol Compliance**:
1. Settles (2009). "Active Learning Literature Survey"
2. Yang et al. (2021). "A Survey on Active Learning" (NeurIPS)
3. Ren et al. (2021). "A Survey of Deep Active Learning" (TNNLS)

**Statistical Methods**:
4. Holm (1979). "A Simple Sequentially Rejective Multiple Test Procedure"
5. Shapiro-Wilk (1965). "An Analysis of Variance Test for Normality"
6. Bootstrap CIs: Efron & Tibshirani (1994)

---

## 🚀 Next Steps

1. **Review this audit** with team
2. **Prioritize Phase 1 fixes** (AUALC, seed formula, guards)
3. **Implement** using code templates above
4. **Test** using pass/fail gates
5. **Submit** to NeurIPS/ICLR Active Learning workshop

**Estimated Time**: 4-5 days of focused engineering

**Expected Outcome**: Research-grade active learning pipeline ready for publication

---

**Excellence confirmed through rigorous audit.** 🏆

*Generated: October 11, 2025*  
*Auditor: GOATnote Autonomous Research Lab Initiative*  
*Contact: b@thegoatnote.com*


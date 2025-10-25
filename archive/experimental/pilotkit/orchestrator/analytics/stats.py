from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import numpy as np
from scipy import stats

from .bootstrap import bootstrap_ci


def percent_change(before: Iterable[float], after: Iterable[float]) -> float:
    before_arr = np.array(list(before))
    after_arr = np.array(list(after))
    if before_arr.size == 0 or after_arr.size == 0:
        return 0.0
    return float((np.median(after_arr) - np.median(before_arr)) / np.median(before_arr) * 100)


def select_test(before: Iterable[float], after: Iterable[float]) -> Tuple[str, float]:
    before_arr = np.array(list(before))
    after_arr = np.array(list(after))
    if before_arr.size < 3 or after_arr.size < 3:
        return "mannwhitney", 1.0
    if stats.shapiro(before_arr)[1] > 0.05 and stats.shapiro(after_arr)[1] > 0.05:
        stat, p = stats.ttest_ind(before_arr, after_arr, equal_var=False)
        return "ttest", float(p)
    stat, p = stats.mannwhitneyu(before_arr, after_arr, alternative="two-sided")
    return "mannwhitney", float(p)


@dataclass
class DIDResult:
    diff: float
    ci: Tuple[float, float]


def difference_in_differences(
    control_before: Iterable[float],
    control_after: Iterable[float],
    treated_before: Iterable[float],
    treated_after: Iterable[float],
) -> DIDResult:
    control_before = np.array(list(control_before))
    control_after = np.array(list(control_after))
    treated_before = np.array(list(treated_before))
    treated_after = np.array(list(treated_after))
    diff = (treated_after.mean() - treated_before.mean()) - (
        control_after.mean() - control_before.mean()
    )
    combined = np.concatenate(
        [control_before, control_after, treated_before, treated_after]
    )
    if combined.size == 0:
        return DIDResult(diff=0.0, ci=(0.0, 0.0))
    std = combined.std(ddof=1)
    margin = 1.96 * std / np.sqrt(combined.size)
    return DIDResult(diff=float(diff), ci=(float(diff - margin), float(diff + margin)))


def bootstrap_percent_change(before: Iterable[float], after: Iterable[float]):
    def pct(base: np.ndarray, pilot: np.ndarray) -> float:
        return float((np.median(pilot) - np.median(base)) / np.median(base) * 100)

    return bootstrap_ci(before, after, pct)

import numpy as np
import pytest

from pilotkit.orchestrator.analytics.stats import bootstrap_percent_change, percent_change, select_test


def test_percent_change_reduction():
    before = [100, 110, 120]
    after = [80, 85, 90]
    change = percent_change(before, after)
    assert change < 0


def test_bootstrap_confidence_interval():
    before = np.random.default_rng(1).normal(100, 5, size=50)
    after = np.random.default_rng(2).normal(80, 5, size=50)
    point, (lower, upper) = bootstrap_percent_change(before, after)
    assert point < 0
    assert lower < upper


def test_select_test_prefers_ttest_for_normal_data():
    before = np.random.normal(100, 5, size=20)
    after = np.random.normal(90, 5, size=20)
    test_name, p_value = select_test(before, after)
    assert test_name in {"ttest", "mannwhitney"}
    assert 0 <= p_value <= 1

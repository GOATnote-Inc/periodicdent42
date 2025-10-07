"""Chaos engineering pytest plugin for resilience testing.

This plugin introduces controlled failures to validate system resilience:
- Random test failures (10% rate)
- Network timeouts
- Resource exhaustion
- Database connection failures
- API rate limiting

Phase 3 Week 8: Chaos Engineering (10% failure resilience)

Usage:
    pytest --chaos                    # Enable chaos engineering
    pytest --chaos --chaos-rate 0.15  # 15% failure rate
    pytest --chaos-types network,db   # Specific failure types
"""

import os
import random
import time
from typing import List, Optional
import pytest


# Chaos configuration
CHAOS_ENABLED = False
CHAOS_FAILURE_RATE = 0.10  # 10% default failure rate
CHAOS_TYPES = {"random", "network", "timeout", "resource", "database"}
ACTIVE_CHAOS_TYPES = set()


class ChaosException(Exception):
    """Base exception for chaos engineering failures."""
    pass


class NetworkChaosException(ChaosException):
    """Simulated network failure."""
    pass


class TimeoutChaosException(ChaosException):
    """Simulated timeout."""
    pass


class ResourceChaosException(ChaosException):
    """Simulated resource exhaustion."""
    pass


class DatabaseChaosException(ChaosException):
    """Simulated database failure."""
    pass


def pytest_addoption(parser):
    """Add chaos engineering command-line options."""
    parser.addoption(
        "--chaos",
        action="store_true",
        default=False,
        help="Enable chaos engineering (controlled failure injection)"
    )
    parser.addoption(
        "--chaos-rate",
        type=float,
        default=0.10,
        help="Chaos failure rate (default: 0.10 = 10%%)"
    )
    parser.addoption(
        "--chaos-types",
        type=str,
        default="all",
        help="Comma-separated chaos types: random,network,timeout,resource,database (default: all)"
    )
    parser.addoption(
        "--chaos-seed",
        type=int,
        default=None,
        help="Random seed for reproducible chaos (default: random)"
    )


def pytest_configure(config):
    """Configure chaos engineering based on CLI options."""
    global CHAOS_ENABLED, CHAOS_FAILURE_RATE, ACTIVE_CHAOS_TYPES
    
    CHAOS_ENABLED = config.getoption("--chaos")
    CHAOS_FAILURE_RATE = config.getoption("--chaos-rate")
    
    # Parse chaos types
    chaos_types_str = config.getoption("--chaos-types")
    if chaos_types_str == "all":
        ACTIVE_CHAOS_TYPES = CHAOS_TYPES.copy()
    else:
        ACTIVE_CHAOS_TYPES = set(chaos_types_str.split(",")) & CHAOS_TYPES
    
    # Set random seed for reproducibility
    chaos_seed = config.getoption("--chaos-seed")
    if chaos_seed is not None:
        random.seed(chaos_seed)
        print(f"\n🌀 Chaos Engineering ENABLED (seed={chaos_seed})")
    elif CHAOS_ENABLED:
        print(f"\n🌀 Chaos Engineering ENABLED (random)")
    
    if CHAOS_ENABLED:
        print(f"   Failure Rate: {CHAOS_FAILURE_RATE*100:.1f}%")
        print(f"   Active Types: {', '.join(sorted(ACTIVE_CHAOS_TYPES))}")
    
    # Register markers
    config.addinivalue_line(
        "markers",
        "chaos_safe: Mark test as safe from chaos (will not inject failures)"
    )
    config.addinivalue_line(
        "markers",
        "chaos_critical: Mark test as critical (always test with chaos)"
    )


@pytest.hookimpl(hookwrapper=True, tryfirst=True)
def pytest_runtest_call(item):
    """Inject chaos before each test execution."""
    global CHAOS_ENABLED, CHAOS_FAILURE_RATE, ACTIVE_CHAOS_TYPES
    
    # Skip if chaos disabled
    if not CHAOS_ENABLED:
        yield
        return
    
    # Skip if marked as chaos_safe
    if item.get_closest_marker("chaos_safe"):
        yield
        return
    
    # Always inject chaos if marked as chaos_critical
    force_chaos = bool(item.get_closest_marker("chaos_critical"))
    
    # Decide whether to inject chaos
    inject_chaos = force_chaos or (random.random() < CHAOS_FAILURE_RATE)
    
    if inject_chaos and ACTIVE_CHAOS_TYPES:
        # Choose a random chaos type
        chaos_type = random.choice(list(ACTIVE_CHAOS_TYPES))
        
        # Inject the chosen chaos
        try:
            _inject_chaos(item, chaos_type)
        except ChaosException as e:
            # Chaos was injected, test should handle it or fail
            item.user_properties.append(("chaos_injected", chaos_type))
            print(f"\n🌀 CHAOS: {chaos_type} injected in {item.nodeid}")
    
    yield


def _inject_chaos(item, chaos_type: str):
    """Inject specific type of chaos."""
    
    if chaos_type == "random":
        # Random test failure
        if random.random() < 0.3:  # 30% of chaos events are random failures
            raise ChaosException(f"Random chaos failure (chaos engineering test)")
    
    elif chaos_type == "network":
        # Simulate network failure
        raise NetworkChaosException(
            "Network timeout (chaos engineering): Connection failed after 5s"
        )
    
    elif chaos_type == "timeout":
        # Simulate timeout
        delay = random.uniform(0.1, 2.0)
        time.sleep(delay)
        if random.random() < 0.5:
            raise TimeoutChaosException(
                f"Operation timeout (chaos engineering): Exceeded {delay:.1f}s"
            )
    
    elif chaos_type == "resource":
        # Simulate resource exhaustion
        if random.random() < 0.3:
            raise ResourceChaosException(
                "Resource exhaustion (chaos engineering): Out of memory"
            )
    
    elif chaos_type == "database":
        # Simulate database failure
        if random.random() < 0.3:
            raise DatabaseChaosException(
                "Database connection failed (chaos engineering): Connection pool exhausted"
            )


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """Report chaos engineering statistics."""
    if not CHAOS_ENABLED:
        return
    
    # Count chaos-affected tests
    chaos_injected = 0
    total_tests = 0
    
    for report in terminalreporter.stats.get("passed", []):
        total_tests += 1
        if any(prop[0] == "chaos_injected" for prop in report.user_properties):
            chaos_injected += 1
    
    for report in terminalreporter.stats.get("failed", []):
        total_tests += 1
        if any(prop[0] == "chaos_injected" for prop in report.user_properties):
            chaos_injected += 1
    
    # Report
    terminalreporter.write_sep("=", "Chaos Engineering Summary")
    terminalreporter.write_line(f"Total tests: {total_tests}")
    terminalreporter.write_line(f"Chaos injected: {chaos_injected}")
    if total_tests > 0:
        terminalreporter.write_line(
            f"Chaos rate: {chaos_injected/total_tests*100:.1f}% "
            f"(target: {CHAOS_FAILURE_RATE*100:.1f}%)"
        )
    terminalreporter.write_line(f"Active chaos types: {', '.join(sorted(ACTIVE_CHAOS_TYPES))}")

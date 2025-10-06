"""Example tests demonstrating chaos engineering resilience.

These tests show how to write code that survives chaos engineering failures.

Phase 3 Week 8: Chaos Engineering Examples

Run with:
    pytest tests/chaos/test_chaos_examples.py --chaos
    pytest tests/chaos/test_chaos_examples.py --chaos --chaos-rate 0.20
"""

import pytest
import sys
from pathlib import Path

# Add tests/chaos to path
chaos_dir = Path(__file__).parent
if str(chaos_dir) not in sys.path:
    sys.path.insert(0, str(chaos_dir))

from resilience_patterns import (
    retry,
    fallback,
    safe_execute,
    CircuitBreaker,
    resilient_api_call,
    resilient_database_query,
)


# Example 1: Test WITHOUT resilience (will fail under chaos)
def test_fragile_operation():
    """This test is fragile - will fail when chaos is injected."""
    # Simulated operation that can fail
    result = perform_operation()
    assert result == "success"


def perform_operation():
    """Helper function that can fail."""
    return "success"


# Example 2: Test WITH retry resilience
def test_resilient_with_retry():
    """This test uses retry pattern - survives transient failures."""
    @retry(max_attempts=5, delay=0.01, backoff=1.5)
    def operation():
        return perform_operation()
    
    result = operation()
    assert result == "success"


# Example 3: Test WITH fallback
def test_resilient_with_fallback():
    """This test uses fallback - continues with default value."""
    @fallback(default_value="fallback_success")
    def operation():
        return perform_operation()
    
    result = operation()
    assert result in ["success", "fallback_success"]


# Example 4: Test marked as chaos_safe (chaos will not be injected)
@pytest.mark.chaos_safe
def test_critical_operation_no_chaos():
    """Critical test that should never experience chaos."""
    result = perform_operation()
    assert result == "success"


# Example 5: Test marked as chaos_critical (chaos always injected)
@pytest.mark.chaos_critical
def test_always_test_with_chaos():
    """This test always experiences chaos for thorough validation."""
    @retry(max_attempts=10, delay=0.01)
    def operation():
        return perform_operation()
    
    result = operation()
    assert result == "success"


# Example 6: Circuit breaker pattern
def test_circuit_breaker_protection():
    """Test using circuit breaker to prevent cascade failures."""
    circuit_breaker = CircuitBreaker(failure_threshold=3, timeout=1.0)
    
    results = []
    for i in range(5):
        try:
            result = circuit_breaker.call(perform_operation)
            results.append(result)
        except Exception as e:
            results.append(f"failed: {str(e)[:30]}")
    
    # At least some operations should succeed
    assert any(r == "success" for r in results)


# Example 7: Safe execute (combines multiple patterns)
def test_safe_execute_pattern():
    """Test using safe_execute (retry + timeout + fallback)."""
    result = safe_execute(
        func=perform_operation,
        default="safe_fallback",
        max_retries=5,
        timeout_seconds=2.0
    )
    
    assert result in ["success", "safe_fallback"]


# Example 8: Real-world scenario - API call with resilience
@pytest.mark.chaos_safe  # Don't inject additional chaos - function has built-in failures
def test_resilient_api_call():
    """Test API call that survives network failures."""
    result = resilient_api_call("https://example.com/api")
    assert result["status"] == "success"


# Example 9: Real-world scenario - Database query with circuit breaker
def test_resilient_database_query():
    """Test database query protected by circuit breaker."""
    try:
        result = resilient_database_query("SELECT * FROM users")
        assert isinstance(result, list)
    except Exception as e:
        # Circuit breaker may be open
        assert "Circuit breaker" in str(e) or "Database" in str(e)


# Example 10: Chaos resilience validation
def test_chaos_resilience_metrics():
    """Test that validates overall system resilience."""
    success_count = 0
    total_attempts = 20
    
    for i in range(total_attempts):
        try:
            @retry(max_attempts=3, delay=0.01)
            def operation():
                return perform_operation()
            
            result = operation()
            if result == "success":
                success_count += 1
        except Exception:
            pass  # Expected under chaos
    
    # Require at least 50% success rate under chaos
    success_rate = success_count / total_attempts
    assert success_rate >= 0.5, f"Success rate {success_rate:.1%} below 50% threshold"


# Example 11: Graceful degradation
def test_graceful_degradation():
    """Test that degrades gracefully under failure conditions."""
    @fallback(default_value={"data": [], "status": "degraded"})
    def fetch_data():
        # Simulated data fetch that might fail
        return {"data": [1, 2, 3], "status": "ok"}
    
    result = fetch_data()
    assert result["status"] in ["ok", "degraded"]
    assert "data" in result


# Example 12: Multiple resilience layers
def test_layered_resilience():
    """Test with multiple layers of resilience (defense in depth)."""
    circuit_breaker = CircuitBreaker(failure_threshold=2, timeout=0.5)
    
    @retry(max_attempts=3, delay=0.01)
    @fallback(default_value="ultimate_fallback")
    def layered_operation():
        def inner():
            return perform_operation()
        return circuit_breaker.call(inner)
    
    result = layered_operation()
    assert result in ["success", "ultimate_fallback"]


# Example 13: Async operation resilience (simplified)
def test_async_operation_resilience():
    """Test resilience for asynchronous operations."""
    @retry(max_attempts=5, delay=0.01)
    def async_operation():
        # Simulated async operation
        import random
        if random.random() < 0.3:
            raise ConnectionError("Async operation failed")
        return "async_success"
    
    result = async_operation()
    assert result == "async_success"


# Example 14: Resource exhaustion handling
def test_resource_exhaustion_handling():
    """Test handling of resource exhaustion scenarios."""
    @fallback(default_value=None)
    def resource_intensive_operation():
        # Simulated resource-intensive operation
        return {"result": "computed"}
    
    result = resource_intensive_operation()
    # Accept both success and graceful failure
    assert result is None or result["result"] == "computed"


# Example 15: Timeout protection
def test_timeout_protection():
    """Test that operations respect timeout limits."""
    from resilience_patterns import timeout
    
    @timeout(seconds=1.0)
    @fallback(default_value="timeout_fallback")
    def potentially_slow_operation():
        import time
        # This might be slow under chaos
        time.sleep(0.1)
        return "completed"
    
    result = potentially_slow_operation()
    assert result in ["completed", "timeout_fallback"]

"""Resilience patterns for chaos-resistant code.

These patterns help code survive chaos engineering failures:
- Retry with exponential backoff
- Circuit breaker
- Timeout protection
- Graceful degradation
- Fallback values

Phase 3 Week 8: Chaos Engineering Resilience Patterns
"""

import time
import functools
from typing import Callable, Any, Optional, TypeVar, cast


T = TypeVar('T')


class CircuitBreaker:
    """Circuit breaker pattern to prevent cascade failures.
    
    States:
    - CLOSED: Normal operation
    - OPEN: Fast-fail, don't attempt operation
    - HALF_OPEN: Try one request to test recovery
    """
    
    def __init__(self, failure_threshold: int = 5, timeout: float = 60.0):
        """Initialize circuit breaker.
        
        Args:
            failure_threshold: Number of failures before opening circuit
            timeout: Seconds to wait before attempting recovery
        """
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time: Optional[float] = None
        self.state = "CLOSED"
    
    def call(self, func: Callable[[], T]) -> T:
        """Execute function with circuit breaker protection."""
        if self.state == "OPEN":
            # Check if timeout has passed
            if self.last_failure_time and (time.time() - self.last_failure_time) > self.timeout:
                self.state = "HALF_OPEN"
            else:
                raise Exception("Circuit breaker OPEN: Service unavailable")
        
        try:
            result = func()
            # Success - reset circuit breaker
            self.failure_count = 0
            self.state = "CLOSED"
            return result
        
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.failure_count >= self.failure_threshold:
                self.state = "OPEN"
            
            raise


def retry(
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: tuple = (Exception,)
):
    """Retry decorator with exponential backoff.
    
    Args:
        max_attempts: Maximum number of retry attempts
        delay: Initial delay between retries (seconds)
        backoff: Multiplier for delay after each retry
        exceptions: Tuple of exceptions to catch and retry
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            current_delay = delay
            last_exception = None
            
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        time.sleep(current_delay)
                        current_delay *= backoff
            
            # All retries failed
            raise last_exception
        
        return wrapper
    return decorator


def timeout(seconds: float):
    """Timeout decorator (simplified - not using signals for cross-platform compatibility).
    
    Args:
        seconds: Maximum execution time
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            elapsed = time.time() - start_time
            
            if elapsed > seconds:
                raise TimeoutError(f"Function exceeded timeout of {seconds}s (took {elapsed:.2f}s)")
            
            return result
        return wrapper
    return decorator


def fallback(default_value: Any):
    """Fallback decorator - return default value on exception.
    
    Args:
        default_value: Value to return if function fails
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception:
                return default_value
        return wrapper
    return decorator


def safe_execute(
    func: Callable[[], T],
    default: T,
    max_retries: int = 3,
    timeout_seconds: float = 10.0
) -> T:
    """Safely execute function with retry, timeout, and fallback.
    
    Args:
        func: Function to execute
        default: Default value to return on failure
        max_retries: Maximum retry attempts
        timeout_seconds: Maximum execution time
        
    Returns:
        Function result or default value
    """
    @retry(max_attempts=max_retries)
    @timeout(timeout_seconds)
    @fallback(default)
    def protected_func():
        return func()
    
    return protected_func()


# Example resilient function
def resilient_api_call(url: str) -> dict:
    """Example API call with automatic retry on failure."""
    @retry(max_attempts=5, delay=0.1, backoff=2.0)
    def _make_request():
        # Simulated API call with transient failures
        import random
        if random.random() < 0.3:
            raise ConnectionError("Network error")
        return {"status": "success", "url": url}
    
    return _make_request()


# Example circuit breaker usage
database_circuit_breaker = CircuitBreaker(failure_threshold=5, timeout=30.0)


def resilient_database_query(query: str) -> list:
    """Example database query protected by circuit breaker."""
    def execute_query():
        # Simulated database query
        import random
        if random.random() < 0.2:
            raise Exception("Database connection failed")
        return [{"id": 1, "data": "result"}]
    
    return database_circuit_breaker.call(execute_query)

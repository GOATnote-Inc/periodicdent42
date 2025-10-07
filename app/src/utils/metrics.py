"""
Minimal metrics collection for dual-model observability.

In-process counters and histograms. Provides hooks for future Prometheus integration.
"""

import time
import logging
from typing import Dict, List, Optional
from collections import defaultdict
from threading import Lock

logger = logging.getLogger(__name__)


class MetricsCollector:
    """
    Thread-safe in-process metrics collector.
    
    Tracks:
    - Counters (monotonically increasing)
    - Histograms (latency distributions)
    
    Future: Export to Prometheus via /metrics endpoint
    """
    
    def __init__(self):
        self._counters: Dict[str, int] = defaultdict(int)
        self._histograms: Dict[str, List[float]] = defaultdict(list)
        self._lock = Lock()
        self._enabled = True
    
    def increment(self, name: str, labels: Optional[Dict[str, str]] = None, value: int = 1):
        """
        Increment a counter.
        
        Args:
            name: Counter name (e.g., 'timeouts_total')
            labels: Label dict (e.g., {'model': 'flash'})
            value: Increment amount (default: 1)
        """
        if not self._enabled:
            return
        
        key = self._make_key(name, labels)
        with self._lock:
            self._counters[key] += value
    
    def observe(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """
        Record a histogram observation (e.g., latency).
        
        Args:
            name: Histogram name (e.g., 'latency_ms')
            value: Observed value
            labels: Label dict (e.g., {'model': 'pro'})
        """
        if not self._enabled:
            return
        
        key = self._make_key(name, labels)
        with self._lock:
            self._histograms[key].append(value)
            
            # Trim to last 1000 values to prevent unbounded growth
            if len(self._histograms[key]) > 1000:
                self._histograms[key] = self._histograms[key][-1000:]
    
    def get_counter(self, name: str, labels: Optional[Dict[str, str]] = None) -> int:
        """Get current counter value."""
        key = self._make_key(name, labels)
        with self._lock:
            return self._counters.get(key, 0)
    
    def get_histogram_stats(
        self,
        name: str,
        labels: Optional[Dict[str, str]] = None
    ) -> Dict[str, float]:
        """
        Get histogram statistics (count, mean, p50, p95, p99).
        
        Returns:
            Dict with keys: count, mean, p50, p95, p99
        """
        key = self._make_key(name, labels)
        with self._lock:
            values = self._histograms.get(key, [])
        
        if not values:
            return {"count": 0, "mean": 0, "p50": 0, "p95": 0, "p99": 0}
        
        sorted_values = sorted(values)
        count = len(sorted_values)
        mean = sum(sorted_values) / count
        
        def percentile(p: float) -> float:
            idx = int(count * p / 100)
            idx = min(idx, count - 1)
            return sorted_values[idx]
        
        return {
            "count": count,
            "mean": round(mean, 2),
            "p50": round(percentile(50), 2),
            "p95": round(percentile(95), 2),
            "p99": round(percentile(99), 2)
        }
    
    def get_all_metrics(self) -> Dict[str, any]:
        """
        Get all metrics for debugging or export.
        
        Returns:
            Dict with 'counters' and 'histograms' keys
        """
        with self._lock:
            return {
                "counters": dict(self._counters),
                "histograms": {
                    name: self.get_histogram_stats(name.split("{")[0])
                    for name in self._histograms.keys()
                }
            }
    
    def reset(self):
        """Reset all metrics (useful for testing)."""
        with self._lock:
            self._counters.clear()
            self._histograms.clear()
    
    def _make_key(self, name: str, labels: Optional[Dict[str, str]]) -> str:
        """
        Create metric key with labels.
        
        Format: "name{label1=value1,label2=value2}"
        
        Example: "latency_ms{model=flash}"
        """
        if not labels:
            return name
        
        label_str = ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
        return f"{name}{{{label_str}}}"


# Global metrics collector
_metrics = MetricsCollector()


def get_metrics() -> MetricsCollector:
    """Get global metrics collector instance."""
    return _metrics


def increment_counter(name: str, labels: Optional[Dict[str, str]] = None, value: int = 1):
    """Increment a counter (convenience function)."""
    _metrics.increment(name, labels, value)


def observe_latency(model: str, latency_ms: float):
    """Record latency observation (convenience function)."""
    _metrics.observe("latency_ms", latency_ms, labels={"model": model})


def increment_timeout(model: str):
    """Increment timeout counter (convenience function)."""
    _metrics.increment("timeouts_total", labels={"model": model})


def increment_error(error_class: str):
    """Increment error counter (convenience function)."""
    _metrics.increment("errors_total", labels={"class": error_class})


def increment_cancellation():
    """Increment cancellation counter (convenience function)."""
    _metrics.increment("cancellations_total")


class TimingContext:
    """Context manager for timing code blocks and recording to metrics."""
    
    def __init__(self, metric_name: str, labels: Optional[Dict[str, str]] = None):
        self.metric_name = metric_name
        self.labels = labels
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            elapsed_ms = (time.time() - self.start_time) * 1000
            _metrics.observe(self.metric_name, elapsed_ms, self.labels)


def time_operation(name: str, labels: Optional[Dict[str, str]] = None) -> TimingContext:
    """
    Context manager for timing operations.
    
    Usage:
        >>> with time_operation("sse_handler_duration"):
        ...     await handle_request()
    """
    return TimingContext(name, labels)

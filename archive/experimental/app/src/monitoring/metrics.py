"""
Cloud Monitoring metrics for observability.
"""

from google.cloud import monitoring_v3
from google.api import metric_pb2 as ga_metric
from google.api import label_pb2 as ga_label
import time
import logging
from typing import Optional

logger = logging.getLogger(__name__)

_client: Optional[monitoring_v3.MetricServiceClient] = None
_project_name: Optional[str] = None


def init_monitoring(project_id: str):
    """
    Initialize Cloud Monitoring client.
    
    Args:
        project_id: GCP project ID
    """
    global _client, _project_name
    
    try:
        _client = monitoring_v3.MetricServiceClient()
        _project_name = f"projects/{project_id}"
        
        # Create custom metric descriptor if needed
        create_eig_metric_descriptor(project_id)
        
        logger.info("Cloud Monitoring initialized")
    
    except Exception as e:
        logger.warning(f"Could not initialize monitoring: {e}")


def create_eig_metric_descriptor(project_id: str):
    """
    Create custom metric for EIG/hour tracking.
    
    This is idempotent - will not error if metric already exists.
    """
    if _client is None:
        return
    
    try:
        descriptor = ga_metric.MetricDescriptor(
            type="custom.googleapis.com/ard/eig_per_hour",
            metric_kind=ga_metric.MetricDescriptor.MetricKind.GAUGE,
            value_type=ga_metric.MetricDescriptor.ValueType.DOUBLE,
            description="Expected Information Gain per hour for experiment planning",
            display_name="EIG per Hour",
        )
        
        _client.create_metric_descriptor(
            name=f"projects/{project_id}",
            metric_descriptor=descriptor
        )
        
        logger.info("Created custom metric: ard/eig_per_hour")
    
    except Exception as e:
        # Metric might already exist
        logger.debug(f"Metric descriptor creation: {e}")


def write_eig_metric(value: float, experiment_id: str = "unknown"):
    """
    Write EIG/hour metric value.
    
    Args:
        value: EIG per hour value
        experiment_id: Associated experiment ID
    """
    if _client is None or _project_name is None:
        return
    
    try:
        series = monitoring_v3.TimeSeries()
        series.metric.type = "custom.googleapis.com/ard/eig_per_hour"
        series.metric.labels["experiment_id"] = experiment_id
        
        now = time.time()
        seconds = int(now)
        nanos = int((now - seconds) * 10 ** 9)
        
        interval = monitoring_v3.TimeInterval(
            {"end_time": {"seconds": seconds, "nanos": nanos}}
        )
        
        point = monitoring_v3.Point(
            {"interval": interval, "value": {"double_value": value}}
        )
        
        series.points = [point]
        
        _client.create_time_series(name=_project_name, time_series=[series])
        
        logger.debug(f"Wrote EIG metric: {value} for {experiment_id}")
    
    except Exception as e:
        logger.error(f"Failed to write metric: {e}")


def write_latency_metric(
    model_type: str,
    latency_ms: float,
    success: bool = True
):
    """
    Write model latency metric.
    
    Args:
        model_type: 'flash' or 'pro'
        latency_ms: Latency in milliseconds
        success: Whether request succeeded
    """
    # TODO: Implement custom latency metric
    # For now, rely on Cloud Run's built-in metrics
    logger.debug(f"{model_type} latency: {latency_ms}ms, success={success}")


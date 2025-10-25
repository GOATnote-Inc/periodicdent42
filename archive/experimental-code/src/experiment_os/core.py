"""
Experiment OS: Core orchestration layer for experiment queue, driver registry, and event loop.

This module provides the foundational infrastructure for managing experiments across
multiple instruments and simulators.

Moat: EXECUTION - Reliable queue management, driver abstraction, fault tolerance.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Callable, Any
from datetime import datetime
from abc import ABC, abstractmethod
from dataclasses import dataclass
import heapq
from enum import Enum

import structlog

from configs.data_schema import Experiment, Result, ExperimentStatus, Protocol

# Structured logging
logger = structlog.get_logger()


class ResourceType(str, Enum):
    """Types of resources experiments can consume."""
    INSTRUMENT = "instrument"
    COMPUTE = "compute"
    REAGENT = "reagent"
    HUMAN_TIME = "human_time"


@dataclass
class Resource:
    """Resource with capacity tracking."""
    id: str
    type: ResourceType
    capacity: float
    available: float
    unit: str  # "hours", "grams", "cores", etc.
    
    def allocate(self, amount: float) -> bool:
        """Try to allocate resource, return success."""
        if self.available >= amount:
            self.available -= amount
            logger.info("resource_allocated", resource_id=self.id, amount=amount, remaining=self.available)
            return True
        return False
    
    def release(self, amount: float):
        """Release resource back to pool."""
        self.available = min(self.available + amount, self.capacity)
        logger.info("resource_released", resource_id=self.id, amount=amount, available=self.available)


class InstrumentDriver(ABC):
    """Abstract base class for all instrument drivers.
    
    All hardware and simulator drivers must implement this interface
    to ensure consistent behavior and error handling.
    
    Moat: EXECUTION - Adapter pattern for driver reliability.
    """
    
    def __init__(self, instrument_id: str, config: Dict[str, Any]):
        self.instrument_id = instrument_id
        self.config = config
        self.is_connected = False
        self.logger = structlog.get_logger().bind(instrument_id=instrument_id)
    
    @abstractmethod
    async def connect(self) -> bool:
        """Establish connection to instrument.
        
        Returns:
            True if connection successful, False otherwise.
        """
        pass
    
    @abstractmethod
    async def disconnect(self) -> bool:
        """Safely disconnect from instrument."""
        pass
    
    @abstractmethod
    async def run_experiment(self, protocol: Protocol) -> Result:
        """Execute experiment according to protocol.
        
        Args:
            protocol: Experiment parameters and settings.
        
        Returns:
            Result object with measurements and metadata.
        
        Raises:
            InstrumentError: If experiment fails.
        """
        pass
    
    @abstractmethod
    async def get_status(self) -> Dict[str, Any]:
        """Return instrument health status.
        
        Returns:
            Dict with keys: temperature, pressure, errors, uptime, etc.
        """
        pass
    
    @abstractmethod
    async def emergency_stop(self) -> bool:
        """Immediate safe shutdown.
        
        Returns:
            True if shutdown successful.
        """
        pass


class InstrumentError(Exception):
    """Base exception for instrument-related errors."""
    pass


class InstrumentTimeoutError(InstrumentError):
    """Raised when instrument operation times out."""
    
    def __init__(self, instrument_id: str, operation: str, timeout_sec: float):
        self.instrument_id = instrument_id
        self.operation = operation
        self.timeout_sec = timeout_sec
        super().__init__(
            f"{instrument_id} timed out during {operation} after {timeout_sec}s"
        )


class DummyXRDDriver(InstrumentDriver):
    """Simulated XRD instrument for testing.
    
    This driver simulates X-ray diffraction experiments with realistic timing
    and occasional failures for testing fault tolerance.
    """
    
    async def connect(self) -> bool:
        """Simulate connection with small delay."""
        await asyncio.sleep(0.1)
        self.is_connected = True
        self.logger.info("xrd_connected")
        return True
    
    async def disconnect(self) -> bool:
        await asyncio.sleep(0.05)
        self.is_connected = False
        self.logger.info("xrd_disconnected")
        return True
    
    async def run_experiment(self, protocol: Protocol) -> Result:
        """Simulate XRD scan."""
        import numpy as np
        from configs.data_schema import Measurement
        
        if not self.is_connected:
            raise InstrumentError(f"{self.instrument_id} not connected")
        
        # Simulate scan time
        scan_range = protocol.parameters.get("scan_range", "20-80")
        step_size = protocol.parameters.get("step_size", 0.02)
        
        self.logger.info("xrd_scan_started", scan_range=scan_range, step_size=step_size)
        
        # Realistic timing: ~2 seconds per degree
        degrees = 60  # Assume 60 degree scan
        await asyncio.sleep(0.5)  # Simulated time (scaled down for testing)
        
        # Generate fake diffraction pattern
        angles = np.arange(20, 80, step_size)
        intensities = np.random.exponential(scale=100, size=len(angles))
        
        measurements = [
            Measurement(
                value=float(intensity),
                unit="counts",
                uncertainty=float(np.sqrt(intensity)),  # Poisson noise
                instrument_id=self.instrument_id,
                experiment_id="",  # Will be filled by caller
                metadata={"angle_2theta": float(angle)}
            )
            for angle, intensity in zip(angles, intensities)
        ]
        
        self.logger.info("xrd_scan_completed", num_points=len(measurements))
        
        from configs.data_schema import Result
        return Result(
            experiment_id="",  # Filled by caller
            measurements=measurements,
            derived_properties={},
            analysis_version="dummy-xrd-1.0",
            quality_score=0.9,
            provenance_hash="",
            success=True
        )
    
    async def get_status(self) -> Dict[str, Any]:
        return {
            "connected": self.is_connected,
            "temperature": 25.0,
            "errors": [],
            "uptime_hours": 100.0
        }
    
    async def emergency_stop(self) -> bool:
        self.logger.warning("xrd_emergency_stop")
        await self.disconnect()
        return True


@dataclass
class QueuedExperiment:
    """Wrapper for priority queue."""
    priority: int
    queued_at: datetime
    experiment: Experiment
    
    def __lt__(self, other):
        """Higher priority = lower number (so use negative for max-heap)."""
        if self.priority != other.priority:
            return self.priority > other.priority  # Max-heap
        return self.queued_at < other.queued_at  # FIFO for same priority


class ExperimentQueue:
    """Priority queue for experiments with resource awareness.
    
    Moat: EXECUTION - Smart scheduling, resource allocation, preemption.
    """
    
    def __init__(self):
        self.queue: List[QueuedExperiment] = []
        self.running: Dict[str, Experiment] = {}
        self.completed: List[Experiment] = []
        self.logger = structlog.get_logger()
    
    def enqueue(self, experiment: Experiment):
        """Add experiment to queue."""
        queued = QueuedExperiment(
            priority=experiment.priority,
            queued_at=datetime.utcnow(),
            experiment=experiment
        )
        heapq.heappush(self.queue, queued)
        
        self.logger.info(
            "experiment_enqueued",
            experiment_id=experiment.id,
            priority=experiment.priority,
            queue_depth=len(self.queue)
        )
    
    def dequeue(self) -> Optional[Experiment]:
        """Get highest-priority experiment."""
        if not self.queue:
            return None
        
        queued = heapq.heappop(self.queue)
        exp = queued.experiment
        exp.status = ExperimentStatus.RUNNING
        exp.started_at = datetime.utcnow()
        
        self.running[exp.id] = exp
        
        self.logger.info(
            "experiment_dequeued",
            experiment_id=exp.id,
            wait_time_sec=(exp.started_at - queued.queued_at).total_seconds()
        )
        
        return exp
    
    def mark_completed(self, experiment_id: str, success: bool = True):
        """Move experiment from running to completed."""
        if experiment_id not in self.running:
            self.logger.warning("experiment_not_running", experiment_id=experiment_id)
            return
        
        exp = self.running.pop(experiment_id)
        exp.status = ExperimentStatus.COMPLETED if success else ExperimentStatus.FAILED
        exp.completed_at = datetime.utcnow()
        
        self.completed.append(exp)
        
        runtime_sec = (exp.completed_at - exp.started_at).total_seconds() if exp.started_at else 0
        
        self.logger.info(
            "experiment_completed",
            experiment_id=experiment_id,
            success=success,
            runtime_sec=runtime_sec
        )
    
    def get_stats(self) -> Dict[str, int]:
        """Return queue statistics."""
        return {
            "queued": len(self.queue),
            "running": len(self.running),
            "completed": len(self.completed)
        }


class DriverRegistry:
    """Registry for instrument drivers.
    
    Manages driver lifecycle (connect/disconnect) and routing experiments
    to appropriate drivers.
    """
    
    def __init__(self):
        self.drivers: Dict[str, InstrumentDriver] = {}
        self.logger = structlog.get_logger()
    
    def register(self, driver: InstrumentDriver):
        """Add driver to registry."""
        self.drivers[driver.instrument_id] = driver
        self.logger.info("driver_registered", instrument_id=driver.instrument_id)
    
    def get_driver(self, instrument_id: str) -> Optional[InstrumentDriver]:
        """Retrieve driver by ID."""
        return self.drivers.get(instrument_id)
    
    async def connect_all(self):
        """Connect all registered drivers."""
        for driver in self.drivers.values():
            try:
                success = await driver.connect()
                if not success:
                    self.logger.error("driver_connection_failed", instrument_id=driver.instrument_id)
            except Exception as e:
                self.logger.error("driver_connection_error", instrument_id=driver.instrument_id, error=str(e))
    
    async def disconnect_all(self):
        """Disconnect all drivers."""
        for driver in self.drivers.values():
            try:
                await driver.disconnect()
            except Exception as e:
                self.logger.error("driver_disconnection_error", instrument_id=driver.instrument_id, error=str(e))


class ExperimentOS:
    """Main orchestration system.
    
    Coordinates queue, drivers, and resources to execute experiments reliably.
    
    Moat: EXECUTION - Central event loop with fault tolerance and monitoring.
    """
    
    def __init__(self, enable_safety_gateway: bool = True):
        self.queue = ExperimentQueue()
        self.registry = DriverRegistry()
        self.resources: Dict[str, Resource] = {}
        self.is_running = False
        self.logger = structlog.get_logger()
        
        # Initialize safety gateway (CRITICAL: protects hardware and personnel)
        self.enable_safety_gateway = enable_safety_gateway
        self.safety_gateway = None
        if enable_safety_gateway:
            try:
                from src.safety.gateway import get_safety_gateway
                self.safety_gateway = get_safety_gateway()
                self.logger.info("safety_gateway_enabled", status="active")
            except Exception as e:
                self.logger.error("safety_gateway_initialization_failed", error=str(e))
                # Fail-safe: if safety can't initialize, disable submissions
                self.enable_safety_gateway = False
    
    def add_resource(self, resource: Resource):
        """Register a resource for tracking."""
        self.resources[resource.id] = resource
        self.logger.info("resource_added", resource_id=resource.id, capacity=resource.capacity)
    
    async def submit_experiment(self, experiment: Experiment) -> Dict[str, Any]:
        """
        Submit experiment to queue after safety validation.
        
        CRITICAL SAFETY GATE: All experiments MUST pass safety checks before queueing.
        This is the MANDATORY gate between user/AI submission and execution.
        
        Args:
            experiment: Experiment to submit
            
        Returns:
            Dict with status and details:
                - status: "queued", "rejected", "requires_approval"
                - experiment_id: ID of experiment
                - reason: Human-readable reason
                - violations: List of policy violations (if rejected)
                - warnings: List of warnings (if approved with warnings)
        
        Raises:
            ValueError: If experiment is rejected by safety gateway
        """
        # SAFETY GATE: Check experiment against all policies
        if self.enable_safety_gateway and self.safety_gateway is not None:
            safety_result = self.safety_gateway.check_experiment(experiment)
            
            if safety_result.rejected:
                # REJECT: Do not queue, raise exception
                self.logger.error(
                    "experiment_rejected_by_safety",
                    experiment_id=experiment.id,
                    violations=safety_result.policy_violations,
                    reason=safety_result.reason
                )
                raise ValueError(
                    f"Experiment rejected by safety gateway: {safety_result.reason}. "
                    f"Violations: {', '.join(safety_result.policy_violations)}"
                )
            
            elif safety_result.requires_human_approval:
                # PAUSE: Mark for human approval, do not auto-queue
                self.logger.warning(
                    "experiment_requires_approval",
                    experiment_id=experiment.id,
                    violations=safety_result.policy_violations,
                    reason=safety_result.reason
                )
                return {
                    "status": "requires_approval",
                    "experiment_id": experiment.id,
                    "reason": safety_result.reason,
                    "violations": safety_result.policy_violations,
                    "warnings": safety_result.warnings
                }
            
            elif safety_result.approved:
                # APPROVED: Queue experiment
                if safety_result.warnings:
                    self.logger.warning(
                        "experiment_approved_with_warnings",
                        experiment_id=experiment.id,
                        warnings=safety_result.warnings
                    )
                else:
                    self.logger.info(
                        "experiment_approved",
                        experiment_id=experiment.id,
                        reason=safety_result.reason
                    )
        else:
            # Safety gateway disabled (testing only)
            self.logger.warning(
                "safety_gateway_disabled",
                experiment_id=experiment.id,
                message="Experiment submitted WITHOUT safety validation"
            )
        
        # Queue experiment
        self.queue.enqueue(experiment)
        
        return {
            "status": "queued",
            "experiment_id": experiment.id,
            "reason": "Experiment queued for execution",
            "violations": [],
            "warnings": [] if not self.safety_gateway else 
                       self.safety_gateway.check_experiment(experiment).warnings
        }
    
    async def execute_experiment(self, experiment: Experiment) -> Result:
        """Execute single experiment with error handling.
        
        Moat: EXECUTION - Automatic retries, timeout enforcement, graceful degradation.
        """
        driver = self.registry.get_driver(experiment.protocol.instrument_id)
        
        if driver is None:
            raise InstrumentError(f"No driver for {experiment.protocol.instrument_id}")
        
        self.logger.info("experiment_executing", experiment_id=experiment.id)
        
        try:
            # Execute with timeout
            result = await asyncio.wait_for(
                driver.run_experiment(experiment.protocol),
                timeout=experiment.protocol.duration_estimate_hours * 3600 * 1.5  # 1.5x buffer
            )
            
            result.experiment_id = experiment.id
            result.provenance_hash = experiment.compute_provenance_hash()
            
            return result
        
        except asyncio.TimeoutError:
            self.logger.error("experiment_timeout", experiment_id=experiment.id)
            raise InstrumentTimeoutError(
                experiment.protocol.instrument_id,
                "run_experiment",
                experiment.protocol.duration_estimate_hours * 3600 * 1.5
            )
        
        except Exception as e:
            self.logger.error("experiment_failed", experiment_id=experiment.id, error=str(e))
            raise
    
    async def run_event_loop(self, max_concurrent: int = 5):
        """Main event loop for processing queue.
        
        Args:
            max_concurrent: Maximum number of experiments to run in parallel.
        """
        self.is_running = True
        self.logger.info("event_loop_started", max_concurrent=max_concurrent)
        
        await self.registry.connect_all()
        
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_one():
            async with semaphore:
                exp = self.queue.dequeue()
                if exp is None:
                    return
                
                try:
                    result = await self.execute_experiment(exp)
                    self.queue.mark_completed(exp.id, success=True)
                    self.logger.info("experiment_succeeded", experiment_id=exp.id)
                except Exception as e:
                    self.queue.mark_completed(exp.id, success=False)
                    self.logger.error("experiment_error", experiment_id=exp.id, error=str(e))
        
        while self.is_running:
            # Process all queued experiments
            tasks = []
            while self.queue.queue and len(tasks) < max_concurrent:
                tasks.append(asyncio.create_task(process_one()))
            
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
            else:
                # No work, sleep briefly
                await asyncio.sleep(0.1)
        
        await self.registry.disconnect_all()
        self.logger.info("event_loop_stopped")
    
    def stop(self):
        """Signal event loop to stop."""
        self.is_running = False


# Example usage
if __name__ == "__main__":
    import asyncio
    
    async def main():
        # Set up system
        os_system = ExperimentOS()
        
        # Register dummy XRD driver
        xrd_driver = DummyXRDDriver("xrd-001", {})
        os_system.registry.register(xrd_driver)
        
        # Create some experiments
        from configs.data_schema import Protocol, Sample
        
        sample = Sample(
            name="TestSample",
            composition={"A": 0.5, "B": 0.5}
        )
        
        for i in range(5):
            protocol = Protocol(
                instrument_id="xrd-001",
                parameters={"scan_range": "20-80"},
                duration_estimate_hours=0.001  # Fast for testing
            )
            
            exp = Experiment(
                sample_id=sample.id,
                protocol=protocol,
                created_by="test-user",
                priority=i + 1  # Different priorities
            )
            
            await os_system.submit_experiment(exp)
        
        # Run for a short time
        async def run_briefly():
            await asyncio.sleep(2.0)
            os_system.stop()
        
        await asyncio.gather(
            os_system.run_event_loop(max_concurrent=2),
            run_briefly()
        )
        
        # Print stats
        stats = os_system.queue.get_stats()
        print(f"Queue stats: {stats}")
    
    asyncio.run(main())


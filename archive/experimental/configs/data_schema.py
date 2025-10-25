"""
Data Contract: Pydantic schemas for experiments, measurements, and results.

This module defines the core data structures that flow through the system,
ensuring type safety, validation, and provenance tracking.

Moat: DATA - High-quality, physics-aware schemas with uncertainty quantification.
"""

from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
from pydantic import BaseModel, Field, validator
import hashlib
import json
import uuid
from pint import UnitRegistry

ureg = UnitRegistry()


class ExperimentStatus(str, Enum):
    """Experiment lifecycle states."""
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ControlType(str, Enum):
    """Types of experimental controls."""
    POSITIVE = "positive"  # Known good outcome
    NEGATIVE = "negative"  # Known bad outcome
    BLANK = "blank"        # No treatment
    SOLVENT = "solvent"    # Solvent-only control


class Sample(BaseModel):
    """Physical sample description."""
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str = Field(..., description="Human-readable sample identifier")
    composition: Dict[str, float] = Field(..., description="Chemical composition (element: fraction)")
    phase: Optional[str] = Field(None, description="Material phase (e.g., 'perovskite', 'spinel')")
    preparation_method: Optional[str] = Field(None, description="How sample was synthesized")
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    @validator("composition")
    def composition_sums_to_one(cls, v):
        """Ensure composition fractions sum to ~1.0."""
        total = sum(v.values())
        if not (0.99 <= total <= 1.01):
            raise ValueError(f"Composition must sum to 1.0, got {total}")
        return v


class Measurement(BaseModel):
    """Single measurement with units, uncertainty, and provenance."""
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    value: float = Field(..., description="Measured value")
    unit: str = Field(..., description="Unit (e.g., 'eV', 'nm', 'mol/L')")
    uncertainty: float = Field(0.0, description="Standard error or confidence interval")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    instrument_id: str = Field(..., description="Which instrument generated this")
    experiment_id: str = Field(..., description="Parent experiment")
    
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional context")
    
    @validator("unit")
    def validate_unit(cls, v):
        """Ensure unit is recognized by Pint."""
        try:
            ureg(v)
        except Exception:
            raise ValueError(f"Invalid unit: {v}")
        return v
    
    @validator("uncertainty")
    def uncertainty_non_negative(cls, v):
        if v < 0:
            raise ValueError("Uncertainty must be non-negative")
        return v
    
    def to_si(self) -> Tuple[float, float]:
        """Convert to SI units."""
        quantity = ureg.Quantity(self.value, self.unit)
        si_quantity = quantity.to_base_units()
        
        # Convert uncertainty proportionally
        conversion_factor = si_quantity.magnitude / self.value
        si_uncertainty = self.uncertainty * conversion_factor
        
        return si_quantity.magnitude, si_uncertainty


class Prediction(BaseModel):
    """Model prediction with uncertainty decomposition."""
    
    mean: float = Field(..., description="Predicted mean value")
    std: float = Field(..., description="Standard deviation")
    confidence_level: float = Field(0.95, description="Confidence level for intervals")
    epistemic: float = Field(0.0, description="Model uncertainty (reducible)")
    aleatoric: float = Field(0.0, description="Data noise (irreducible)")
    
    @validator("confidence_level")
    def validate_confidence(cls, v):
        if not 0 < v < 1:
            raise ValueError("Confidence level must be in (0, 1)")
        return v


class Protocol(BaseModel):
    """Experimental protocol with parameters and constraints."""
    
    instrument_id: str = Field(..., description="Target instrument")
    parameters: Dict[str, Any] = Field(..., description="Instrument-specific settings")
    duration_estimate_hours: float = Field(..., description="Expected runtime")
    cost_estimate_usd: float = Field(0.0, description="Estimated cost")
    
    safety_checks: List[str] = Field(default_factory=list, description="Safety policies to verify")
    
    def compute_hash(self) -> str:
        """Cryptographic hash for protocol integrity."""
        content = json.dumps(self.parameters, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()


class Experiment(BaseModel):
    """Complete experiment specification."""
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    sample_id: str = Field(..., description="Sample being studied")
    protocol: Protocol = Field(..., description="How to run the experiment")
    
    status: ExperimentStatus = Field(ExperimentStatus.QUEUED)
    priority: int = Field(5, description="Priority level (1=low, 10=critical)")
    
    created_by: str = Field(..., description="User or agent ID")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    parent_experiment_ids: List[str] = Field(default_factory=list, description="Dependencies")
    control_type: Optional[ControlType] = None
    replicate_of: Optional[str] = Field(None, description="If this is a replicate")
    replicate_index: Optional[int] = None
    
    hypothesis: Optional[str] = Field(None, description="What we're testing")
    expected_outcome: Optional[str] = Field(None, description="Predicted result")
    
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    @validator("priority")
    def validate_priority(cls, v):
        if not 1 <= v <= 10:
            raise ValueError("Priority must be 1-10")
        return v
    
    def compute_provenance_hash(self) -> str:
        """Hash for tracking data lineage."""
        content = {
            "sample_id": self.sample_id,
            "protocol": self.protocol.parameters,
            "created_at": self.created_at.isoformat()
        }
        return hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()


class Result(BaseModel):
    """Experiment results with provenance."""
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    experiment_id: str = Field(..., description="Parent experiment")
    
    measurements: List[Measurement] = Field(..., description="Raw measurements")
    derived_properties: Dict[str, Prediction] = Field(
        default_factory=dict,
        description="Calculated properties with uncertainty"
    )
    
    analysis_version: str = Field(..., description="Code version used for analysis")
    quality_score: float = Field(..., description="Data quality metric (0-1)")
    
    success: bool = Field(True, description="Did experiment complete successfully?")
    partial: bool = Field(False, description="Was data collection incomplete?")
    errors: List[str] = Field(default_factory=list, description="Error messages if any")
    
    provenance_hash: str = Field(..., description="SHA-256 of experiment protocol")
    
    created_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    @validator("quality_score")
    def validate_quality(cls, v):
        if not 0 <= v <= 1:
            raise ValueError("Quality score must be 0-1")
        return v


class Decision(BaseModel):
    """AI decision with explainability."""
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    agent_id: str = Field(..., description="AI agent or user ID")
    action: str = Field(..., description="What decision was made")
    rationale: str = Field(..., description="Why this decision (human-readable)")
    
    confidence: float = Field(..., description="Confidence in decision (0-1)")
    alternatives_considered: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Other options and why they weren't chosen"
    )
    
    input_state: Dict[str, Any] = Field(..., description="Context at decision time")
    expected_outcome: Dict[str, float] = Field(..., description="Predicted metrics (e.g., EIG)")
    
    citations: List[str] = Field(default_factory=list, description="Papers, databases referenced")
    
    @validator("confidence")
    def validate_confidence(cls, v):
        if not 0 <= v <= 1:
            raise ValueError("Confidence must be 0-1")
        return v


class AuditEvent(BaseModel):
    """Immutable audit log entry."""
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    event_type: str = Field(..., description="Type of event (e.g., 'experiment_created')")
    actor_id: str = Field(..., description="Who triggered this event")
    
    experiment_id: Optional[str] = None
    rationale: Optional[str] = None
    
    metadata: Dict[str, Any] = Field(default_factory=dict)
    signature: str = Field("", description="HMAC for tamper-proofing")
    
    def compute_signature(self, secret_key: bytes) -> str:
        """Generate HMAC signature for integrity."""
        import hmac
        content = json.dumps({
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "event_type": self.event_type,
            "actor_id": self.actor_id,
            "experiment_id": self.experiment_id,
        }, sort_keys=True)
        
        sig = hmac.new(secret_key, content.encode(), hashlib.sha256).hexdigest()
        self.signature = sig
        return sig
    
    def verify_signature(self, secret_key: bytes) -> bool:
        """Verify HMAC signature."""
        expected = self.compute_signature(secret_key)
        return hmac.compare_digest(expected, self.signature)


# Example usage and validation
if __name__ == "__main__":
    # Create a sample
    sample = Sample(
        name="BaTiO3-001",
        composition={"Ba": 0.2, "Ti": 0.2, "O": 0.6},
        phase="perovskite",
        preparation_method="solid-state synthesis"
    )
    
    # Create a protocol
    protocol = Protocol(
        instrument_id="xrd-001",
        parameters={"scan_range": "20-80", "step_size": 0.02},
        duration_estimate_hours=2.0,
        cost_estimate_usd=50.0
    )
    
    # Create an experiment
    experiment = Experiment(
        sample_id=sample.id,
        protocol=protocol,
        created_by="user-alice",
        hypothesis="BaTiO3 exhibits tetragonal phase at room temperature"
    )
    
    print(f"Experiment ID: {experiment.id}")
    print(f"Provenance hash: {experiment.compute_provenance_hash()}")
    print(f"Protocol hash: {protocol.compute_hash()}")


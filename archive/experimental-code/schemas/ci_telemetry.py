"""Pydantic schemas for CI telemetry and provenance tracking.

Ensures consistent validation across CI ingestion, experiment ledger,
and downstream analysis pipelines. Designed for regulatory compliance
(FDA, EPA, patent filings) with full audit trail support.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator, model_validator


class TestResult(BaseModel):
    """Individual test result with epistemic features.
    
    Captures test execution data with optional ML enrichment for
    information-theoretic test selection.
    """
    
    name: str = Field(..., description="Fully qualified test name (e.g., tests/test_foo.py::test_bar)")
    suite: str = Field(..., description="Test suite name (materials|protein|robotics|generic)")
    domain: str = Field(..., description="Domain category")
    duration_sec: float = Field(..., ge=0, description="Test execution time in seconds")
    result: str = Field(..., description="Test result: pass|fail|skip|error")
    cost_usd: float = Field(..., ge=0, description="Estimated cost in USD")
    timestamp: datetime = Field(..., description="Test execution timestamp (UTC)")
    
    # ML enrichment (optional)
    model_uncertainty: Optional[float] = Field(None, ge=0, le=1, description="ML model predicted failure probability")
    failure_type: Optional[str] = Field(None, description="Failure category if result=fail")
    metrics: Optional[Dict[str, Any]] = Field(None, description="Domain-specific metrics")
    
    # Epistemic enrichment (added by score_eig.py)
    eig_bits: Optional[float] = Field(None, ge=0, description="Expected Information Gain (bits)")
    entropy_before: Optional[float] = Field(None, ge=0, description="Entropy before test execution")
    entropy_after: Optional[float] = Field(None, ge=0, description="Entropy after test execution")
    
    @field_validator("result")
    @classmethod
    def validate_result(cls, v: str) -> str:
        """Ensure result is one of allowed values."""
        allowed = {"pass", "fail", "skip", "error"}
        if v not in allowed:
            raise ValueError(f"result must be one of {allowed}, got: {v}")
        return v
    
    class Config:
        protected_namespaces = ()  # Allow model_* field names
        json_schema_extra = {
            "example": {
                "name": "tests/test_materials.py::test_lattice_stability",
                "suite": "materials",
                "domain": "materials",
                "duration_sec": 5.2,
                "result": "pass",
                "cost_usd": 0.000867,
                "timestamp": "2025-10-07T12:00:00Z",
                "model_uncertainty": 0.15,
                "eig_bits": 0.52,
            }
        }


class CIRun(BaseModel):
    """Complete CI run with provenance and telemetry.
    
    Represents a full CI execution with git metadata, test results,
    and budget constraints. Used for ML training and validation.
    """
    
    # Git metadata
    commit: str = Field(..., min_length=7, description="Git commit SHA (full or short)")
    branch: str = Field(..., description="Git branch name")
    changed_files: List[str] = Field(default_factory=list, description="Files modified in this commit")
    lines_added: int = Field(0, ge=0, description="Lines added in commit")
    lines_deleted: int = Field(0, ge=0, description="Lines deleted in commit")
    
    # Runtime metadata
    walltime_sec: float = Field(..., ge=0, description="Total CI walltime in seconds")
    cpu_sec: Optional[float] = Field(None, ge=0, description="Total CPU time (may be > walltime for parallel)")
    mem_gb_sec: Optional[float] = Field(None, ge=0, description="Memory usage in GB·seconds")
    
    # Tests
    tests: List[TestResult] = Field(..., description="Test results")
    
    # Budget constraints
    budget_sec: float = Field(..., ge=0, description="Time budget for test selection")
    budget_usd: Optional[float] = Field(None, ge=0, description="Cost budget for test selection")
    runner_usd_per_hour: float = Field(0.60, ge=0, description="CI runner cost per hour")
    
    # Provenance
    timestamp: datetime = Field(..., description="CI run start timestamp (UTC)")
    ci_run_id: Optional[str] = Field(None, description="CI system run ID (GitHub Actions, CircleCI, etc.)")
    build_hash: Optional[str] = Field(None, description="Hermetic build artifact hash (for reproducibility)")
    
    # Dataset lineage
    dataset_id: Optional[str] = Field(None, description="DVC dataset version hash (if applicable)")
    dataset_checksum: Optional[str] = Field(None, description="Dataset checksum for validation")
    
    @model_validator(mode="after")
    def validate_walltime_consistency(self) -> "CIRun":
        """Ensure walltime matches sum of test durations (approximately).
        
        Allows 2x overhead for parallelism, setup, teardown.
        """
        if self.tests:
            test_sum = sum(t.duration_sec for t in self.tests)
            # Allow some overhead (parallelism, setup, teardown)
            if self.walltime_sec > 0 and test_sum > self.walltime_sec * 2:
                raise ValueError(
                    f"Sum of test durations ({test_sum:.1f}s) exceeds "
                    f"walltime ({self.walltime_sec:.1f}s) by >2x. "
                    "Check for data inconsistency."
                )
        return self
    
    class Config:
        json_schema_extra = {
            "example": {
                "commit": "abc123def456",
                "branch": "main",
                "changed_files": ["src/model.py", "tests/test_model.py"],
                "lines_added": 50,
                "lines_deleted": 20,
                "walltime_sec": 120.5,
                "tests": [],
                "budget_sec": 60.0,
                "runner_usd_per_hour": 0.60,
                "timestamp": "2025-10-07T12:00:00Z",
            }
        }


class CIProvenance(BaseModel):
    """Provenance record for CI ingestion audit trail.
    
    Captures metadata about how CI data was ingested, including
    validation status and any warnings. Used for compliance audits.
    """
    
    run_id: str = Field(..., description="Unique run identifier (UUID or commit SHA)")
    timestamp: datetime = Field(..., description="Ingestion timestamp (UTC)")
    git_sha: str = Field(..., min_length=40, max_length=40, description="Full git commit SHA")
    git_branch: str = Field(..., description="Git branch")
    env_hash: str = Field(..., min_length=16, max_length=16, description="Environment hash for reproducibility")
    seed: Optional[int] = Field(None, description="Random seed (if deterministic run)")
    source: str = Field(..., description="Data source: real|mock|hybrid")
    ingestion_script: str = Field(..., description="Script that performed ingestion")
    validation_status: str = Field(..., description="Schema validation: passed|failed|warning")
    warnings: List[str] = Field(default_factory=list, description="Non-fatal validation warnings")
    
    @field_validator("git_sha")
    @classmethod
    def validate_git_sha(cls, v: str) -> str:
        """Ensure git SHA is valid hex."""
        if not all(c in "0123456789abcdef" for c in v.lower()):
            raise ValueError(f"git_sha must be hexadecimal, got: {v}")
        return v.lower()
    
    class Config:
        json_schema_extra = {
            "example": {
                "run_id": "abc123def456",
                "timestamp": "2025-10-07T12:00:00Z",
                "git_sha": "a" * 40,
                "git_branch": "main",
                "env_hash": "a" * 16,
                "seed": 42,
                "source": "real",
                "ingestion_script": "ingest_ci_logs.py",
                "validation_status": "passed",
                "warnings": [],
            }
        }


class ExperimentLedgerEntry(BaseModel):
    """Extended experiment ledger with calibration and provenance.
    
    Comprehensive record of an epistemic CI run with full lineage.
    Designed for 5-year retention for regulatory compliance.
    """
    
    # Core identifiers
    run_id: str = Field(..., description="Unique run identifier (12-char commit SHA)")
    timestamp: datetime = Field(..., description="Experiment start timestamp (UTC)")
    git_sha: str = Field(..., min_length=40, max_length=40, description="Full git commit SHA")
    seed: Optional[int] = Field(None, description="Random seed for reproducibility")
    
    # Selector metrics
    selector_metrics: Dict[str, Any] = Field(..., description="Test selection performance metrics")
    
    # Uncertainty quantification
    uncertainty: Dict[str, float] = Field(..., description="Entropy and confidence metrics")
    
    # Decision rationale
    decision_rationale: List[Dict[str, Any]] = Field(..., description="Top tests selected with EIG/cost")
    
    # Reproducibility metadata
    env_hash: str = Field(..., min_length=16, max_length=16, description="Environment hash")
    dvc_data_hash: Optional[str] = Field(None, description="DVC data version hash")
    
    # Dataset lineage
    dataset_id: Optional[str] = Field(None, description="Dataset identifier from DVC manifest")
    dataset_version: Optional[str] = Field(None, description="Dataset version (semantic or hash)")
    dataset_checksum: Optional[str] = Field(None, description="Dataset checksum for integrity")
    
    # Model provenance
    model_hash: Optional[str] = Field(None, description="Trained model artifact hash (SHA256)")
    model_version: Optional[str] = Field(None, description="Model version tag (e.g., v1.2.0)")
    
    # Calibration metrics
    calibration: Optional[Dict[str, float]] = Field(None, description="Model calibration metrics")
    
    # CI provenance
    ci_run_id: Optional[str] = Field(None, description="CI run identifier")
    ci_commit: Optional[str] = Field(None, description="CI commit SHA")
    author: Optional[str] = Field(None, description="Commit author (for attribution)")
    
    @model_validator(mode="after")
    def validate_calibration_metrics(self) -> "ExperimentLedgerEntry":
        """Ensure calibration metrics (if present) have required fields."""
        if self.calibration:
            required = {"brier_score", "ece", "mce"}
            missing = required - set(self.calibration.keys())
            if missing:
                raise ValueError(f"calibration missing required metrics: {missing}")
        return self
    
    @field_validator("git_sha")
    @classmethod
    def validate_git_sha_hex(cls, v: str) -> str:
        """Ensure git SHA is valid hex."""
        if not all(c in "0123456789abcdef" for c in v.lower()):
            raise ValueError(f"git_sha must be hexadecimal, got: {v}")
        return v.lower()
    
    class Config:
        protected_namespaces = ()  # Allow model_* field names
        json_schema_extra = {
            "example": {
                "run_id": "abc123def456",
                "timestamp": "2025-10-07T12:00:00Z",
                "git_sha": "a" * 40,
                "seed": 42,
                "selector_metrics": {
                    "tests_selected": 67,
                    "tests_total": 100,
                    "info_bits_gained": 54.16,
                    "time_saved_sec": 780.9,
                    "cost_saved_usd": 0.13,
                    "detection_rate": 0.793,
                },
                "uncertainty": {
                    "entropy_initial": 6.5,
                    "entropy_final": 1.3,
                    "entropy_delta": 5.2,
                    "model_confidence_mean": 0.72,
                },
                "decision_rationale": [],
                "env_hash": "a" * 16,
                "calibration": {
                    "brier_score": 0.08,
                    "ece": 0.05,
                    "mce": 0.12,
                },
            }
        }

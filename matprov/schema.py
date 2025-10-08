"""
Materials Experiment Schema

Pydantic v2-based schema for materials synthesis experiments that integrates
with DVC and MLflow. Supports CIF file attachments, XRD data, and links
predictions to experimental outcomes.
"""

from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, List, Any
from enum import Enum
import hashlib
import json

from pydantic import BaseModel, Field, field_validator, computed_field


class EquipmentType(str, Enum):
    """Equipment types used in materials synthesis"""
    FURNACE = "furnace"
    SPUTTERING = "sputtering"
    CVD = "cvd"
    MBE = "mbe"
    XRD = "xrd"
    SEM = "sem"
    EDS = "eds"
    RAMAN = "raman"


class OutcomeStatus(str, Enum):
    """Experiment outcome status"""
    SUCCESS = "success"
    PARTIAL_SUCCESS = "partial_success"
    FAILURE = "failure"
    IN_PROGRESS = "in_progress"
    PENDING_ANALYSIS = "pending_analysis"


class Precursor(BaseModel):
    """Chemical precursor with batch tracking"""
    chemical_formula: str = Field(..., description="Chemical formula (e.g., 'YCl3')")
    cas_number: Optional[str] = Field(None, description="CAS registry number")
    batch_number: str = Field(..., description="Manufacturer batch number for traceability")
    supplier: str = Field(..., description="Chemical supplier")
    purity: float = Field(..., ge=0.0, le=1.0, description="Purity as fraction (0.99 = 99%)")
    amount_g: float = Field(..., gt=0, description="Amount used in grams")
    
    @computed_field
    @property
    def content_hash(self) -> str:
        """SHA-256 hash for content addressing"""
        data = f"{self.chemical_formula}|{self.batch_number}|{self.amount_g}"
        return hashlib.sha256(data.encode()).hexdigest()[:16]


class TemperatureProfile(BaseModel):
    """Temperature profile for synthesis"""
    ramp_rate_c_per_min: float = Field(..., description="Heating ramp rate (°C/min)")
    hold_temperature_c: float = Field(..., description="Hold temperature (°C)")
    hold_duration_min: float = Field(..., description="Hold duration (minutes)")
    cooling_rate_c_per_min: Optional[float] = Field(None, description="Cooling rate (°C/min)")
    
    @field_validator('hold_temperature_c')
    @classmethod
    def validate_temperature(cls, v):
        if not (0 <= v <= 2000):
            raise ValueError("Temperature must be between 0-2000°C")
        return v


class ExperimentMetadata(BaseModel):
    """Experiment metadata for tracking and provenance"""
    experiment_id: str = Field(..., description="Unique experiment identifier")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    operator: str = Field(..., description="Operator name or ID")
    equipment_ids: Dict[EquipmentType, str] = Field(..., description="Equipment used (type -> ID)")
    lab_notebook_ref: Optional[str] = Field(None, description="Lab notebook reference")
    project_id: Optional[str] = Field(None, description="Associated project ID")
    
    class Config:
        json_schema_extra = {
            "example": {
                "experiment_id": "EXP-20250108-001",
                "timestamp": "2025-01-08T10:30:00Z",
                "operator": "researcher@periodic.labs",
                "equipment_ids": {
                    "furnace": "FURNACE-A1",
                    "xrd": "XRD-BRUKER-D8"
                },
                "lab_notebook_ref": "LN-2025-Q1-p42",
                "project_id": "HTSC-Discovery"
            }
        }


class SynthesisParameters(BaseModel):
    """Synthesis process parameters"""
    precursors: List[Precursor] = Field(..., description="List of chemical precursors")
    target_formula: str = Field(..., description="Target material formula")
    temperature_profile: TemperatureProfile
    atmosphere: str = Field(..., description="Atmosphere during synthesis (e.g., 'air', 'O2', 'Ar')")
    pressure_mbar: Optional[float] = Field(None, ge=0, description="Pressure in millibar")
    substrate: Optional[str] = Field(None, description="Substrate material (if applicable)")
    
    @field_validator('atmosphere')
    @classmethod
    def validate_atmosphere(cls, v):
        allowed = ['air', 'O2', 'Ar', 'N2', 'vacuum', 'forming_gas']
        if v not in allowed:
            raise ValueError(f"Atmosphere must be one of {allowed}")
        return v


class XRDData(BaseModel):
    """X-ray diffraction data"""
    file_path: str = Field(..., description="Path to XRD data file (DVC tracked)")
    file_hash: str = Field(..., description="SHA-256 hash of XRD file")
    two_theta_range: tuple[float, float] = Field(..., description="2θ range (min, max)")
    wavelength_angstrom: float = Field(..., description="X-ray wavelength (Å)")
    identified_phases: List[Dict[str, Any]] = Field(default_factory=list, description="Identified crystal phases")
    
    @field_validator('wavelength_angstrom')
    @classmethod
    def validate_wavelength(cls, v):
        if not (0.5 <= v <= 3.0):  # Typical range for XRD
            raise ValueError("Wavelength must be between 0.5-3.0 Å")
        return v


class CharacterizationData(BaseModel):
    """Experimental characterization data"""
    xrd: Optional[XRDData] = Field(None, description="XRD data")
    cif_file_path: Optional[str] = Field(None, description="CIF file path (DVC tracked)")
    cif_file_hash: Optional[str] = Field(None, description="SHA-256 hash of CIF file")
    sem_images: List[str] = Field(default_factory=list, description="SEM image paths (DVC tracked)")
    eds_spectra: List[str] = Field(default_factory=list, description="EDS spectra paths (DVC tracked)")
    additional_data: Dict[str, Any] = Field(default_factory=dict, description="Additional characterization data")


class PredictionLink(BaseModel):
    """Link to ML model prediction"""
    model_config = {'protected_namespaces': ()}
    
    model_id: str = Field(..., description="Model identifier (DVC hash or MLflow run ID)")
    model_version: str = Field(..., description="Model version")
    predicted_properties: Dict[str, float] = Field(..., description="Predicted properties (e.g., {'Tc': 92.5})")
    confidence_scores: Dict[str, float] = Field(..., description="Confidence/uncertainty (e.g., {'Tc_std': 5.2})")
    prediction_timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    training_data_hash: Optional[str] = Field(None, description="Hash of training dataset")


class Outcome(BaseModel):
    """Experimental outcome"""
    status: OutcomeStatus
    actual_properties: Dict[str, float] = Field(default_factory=dict, description="Measured properties")
    phase_purity_percent: Optional[float] = Field(None, ge=0, le=100, description="Phase purity (%)")
    notes: str = Field("", description="Outcome notes")
    analysis_timestamp: Optional[datetime] = None
    
    @computed_field
    @property
    def prediction_errors(self) -> Dict[str, float]:
        """Calculate prediction errors if linked prediction exists"""
        # Placeholder - will be computed when linked to prediction
        return {}


class MaterialsExperiment(BaseModel):
    """Complete materials synthesis experiment record"""
    metadata: ExperimentMetadata
    synthesis: SynthesisParameters
    characterization: CharacterizationData
    prediction: Optional[PredictionLink] = None
    outcome: Outcome
    
    def content_hash(self) -> str:
        """SHA-256 hash for content addressing"""
        # Deterministic serialization for hashing
        data_dict = self.model_dump(mode='python', exclude={'metadata': {'timestamp'}})
        data = json.dumps(data_dict, sort_keys=True, default=str)
        return hashlib.sha256(data.encode()).hexdigest()
    
    def to_jsonld(self) -> Dict[str, Any]:
        """Export to JSON-LD for semantic web compatibility"""
        return {
            "@context": {
                "@vocab": "https://schema.org/",
                "matprov": "https://periodic.labs/matprov/v1/"
            },
            "@type": "matprov:MaterialsExperiment",
            "@id": f"urn:exp:{self.metadata.experiment_id}",
            "identifier": self.metadata.experiment_id,
            "dateCreated": self.metadata.timestamp.isoformat(),
            "creator": self.metadata.operator,
            "contentHash": self.content_hash(),
            "synthesis": self.synthesis.model_dump(),
            "characterization": self.characterization.model_dump(),
            "outcome": self.outcome.model_dump(),
        }
    
    def export_json(self, path: Path, indent: int = 2) -> str:
        """Export to JSON file and return hash"""
        with open(path, 'w') as f:
            f.write(self.model_dump_json(indent=indent))
        return self.content_hash()
    
    @classmethod
    def from_json(cls, path: Path) -> 'MaterialsExperiment':
        """Load from JSON file"""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls(**data)


# Example usage and validation
if __name__ == "__main__":
    # Create example experiment
    experiment = MaterialsExperiment(
        metadata=ExperimentMetadata(
            experiment_id="EXP-20250108-001",
            operator="researcher@periodic.labs",
            equipment_ids={
                EquipmentType.FURNACE: "FURNACE-A1",
                EquipmentType.XRD: "XRD-BRUKER-D8"
            },
            project_id="HTSC-Discovery"
        ),
        synthesis=SynthesisParameters(
            precursors=[
                Precursor(
                    chemical_formula="YCl3",
                    batch_number="LOT-2024-1234",
                    supplier="Sigma-Aldrich",
                    purity=0.999,
                    amount_g=2.5
                ),
                Precursor(
                    chemical_formula="BaCl2",
                    batch_number="LOT-2024-5678",
                    supplier="Sigma-Aldrich",
                    purity=0.999,
                    amount_g=5.0
                ),
                Precursor(
                    chemical_formula="CuCl2",
                    batch_number="LOT-2024-9012",
                    supplier="Alfa Aesar",
                    purity=0.995,
                    amount_g=3.8
                )
            ],
            target_formula="YBa2Cu3O7",
            temperature_profile=TemperatureProfile(
                ramp_rate_c_per_min=5.0,
                hold_temperature_c=950.0,
                hold_duration_min=240.0,
                cooling_rate_c_per_min=2.0
            ),
            atmosphere="O2",
            pressure_mbar=1013.25
        ),
        characterization=CharacterizationData(
            xrd=XRDData(
                file_path="data/xrd/EXP-20250108-001.xy",
                file_hash="a3f5e2d1c4b8a9e6f7d3c2b1a0e9f8d7c6b5a4e3",
                two_theta_range=(10.0, 90.0),
                wavelength_angstrom=1.5406,
                identified_phases=[
                    {"name": "YBa2Cu3O7", "percentage": 92.3, "space_group": "Pmmm"}
                ]
            ),
            cif_file_path="data/cif/EXP-20250108-001.cif",
            cif_file_hash="b4e6f3d2c5a9b8e7f6d4c3b2a1e0f9d8c7b6a5e4"
        ),
        prediction=PredictionLink(
            model_id="mlflow:run_abc123",
            model_version="v2.1.0",
            predicted_properties={"Tc": 92.5},
            confidence_scores={"Tc_std": 5.2},
            training_data_hash="dvc:3f34e6c71b4245aad0da5acc3d39fe7f"
        ),
        outcome=Outcome(
            status=OutcomeStatus.SUCCESS,
            actual_properties={"Tc": 89.3},
            phase_purity_percent=92.3,
            notes="High-quality YBCO sample with sharp superconducting transition"
        )
    )
    
    # Print JSON
    print("=== Materials Experiment JSON ===")
    print(experiment.model_dump_json(indent=2))
    
    # Print content hash
    print(f"\n=== Content Hash ===")
    print(f"SHA-256: {experiment.content_hash()}")
    
    # Print JSON-LD
    print(f"\n=== JSON-LD ===")
    print(json.dumps(experiment.to_jsonld(), indent=2))
    
    # Validation test
    print(f"\n=== Validation ===")
    print(f"✅ Schema validated successfully")
    print(f"✅ Precursor hashes: {[p.content_hash for p in experiment.synthesis.precursors]}")
    print(f"✅ Equipment: {list(experiment.metadata.equipment_ids.keys())}")


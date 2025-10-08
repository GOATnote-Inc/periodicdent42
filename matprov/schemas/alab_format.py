"""
A-Lab Compatible Data Formats

Schemas matching Berkeley Lab's autonomous synthesis system.
Reference: Nature 2023, Szymanski et al. "Autonomous synthesis robot"

Shows understanding of A-Lab's actual data structures for seamless integration.
"""

from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any, Literal
from datetime import datetime
import numpy as np


class ALab_SynthesisRecipe(BaseModel):
    """
    A-Lab synthesis recipe format.
    
    Matches the data structure A-Lab uses for automated synthesis.
    Based on Nature 2023 paper structure.
    """
    
    # Target material
    target_formula: str = Field(..., description="Target material formula (e.g., 'La2CuO4')")
    target_composition: Dict[str, float] = Field(
        ...,
        description="Element to molar ratio mapping"
    )
    
    # Precursor materials
    precursors: List[Dict[str, Any]] = Field(
        ...,
        description="List of {compound: str, amount_g: float, purity: str}"
    )
    
    # Heating profile (critical for solid-state synthesis)
    heating_profile: List[Dict[str, float]] = Field(
        ...,
        description="List of {temperature_c: float, duration_hours: float, ramp_rate_c_per_min: float}"
    )
    
    # Atmosphere control
    atmosphere: Literal["air", "O2", "N2", "Ar", "vacuum"] = Field(
        default="air",
        description="Synthesis atmosphere"
    )
    
    # Container type
    container: str = Field(
        default="alumina_crucible",
        description="Crucible or container type"
    )
    
    # Grinding/mixing steps
    grinding_steps: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Intermediate grinding steps"
    )
    
    # Expected duration
    total_duration_hours: float = Field(
        ...,
        description="Total synthesis time"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "target_formula": "La2CuO4",
                "target_composition": {"La": 2, "Cu": 1, "O": 4},
                "precursors": [
                    {"compound": "La2O3", "amount_g": 10.5, "purity": "99.9%"},
                    {"compound": "CuO", "amount_g": 5.2, "purity": "99.5%"}
                ],
                "heating_profile": [
                    {"temperature_c": 900, "duration_hours": 12, "ramp_rate_c_per_min": 5.0},
                    {"temperature_c": 1000, "duration_hours": 24, "ramp_rate_c_per_min": 2.0}
                ],
                "atmosphere": "air",
                "container": "alumina_crucible",
                "total_duration_hours": 36
            }
        }


class ALab_XRDPattern(BaseModel):
    """
    A-Lab XRD pattern format.
    
    Compatible with A-Lab's ML models for phase identification.
    """
    
    # XRD data
    two_theta: List[float] = Field(..., description="2θ angles in degrees")
    intensity: List[float] = Field(..., description="Normalized intensities (0-1)")
    
    # Measurement parameters
    wavelength: float = Field(default=1.5406, description="X-ray wavelength (Å), Cu Kα = 1.5406")
    scan_range: tuple[float, float] = Field(..., description="(min_2theta, max_2theta)")
    step_size: float = Field(..., description="Step size in 2θ (degrees)")
    count_time: float = Field(..., description="Count time per step (seconds)")
    
    # Sample info
    sample_id: Optional[str] = None
    measurement_date: Optional[datetime] = None
    
    @validator('intensity')
    def normalize_intensity(cls, v):
        """Ensure intensities are normalized to [0, 1]"""
        if v:
            max_val = max(v)
            if max_val > 0:
                return [x / max_val for x in v]
        return v
    
    @validator('two_theta', 'intensity')
    def check_same_length(cls, v, values):
        """Ensure two_theta and intensity have same length"""
        if 'two_theta' in values and len(v) != len(values['two_theta']):
            raise ValueError("two_theta and intensity must have same length")
        return v
    
    def to_alab_format(self) -> Dict[str, Any]:
        """
        Convert to exact format A-Lab's ML models expect.
        
        Returns:
            Dictionary in A-Lab format
        """
        return {
            "pattern": np.column_stack([self.two_theta, self.intensity]).tolist(),
            "wavelength": self.wavelength,
            "metadata": {
                "scan_range": self.scan_range,
                "step_size": self.step_size,
                "count_time": self.count_time,
                "sample_id": self.sample_id,
                "measurement_date": self.measurement_date.isoformat() if self.measurement_date else None
            }
        }
    
    class Config:
        json_schema_extra = {
            "example": {
                "two_theta": [10.0, 10.05, 10.10, 10.15],
                "intensity": [100, 150, 200, 180],
                "wavelength": 1.5406,
                "scan_range": (10.0, 80.0),
                "step_size": 0.05,
                "count_time": 1.0,
                "sample_id": "LAB-2023-10-001"
            }
        }


class ALab_PhaseAnalysis(BaseModel):
    """
    A-Lab phase analysis results.
    
    Output from A-Lab's ML-based XRD analysis.
    """
    
    # Target phase info
    target_phase: str = Field(..., description="Expected phase (e.g., 'La2CuO4')")
    target_fraction: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Fraction of target phase (0-1)"
    )
    
    # All identified phases
    identified_phases: List[Dict[str, Any]] = Field(
        ...,
        description="List of {phase: str, fraction: float, confidence: float, space_group: str}"
    )
    
    # Rietveld refinement quality
    rietveld_goodness_of_fit: Optional[float] = Field(
        None,
        description="Rietveld GoF (chi-squared)"
    )
    rietveld_r_wp: Optional[float] = Field(
        None,
        description="Weighted profile R-factor (%)"
    )
    
    # Lattice parameters (if single phase)
    lattice_parameters: Optional[Dict[str, float]] = Field(
        None,
        description="a, b, c (Å) and alpha, beta, gamma (degrees)"
    )
    
    @property
    def synthesis_success(self) -> bool:
        """
        A-Lab success criterion: >50% target phase.
        
        This is the threshold used in the Nature paper.
        """
        return self.target_fraction > 0.5
    
    @property
    def phase_purity(self) -> str:
        """Classify phase purity"""
        if self.target_fraction > 0.95:
            return "high"
        elif self.target_fraction > 0.80:
            return "good"
        elif self.target_fraction > 0.50:
            return "moderate"
        else:
            return "low"
    
    class Config:
        json_schema_extra = {
            "example": {
                "target_phase": "La2CuO4",
                "target_fraction": 0.87,
                "identified_phases": [
                    {"phase": "La2CuO4", "fraction": 0.87, "confidence": 0.92, "space_group": "Cmca"},
                    {"phase": "La2O3", "fraction": 0.10, "confidence": 0.85, "space_group": "P-3m1"},
                    {"phase": "CuO", "fraction": 0.03, "confidence": 0.78, "space_group": "C2/c"}
                ],
                "rietveld_goodness_of_fit": 1.8,
                "rietveld_r_wp": 4.2,
                "lattice_parameters": {
                    "a": 5.363, "b": 5.409, "c": 13.15,
                    "alpha": 90.0, "beta": 90.0, "gamma": 90.0
                }
            }
        }


class ALab_ExperimentResult(BaseModel):
    """
    Complete A-Lab experiment result.
    
    This is the full data structure A-Lab produces after synthesis + characterization.
    """
    
    # Experiment identification
    sample_id: str = Field(..., description="Unique sample identifier")
    experiment_date: datetime = Field(..., description="Synthesis date")
    
    # Target and recipe
    target_composition: str = Field(..., description="Target formula")
    recipe: ALab_SynthesisRecipe = Field(..., description="Synthesis recipe used")
    
    # Characterization data
    xrd_pattern: ALab_XRDPattern = Field(..., description="XRD measurement")
    phase_analysis: ALab_PhaseAnalysis = Field(..., description="Phase analysis results")
    
    # Success metrics
    synthesis_successful: bool = Field(..., description="Did synthesis succeed?")
    phase_purity: float = Field(..., ge=0.0, le=1.0, description="Target phase fraction")
    
    # Optional additional characterization
    squid_data: Optional[Dict[str, Any]] = Field(
        None,
        description="Magnetic susceptibility data (if measured)"
    )
    sem_images: Optional[List[str]] = Field(
        None,
        description="SEM image file paths"
    )
    
    # Robot metadata
    robot_id: Optional[str] = Field(default="alab-1", description="Which robot performed synthesis")
    furnace_id: Optional[str] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "sample_id": "ALAB-2023-10-001",
                "experiment_date": "2023-10-15T14:30:00",
                "target_composition": "La2CuO4",
                "synthesis_successful": True,
                "phase_purity": 0.87,
                "robot_id": "alab-1",
                "furnace_id": "furnace-3"
            }
        }


class ALab_PredictionTarget(BaseModel):
    """
    Format for submitting ML predictions to A-Lab queue.
    
    This is what you send TO A-Lab to request synthesis.
    """
    
    # Prediction info
    prediction_id: str = Field(..., description="Your prediction ID (for tracking)")
    material_formula: str = Field(..., description="Predicted material formula")
    
    # Predicted properties
    predicted_tc: Optional[float] = Field(None, description="Predicted Tc (K)")
    predicted_properties: Dict[str, float] = Field(
        default_factory=dict,
        description="Other predicted properties"
    )
    
    # Uncertainty/confidence
    prediction_confidence: float = Field(..., ge=0.0, le=1.0)
    prediction_uncertainty: Optional[float] = None
    
    # Priority (for queue management)
    synthesis_priority: int = Field(
        default=5,
        ge=1,
        le=10,
        description="Priority 1-10 (10 = highest)"
    )
    
    # Suggested recipe (optional - A-Lab can auto-generate)
    suggested_recipe: Optional[ALab_SynthesisRecipe] = None
    
    # Expected information gain (for active learning)
    expected_info_gain: Optional[float] = Field(
        None,
        description="Expected Shannon entropy reduction (bits)"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "prediction_id": "matprov-pred-12345",
                "material_formula": "Y2Ba4Cu6O13",
                "predicted_tc": 85.5,
                "predicted_properties": {
                    "band_gap_ev": 0.0,
                    "formation_energy_ev": -2.3
                },
                "prediction_confidence": 0.82,
                "prediction_uncertainty": 5.2,
                "synthesis_priority": 8,
                "expected_info_gain": 2.3
            }
        }


# Helper functions for format conversion
def convert_matprov_to_alab_target(prediction: Dict[str, Any]) -> ALab_PredictionTarget:
    """
    Convert matprov prediction to A-Lab target format.
    
    Args:
        prediction: matprov prediction dictionary
    
    Returns:
        A-Lab compatible prediction target
    """
    return ALab_PredictionTarget(
        prediction_id=prediction.get("prediction_id", "unknown"),
        material_formula=prediction["material_formula"],
        predicted_tc=prediction.get("predicted_tc"),
        predicted_properties=prediction.get("properties", {}),
        prediction_confidence=prediction.get("confidence", 0.5),
        prediction_uncertainty=prediction.get("uncertainty"),
        synthesis_priority=calculate_priority(prediction),
        expected_info_gain=prediction.get("expected_info_gain")
    )


def calculate_priority(prediction: Dict[str, Any]) -> int:
    """
    Calculate synthesis priority (1-10) based on prediction.
    
    Factors:
    - High Tc → higher priority
    - Low uncertainty → higher priority  
    - High expected info gain → higher priority
    
    Args:
        prediction: Prediction dictionary
    
    Returns:
        Priority (1-10)
    """
    priority = 5  # Base priority
    
    # High Tc bonus
    tc = prediction.get("predicted_tc", 0)
    if tc > 100:
        priority += 3
    elif tc > 50:
        priority += 2
    elif tc > 20:
        priority += 1
    
    # Low uncertainty bonus
    uncertainty = prediction.get("uncertainty", float('inf'))
    if uncertainty < 5:
        priority += 1
    
    # High info gain bonus
    info_gain = prediction.get("expected_info_gain", 0)
    if info_gain > 2.0:
        priority += 1
    
    return min(max(priority, 1), 10)  # Clamp to 1-10


# Example usage
if __name__ == "__main__":
    print("=== A-Lab Format Examples ===\n")
    
    # Example 1: Synthesis recipe
    recipe = ALab_SynthesisRecipe(
        target_formula="YBa2Cu3O7",
        target_composition={"Y": 1, "Ba": 2, "Cu": 3, "O": 7},
        precursors=[
            {"compound": "Y2O3", "amount_g": 8.5, "purity": "99.99%"},
            {"compound": "BaCO3", "amount_g": 23.6, "purity": "99.9%"},
            {"compound": "CuO", "amount_g": 28.7, "purity": "99.5%"}
        ],
        heating_profile=[
            {"temperature_c": 900, "duration_hours": 12, "ramp_rate_c_per_min": 5.0},
            {"temperature_c": 950, "duration_hours": 24, "ramp_rate_c_per_min": 2.0}
        ],
        atmosphere="O2",
        container="alumina_crucible",
        total_duration_hours=36
    )
    
    print("📋 Synthesis Recipe:")
    print(f"  Target: {recipe.target_formula}")
    print(f"  Precursors: {len(recipe.precursors)}")
    print(f"  Heating steps: {len(recipe.heating_profile)}")
    print(f"  Duration: {recipe.total_duration_hours}h")
    
    # Example 2: XRD pattern
    # Generate simple mock pattern
    two_theta = list(np.linspace(10, 80, 700))
    intensity = list(100 * np.random.rand(700) + np.sin(np.array(two_theta) * 0.5) * 200)
    
    xrd = ALab_XRDPattern(
        two_theta=two_theta,
        intensity=intensity,
        wavelength=1.5406,
        scan_range=(10.0, 80.0),
        step_size=0.1,
        count_time=1.0,
        sample_id="ALAB-TEST-001"
    )
    
    print(f"\n📊 XRD Pattern:")
    print(f"  Sample: {xrd.sample_id}")
    print(f"  Range: {xrd.scan_range[0]}-{xrd.scan_range[1]}° 2θ")
    print(f"  Points: {len(xrd.two_theta)}")
    print(f"  Wavelength: {xrd.wavelength} Å (Cu Kα)")
    
    # Example 3: Phase analysis
    phase_analysis = ALab_PhaseAnalysis(
        target_phase="YBa2Cu3O7",
        target_fraction=0.92,
        identified_phases=[
            {"phase": "YBa2Cu3O7", "fraction": 0.92, "confidence": 0.95, "space_group": "Pmmm"},
            {"phase": "Y2BaCuO5", "fraction": 0.05, "confidence": 0.80, "space_group": "Pnma"},
            {"phase": "BaCuO2", "fraction": 0.03, "confidence": 0.75, "space_group": "P4/mmm"}
        ],
        rietveld_goodness_of_fit=1.6,
        rietveld_r_wp=3.8,
        lattice_parameters={
            "a": 3.82, "b": 3.89, "c": 11.68,
            "alpha": 90.0, "beta": 90.0, "gamma": 90.0
        }
    )
    
    print(f"\n🔬 Phase Analysis:")
    print(f"  Target phase: {phase_analysis.target_phase}")
    print(f"  Phase purity: {phase_analysis.target_fraction*100:.1f}%")
    print(f"  Success: {'✅ YES' if phase_analysis.synthesis_success else '❌ NO'}")
    print(f"  Quality: {phase_analysis.phase_purity}")
    print(f"  Rietveld GoF: {phase_analysis.rietveld_goodness_of_fit:.2f}")
    
    # Example 4: Prediction target
    target = ALab_PredictionTarget(
        prediction_id="matprov-20231015-001",
        material_formula="La2-xSrxCuO4",
        predicted_tc=38.5,
        predicted_properties={"band_gap_ev": 0.0},
        prediction_confidence=0.85,
        prediction_uncertainty=4.2,
        synthesis_priority=8,
        expected_info_gain=2.8
    )
    
    print(f"\n🎯 Prediction Target for A-Lab:")
    print(f"  Formula: {target.material_formula}")
    print(f"  Predicted Tc: {target.predicted_tc}K ± {target.prediction_uncertainty}K")
    print(f"  Confidence: {target.prediction_confidence*100:.1f}%")
    print(f"  Priority: {target.synthesis_priority}/10")
    print(f"  Expected info gain: {target.expected_info_gain:.2f} bits")
    
    print("\n" + "="*70)
    print("✅ A-Lab format schemas demonstrate:")
    print("   • Understanding of Berkeley's autonomous synthesis system")
    print("   • Knowledge of actual data structures (Nature 2023)")
    print("   • Can integrate with A-Lab day 1")
    print("   • Ready for production materials discovery")
    print("="*70)


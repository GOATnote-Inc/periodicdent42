"""
SQLAlchemy models for prediction registry database
"""

from datetime import datetime, timezone
from typing import Optional
from sqlalchemy import String, Float, Integer, DateTime, ForeignKey, Boolean, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Base class for all models"""
    pass


class Model(Base):
    """ML model metadata"""
    __tablename__ = "models"
    
    id: Mapped[int] = mapped_column(primary_key=True)
    model_id: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    version: Mapped[str] = mapped_column(String(50))
    checkpoint_hash: Mapped[str] = mapped_column(String(64), comment="DVC hash of model checkpoint")
    training_dataset_hash: Mapped[str] = mapped_column(String(64), comment="DVC hash of training dataset")
    architecture: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    hyperparameters: Mapped[Optional[str]] = mapped_column(Text, nullable=True, comment="JSON string")
    training_date: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))
    notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    
    # Relationships
    predictions: Mapped[list["Prediction"]] = relationship(back_populates="model", cascade="all, delete-orphan")
    
    def __repr__(self) -> str:
        return f"<Model(id={self.model_id}, version={self.version})>"


class Prediction(Base):
    """Model prediction for a material"""
    __tablename__ = "predictions"
    
    id: Mapped[int] = mapped_column(primary_key=True)
    prediction_id: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    model_id: Mapped[int] = mapped_column(ForeignKey("models.id"))
    
    # Material info
    material_formula: Mapped[str] = mapped_column(String(100), index=True)
    material_id: Mapped[Optional[str]] = mapped_column(String(100), nullable=True, comment="Materials Project ID or similar")
    
    # Prediction
    predicted_tc: Mapped[float] = mapped_column(Float, comment="Predicted critical temperature (K)")
    uncertainty: Mapped[Optional[float]] = mapped_column(Float, nullable=True, comment="Prediction uncertainty (K)")
    prediction_date: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc), index=True)
    
    # Additional predicted properties
    predicted_class: Mapped[Optional[str]] = mapped_column(String(50), nullable=True, comment="Classification (e.g., 'low_Tc', 'high_Tc')")
    confidence: Mapped[Optional[float]] = mapped_column(Float, nullable=True, comment="Confidence score (0-1)")
    
    # Relationships
    model: Mapped["Model"] = relationship(back_populates="predictions")
    outcome: Mapped[Optional["ExperimentOutcome"]] = relationship(back_populates="prediction", uselist=False)
    error: Mapped[Optional["PredictionError"]] = relationship(back_populates="prediction", uselist=False)
    
    def __repr__(self) -> str:
        return f"<Prediction(id={self.prediction_id}, formula={self.material_formula}, Tc={self.predicted_tc:.1f}K)>"


class ExperimentOutcome(Base):
    """Experimental validation of a prediction"""
    __tablename__ = "experiment_outcomes"
    
    id: Mapped[int] = mapped_column(primary_key=True)
    prediction_id: Mapped[int] = mapped_column(ForeignKey("predictions.id"), unique=True)
    experiment_id: Mapped[str] = mapped_column(String(255), index=True, comment="Reference to matprov experiment")
    
    # Outcome
    actual_tc: Mapped[Optional[float]] = mapped_column(Float, nullable=True, comment="Measured critical temperature (K)")
    validation_status: Mapped[str] = mapped_column(String(50), comment="success, failure, partial_success, inconclusive")
    phase_purity: Mapped[Optional[float]] = mapped_column(Float, nullable=True, comment="Phase purity (%)")
    
    # Timing
    experiment_date: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))
    validation_date: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    
    # Classification outcomes
    is_superconductor: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)
    tc_class_actual: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    
    notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    
    # Relationships
    prediction: Mapped["Prediction"] = relationship(back_populates="outcome")
    
    def __repr__(self) -> str:
        return f"<ExperimentOutcome(exp_id={self.experiment_id}, actual_Tc={self.actual_tc})>"


class PredictionError(Base):
    """Prediction error analysis"""
    __tablename__ = "prediction_errors"
    
    id: Mapped[int] = mapped_column(primary_key=True)
    prediction_id: Mapped[int] = mapped_column(ForeignKey("predictions.id"), unique=True)
    
    # Error metrics
    absolute_error: Mapped[float] = mapped_column(Float, comment="Predicted - Actual (K)")
    relative_error: Mapped[Optional[float]] = mapped_column(Float, nullable=True, comment="Relative error (%)")
    squared_error: Mapped[float] = mapped_column(Float, comment="(Predicted - Actual)^2")
    
    # Classification metrics (if applicable)
    is_true_positive: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)
    is_true_negative: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)
    is_false_positive: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)
    is_false_negative: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)
    
    # Metadata
    computed_date: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))
    
    # Relationships
    prediction: Mapped["Prediction"] = relationship(back_populates="error")
    
    def __repr__(self) -> str:
        return f"<PredictionError(abs_error={self.absolute_error:.2f}K)>"


# Example usage and schema documentation
if __name__ == "__main__":
    from sqlalchemy import create_engine
    from sqlalchemy.orm import Session
    
    # Create in-memory database for testing
    engine = create_engine("sqlite:///:memory:", echo=True)
    Base.metadata.create_all(engine)
    
    with Session(engine) as session:
        # Create a model
        model = Model(
            model_id="rf_v2.1_uci_21k",
            version="2.1.0",
            checkpoint_hash="dvc:abc123...",
            training_dataset_hash="dvc:3f34e6c71b4245aad0da5acc3d39fe7f",
            architecture="RandomForest",
            hyperparameters='{"n_estimators": 100, "max_depth": 20}',
            notes="Trained on UCI Superconductor dataset (21,263 samples)"
        )
        session.add(model)
        session.flush()
        
        # Create a prediction
        prediction = Prediction(
            prediction_id="PRED-20250108-001",
            model_id=model.id,
            material_formula="YBa2Cu3O7",
            predicted_tc=92.5,
            uncertainty=5.2,
            predicted_class="high_Tc",
            confidence=0.89
        )
        session.add(prediction)
        session.flush()
        
        # Add experimental outcome
        outcome = ExperimentOutcome(
            prediction_id=prediction.id,
            experiment_id="EXP-20250108-001",
            actual_tc=89.3,
            validation_status="success",
            phase_purity=92.3,
            is_superconductor=True,
            tc_class_actual="high_Tc"
        )
        session.add(outcome)
        session.flush()
        
        # Compute error
        error = PredictionError(
            prediction_id=prediction.id,
            absolute_error=prediction.predicted_tc - outcome.actual_tc,
            relative_error=((prediction.predicted_tc - outcome.actual_tc) / outcome.actual_tc) * 100,
            squared_error=(prediction.predicted_tc - outcome.actual_tc) ** 2,
            is_true_positive=True
        )
        session.add(error)
        
        session.commit()
        
        # Query
        print("\n=== Database Schema Created ===")
        print(f"✅ Model: {model}")
        print(f"✅ Prediction: {prediction}")
        print(f"✅ Outcome: {outcome}")
        print(f"✅ Error: {error}")
        print(f"\n✅ Absolute error: {error.absolute_error:.2f} K")


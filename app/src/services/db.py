"""
Database service for experiment tracking.

Uses Cloud SQL (PostgreSQL) for structured data.
"""

from sqlalchemy import create_engine, Column, String, DateTime, JSON, Float, Integer, Enum as SQLEnum
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from datetime import datetime
import logging
from typing import Optional, List
from enum import Enum

from src.utils.compliance import sanitize_payload
from src.utils.settings import settings

logger = logging.getLogger(__name__)

Base = declarative_base()


# Enums for status and method types
class ExperimentStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class OptimizationMethod(str, Enum):
    BAYESIAN_OPTIMIZATION = "bayesian_optimization"
    REINFORCEMENT_LEARNING = "reinforcement_learning"
    ADAPTIVE_ROUTER = "adaptive_router"


class ExperimentRun(Base):
    """
    Experiment run record for tracking and audit.
    """
    __tablename__ = "experiment_runs"
    
    id = Column(String, primary_key=True)
    query = Column(String, nullable=False)
    context = Column(JSON)
    flash_response = Column(JSON)
    pro_response = Column(JSON)
    flash_latency_ms = Column(Float)
    pro_latency_ms = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)
    user_id = Column(String, default="anonymous")  # TODO: Auth integration


class InstrumentRun(Base):
    """Hardware instrument run log for auditability."""

    __tablename__ = "instrument_runs"

    id = Column(String, primary_key=True)
    instrument_id = Column(String, nullable=False)
    sample_id = Column(String, nullable=False)
    campaign_id = Column(String, nullable=True)
    status = Column(String, default="completed")
    metadata_json = Column(JSON, default=dict)
    notes = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class Experiment(Base):
    """Individual experiment within an optimization run."""
    __tablename__ = "experiments"
    
    id = Column(String, primary_key=True)
    optimization_run_id = Column(String, nullable=True)  # NULL for standalone experiments
    method = Column(String, nullable=True)  # Which optimizer suggested this
    parameters = Column(JSON, nullable=False)  # Experimental parameters
    context = Column(JSON, nullable=True)  # Domain context
    noise_estimate = Column(Float, nullable=True)  # Estimated noise level
    results = Column(JSON, nullable=True)  # Measurement results
    status = Column(String, default="pending")  # pending, running, completed, failed
    start_time = Column(DateTime, nullable=True)
    end_time = Column(DateTime, nullable=True)
    error_message = Column(String, nullable=True)
    created_by = Column(String, default="anonymous")
    created_at = Column(DateTime, default=datetime.utcnow)


class OptimizationRun(Base):
    """Optimization campaign with multiple experiments."""
    __tablename__ = "optimization_runs"
    
    id = Column(String, primary_key=True)
    method = Column(String, nullable=False)  # RL, BO, or Adaptive
    context = Column(JSON, nullable=True)  # Optimization context
    status = Column(String, default="pending")
    start_time = Column(DateTime, nullable=True)
    end_time = Column(DateTime, nullable=True)
    error_message = Column(String, nullable=True)
    created_by = Column(String, default="anonymous")
    created_at = Column(DateTime, default=datetime.utcnow)


class AIQuery(Base):
    """AI model query tracking for cost analysis."""
    __tablename__ = "ai_queries"

    id = Column(String, primary_key=True)
    query = Column(String, nullable=False)
    context = Column(JSON, nullable=True)
    selected_model = Column(String, nullable=False)  # flash, pro, adaptive_router
    latency_ms = Column(Float, nullable=True)
    input_tokens = Column(Integer, nullable=True)
    output_tokens = Column(Integer, nullable=True)
    cost_usd = Column(Float, nullable=True)
    created_by = Column(String, default="anonymous")
    created_at = Column(DateTime, default=datetime.utcnow)


def get_database_url() -> str:
    """
    Construct database URL based on environment.
    
    For Cloud SQL, uses Unix socket or TCP depending on environment.
    """
    if settings.GCP_SQL_INSTANCE:
        # Cloud SQL Unix socket connection
        # Format: /cloudsql/PROJECT:REGION:INSTANCE
        socket_path = f"/cloudsql/{settings.GCP_SQL_INSTANCE}"
        return f"postgresql://{settings.DB_USER}:{settings.DB_PASSWORD}@/{settings.DB_NAME}?host={socket_path}"
    else:
        # Local PostgreSQL or TCP connection
        return f"postgresql://{settings.DB_USER}:{settings.DB_PASSWORD}@{settings.DB_HOST}:{settings.DB_PORT}/{settings.DB_NAME}"


# Global engine and session factory
_engine = None
_SessionLocal = None


def _isoformat(value: Optional[datetime]) -> Optional[str]:
    """Return an ISO formatted string when a datetime is provided."""

    return value.isoformat() if value else None


def experiment_to_dict(experiment: Experiment) -> dict:
    """Serialize an Experiment ORM object into a JSON-friendly dictionary."""

    return {
        "id": experiment.id,
        "optimization_run_id": experiment.optimization_run_id,
        "method": experiment.method,
        "parameters": experiment.parameters,
        "context": experiment.context,
        "noise_estimate": experiment.noise_estimate,
        "results": experiment.results,
        "status": experiment.status,
        "start_time": _isoformat(experiment.start_time),
        "end_time": _isoformat(experiment.end_time),
        "error_message": experiment.error_message,
        "created_by": experiment.created_by,
        "created_at": _isoformat(experiment.created_at),
    }


def optimization_run_to_dict(run: OptimizationRun) -> dict:
    """Serialize an OptimizationRun ORM object into a JSON-friendly dictionary."""

    return {
        "id": run.id,
        "method": run.method,
        "context": run.context,
        "status": run.status,
        "start_time": _isoformat(run.start_time),
        "end_time": _isoformat(run.end_time),
        "error_message": run.error_message,
        "created_by": run.created_by,
        "created_at": _isoformat(run.created_at),
    }


def ai_query_to_dict(query: AIQuery) -> dict:
    """Serialize an AIQuery ORM object into a JSON-friendly dictionary."""

    return {
        "id": query.id,
        "query": query.query,
        "context": query.context,
        "selected_model": query.selected_model,
        "latency_ms": query.latency_ms,
        "input_tokens": query.input_tokens,
        "output_tokens": query.output_tokens,
        "cost_usd": query.cost_usd,
        "created_by": query.created_by,
        "created_at": _isoformat(query.created_at),
    }


def init_database():
    """
    Initialize database connection and create tables.
    """
    global _engine, _SessionLocal
    
    try:
        if settings.DB_PASSWORD is None and settings.GCP_SQL_INSTANCE is None:
            logger.info("Database credentials not provided; skipping database initialization")
            return

        database_url = get_database_url()
        logger.info(f"Connecting to database...")
        
        _engine = create_engine(
            database_url,
            pool_size=5,
            max_overflow=10,
            pool_pre_ping=True,  # Verify connections before using
        )
        
        _SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=_engine)
        
        # Create tables
        Base.metadata.create_all(bind=_engine)
        
        logger.info("Database initialized successfully")
    
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        logger.warning("Continuing without database (data will not be persisted)")


def get_session() -> Optional[Session]:
    """
    Get database session.
    
    Returns:
        SQLAlchemy session or None if database not initialized
    """
    if _SessionLocal is None:
        return None
    
    return _SessionLocal()


def log_experiment_run(
    experiment_id: str,
    query: str,
    context: dict,
    flash_response: dict,
    pro_response: dict
) -> None:
    """
    Log experiment run to database.
    
    Args:
        experiment_id: Unique identifier
        query: User query
        context: Query context
        flash_response: Flash model response
        pro_response: Pro model response
    """
    session = get_session()
    if session is None:
        logger.warning("Database not available, skipping log")
        return
    
    try:
        run = ExperimentRun(
            id=experiment_id,
            query=sanitize_payload(query),
            context=sanitize_payload(context),
            flash_response=sanitize_payload(flash_response),
            pro_response=sanitize_payload(pro_response),
            flash_latency_ms=flash_response.get("latency_ms"),
            pro_latency_ms=pro_response.get("latency_ms"),
        )
        
        session.add(run)
        session.commit()
        
        logger.info(f"Logged experiment run: {experiment_id}")
    
    except Exception as e:
        logger.error(f"Failed to log experiment: {e}")
        session.rollback()

    finally:
        session.close()


def close_database():
    """
    Close database connections and dispose of engine.
    """
    global _engine, _SessionLocal
    if _engine:
        _engine.dispose()
        _engine = None
        _SessionLocal = None
        logger.info("Database connections closed")


def get_experiments(
    status: Optional[str] = None,
    optimization_run_id: Optional[str] = None,
    created_by: Optional[str] = None,
    limit: int = 100
) -> List[Experiment]:
    """
    Query experiments with optional filters.
    
    Args:
        status: Filter by status
        optimization_run_id: Filter by optimization run
        created_by: Filter by creator
        limit: Maximum number of results
        
    Returns:
        List of Experiment objects
    """
    session = get_session()
    if session is None:
        return []
    
    try:
        query = session.query(Experiment)
        
        if status:
            query = query.filter(Experiment.status == status)
        if optimization_run_id:
            query = query.filter(Experiment.optimization_run_id == optimization_run_id)
        if created_by:
            query = query.filter(Experiment.created_by == created_by)
        
        return query.order_by(Experiment.created_at.desc()).limit(limit).all()
    
    finally:
        session.close()


def get_optimization_runs(
    status: Optional[str] = None,
    method: Optional[str] = None,
    created_by: Optional[str] = None,
    limit: int = 50
) -> List[OptimizationRun]:
    """
    Query optimization runs with optional filters.
    
    Args:
        status: Filter by status
        method: Filter by optimization method
        created_by: Filter by creator
        limit: Maximum number of results
        
    Returns:
        List of OptimizationRun objects
    """
    session = get_session()
    if session is None:
        return []
    
    try:
        query = session.query(OptimizationRun)
        
        if status:
            query = query.filter(OptimizationRun.status == status)
        if method:
            query = query.filter(OptimizationRun.method == method)
        if created_by:
            query = query.filter(OptimizationRun.created_by == created_by)
        
        return query.order_by(OptimizationRun.created_at.desc()).limit(limit).all()
    
    finally:
        session.close()


def log_instrument_run(
    *,
    run_id: str,
    instrument_id: str,
    sample_id: str,
    campaign_id: Optional[str],
    status: str,
    notes: Optional[str],
    metadata: Optional[dict] = None,
) -> None:
    """Persist hardware campaign run information."""

    session = get_session()
    if session is None:
        logger.warning("Database not available, skipping instrument log")
        return

    try:
        run = InstrumentRun(
            id=run_id,
            instrument_id=instrument_id,
            sample_id=sample_id,
            campaign_id=campaign_id,
            status=status,
            notes=notes,
            metadata_json=sanitize_payload(metadata or {}),
        )
        session.add(run)
        session.commit()
    except Exception as exc:  # pragma: no cover - defensive
        logger.error(f"Failed to log instrument run: {exc}")
        session.rollback()
    finally:
        session.close()


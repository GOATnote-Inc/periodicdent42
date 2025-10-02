"""
Database service for experiment tracking and metadata persistence.

Uses Cloud SQL (PostgreSQL) for structured data with proper schema design.
"""

from sqlalchemy import (
    create_engine, Column, String, DateTime, JSON, Float, Integer,
    Boolean, Text, ForeignKey, Index, Enum as SQLEnum
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session, relationship
from datetime import datetime
import logging
from typing import Optional, List, Dict, Any
import enum

from src.utils.settings import settings

logger = logging.getLogger(__name__)

Base = declarative_base()


# Enums
class ExperimentStatus(str, enum.Enum):
    """Experiment status enumeration."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class OptimizationMethod(str, enum.Enum):
    """Optimization method enumeration."""
    BAYESIAN_OPTIMIZATION = "bayesian_optimization"
    REINFORCEMENT_LEARNING = "reinforcement_learning"
    RANDOM_SEARCH = "random_search"
    GRID_SEARCH = "grid_search"
    ADAPTIVE_ROUTER = "adaptive_router"
    HYBRID = "hybrid"


# Database Models
class Experiment(Base):
    """
    Experiment metadata and configuration.
    
    An experiment represents a single physical measurement or simulation run.
    """
    __tablename__ = "experiments"
    
    id = Column(String(255), primary_key=True)
    name = Column(String(255), nullable=True)
    description = Column(Text, nullable=True)
    status = Column(SQLEnum(ExperimentStatus), default=ExperimentStatus.PENDING, nullable=False)
    
    # Configuration
    parameters = Column(JSON, nullable=True)  # Input parameters
    config = Column(JSON, nullable=True)  # Experiment configuration
    
    # Results
    result_value = Column(Float, nullable=True)  # Primary objective value
    result_data = Column(JSON, nullable=True)  # Full result data
    result_uri = Column(String(512), nullable=True)  # GCS URI for large results
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    created_by = Column(String(255), default="anonymous")
    
    # Foreign keys
    optimization_run_id = Column(String(255), ForeignKey("optimization_runs.id"), nullable=True)
    
    # Relationships
    optimization_run = relationship("OptimizationRun", back_populates="experiments")
    
    # Indexes for common queries
    __table_args__ = (
        Index('idx_experiments_status', 'status'),
        Index('idx_experiments_created_at', 'created_at'),
        Index('idx_experiments_optimization_run', 'optimization_run_id'),
    )


class OptimizationRun(Base):
    """
    Optimization run tracking.
    
    An optimization run consists of multiple experiments guided by an optimization algorithm.
    """
    __tablename__ = "optimization_runs"
    
    id = Column(String(255), primary_key=True)
    name = Column(String(255), nullable=True)
    description = Column(Text, nullable=True)
    
    # Optimization configuration
    method = Column(SQLEnum(OptimizationMethod), nullable=False)
    objective = Column(String(255), nullable=False)  # "minimize" or "maximize"
    search_space = Column(JSON, nullable=False)  # Parameter bounds/constraints
    config = Column(JSON, nullable=True)  # Algorithm-specific config
    
    # Progress tracking
    num_experiments = Column(Integer, default=0)
    best_value = Column(Float, nullable=True)
    best_experiment_id = Column(String(255), nullable=True)
    
    # Status
    status = Column(SQLEnum(ExperimentStatus), default=ExperimentStatus.PENDING, nullable=False)
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    created_by = Column(String(255), default="anonymous")
    
    # Relationships
    experiments = relationship("Experiment", back_populates="optimization_run")
    
    # Indexes
    __table_args__ = (
        Index('idx_optimization_runs_method', 'method'),
        Index('idx_optimization_runs_status', 'status'),
        Index('idx_optimization_runs_created_at', 'created_at'),
    )


class AIQuery(Base):
    """
    AI reasoning query tracking.
    
    Tracks queries to Gemini models for AI-driven experiment design and reasoning.
    """
    __tablename__ = "ai_queries"
    
    id = Column(String(255), primary_key=True)
    query_text = Column(Text, nullable=False)
    context = Column(JSON, nullable=True)
    
    # Model responses
    flash_response = Column(JSON, nullable=True)
    pro_response = Column(JSON, nullable=True)
    selected_model = Column(String(50), nullable=True)  # "flash" or "pro"
    
    # Performance metrics
    flash_latency_ms = Column(Float, nullable=True)
    pro_latency_ms = Column(Float, nullable=True)
    flash_tokens = Column(Integer, nullable=True)
    pro_tokens = Column(Integer, nullable=True)
    
    # Cost tracking
    estimated_cost_usd = Column(Float, nullable=True)
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    created_by = Column(String(255), default="anonymous")
    
    # Relationships
    experiment_id = Column(String(255), ForeignKey("experiments.id"), nullable=True)
    experiment = relationship("Experiment")
    
    # Indexes
    __table_args__ = (
        Index('idx_ai_queries_created_at', 'created_at'),
        Index('idx_ai_queries_selected_model', 'selected_model'),
    )


class NoiseEstimate(Base):
    """
    Noise estimation tracking for adaptive routing.
    
    Stores noise estimates from pilot experiments for adaptive algorithm selection.
    """
    __tablename__ = "noise_estimates"
    
    id = Column(String(255), primary_key=True)
    
    # Estimation method
    method = Column(String(100), nullable=False)  # "replicate_pooled", "residuals", "sequential"
    
    # Results
    noise_std = Column(Float, nullable=False)
    confidence_interval_lower = Column(Float, nullable=True)
    confidence_interval_upper = Column(Float, nullable=True)
    sample_size = Column(Integer, nullable=False)
    reliable = Column(Boolean, default=False)
    
    # Context
    pilot_data = Column(JSON, nullable=True)  # Store pilot experiment data
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    optimization_run_id = Column(String(255), ForeignKey("optimization_runs.id"), nullable=True)
    
    # Indexes
    __table_args__ = (
        Index('idx_noise_estimates_created_at', 'created_at'),
        Index('idx_noise_estimates_method', 'method'),
    )


class RoutingDecision(Base):
    """
    Adaptive routing decisions.
    
    Tracks decisions made by the adaptive router (BO vs RL selection).
    """
    __tablename__ = "routing_decisions"
    
    id = Column(String(255), primary_key=True)
    
    # Decision
    selected_method = Column(SQLEnum(OptimizationMethod), nullable=False)
    confidence = Column(Float, nullable=False)  # 0.0 to 1.0
    reasoning = Column(Text, nullable=True)
    threshold_used = Column(Float, nullable=True)
    
    # Alternative methods considered
    alternatives = Column(JSON, nullable=True)  # {method: score}
    warnings = Column(JSON, nullable=True)  # List of warning messages
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    noise_estimate_id = Column(String(255), ForeignKey("noise_estimates.id"), nullable=True)
    optimization_run_id = Column(String(255), ForeignKey("optimization_runs.id"), nullable=True)
    
    # Indexes
    __table_args__ = (
        Index('idx_routing_decisions_method', 'selected_method'),
        Index('idx_routing_decisions_created_at', 'created_at'),
    )


# Database connection management
_engine = None
_SessionLocal = None


def get_database_url() -> str:
    """
    Construct database URL based on environment.
    
    For Cloud SQL, uses Unix socket or TCP depending on environment.
    For local development, uses standard PostgreSQL connection.
    """
    if settings.GCP_SQL_INSTANCE:
        # Cloud SQL Unix socket connection (Cloud Run)
        # Format: /cloudsql/PROJECT:REGION:INSTANCE
        socket_path = f"/cloudsql/{settings.GCP_SQL_INSTANCE}"
        return f"postgresql://{settings.DB_USER}:{settings.DB_PASSWORD}@/{settings.DB_NAME}?host={socket_path}"
    else:
        # Local PostgreSQL or TCP connection
        return f"postgresql://{settings.DB_USER}:{settings.DB_PASSWORD}@{settings.DB_HOST}:{settings.DB_PORT}/{settings.DB_NAME}"


def init_database():
    """
    Initialize database connection and create tables.
    
    This should be called at application startup.
    """
    global _engine, _SessionLocal
    
    if not settings.DB_USER or not settings.DB_PASSWORD or not settings.DB_NAME:
        logger.warning("Database credentials not configured, skipping database initialization")
        return
    
    try:
        database_url = get_database_url()
        logger.info("Connecting to Cloud SQL database...")
        
        _engine = create_engine(
            database_url,
            pool_size=5,
            max_overflow=10,
            pool_pre_ping=True,  # Verify connections before using
            echo=False,  # Set to True for SQL query logging
        )
        
        _SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=_engine)
        
        # Create tables (use Alembic migrations in production)
        Base.metadata.create_all(bind=_engine)
        
        logger.info("✅ Cloud SQL database initialized successfully")
    
    except Exception as e:
        logger.error(f"❌ Database initialization failed: {e}")
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


def close_database():
    """
    Close database connections.
    
    Should be called at application shutdown.
    """
    global _engine, _SessionLocal
    
    if _engine:
        _engine.dispose()
        logger.info("Database connections closed")
    
    _engine = None
    _SessionLocal = None


# Database operations
def create_experiment(
    experiment_id: str,
    parameters: Dict[str, Any],
    config: Optional[Dict[str, Any]] = None,
    optimization_run_id: Optional[str] = None,
    created_by: str = "anonymous"
) -> Optional[Experiment]:
    """
    Create a new experiment record.
    
    Args:
        experiment_id: Unique experiment identifier
        parameters: Experiment parameters (input)
        config: Experiment configuration
        optimization_run_id: Associated optimization run (if any)
        created_by: User identifier
    
    Returns:
        Experiment object or None if database not available
    """
    session = get_session()
    if session is None:
        logger.warning("Database not available, skipping experiment creation")
        return None
    
    try:
        experiment = Experiment(
            id=experiment_id,
            parameters=parameters,
            config=config,
            optimization_run_id=optimization_run_id,
            created_by=created_by,
            status=ExperimentStatus.PENDING
        )
        
        session.add(experiment)
        session.commit()
        session.refresh(experiment)
        
        logger.info(f"Created experiment: {experiment_id}")
        return experiment
    
    except Exception as e:
        logger.error(f"Failed to create experiment: {e}")
        session.rollback()
        return None
    
    finally:
        session.close()


def update_experiment(
    experiment_id: str,
    status: Optional[ExperimentStatus] = None,
    result_value: Optional[float] = None,
    result_data: Optional[Dict[str, Any]] = None,
    result_uri: Optional[str] = None
) -> bool:
    """
    Update experiment with results.
    
    Args:
        experiment_id: Experiment identifier
        status: New status
        result_value: Primary objective value
        result_data: Full result data
        result_uri: GCS URI for large results
    
    Returns:
        True if successful, False otherwise
    """
    session = get_session()
    if session is None:
        return False
    
    try:
        experiment = session.query(Experiment).filter(Experiment.id == experiment_id).first()
        if not experiment:
            logger.warning(f"Experiment not found: {experiment_id}")
            return False
        
        if status:
            experiment.status = status
            if status == ExperimentStatus.RUNNING and not experiment.started_at:
                experiment.started_at = datetime.utcnow()
            elif status in (ExperimentStatus.COMPLETED, ExperimentStatus.FAILED):
                experiment.completed_at = datetime.utcnow()
        
        if result_value is not None:
            experiment.result_value = result_value
        
        if result_data is not None:
            experiment.result_data = result_data
        
        if result_uri:
            experiment.result_uri = result_uri
        
        session.commit()
        logger.info(f"Updated experiment: {experiment_id}")
        return True
    
    except Exception as e:
        logger.error(f"Failed to update experiment: {e}")
        session.rollback()
        return False
    
    finally:
        session.close()


def log_ai_query(
    query_id: str,
    query_text: str,
    context: Dict[str, Any],
    flash_response: Optional[Dict[str, Any]] = None,
    pro_response: Optional[Dict[str, Any]] = None,
    selected_model: Optional[str] = None,
    created_by: str = "anonymous"
) -> bool:
    """
    Log AI query to database.
    
    Args:
        query_id: Unique query identifier
        query_text: User query
        context: Query context
        flash_response: Flash model response
        pro_response: Pro model response
        selected_model: Which model was used ("flash" or "pro")
        created_by: User identifier
    
    Returns:
        True if successful, False otherwise
    """
    session = get_session()
    if session is None:
        logger.warning("Database not available, skipping AI query log")
        return False
    
    try:
        query = AIQuery(
            id=query_id,
            query_text=query_text,
            context=context,
            flash_response=flash_response,
            pro_response=pro_response,
            selected_model=selected_model,
            flash_latency_ms=flash_response.get("latency_ms") if flash_response else None,
            pro_latency_ms=pro_response.get("latency_ms") if pro_response else None,
            flash_tokens=flash_response.get("tokens") if flash_response else None,
            pro_tokens=pro_response.get("tokens") if pro_response else None,
            created_by=created_by,
        )
        
        session.add(query)
        session.commit()
        
        logger.info(f"Logged AI query: {query_id}")
        return True
    
    except Exception as e:
        logger.error(f"Failed to log AI query: {e}")
        session.rollback()
        return False
    
    finally:
        session.close()


def get_experiments(
    optimization_run_id: Optional[str] = None,
    status: Optional[ExperimentStatus] = None,
    limit: int = 100
) -> List[Experiment]:
    """
    Query experiments from database.
    
    Args:
        optimization_run_id: Filter by optimization run
        status: Filter by status
        limit: Maximum number of results
    
    Returns:
        List of Experiment objects
    """
    session = get_session()
    if session is None:
        return []
    
    try:
        query = session.query(Experiment)
        
        if optimization_run_id:
            query = query.filter(Experiment.optimization_run_id == optimization_run_id)
        
        if status:
            query = query.filter(Experiment.status == status)
        
        query = query.order_by(Experiment.created_at.desc()).limit(limit)
        
        return query.all()
    
    except Exception as e:
        logger.error(f"Failed to query experiments: {e}")
        return []
    
    finally:
        session.close()


def get_optimization_runs(
    status: Optional[ExperimentStatus] = None,
    method: Optional[OptimizationMethod] = None,
    limit: int = 50
) -> List[OptimizationRun]:
    """
    Query optimization runs from database.
    
    Args:
        status: Filter by status
        method: Filter by optimization method
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
        
        query = query.order_by(OptimizationRun.created_at.desc()).limit(limit)
        
        return query.all()
    
    except Exception as e:
        logger.error(f"Failed to query optimization runs: {e}")
        return []
    
    finally:
        session.close()

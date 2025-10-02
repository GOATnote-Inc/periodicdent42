"""
Database service for experiment tracking.

Uses Cloud SQL (PostgreSQL) for structured data.
"""

from sqlalchemy import create_engine, Column, String, DateTime, JSON, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from datetime import datetime
import logging
from typing import Optional

from src.utils.compliance import sanitize_payload
from src.utils.settings import settings

logger = logging.getLogger(__name__)

Base = declarative_base()


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


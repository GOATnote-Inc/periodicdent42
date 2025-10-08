"""
Database Configuration for API

SQLAlchemy setup for the FastAPI service.
"""

from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
import os

# Database URL from environment variable
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "sqlite:///./matprov_api.db"  # Default to SQLite for development
)

# Create engine
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {},
    pool_pre_ping=True,  # Enable connection health checks
    pool_recycle=3600,  # Recycle connections after 1 hour
)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for models
Base = declarative_base()


def get_db() -> Session:
    """
    Dependency for getting database session
    
    Usage:
        @app.get("/users/me")
        def read_users_me(db: Session = Depends(get_db)):
            ...
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """Initialize database (create all tables)"""
    # Import all models here to ensure they're registered
    from .models.user import User
    
    Base.metadata.create_all(bind=engine)
    print("✅ Database initialized")

